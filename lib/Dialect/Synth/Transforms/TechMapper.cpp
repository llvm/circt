//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the TechMapper pass, which performs technology mapping
// by converting logic network representations (AIG operations) into
// technology-specific gate implementations using cut-based rewriting.
//
// The pass uses a cut-based algorithm with priority cuts and NPN canonical
// forms for efficient pattern matching. It processes HWModuleOp instances with
// "synth.mapping_cost" attributes as technology library patterns and maps
// non-library modules to optimal gate implementations based on area and timing
// optimization strategies.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthAttributes.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/WalkResult.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <atomic>
#include <tuple>

namespace circt {
namespace synth {
#define GEN_PASS_DEF_TECHMAPPER
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;

#define DEBUG_TYPE "synth-tech-mapper"

//===----------------------------------------------------------------------===//
// Tech Mapper Pass
//===----------------------------------------------------------------------===//

static llvm::FailureOr<NPNClass> getNPNClassFromModule(hw::HWModuleOp module) {
  // Get input and output ports
  auto inputTypes = module.getInputTypes();
  auto outputTypes = module.getOutputTypes();

  unsigned numInputs = inputTypes.size();
  unsigned numOutputs = outputTypes.size();
  if (numOutputs != 1)
    return module->emitError(
        "Modules with multiple outputs are not supported yet");

  // Verify all ports are single bit
  for (auto type : inputTypes) {
    if (!type.isInteger(1))
      return module->emitError("All input ports must be single bit");
  }
  for (auto type : outputTypes) {
    if (!type.isInteger(1))
      return module->emitError("All output ports must be single bit");
  }

  if (numInputs > maxTruthTableInputs)
    return module->emitError("Too many inputs for truth table generation");

  SmallVector<Value> results;
  results.reserve(numOutputs);
  // Get the body block of the module
  auto *bodyBlock = module.getBodyBlock();
  assert(bodyBlock && "Module must have a body block");
  // Collect output values from the body block
  for (auto result : bodyBlock->getTerminator()->getOperands())
    results.push_back(result);

  // Create a truth table for the module
  FailureOr<BinaryTruthTable> truthTable = getTruthTable(results, bodyBlock);
  if (failed(truthTable))
    return failure();

  return NPNClass::computeNPNCanonicalForm(*truthTable);
}

/// An inverter available for local input/output phase adjustments.
struct Inverter {
  hw::HWModuleOp module;
  double area;
  DelayType delay;
};

/// The exact realization priced by a phase-adjusted library match. Inverters
/// are local to this match: each selected inverter is charged and instantiated
/// once, without assuming sharing with other cuts.
struct TechMappingImplementation : public PatternImplementation {
  SmallVector<hw::HWModuleOp, 6> inputInverters;
  hw::HWModuleOp outputInverter;
};

/// Simple technology library encoded as a HWModuleOp.
struct TechLibraryPattern : public CutRewritePattern {
  TechLibraryPattern(hw::HWModuleOp module, double area,
                     SmallVector<DelayType> delay, NPNClass npnClass,
                     const SmallVector<Inverter> &inverters)
      : CutRewritePattern(module->getContext()), area(area),
        delay(std::move(delay)), module(module), npnClass(std::move(npnClass)),
        inverters(inverters) {

    LLVM_DEBUG({
      llvm::dbgs() << "Created Tech Library Pattern for module: "
                   << module.getModuleName() << "\n"
                   << "NPN Class: " << this->npnClass.truthTable.table << "\n"
                   << "Inputs: " << this->npnClass.inputPermutation.size()
                   << "\n"
                   << "Input Negation: " << this->npnClass.inputNegation << "\n"
                   << "Output Negation: " << this->npnClass.outputNegation
                   << "\n";
    });
  }

  StringRef getPatternName() const override {
    auto moduleCp = module;
    return moduleCp.getModuleName();
  }

  /// Match the cut set against this library primitive
  std::optional<MatchResult> match(CutEnumerator &enumerator, const Cut &cut,
                                   const MatchBinding &binding) const override {
    const auto &cutNPN = cut.getNPNClass(enumerator.getOptions().npnTable);
    if (!(cutNPN.truthTable == npnClass.truthTable))
      return std::nullopt;
    if (!binding.hasNegation())
      return MatchResult(area, delay);
    // The per-input delay model cannot represent an output inverter on a
    // zero-input constant cell. Keep such matches on the direct path above.
    if (inverters.empty() || cut.inputs.empty())
      return std::nullopt;

    auto implementation = std::make_shared<TechMappingImplementation>();
    implementation->inputInverters.resize(cut.getInputSize());
    bool invertedOutput = binding.isOutputNegated(0);

    const Inverter *cheapest = &inverters.front();
    const Inverter *fastest = cheapest;
    for (const auto &inverter : inverters) {
      if (std::tie(inverter.area, inverter.delay) <
          std::tie(cheapest->area, cheapest->delay))
        cheapest = &inverter;
      if (std::tie(inverter.delay, inverter.area) <
          std::tie(fastest->delay, fastest->area))
        fastest = &inverter;
    }

    bool optimizeTiming =
        enumerator.getOptions().strategy == OptimizationStrategyTiming;
    const Inverter *outputInverter = optimizeTiming ? fastest : cheapest;
    DelayType outputDelay = invertedOutput ? outputInverter->delay : 0;
    double totalArea = area;
    if (invertedOutput) {
      implementation->outputInverter = outputInverter->module;
      totalArea += outputInverter->area;
    }

    // Find the fastest achievable arrival at the cell output. For timing,
    // choose the cheapest inverter on each pin that preserves this arrival,
    // including pins with slack. This avoids enumerating combinations of
    // inverter drive strengths.
    SmallVector<DelayType, 6> inputArrivalTimes;
    DelayType fastestArrival = 0;
    if (optimizeTiming) {
      if (failed(cut.getInputArrivalTimes(enumerator, inputArrivalTimes)))
        return std::nullopt;
      for (unsigned pin = 0; pin < cut.getInputSize(); ++pin)
        fastestArrival = std::max(
            fastestArrival,
            inputArrivalTimes[binding.getCutInputForPatternInput(pin)] +
                delay[pin] +
                (binding.isPatternInputNegated(pin) ? fastest->delay : 0));
    }

    SmallVector<DelayType, 6> delays(delay.begin(), delay.end());
    for (unsigned pin = 0; pin < cut.getInputSize(); ++pin) {
      if (binding.isPatternInputNegated(pin)) {
        const Inverter *selected = cheapest;
        if (optimizeTiming) {
          selected = fastest;
          DelayType available =
              fastestArrival -
              inputArrivalTimes[binding.getCutInputForPatternInput(pin)] -
              delay[pin];
          for (const auto &inverter : inverters)
            if (inverter.delay <= available &&
                std::tie(inverter.area, inverter.delay) <
                    std::tie(selected->area, selected->delay))
              selected = &inverter;
        }
        implementation->inputInverters[pin] = selected->module;
        totalArea += selected->area;
        delays[pin] += selected->delay;
      }
      delays[pin] += outputDelay;
    }
    MatchResult result;
    result.area = totalArea;
    result.setOwnedDelays(std::move(delays));
    result.setImplementation(std::move(implementation));
    return result;
  }

  /// Enable truth table matching for this pattern
  bool useTruthTableMatcher(
      SmallVectorImpl<NPNClass> &matchingNPNClasses) const override {
    matchingNPNClasses.push_back(npnClass);
    return true;
  }

  /// Rewrite the cut set using this library primitive
  llvm::FailureOr<Operation *>
  rewrite(mlir::OpBuilder &builder, CutEnumerator &enumerator, const Cut &cut,
          const MatchedPattern &matched) const override {
    const auto &network = enumerator.getLogicNetwork();
    const MatchBinding &binding = matched.getBinding();
    auto *implementation = static_cast<const TechMappingImplementation *>(
        matched.getImplementation());

    auto *rootOp = network.getGate(cut.getRootIndex()).getOperation();
    assert(rootOp && "cut root must be a valid operation");
    auto createInstance = [&](hw::HWModuleOp cell, ArrayRef<Value> inputs) {
      // TODO: Give a better name to the instance.
      return hw::InstanceOp::create(builder, rootOp->getLoc(), cell, "mapped",
                                    inputs);
    };

    SmallVector<Value> inputs;
    inputs.reserve(binding.getNumInputs());
    for (unsigned pin = 0; pin < binding.getNumInputs(); ++pin) {
      unsigned idx = binding.getCutInputForPatternInput(pin);
      assert(idx < cut.inputs.size() && "input permutation index out of range");
      Value input = network.getValue(cut.inputs[idx]);
      if (implementation && implementation->inputInverters[pin])
        input = createInstance(implementation->inputInverters[pin], input)
                    .getResult(0);
      inputs.push_back(input);
    }

    auto instanceOp = createInstance(module, inputs);
    if (implementation && implementation->outputInverter)
      instanceOp = createInstance(implementation->outputInverter,
                                  instanceOp.getResult(0));
    return instanceOp.getOperation();
  }

  unsigned getNumInputs() const {
    return static_cast<hw::HWModuleOp>(module).getNumInputPorts();
  }

  unsigned getNumOutputs() const override {
    return static_cast<hw::HWModuleOp>(module).getNumOutputPorts();
  }

  LocationAttr getLoc() const override {
    auto module = this->module;
    return module.getLoc();
  }

private:
  const double area;
  const SmallVector<DelayType> delay;
  hw::HWModuleOp module;
  NPNClass npnClass;
  const SmallVector<Inverter> &inverters;
};

namespace {
struct TechMapperPass : public impl::TechMapperBase<TechMapperPass> {
  using TechMapperBase<TechMapperPass>::TechMapperBase;

  LogicalResult initialize(MLIRContext *context) override {
    (void)context;
    npnTable = std::make_shared<const NPNTable>();
    return success();
  }

  void runOnOperation() override {
    auto module = getOperation();

    SmallVector<std::unique_ptr<CutRewritePattern>> libraryPatterns;

    // Patterns refer to this vector, which is complete before matching starts
    // and remains immutable throughout the parallel rewrite.
    SmallVector<Inverter> inverters;
    auto inverterClass =
        NPNClass::computeNPNCanonicalForm(BinaryTruthTable(1, 1, APInt(2, 1)));

    unsigned maxInputSize = 0;
    // Consider modules with the "synth.mapping_cost" attribute as library
    // modules.
    SmallVector<hw::HWModuleOp> nonLibraryModules;
    for (auto hwModule : module.getOps<hw::HWModuleOp>()) {

      auto mappingCost =
          hwModule->getAttrOfType<MappingCostAttr>("synth.mapping_cost");
      if (!mappingCost) {
        nonLibraryModules.push_back(hwModule);
        continue;
      }

      double area = mappingCost.getArea().getValue().convertToDouble();

      StringAttr outputName;
      hw::ModulePortInfo ports(hwModule.getPortList());
      for (const auto &port : ports.getOutputs()) {
        if (outputName) {
          hwModule.emitError(
              "Modules with multiple outputs are not supported yet");
          signalPassFailure();
          return;
        }
        outputName = port.name;
      }
      if (!outputName) {
        hwModule.emitError("expected library module to have an output");
        signalPassFailure();
        return;
      }

      auto arcs = mappingCost.getArcs();

      SmallVector<hw::PortInfo> inputPorts;
      for (const auto &port : hwModule.getPortList()) {
        if (!port.isInput())
          continue;
        inputPorts.push_back(port);
      }

      if (arcs.size() != inputPorts.size()) {
        hwModule.emitError(
            "synth.mapping_cost arcs do not match module inputs");
        signalPassFailure();
        return;
      }

      SmallVector<DelayType> delay;
      for (auto attr : arcs) {
        auto arc = cast<LinearTimingArcAttr>(attr);

        // TechMapper currently preserves the old integer per-pin delay model.
        // The sensitivity, polarity, and input capacitance fields are carried
        // in the attribute for future load-aware mapping.
        delay.push_back(static_cast<DelayType>(arc.getIntrinsic()));
      }

      // Compute NPN Class for the module.
      auto npnClass = getNPNClassFromModule(hwModule);
      if (failed(npnClass)) {
        signalPassFailure();
        return;
      }

      if (npnClass->equivalentOtherThanPermutation(inverterClass))
        inverters.push_back({hwModule, area, delay.front()});

      // Create a CutRewritePattern for the library module
      std::unique_ptr<TechLibraryPattern> pattern =
          std::make_unique<TechLibraryPattern>(hwModule, area, std::move(delay),
                                               std::move(*npnClass), inverters);

      // Update the maximum input size
      maxInputSize = std::max(maxInputSize, pattern->getNumInputs());

      // Add the pattern to the library
      libraryPatterns.push_back(std::move(pattern));
    }

    if (libraryPatterns.empty())
      return markAllAnalysesPreserved();

    CutRewritePatternSet patternSet(std::move(libraryPatterns));
    CutRewriterOptions options;
    options.strategy = strategy;
    options.maxCutInputSize = maxInputSize;
    options.maxCutSizePerRoot = maxCutsPerRoot;
    options.attachDebugTiming = test;
    options.npnTable = npnTable.get();
    std::atomic<uint64_t> numCutsCreatedCount = 0;
    std::atomic<uint64_t> numCutSetsCreatedCount = 0;
    std::atomic<uint64_t> numCutsRewrittenCount = 0;
    auto result = mlir::failableParallelForEach(
        module.getContext(), nonLibraryModules, [&](hw::HWModuleOp hwModule) {
          LLVM_DEBUG(llvm::dbgs() << "Processing non-library module: "
                                  << hwModule.getName() << "\n");
          CutRewriter rewriter(options, patternSet);
          if (failed(rewriter.run(hwModule)))
            return failure();
          const auto &stats = rewriter.getStats();
          numCutsCreatedCount.fetch_add(stats.numCutsCreated,
                                        std::memory_order_relaxed);
          numCutSetsCreatedCount.fetch_add(stats.numCutSetsCreated,
                                           std::memory_order_relaxed);
          numCutsRewrittenCount.fetch_add(stats.numCutsRewritten,
                                          std::memory_order_relaxed);
          return success();
        });
    if (failed(result))
      signalPassFailure();
    numCutsCreated += numCutsCreatedCount;
    numCutSetsCreated += numCutSetsCreatedCount;
    numCutsRewritten += numCutsRewrittenCount;
  }

private:
  std::shared_ptr<const NPNTable> npnTable;
};

} // namespace
