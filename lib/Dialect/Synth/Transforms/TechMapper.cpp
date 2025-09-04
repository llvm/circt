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
// "hw.techlib.info" attributes as technology library patterns and maps
// non-library modules to optimal gate implementations based on area and timing
// optimization strategies.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/WalkResult.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

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

/// Simple technology library encoded as a HWModuleOp.
struct TechLibraryPattern : public CutRewritePattern {
  TechLibraryPattern(hw::HWModuleOp module, double area,
                     SmallVector<SmallVector<DelayType, 2>, 4> delay,
                     NPNClass npnClass)
      : CutRewritePattern(module->getContext()), area(area),
        delay(std::move(delay)), module(module), npnClass(std::move(npnClass)) {

    LLVM_DEBUG({
      llvm::dbgs() << "Created Tech Library Pattern for module: "
                   << module.getModuleName() << "\n"
                   << "NPN Class: " << npnClass.truthTable.table << "\n"
                   << "Inputs: " << npnClass.inputPermutation.size() << "\n"
                   << "Input Negation: " << npnClass.inputNegation << "\n"
                   << "Output Negation: " << npnClass.outputNegation << "\n";
    });
  }

  StringRef getPatternName() const override {
    auto moduleCp = module;
    return moduleCp.getModuleName();
  }

  /// Match the cut set against this library primitive
  bool match(const Cut &cut) const override {
    return cut.getNPNClass().equivalentOtherThanPermutation(npnClass);
  }

  /// Enable truth table matching for this pattern
  bool useTruthTableMatcher(
      SmallVectorImpl<NPNClass> &matchingNPNClasses) const override {
    matchingNPNClasses.push_back(npnClass);
    return true;
  }

  /// Rewrite the cut set using this library primitive
  llvm::FailureOr<Operation *> rewrite(mlir::OpBuilder &builder,
                                       Cut &cut) const override {
    // Create a new instance of the module
    SmallVector<Value> inputs;
    cut.getPermutatedInputs(npnClass, inputs);

    // TODO: Give a better name to the instance
    auto instanceOp = builder.create<hw::InstanceOp>(
        cut.getRoot()->getLoc(), module, "mapped", ArrayRef<Value>(inputs));
    return instanceOp.getOperation();
  }

  double getArea() const override { return area; }

  DelayType getDelay(unsigned inputIndex, unsigned outputIndex) const override {
    return delay[inputIndex][outputIndex];
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
  const SmallVector<SmallVector<DelayType, 2>, 4> delay;
  hw::HWModuleOp module;
  NPNClass npnClass;
};

namespace {
struct TechMapperPass : public impl::TechMapperBase<TechMapperPass> {
  using TechMapperBase<TechMapperPass>::TechMapperBase;

  void runOnOperation() override {
    auto module = getOperation();

    SmallVector<std::unique_ptr<CutRewritePattern>> libraryPatterns;

    unsigned maxInputSize = 0;
    // Consider modules with the "hw.techlib.info" attribute as library
    // modules.
    // TODO: This attribute should be replaced with a more structured
    // representation of technology library information. Specifically, we should
    // have a dedicated operation for technology library.
    SmallVector<hw::HWModuleOp> nonLibraryModules;
    for (auto hwModule : module.getOps<hw::HWModuleOp>()) {
      auto techInfo =
          hwModule->getAttrOfType<DictionaryAttr>("hw.techlib.info");
      if (!techInfo) {
        // If the module does not have the techlib info, it is not a library
        // TODO: Run mapping only when the module is under the specific
        // hierarchy.
        nonLibraryModules.push_back(hwModule);
        continue;
      }

      // Get area and delay attributes
      auto areaAttr = techInfo.getAs<FloatAttr>("area");
      auto delayAttr = techInfo.getAs<ArrayAttr>("delay");
      if (!areaAttr || !delayAttr) {
        mlir::emitError(hwModule.getLoc())
            << "Library module " << hwModule.getModuleName()
            << " must have 'area'(float) and 'delay' (2d array to represent "
               "input-output pair delay) attributes";
        signalPassFailure();
        return;
      }

      double area = areaAttr.getValue().convertToDouble();

      SmallVector<SmallVector<DelayType, 2>, 4> delay;
      for (auto delayValue : delayAttr) {
        auto delayArray = cast<ArrayAttr>(delayValue);
        SmallVector<DelayType, 2> delayRow;
        for (auto delayElement : delayArray) {
          // FIXME: Currently we assume delay is given as integer attributes,
          // this should be replaced once we have a proper cell op with
          // dedicated timing attributes with units.
          delayRow.push_back(
              cast<mlir::IntegerAttr>(delayElement).getValue().getZExtValue());
        }
        delay.push_back(std::move(delayRow));
      }
      // Compute NPN Class for the module.
      auto npnClass = getNPNClassFromModule(hwModule);
      if (failed(npnClass)) {
        signalPassFailure();
        return;
      }

      // Create a CutRewritePattern for the library module
      std::unique_ptr<TechLibraryPattern> pattern =
          std::make_unique<TechLibraryPattern>(hwModule, area, std::move(delay),
                                               std::move(*npnClass));

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
    auto result = mlir::failableParallelForEach(
        module.getContext(), nonLibraryModules, [&](hw::HWModuleOp hwModule) {
          LLVM_DEBUG(llvm::dbgs() << "Processing non-library module: "
                                  << hwModule.getName() << "\n");
          CutRewriter rewriter(options, patternSet);
          return rewriter.run(hwModule);
        });
    if (failed(result))
      signalPassFailure();
  }
};

} // namespace
