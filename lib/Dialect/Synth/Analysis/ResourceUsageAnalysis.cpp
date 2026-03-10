//===- ResourceUsageAnalysis.cpp - Resource Usage Analysis ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the resource usage analysis for the Synth dialect.
// The analysis computes resource utilization including and-inverter gates,
// DFF bits, and LUTs across module hierarchies.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Analysis/ResourceUsageAnalysis.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/InstanceGraph.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/ToolOutputFile.h"

namespace circt {
namespace synth {
#define GEN_PASS_DEF_PRINTRESOURCEUSAGEANALYSIS
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace synth;

//===----------------------------------------------------------------------===//
// ResourceUsageAnalysis Implementation
//===----------------------------------------------------------------------===//

/// Get resource count for an operation if it's a tracked resource type.
/// Returns (operation name, count) pair, or std::nullopt if not tracked.
/// The operation name may include additional information (e.g., LUT input
/// count).
static std::optional<std::pair<std::string, uint64_t>>
getResourceCount(Operation *op) {
  if (op->getNumResults() != 1 || !op->getResult(0).getType().isInteger())
    return std::nullopt;
  return TypeSwitch<Operation *,
                    std::optional<std::pair<std::string, uint64_t>>>(op)
      // Variadic logic operations (AND, OR, XOR, AIG).
      // Gate count = (num_inputs - 1) * bitwidth
      .Case<synth::aig::AndInverterOp, comb::AndOp, comb::OrOp, comb::XorOp>(
          [](auto logicOp) {
            return std::make_pair(
                logicOp->getName().getStringRef(),
                (logicOp.getNumOperands() - 1) *
                    logicOp.getType().getIntOrFloatBitWidth());
          })
      // Majority-inverter graph (MIG) - include input count in the name.
      // Gate count = (num_inputs / 2) * bitwidth
      // Each MIG gate consumes 3 inputs and produces 1 output, so a variadic
      // MIG operation with N inputs requires N/2 gates (rounded down).
      .Case<synth::mig::MajorityInverterOp>([](auto logicOp) {
        uint64_t count = logicOp.getType().getIntOrFloatBitWidth();
        // Concatenate input count to the operation name.
        std::string name = (Twine(logicOp->getName().getStringRef()) + "_" +
                            Twine(logicOp.getNumOperands()))
                               .str();
        return std::make_pair(std::move(name), count);
      })
      // Truth tables (LUTs) - include input count in the name.
      .Case<comb::TruthTableOp>([](auto op) {
        uint64_t count = op.getType().getIntOrFloatBitWidth();
        // Concatenate LUT input number to the operation name.
        std::string name = (Twine(op->getName().getStringRef()) + "_" +
                            Twine(op.getNumOperands()))
                               .str();
        return std::make_pair(std::move(name), count);
      })
      // Sequential elements.
      // Count = bitwidth
      .Case<seq::CompRegOp, seq::FirRegOp>([](auto op) {
        uint64_t count = op.getType().getIntOrFloatBitWidth();
        return std::make_pair(op->getName().getStringRef().str(), count);
      })
      .Default([](Operation *) { return std::nullopt; });
}

ResourceUsageAnalysis::ResourceUsageAnalysis(Operation *moduleOp,
                                             mlir::AnalysisManager &am)
    : instanceGraph(&am.getAnalysis<igraph::InstanceGraph>()) {}

ResourceUsageAnalysis::ModuleResourceUsage *
ResourceUsageAnalysis::getResourceUsage(StringAttr moduleName) {
  // Check cache first.
  auto it = designUsageCache.find(moduleName);
  if (it != designUsageCache.end())
    return it->second.get();

  // Lookup module in instance graph.
  auto *node = instanceGraph->lookup(moduleName);
  if (!node)
    return nullptr;

  auto module = dyn_cast_or_null<igraph::ModuleOpInterface>(node->getModule());
  if (!module)
    return nullptr;

  return getResourceUsage(module);
}

ResourceUsageAnalysis::ModuleResourceUsage *
ResourceUsageAnalysis::getResourceUsage(igraph::ModuleOpInterface module) {
  // Check cache first.
  auto cacheIt = designUsageCache.find(module.getModuleNameAttr());
  if (cacheIt != designUsageCache.end())
    return cacheIt->second.get();

  auto *node = instanceGraph->lookup(module.getModuleNameAttr());

  // Count local resources by walking all operations in the module.
  llvm::StringMap<uint64_t> counts;
  uint64_t unknownOpCount = 0;
  module->walk([&](Operation *op) {
    if (auto resource = getResourceCount(op))
      counts[resource->first] += resource->second;
    else if (op->getNumResults() > 0 && !isa<hw::HWInstanceLike>(op) &&
             !op->hasTrait<mlir::OpTrait::ConstantLike>()) {
      // Track operations that has one result and is not a constant.
      unknownOpCount++;
    }
  });

  // Add unknown operation count if any were found.
  if (unknownOpCount > 0)
    counts["<unknown>"] = unknownOpCount;

  // Initialize module usage with local counts.
  // Total will be updated as we process child instances.
  ResourceUsage local(std::move(counts));
  auto moduleUsage = std::make_unique<ModuleResourceUsage>(
      module.getModuleNameAttr(), local, local);

  // Recursively process child module instances.
  for (auto *child : *node) {
    auto *targetNode = child->getTarget();
    auto childModule =
        dyn_cast_or_null<igraph::ModuleOpInterface>(targetNode->getModule());
    if (!childModule)
      continue;

    auto *instanceOp = child->getInstance().getOperation();
    // Skip instances with no results or marked as "doNotPrint".
    if (instanceOp->getNumResults() == 0 ||
        instanceOp->hasAttrOfType<UnitAttr>("doNotPrint"))
      continue;

    // Recursively compute child usage and accumulate into total.
    auto *childUsage = getResourceUsage(childModule);
    moduleUsage->total += childUsage->total;
    moduleUsage->instances.emplace_back(
        childModule.getModuleNameAttr(),
        child->getInstance().getInstanceNameAttr(), childUsage);
  }

  // Insert into cache and return.
  auto [it, success] = designUsageCache.try_emplace(module.getModuleNameAttr(),
                                                    std::move(moduleUsage));
  assert(success && "module already exists in cache");

  return it->second.get();
}

//===----------------------------------------------------------------------===//
// JSON Serialization
//===----------------------------------------------------------------------===//

/// Convert ResourceUsage to JSON object.
static llvm::json::Object
getModuleResourceUsageJSON(const ResourceUsageAnalysis::ResourceUsage &usage) {
  llvm::json::Object obj;
  for (const auto &count : usage.getCounts())
    obj[count.getKey()] = count.second;
  return obj;
}

/// Convert ModuleResourceUsage to JSON object with full hierarchy.
/// This creates fully-elaborated information including all child instances.
static llvm::json::Object getModuleResourceUsageJSON(
    const ResourceUsageAnalysis::ModuleResourceUsage &usage) {
  llvm::json::Object obj;
  obj["moduleName"] = usage.moduleName.getValue();
  obj["local"] = getModuleResourceUsageJSON(usage.getLocal());
  obj["total"] = getModuleResourceUsageJSON(usage.getTotal());

  // Serialize child instances recursively.
  SmallVector<llvm::json::Value> instances;
  for (const auto &instance : usage.instances) {
    llvm::json::Object child;
    child["instanceName"] = instance.instanceName.getValue();
    child["moduleName"] = instance.moduleName.getValue();
    child["usage"] = getModuleResourceUsageJSON(*instance.usage);
    instances.push_back(std::move(child));
  }
  obj["instances"] = llvm::json::Array(instances);

  return obj;
}

void ResourceUsageAnalysis::ModuleResourceUsage::emitJSON(
    raw_ostream &os) const {
  os << getModuleResourceUsageJSON(*this);
}

namespace {
struct PrintResourceUsageAnalysisPass
    : public impl::PrintResourceUsageAnalysisBase<
          PrintResourceUsageAnalysisPass> {
  using PrintResourceUsageAnalysisBase::PrintResourceUsageAnalysisBase;

  void runOnOperation() override;

  /// Determine which modules to analyze based on options.
  LogicalResult getTopModules(igraph::InstanceGraph *instanceGraph,
                              SmallVectorImpl<igraph::ModuleOpInterface> &tops);

  /// Print analysis result for a single top module.
  LogicalResult printAnalysisResult(ResourceUsageAnalysis &analysis,
                                    igraph::ModuleOpInterface top,
                                    llvm::raw_ostream *os,
                                    llvm::json::OStream *jsonOS);
};
} // namespace

LogicalResult PrintResourceUsageAnalysisPass::getTopModules(
    igraph::InstanceGraph *instanceGraph,
    SmallVectorImpl<igraph::ModuleOpInterface> &tops) {
  auto mod = getOperation();

  if (topModuleName.getValue().empty()) {
    // Automatically infer top modules from instance graph.
    auto topLevelNodes = instanceGraph->getInferredTopLevelNodes();
    if (failed(topLevelNodes))
      return mod.emitError()
             << "failed to infer top-level modules from instance graph";

    // Collect all ModuleOpInterface instances from top-level nodes.
    for (auto *node : *topLevelNodes) {
      if (auto module = dyn_cast<igraph::ModuleOpInterface>(node->getModule()))
        tops.push_back(module);
    }

    if (tops.empty())
      return mod.emitError() << "no top-level modules found in instance graph";
  } else {
    // Use user-specified top module name.
    auto *node = instanceGraph->lookup(
        mlir::StringAttr::get(mod.getContext(), topModuleName.getValue()));
    if (!node)
      return mod.emitError()
             << "top module '" << topModuleName.getValue() << "' not found";

    auto top = dyn_cast_or_null<igraph::ModuleOpInterface>(node->getModule());
    if (!top)
      return mod.emitError() << "module '" << topModuleName.getValue()
                             << "' is not a ModuleOpInterface";
    tops.push_back(top);
  }

  return success();
}

LogicalResult PrintResourceUsageAnalysisPass::printAnalysisResult(
    ResourceUsageAnalysis &analysis, igraph::ModuleOpInterface top,
    llvm::raw_ostream *os, llvm::json::OStream *jsonOS) {
  auto *usage = analysis.getResourceUsage(top);
  if (!usage)
    return failure();

  if (jsonOS) {
    usage->emitJSON(jsonOS->rawValueBegin());
    jsonOS->rawValueEnd();
  } else if (os) {
    auto &stream = *os;
    stream << "Resource Usage Analysis for module: "
           << usage->moduleName.getValue() << "\n";
    stream << "========================================\n";
    stream << "Total:\n";

    // Sort resource counts by name for consistent output.
    SmallVector<std::pair<StringRef, uint64_t>> sortedCounts;
    for (const auto &count : usage->getTotal().getCounts())
      sortedCounts.emplace_back(count.getKey(), count.second);
    llvm::sort(sortedCounts,
               [](const auto &a, const auto &b) { return a.first < b.first; });

    // Find the maximum name length for aligned formatting.
    size_t maxNameLen = 0;
    for (const auto &[name, count] : sortedCounts)
      maxNameLen = std::max(maxNameLen, name.size());

    // Print with aligned columns.
    for (const auto &[name, count] : sortedCounts)
      stream << "  " << name << ": "
             << std::string(maxNameLen - name.size(), ' ') << count << "\n";
    stream << "\n";
  }

  return success();
}

void PrintResourceUsageAnalysisPass::runOnOperation() {
  auto &resourceUsage = getAnalysis<ResourceUsageAnalysis>();
  auto *instanceGraph = resourceUsage.getInstanceGraph();

  // Determine which modules to analyze.
  SmallVector<igraph::ModuleOpInterface> tops;
  if (failed(getTopModules(instanceGraph, tops)))
    return signalPassFailure();

  // Open output file.
  std::string error;
  auto file = mlir::openOutputFile(outputFile.getValue(), &error);
  if (!file) {
    llvm::errs() << error;
    return signalPassFailure();
  }

  auto &os = file->os();
  std::unique_ptr<llvm::json::OStream> jsonOS;
  if (emitJSON.getValue()) {
    jsonOS = std::make_unique<llvm::json::OStream>(os);
    jsonOS->arrayBegin();
  }

  // Ensure JSON array is properly closed on exit.
  auto closeJson = llvm::scope_exit([&]() {
    if (jsonOS)
      jsonOS->arrayEnd();
  });

  // Print resource usage for each top module.
  for (auto top : tops) {
    if (failed(printAnalysisResult(resourceUsage, top, jsonOS ? nullptr : &os,
                                   jsonOS.get())))
      return signalPassFailure();
  }

  file->keep();
  markAllAnalysesPreserved();
}
