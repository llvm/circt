//===- ResourceUsageAnalysis.cpp - resource usage analysis ---------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the resource usage analysis.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AIG/AIGAnalysis.h"
#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/AIG/AIGPasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/InstanceGraph.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/AnalysisManager.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace aig;

ResourceUsageAnalysis::ResourceUsageAnalysis(Operation *moduleOp,
                                             mlir::AnalysisManager &am)
    : instanceGraph(&am.getAnalysis<igraph::InstanceGraph>()) {}

ResourceUsageAnalysis::ModuleResourceUsage *
circt::aig::ResourceUsageAnalysis::getResourceUsage(hw::HWModuleOp top) {
  {
    auto it = designUsageCache.find(top.getModuleNameAttr());
    if (it != designUsageCache.end())
      return it->second.get();
  }
  auto *topNode = instanceGraph->lookup(top.getModuleNameAttr());
  uint64_t countAndInverterGates = 0, numDFFBits = 0;
  top.walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<aig::AndInverterOp>([&](auto aigOp) {
          // The number of and-inverter gates is the number of inputs minus one,
          // except for the case where there is only one input, which is a
          // pass-through inverter. We multiply by the bitwidth to account for
          // multiple bits.
          countAndInverterGates += std::max(aigOp.getNumOperands() - 1, 1u) *
                                   aigOp.getType().getIntOrFloatBitWidth();
        })
        .Case<seq::CompRegOp, seq::FirRegOp>([&](auto regOp) {
          // NOTE: We might want to reject seq-level registers in AIG at this
          // point and instead only support aig-level flops.
          numDFFBits += regOp.getType().getIntOrFloatBitWidth();

          // TODO: We need to take into account the number of andInverterGates
          // used for reset.
        })
        // TODO: Add support for memory for more accurate resource usage.
        // TODO: Consider rejecting operations that could introduce
        // AndInverter.
        .Default([](auto) {});
  });

  ResourceUsage local(countAndInverterGates, numDFFBits);
  auto moduleUsage = std::make_unique<ModuleResourceUsage>(
      top.getModuleNameAttr(), local, local);

  for (auto *child : *topNode) {
    auto targetMod = child->getTarget();
    if (!isa_and_nonnull<hw::HWModuleOp>(targetMod->getModule()))
      continue;

    auto instance = child->getInstance().getOperation();
    if (instance->getNumResults() == 0 ||
        instance->hasAttrOfType<UnitAttr>("doNotPrint"))
      continue;

    auto childModule = targetMod->getModule<hw::HWModuleOp>();
    const auto &childUsage = getResourceUsage(childModule);
    moduleUsage->total += childUsage->total;
    moduleUsage->instances.emplace_back(
        childModule.getModuleNameAttr(),
        child->getInstance().getInstanceNameAttr(), childUsage);
  }

  auto [it, success] = designUsageCache.try_emplace(top.getModuleNameAttr(),
                                                    std::move(moduleUsage));
  assert(success && "module already exists in cache");
  return it->second.get();
}

static llvm::json::Object
getModuleResourceUsageJSON(const ResourceUsageAnalysis::ResourceUsage &usage) {
  llvm::json::Object obj;
  obj["numAndInverterGates"] = usage.getNumAndInverterGates();
  obj["numDFFBits"] = usage.getNumDFFBits();
  return obj;
}

// This creates a fully-elaborated information but should be ok for now.
static llvm::json::Object getModuleResourceUsageJSON(
    const ResourceUsageAnalysis::ModuleResourceUsage &usage) {
  llvm::json::Object obj;
  obj["local"] = getModuleResourceUsageJSON(usage.local);
  obj["total"] = getModuleResourceUsageJSON(usage.total);
  obj["moduleName"] = usage.moduleName.getValue();
  SmallVector<llvm::json::Value> instances;
  for (const auto &instance : usage.instances) {
    llvm::json::Object child;
    child["instanceName"] = instance.instanceName.getValue();
    child["moduleName"] = instance.moduleName.getValue();
    child["usage"] = getModuleResourceUsageJSON(*instance.usage);
    instances.push_back(std::move(child));
  }
  obj["instances"] = llvm::json::Array(std::move(instances));
  return obj;
}

void ResourceUsageAnalysis::ModuleResourceUsage::emitJSON(
    raw_ostream &os) const {
  os << getModuleResourceUsageJSON(*this);
}

namespace circt {
namespace aig {
#define GEN_PASS_DEF_PRINTRESOURCEUSAGEANALYSIS
#include "circt/Dialect/AIG/AIGPasses.h.inc"
} // namespace aig
} // namespace circt

using namespace circt;
using namespace aig;

namespace {
struct PrintResourceUsageAnalysisPass
    : public impl::PrintResourceUsageAnalysisBase<
          PrintResourceUsageAnalysisPass> {
  using PrintResourceUsageAnalysisBase::PrintResourceUsageAnalysisBase;

  using PrintResourceUsageAnalysisBase::printSummary;
  using PrintResourceUsageAnalysisBase::outputJSONFile;
  using PrintResourceUsageAnalysisBase::topModuleName;
  void runOnOperation() override;
};
} // namespace

void PrintResourceUsageAnalysisPass::runOnOperation() {
  auto mod = getOperation();
  if (topModuleName.empty()) {
    mod.emitError()
        << "'top-name' option is required for PrintResourceUsageAnalysis";
    return signalPassFailure();
  }
  auto &symTbl = getAnalysis<mlir::SymbolTable>();
  auto top = symTbl.lookup<hw::HWModuleOp>(topModuleName);
  if (!top) {
    mod.emitError() << "top module '" << topModuleName << "' not found";
    return signalPassFailure();
  }
  auto &resourceUsageAnalysis = getAnalysis<ResourceUsageAnalysis>();
  auto usage = resourceUsageAnalysis.getResourceUsage(top);

  if (printSummary) {
    llvm::errs() << "// ------ ResourceUsageAnalysis Summary -----\n";
    llvm::errs() << "Top module: " << topModuleName << "\n";
    llvm::errs() << "Total number of and-inverter gates: "
                 << usage->getTotal().getNumAndInverterGates() << "\n";
    llvm::errs() << "Total number of DFF bits: "
                 << usage->getTotal().getNumDFFBits() << "\n";
  
  }
  if (!outputJSONFile.empty()) {
    std::error_code ec;
    llvm::raw_fd_ostream os(outputJSONFile, ec);
    if (ec) {
      emitError(UnknownLoc::get(&getContext()))
          << "failed to open output JSON file '" << outputJSONFile << "': "
          << ec.message();
      return signalPassFailure();
    }
    usage->emitJSON(os);
  }
}
