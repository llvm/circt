//===- SVExtractTestCode.cpp - SV Simulation Extraction Pass --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass extracts simulation constructs to submodules.
//
//===----------------------------------------------------------------------===//

#include "SVPassDetail.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/SV/SVVisitors.h"
#include "circt/Support/ImplicitLocOpBuilder.h"
#include "mlir/IR/BlockAndValueMapping.h"

#include <set>

using namespace circt;
using namespace sv;

//===----------------------------------------------------------------------===//
// StubExternalModules Pass
//===----------------------------------------------------------------------===//

namespace {

struct rootSets {
  SmallVector<AssertOp> asserts;
  SmallVector<AssumeOp> assumes;
  SmallVector<CoverOp> covers;
};

void findRoots(Operation* op, rootSets& roots) {
    for (auto& r : op->getRegions())
    for(auto& b : r.getBlocks())
    for (auto& o : b.getOperations())
    if (auto opInterest = dyn_cast<AssertOp>(o)) {
        roots.asserts.push_back(opInterest);
    }
    else if (auto opInterest = dyn_cast<AssumeOp>(o)) {
      roots.assumes.push_back(opInterest);
    } else if (auto opInterest = dyn_cast<CoverOp>(o)) {
      roots.covers.push_back(opInterest);
    } else if (op->getNumRegions()) {
        findRoots(&o, roots);
    }
}

bool allUsesInSet(Operation* op, SetVector<Operation*> opSet) {
  for (auto en : llvm::enumerate(op->getResults())) {
    auto operand = en.value();
    for (auto use : operand->getUses())
        if (!opSet.count(use.getDefiningOp()))
            return false;
    return true;
  }
}

// Aggressively clone operations to the new module.  This leaves maximum
// flexibility for optimization after removal of the nodes from the old module.
SetVector<Operation*> computeCloneSet(std::set<Operation*>& roots) {
  SetVector<Operation*> results;
  for (auto op : roots) {
    getBackwardsSlice(op, results, [](Operation *testOp) -> bool {
      return !isa<ReadWriteOp>() && !isa<InstanceOp>() && !isa<PAssignOp> &&
             !isa<PBAssignOp>;
    });
  }
  return results;
}

void expandRegion(SmallVector<Operation*>& roots) {

    SetVector<Operation*> slice;
    unsigned currentIndex = 0;

    bool changed = true;
    SmallVector<Operation*> curWL, nextWL;
    curWL.insert(region.begin(), region.end());
    while (changed) {
        changed = false;
        for (auto op : curWL) {
        for (auto arg : op->getOperands()) {
            if (!region.count(arg.getDefiningOp())) {
                //Outside set, record
                inputs.insert(arg);
                op->dump();    
                arg.dump(); 
           }
        }
    }
    return inputs;
}

rtl::RTLModuleOp createModuleForCut(rtl::RTLModuleOp op,
                                    DenseSet<Value> &inputs, StringRef name,
                                    BlockAndValueMapping& cutMap) {
  OpBuilder b(op->getParentOfType<mlir::ModuleOp>()->getContext());
   SmallVector<rtl::ModulePortInfo> ports;
  size_t inputNum = 0;
  for (auto port : inputs) {
    ports.push_back({b.getStringAttr(""), rtl::INPUT, port.getType(),
                     inputNum++});
                     llvm::errs() << inputNum - 1 << ": ";
                     port.dump();
                     port.getType().dump();
                     llvm::errs() << "\n";
  }
  std::array<NamedAttribute, 1> bindAttr = {
      b.getNamedAttr("genAsBind", b.getBoolAttr(true))};
  auto newMod = b.create<rtl::RTLModuleOp>(op.getLoc(), b.getStringAttr(name), ports, bindAttr);
  inputNum = 0;
  for (auto port : inputs) {
      cutMap.map(port, newMod.body().getArgument(inputNum++));
  }
    return newMod;
  }

  void migrateOps(rtl::RTLModuleOp oldMod, rtl::RTLModuleOp newMod,
                  std::set<Operation *> &ops, BlockAndValueMapping &cutMap) {
    OpBuilder b(newMod);
    for (auto op : ops) {
//        if (!cutMap.contains(op->getBlock()))


        b.clone(*op, cutMap);
    }
  }

struct SVExtractTestCodeImplPass : public SVExtractTestCodeBase<SVExtractTestCodeImplPass> {
  void runOnOperation() override;

private:
  void doModule(rtl::RTLModuleOp module) {
    rootSets roots;
    findRoots(module, roots);
    std::set<Operation *> allRoots;
    allRoots.insert(roots.asserts.begin(), roots.asserts.end());
    allRoots.insert(roots.assumes.begin(), roots.assumes.end());
    allRoots.insert(roots.covers.begin(), roots.covers.end());
    if (allRoots.empty())
      return;
    auto opsToClone = computeCloneSet(roots);
    BlockAndValueMapping cutMap;
    auto inputs = computeCut(osToClone, cutMap);
    auto bmod = createModuleForCut(
        module, inputs,
        (module.getVerilogModuleNameAttr().getValue() + "_bind").str(), cutMap);
    bmod.dump();
    migrateOps(module, bmod, allRoots, cutMap);
  }
};

} // end anonymous namespace

void SVExtractTestCodeImplPass::runOnOperation() {
  auto *topLevelModule = getOperation().getBody();
  for (auto &op : topLevelModule->getOperations())
      if (auto rtlmod = dyn_cast<rtl::RTLModuleOp>(op))
            doModule(rtlmod);
}

    std::unique_ptr<Pass>
    circt::sv::createSVExtractTestCodePass() {
  return std::make_unique<SVExtractTestCodeImplPass>();
}
