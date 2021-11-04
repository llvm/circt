//===- Reduction.cpp - Reductions for circt-reduce ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines abstract reduction patterns for the 'circt-reduce' tool.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/InitAllDialects.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Reducer/Tester.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#include "Reduction.h"

#define DEBUG_TYPE "circt-reduce"

using namespace llvm;
using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Reduction
//===----------------------------------------------------------------------===//

Reduction::~Reduction() {}

//===----------------------------------------------------------------------===//
// Pass Reduction
//===----------------------------------------------------------------------===//

PassReduction::PassReduction(MLIRContext *context, std::unique_ptr<Pass> pass,
                             bool canIncreaseSize)
    : context(context), canIncreaseSize(canIncreaseSize) {
  passName = pass->getArgument();
  if (passName.empty())
    passName = pass->getName();

  if (auto opName = pass->getOpName())
    pm = std::make_unique<PassManager>(context, *opName);
  else
    pm = std::make_unique<PassManager>(context);
  pm->addPass(std::move(pass));
}

bool PassReduction::match(Operation *op) const {
  return op->getName().getIdentifier() == pm->getOpName(*context);
}

LogicalResult PassReduction::rewrite(Operation *op) const {
  return pm->run(op);
}

std::string PassReduction::getName() const { return passName.str(); }

//===----------------------------------------------------------------------===//
// Concrete Sample Reductions (to later move into the dialects)
//===----------------------------------------------------------------------===//

/// A sample reduction pattern that maps `firrtl.module` to `firrtl.extmodule`.
struct ModuleExternalizer : public Reduction {
  bool match(Operation *op) const override {
    return isa<firrtl::FModuleOp>(op);
  }
  LogicalResult rewrite(Operation *op) const override {
    auto module = cast<firrtl::FModuleOp>(op);
    OpBuilder builder(module);
    builder.create<firrtl::FExtModuleOp>(
        module->getLoc(),
        module->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()),
        module.getPorts(), StringRef(), module.annotationsAttr());
    module->erase();
    return success();
  }
  std::string getName() const override { return "module-externalizer"; }
};

/// Starting at the given `op`, traverse through it and its operands and erase
/// operations that have no more uses.
static void pruneUnusedOps(Operation *initialOp) {
  SmallVector<Operation *> worklist;
  SmallSet<Operation *, 4> handled;
  worklist.push_back(initialOp);
  while (!worklist.empty()) {
    auto op = worklist.pop_back_val();
    if (!op->use_empty())
      continue;
    for (auto arg : op->getOperands())
      if (auto argOp = arg.getDefiningOp())
        if (handled.insert(argOp).second)
          worklist.push_back(argOp);
    op->erase();
  }
}

/// A sample reduction pattern that replaces the right-hand-side of
/// `firrtl.connect` and `firrtl.partialconnect` operations with a
/// `firrtl.invalidvalue`. This removes uses from the fanin cone to these
/// connects and creates opportunities for reduction in DCE/CSE.
struct ConnectInvalidator : public Reduction {
  bool match(Operation *op) const override {
    return isa<firrtl::ConnectOp, firrtl::PartialConnectOp>(op) &&
           !op->getOperand(1).getDefiningOp<firrtl::InvalidValueOp>();
  }
  LogicalResult rewrite(Operation *op) const override {
    assert(match(op));
    auto rhs = op->getOperand(1);
    OpBuilder builder(op);
    auto invOp =
        builder.create<firrtl::InvalidValueOp>(rhs.getLoc(), rhs.getType());
    auto rhsOp = rhs.getDefiningOp();
    op->setOperand(1, invOp);
    if (rhsOp)
      pruneUnusedOps(rhsOp);
    return success();
  }
  std::string getName() const override { return "connect-invalidator"; }
};

/// A sample reduction pattern that removes operations which either produce no
/// results or their results have no users.
struct OperationPruner : public Reduction {
  bool match(Operation *op) const override {
    return !isa<ModuleOp>(op) &&
           !op->hasAttr(SymbolTable::getSymbolAttrName()) &&
           (op->getNumResults() == 0 || op->use_empty());
  }
  LogicalResult rewrite(Operation *op) const override {
    assert(match(op));
    pruneUnusedOps(op);
    return success();
  }
  std::string getName() const override { return "operation-pruner"; }
};

/// A sample reduction pattern that replaces instances of `firrtl.extmodule`
/// with wires.
struct ExtmoduleInstanceRemover : public Reduction {
  bool match(Operation *op) const override {
    if (auto instOp = dyn_cast<firrtl::InstanceOp>(op))
      return isa<firrtl::FExtModuleOp>(instOp.getReferencedModule());
    return false;
  }
  LogicalResult rewrite(Operation *op) const override {
    auto instOp = cast<firrtl::InstanceOp>(op);
    auto portInfo = instOp.getReferencedModule().getPorts();
    ImplicitLocOpBuilder builder(instOp.getLoc(), instOp);
    SmallVector<Value> replacementWires;
    for (firrtl::PortInfo info : portInfo) {
      auto wire = builder.create<firrtl::WireOp>(
          info.type, (Twine(instOp.name()) + "_" + info.getName()).str());
      if (info.isOutput()) {
        auto inv = builder.create<firrtl::InvalidValueOp>(info.type);
        builder.create<firrtl::ConnectOp>(wire, inv);
      }
      replacementWires.push_back(wire);
    }
    instOp.replaceAllUsesWith(std::move(replacementWires));
    instOp->erase();
    return success();
  }
  std::string getName() const override { return "extmodule-instance-remover"; }
  bool acceptSizeIncrease() const override { return true; }
};

//===----------------------------------------------------------------------===//
// Reduction Registration
//===----------------------------------------------------------------------===//

static std::unique_ptr<Pass> createSimpleCanonicalizerPass() {
  GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.enableRegionSimplification = false;
  return createCanonicalizerPass(config);
}

void circt::createAllReductions(
    MLIRContext *context,
    llvm::function_ref<void(std::unique_ptr<Reduction>)> add) {
  // Gather a list of reduction patterns that we should try. Ideally these are
  // sorted by decreasing reduction potential/benefit. For example, things that
  // can knock out entire modules while being cheap should be tried first,
  // before trying to tweak operands of individual arithmetic ops.
  add(std::make_unique<ModuleExternalizer>());
  add(std::make_unique<PassReduction>(context, firrtl::createInlinerPass()));
  add(std::make_unique<PassReduction>(context,
                                      createSimpleCanonicalizerPass()));
  add(std::make_unique<PassReduction>(context, createCSEPass()));
  add(std::make_unique<ConnectInvalidator>());
  add(std::make_unique<OperationPruner>());
  add(std::make_unique<ExtmoduleInstanceRemover>());
}
