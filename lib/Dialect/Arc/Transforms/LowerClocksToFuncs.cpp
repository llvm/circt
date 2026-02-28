//===- LowerClocksToFuncs.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-lower-clocks-to-funcs"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_LOWERCLOCKSTOFUNCS
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace arc;
using namespace hw;
using mlir::OpTrait::ConstantLike;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct LowerClocksToFuncsPass
    : public arc::impl::LowerClocksToFuncsBase<LowerClocksToFuncsPass> {
  LowerClocksToFuncsPass() = default;
  LowerClocksToFuncsPass(const LowerClocksToFuncsPass &pass)
      : LowerClocksToFuncsPass() {}

  void runOnOperation() override;
  LogicalResult lowerModel(ModelOp modelOp);
  LogicalResult lowerClock(Operation *clockOp, Value modelStorageArg,
                           OpBuilder &funcBuilder);
  LogicalResult isolateClock(Operation *clockOp, Value modelStorageArg,
                             Value clockStorageArg);

  SymbolTable *symbolTable;

  Statistic numOpsCopied{this, "ops-copied", "Ops copied into clock trees"};
  Statistic numOpsMoved{this, "ops-moved", "Ops moved into clock trees"};
};
} // namespace

void LowerClocksToFuncsPass::runOnOperation() {
  symbolTable = &getAnalysis<SymbolTable>();
  for (auto op : getOperation().getOps<ModelOp>())
    if (failed(lowerModel(op)))
      return signalPassFailure();
}

LogicalResult LowerClocksToFuncsPass::lowerModel(ModelOp modelOp) {
  LLVM_DEBUG(llvm::dbgs() << "Lowering clocks in `" << modelOp.getName()
                          << "`\n");

  // Find the clocks to extract.
  SmallVector<InitialOp, 1> initialOps;
  SmallVector<FinalOp, 1> finalOps;
  SmallVector<Operation *> clocks;
  modelOp.walk([&](Operation *op) {
    TypeSwitch<Operation *, void>(op)
        .Case<InitialOp>([&](auto initOp) {
          initialOps.push_back(initOp);
          clocks.push_back(initOp);
        })
        .Case<FinalOp>([&](auto op) {
          finalOps.push_back(op);
          clocks.push_back(op);
        });
  });

  // Sanity check
  if (initialOps.size() > 1) {
    auto diag = modelOp.emitOpError()
                << "containing multiple InitialOps is currently unsupported.";
    for (auto initOp : initialOps)
      diag.attachNote(initOp.getLoc()) << "Conflicting InitialOp:";
  }
  if (finalOps.size() > 1) {
    auto diag = modelOp.emitOpError()
                << "containing multiple FinalOps is currently unsupported.";
    for (auto op : finalOps)
      diag.attachNote(op.getLoc()) << "Conflicting FinalOp:";
  }
  if (initialOps.size() > 1 || finalOps.size() > 1)
    return failure();

  // Perform the actual extraction.
  OpBuilder funcBuilder(modelOp);
  for (auto *op : clocks)
    if (failed(lowerClock(op, modelOp.getBody().getArgument(0), funcBuilder)))
      return failure();

  return success();
}

LogicalResult LowerClocksToFuncsPass::lowerClock(Operation *clockOp,
                                                 Value modelStorageArg,
                                                 OpBuilder &funcBuilder) {
  LLVM_DEBUG(llvm::dbgs() << "- Lowering clock " << clockOp->getName() << "\n");
  assert((isa<InitialOp, FinalOp>(clockOp)));

  // Add a `StorageType` block argument to the clock's body block which we are
  // going to use to pass the storage pointer to the clock once it has been
  // pulled out into a separate function.
  Region &clockRegion = clockOp->getRegion(0);
  Value clockStorageArg = clockRegion.addArgument(modelStorageArg.getType(),
                                                  modelStorageArg.getLoc());

  // Ensure the clock tree does not use any values defined outside of it.
  if (failed(isolateClock(clockOp, modelStorageArg, clockStorageArg)))
    return failure();

  // Add a return op to the end of the body.
  auto builder = OpBuilder::atBlockEnd(&clockRegion.front());
  func::ReturnOp::create(builder, clockOp->getLoc());

  // Pick a name for the clock function.
  SmallString<32> funcName;
  auto modelOp = clockOp->getParentOfType<ModelOp>();
  funcName.append(modelOp.getName());

  if (isa<InitialOp>(clockOp))
    funcName.append("_initial");
  else if (isa<FinalOp>(clockOp))
    funcName.append("_final");

  auto funcOp = func::FuncOp::create(
      funcBuilder, clockOp->getLoc(), funcName,
      builder.getFunctionType({modelStorageArg.getType()}, {}));
  symbolTable->insert(funcOp); // uniquifies the name
  LLVM_DEBUG(llvm::dbgs() << "  - Created function `" << funcOp.getSymName()
                          << "`\n");

  // Create a call to the function within the model.
  builder.setInsertionPoint(clockOp);
  TypeSwitch<Operation *, void>(clockOp)
      .Case<InitialOp>([&](auto) {
        if (modelOp.getInitialFn().has_value())
          modelOp.emitWarning() << "Existing model initializer '"
                                << modelOp.getInitialFnAttr().getValue()
                                << "' will be overridden.";
        modelOp.setInitialFnAttr(
            FlatSymbolRefAttr::get(funcOp.getSymNameAttr()));
      })
      .Case<FinalOp>([&](auto) {
        if (modelOp.getFinalFn().has_value())
          modelOp.emitWarning()
              << "Existing model finalizer '"
              << modelOp.getFinalFnAttr().getValue() << "' will be overridden.";
        modelOp.setFinalFnAttr(FlatSymbolRefAttr::get(funcOp.getSymNameAttr()));
      });

  // Move the clock's body block to the function and remove the old clock op.
  funcOp.getBody().takeBody(clockRegion);

  clockOp->erase();
  return success();
}

/// Copy any external constants that the clock tree might be using into its
/// body. Anything besides constants should no longer exist after a proper run
/// of the pipeline.
LogicalResult LowerClocksToFuncsPass::isolateClock(Operation *clockOp,
                                                   Value modelStorageArg,
                                                   Value clockStorageArg) {
  auto *clockRegion = &clockOp->getRegion(0);
  auto builder = OpBuilder::atBlockBegin(&clockRegion->front());
  DenseMap<Value, Value> copiedValues;
  auto result = clockRegion->walk([&](Operation *op) {
    for (auto &operand : op->getOpOperands()) {
      // Block arguments are okay, since there's nothing we can move.
      if (operand.get() == modelStorageArg) {
        operand.set(clockStorageArg);
        continue;
      }
      if (isa<BlockArgument>(operand.get())) {
        auto d = op->emitError(
            "operation in clock tree uses external block argument");
        d.attachNote() << "clock trees can only use external constant values";
        d.attachNote() << "see operand #" << operand.getOperandNumber();
        d.attachNote(clockOp->getLoc()) << "clock tree:";
        return WalkResult::interrupt();
      }

      // Check if the value is defined outside of the clock op.
      auto *definingOp = operand.get().getDefiningOp();
      assert(definingOp && "block arguments ruled out above");
      Region *definingRegion = definingOp->getParentRegion();
      if (clockRegion->isAncestor(definingRegion))
        continue;

      // The op is defined outside the clock, so we need to create a copy of the
      // defining inside the clock tree.
      if (auto copiedValue = copiedValues.lookup(operand.get())) {
        operand.set(copiedValue);
        continue;
      }

      // Check that we can actually copy this definition inside.
      if (!definingOp->hasTrait<ConstantLike>()) {
        auto d = op->emitError("operation in clock tree uses external value");
        d.attachNote() << "clock trees can only use external constant values";
        d.attachNote(definingOp->getLoc()) << "external value defined here:";
        d.attachNote(clockOp->getLoc()) << "clock tree:";
        return WalkResult::interrupt();
      }

      // Copy the op inside the clock tree (or move it if all uses are within
      // the clock tree).
      bool canMove = llvm::all_of(definingOp->getUsers(), [&](Operation *user) {
        return clockRegion->isAncestor(user->getParentRegion());
      });
      Operation *clonedOp;
      if (canMove) {
        definingOp->remove();
        clonedOp = definingOp;
        ++numOpsMoved;
      } else {
        clonedOp = definingOp->cloneWithoutRegions();
        ++numOpsCopied;
      }
      builder.insert(clonedOp);
      if (!canMove) {
        for (auto [outerResult, innerResult] :
             llvm::zip(definingOp->getResults(), clonedOp->getResults())) {
          copiedValues.insert({outerResult, innerResult});
          if (operand.get() == outerResult)
            operand.set(innerResult);
        }
      }
    }
    return WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}

std::unique_ptr<Pass> arc::createLowerClocksToFuncsPass() {
  return std::make_unique<LowerClocksToFuncsPass>();
}
