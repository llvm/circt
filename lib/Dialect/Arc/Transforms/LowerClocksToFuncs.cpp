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
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
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
  LogicalResult lowerClock(Operation *clockOp, ValueRange modelArgs,
                           OpBuilder &funcBuilder);

  SymbolTable *symbolTable;
};
} // namespace

void LowerClocksToFuncsPass::runOnOperation() {
  symbolTable = &getAnalysis<SymbolTable>();
  for (auto op : getOperation().getOps<ModelOp>())
    if (failed(lowerModel(op)))
      return signalPassFailure();
}

LogicalResult LowerClocksToFuncsPass::lowerModel(ModelOp modelOp) {
  LLVM_DEBUG(llvm::dbgs() << "Extracting Initial/Final Functions in `"
                          << modelOp.getName() << "`\n");

  SmallVector<InitialOp, 1> initialOps;
  SmallVector<FinalOp, 1> finalOps;
  for (auto &op : modelOp.getBody().getOps())
    TypeSwitch<Operation *, void>(&op)
        .Case<InitialOp>([&](auto initOp) { initialOps.push_back(initOp); })
        .Case<FinalOp>([&](auto op) { finalOps.push_back(op); });

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
  for (auto *op : llvm::concat<Operation *>(initialOps, finalOps))
    if (failed(lowerClock(op, modelOp.getBody().getArguments(), funcBuilder)))
      return failure();

  return success();
}

LogicalResult LowerClocksToFuncsPass::lowerClock(Operation *clockOp,
                                                 ValueRange modelArgs,
                                                 OpBuilder &funcBuilder) {
  LLVM_DEBUG(llvm::dbgs() << "- Lowering clock " << clockOp->getName() << "\n");
  assert((isa<InitialOp, FinalOp>(clockOp)));
  auto modelOp = clockOp->getParentOfType<ModelOp>();
  Region &clockRegion = clockOp->getRegion(0);

  // Make the clock's body isolated from above. Pure operations defined outside
  // the clock are cloned into it. The initializer and finalizer only run once,
  // so duplicating these computations is cheap. All other values defined
  // outside the clock become arguments of the clock's body, and must be
  // arguments of the model.
  IRRewriter rewriter(clockOp->getContext());
  auto capturedValues =
      makeRegionIsolatedFromAbove(rewriter, clockRegion, [](Operation *op) {
        return op->getNumRegions() == 0 && isPure(op);
      });
  SmallVector<unsigned> argIndices;
  for (auto value : capturedValues) {
    auto it = llvm::find(modelArgs, value);
    if (it == modelArgs.end()) {
      auto d = clockOp->emitError("clock tree uses external value");
      d.attachNote() << "clock trees can only use pure external values";
      d.attachNote(value.getLoc()) << "external value defined here:";
      return d;
    }
    argIndices.push_back(it - modelArgs.begin());
  }

  // Pick a name for the clock function.
  SmallString<32> funcName;
  funcName.append(modelOp.getName());
  if (isa<InitialOp>(clockOp))
    funcName.append("_initial");
  else if (isa<FinalOp>(clockOp))
    funcName.append("_final");

  // Create the function, which takes the same arguments as the model.
  auto funcOp = func::FuncOp::create(
      funcBuilder, clockOp->getLoc(), funcName,
      funcBuilder.getFunctionType(modelArgs.getTypes(), {}));
  symbolTable->insert(funcOp); // uniquifies the name
  LLVM_DEBUG(llvm::dbgs() << "  - Created function `" << funcOp.getSymName()
                          << "`\n");

  // Move the clock's body into the function, mapping the captured values to
  // the corresponding function arguments.
  Block *entryBlock = funcOp.addEntryBlock();
  SmallVector<Value> argReplacements;
  for (auto index : argIndices)
    argReplacements.push_back(entryBlock->getArgument(index));
  rewriter.mergeBlocks(&clockRegion.front(), entryBlock, argReplacements);
  rewriter.setInsertionPointToEnd(entryBlock);
  func::ReturnOp::create(rewriter, clockOp->getLoc());

  // Register the function with the model.
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

  clockOp->erase();
  return success();
}
