//===- ExtractInitialAndFinalFunctions.cpp --------------------------------===//
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

#define DEBUG_TYPE "arc-extract-initial-and-final-functions"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_EXTRACTINITIALANDFINALFUNCTIONS
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
struct ExtractInitialAndFinalFunctionsPass
    : public arc::impl::ExtractInitialAndFinalFunctionsBase<
          ExtractInitialAndFinalFunctionsPass> {
  ExtractInitialAndFinalFunctionsPass() = default;
  ExtractInitialAndFinalFunctionsPass(
      const ExtractInitialAndFinalFunctionsPass &pass)
      : ExtractInitialAndFinalFunctionsPass() {}

protected:
  void runOnOperation() override;

private:
  LogicalResult extractFromModel(ModelOp modelOp);
  LogicalResult extractFunction(Operation *regionOp, ValueRange modelArgs,
                                OpBuilder &funcBuilder);

  SymbolTable *symbolTable;
};
} // namespace

void ExtractInitialAndFinalFunctionsPass::runOnOperation() {
  symbolTable = &getAnalysis<SymbolTable>();
  for (auto op : getOperation().getOps<ModelOp>())
    if (failed(extractFromModel(op)))
      return signalPassFailure();
}

LogicalResult
ExtractInitialAndFinalFunctionsPass::extractFromModel(ModelOp modelOp) {
  LLVM_DEBUG(llvm::dbgs() << "Extracting Initial/Final Functions in `"
                          << modelOp.getName() << "`\n");

  SmallVector<InitialOp, 1> initialOps;
  SmallVector<FinalOp, 1> finalOps;
  for (auto &op : modelOp.getBody().getOps())
    TypeSwitch<Operation *, void>(&op)
        .Case([&](InitialOp initOp) { initialOps.push_back(initOp); })
        .Case([&](FinalOp op) { finalOps.push_back(op); });

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
    if (failed(
            extractFunction(op, modelOp.getBody().getArguments(), funcBuilder)))
      return failure();

  return success();
}

LogicalResult ExtractInitialAndFinalFunctionsPass::extractFunction(
    Operation *regionOp, ValueRange modelArgs, OpBuilder &funcBuilder) {
  LLVM_DEBUG(llvm::dbgs() << "- Extracting " << regionOp->getName() << "\n");
  assert((isa<InitialOp, FinalOp>(regionOp)));
  auto modelOp = regionOp->getParentOfType<ModelOp>();
  Region &region = regionOp->getRegion(0);

  // Make the region isolated from above. Pure operations defined outside the
  // region are cloned into it. The initializer and finalizer only run once,
  // so duplicating these computations is cheap. All other values defined
  // outside the region become arguments of the region's body, and must be
  // arguments of the model.
  IRRewriter rewriter(regionOp->getContext());
  auto capturedValues =
      makeRegionIsolatedFromAbove(rewriter, region, [](Operation *op) {
        return op->getNumRegions() == 0 && isPure(op);
      });
  SmallVector<unsigned> argIndices;
  for (auto value : capturedValues) {
    auto it = llvm::find(modelArgs, value);
    if (it == modelArgs.end()) {
      auto d = regionOp->emitError("region uses external value");
      d.attachNote() << "only pure external values can be used";
      d.attachNote(value.getLoc()) << "external value defined here:";
      return d;
    }
    argIndices.push_back(it - modelArgs.begin());
  }

  // Pick a name for the function.
  SmallString<32> funcName;
  funcName.append(modelOp.getName());
  if (isa<InitialOp>(regionOp))
    funcName.append("_initial");
  else if (isa<FinalOp>(regionOp))
    funcName.append("_final");

  // Create the function, which takes the same arguments as the model.
  auto funcOp = func::FuncOp::create(
      funcBuilder, regionOp->getLoc(), funcName,
      funcBuilder.getFunctionType(modelArgs.getTypes(), {}));
  symbolTable->insert(funcOp); // uniquifies the name
  LLVM_DEBUG(llvm::dbgs() << "  - Created function `" << funcOp.getSymName()
                          << "`\n");

  // Move the region's body into the function, mapping the captured values to
  // the corresponding function arguments.
  Block *entryBlock = funcOp.addEntryBlock();
  SmallVector<Value> argReplacements;
  for (auto index : argIndices)
    argReplacements.push_back(entryBlock->getArgument(index));
  rewriter.mergeBlocks(&region.front(), entryBlock, argReplacements);
  rewriter.setInsertionPointToEnd(entryBlock);
  func::ReturnOp::create(rewriter, regionOp->getLoc());

  // Register the function with the model.
  TypeSwitch<Operation *, void>(regionOp)
      .Case([&](InitialOp) {
        if (modelOp.getInitialFn().has_value())
          modelOp.emitWarning() << "Existing model initializer '"
                                << modelOp.getInitialFnAttr().getValue()
                                << "' will be overridden.";
        modelOp.setInitialFnAttr(
            FlatSymbolRefAttr::get(funcOp.getSymNameAttr()));
      })
      .Case([&](FinalOp) {
        if (modelOp.getFinalFn().has_value())
          modelOp.emitWarning()
              << "Existing model finalizer '"
              << modelOp.getFinalFnAttr().getValue() << "' will be overridden.";
        modelOp.setFinalFnAttr(FlatSymbolRefAttr::get(funcOp.getSymNameAttr()));
      });

  regionOp->erase();
  return success();
}
