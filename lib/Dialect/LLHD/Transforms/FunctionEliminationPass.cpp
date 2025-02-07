//===- FunctionEliminationPass.cpp - Implement Function Elimination Pass --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement pass to check that all functions got inlined and delete them.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/Support/LogicalResult.h"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_FUNCTIONELIMINATION
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;

namespace {

struct FunctionInliner : public InlinerInterface {
  FunctionInliner(MLIRContext *context) : InlinerInterface(context) {}

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const override {
    return dest->getParentOfType<llhd::ProcessOp>() &&
           isa<func::FuncOp>(src->getParentOp());
  }
  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &valueMapping) const override {
    return true;
  }
};

struct FunctionEliminationPass
    : public circt::llhd::impl::FunctionEliminationBase<
          FunctionEliminationPass> {
  void runOnOperation() override;
  LogicalResult runOnModule(hw::HWModuleOp module);
};

void FunctionEliminationPass::runOnOperation() {
  for (auto module : getOperation().getOps<hw::HWModuleOp>())
    if (failed(runOnModule(module)))
      return signalPassFailure();

  getOperation().walk([&](mlir::func::FuncOp op) {
    if (op.symbolKnownUseEmpty(getOperation()))
      op.erase();
  });
}

LogicalResult FunctionEliminationPass::runOnModule(hw::HWModuleOp module) {
  FunctionInliner inliner(&getContext());
  SymbolTableCollection table;

  SmallVector<CallOpInterface> calls;
  module.walk([&](func::CallOp op) { calls.push_back(op); });

  for (auto call : calls) {
    auto symbol = call.getCallableForCallee().dyn_cast<SymbolRefAttr>();
    if (!symbol)
      return call.emitError(
          "functions not referred to by symbol are not supported");

    auto func = cast<CallableOpInterface>(
        table.lookupNearestSymbolFrom(module, symbol.getLeafReference()));

    if (succeeded(
            mlir::inlineCall(inliner, call, func, func.getCallableRegion()))) {
      call->erase();
      continue;
    }

    return call.emitError(
        "Not all functions are inlined, there is at least "
        "one function call left within a llhd.process or hw.module.");
  }

  return success();
}
} // namespace
