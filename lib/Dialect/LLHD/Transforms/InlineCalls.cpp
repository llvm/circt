//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/LLHDOps.h"
#include "circt/Dialect/LLHD/LLHDPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Threading.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-inline-calls"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_INLINECALLSPASS
#include "circt/Dialect/LLHD/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace llhd;
using llvm::SmallSetVector;

namespace {
/// Implementation of the `InlinerInterface` that allows calls in SSACFG regions
/// nested within `llhd.process`, `llhd.final`, and `llhd.combinational` ops to
/// be inlined.
struct FunctionInliner : public InlinerInterface {
  using InlinerInterface::InlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const override {
    // Only inline `func.func` ops.
    if (!isa<func::FuncOp>(callable))
      return false;

    // Only inline into SSACFG regions embedded within LLHD processes.
    if (!mayHaveSSADominance(*call->getParentRegion()))
      return false;
    return call->getParentWithTrait<ProceduralRegion>();
  }

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const override {
    return true;
  }

  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &valueMapping) const override {
    return true;
  }

  bool shouldAnalyzeRecursively(Operation *op) const override { return false; }
};

/// Pass implementation.
struct InlineCallsPass
    : public llhd::impl::InlineCallsPassBase<InlineCallsPass> {
  using CallStack = SmallSetVector<func::FuncOp, 8>;
  void runOnOperation() override;
  LogicalResult runOnRegion(Region &region, const SymbolTable &symbolTable,
                            CallStack &callStack);
};
} // namespace

void InlineCallsPass::runOnOperation() {
  auto &symbolTable = getAnalysis<SymbolTable>();
  if (failed(failableParallelForEach(
          &getContext(), getOperation().getOps<hw::HWModuleOp>(),
          [&](auto module) {
            CallStack callStack;
            return runOnRegion(module.getBody(), symbolTable, callStack);
          })))
    signalPassFailure();
}

LogicalResult InlineCallsPass::runOnRegion(Region &region,
                                           const SymbolTable &symbolTable,
                                           CallStack &callStack) {
  FunctionInliner inliner(&getContext());
  InlinerConfig config;
  SmallVector<Operation *> callsToErase;
  SmallVector<std::pair<Operation *, func::FuncOp>> inlineEndMarkers;

  // Walk all calls in the HW module and inline each. Emit a diagnostic if a
  // call does not target a `func.func` op or the inliner fails for some reason.
  // We use a custom version of `Operation::walk` here to ensure that we visit
  // the inlined operations immediately after visiting the call.
  for (auto &block : region) {
    for (auto &op : block) {
      // Pop all calls that are followed by this op off the call stack.
      while (!inlineEndMarkers.empty() &&
             inlineEndMarkers.back().first == &op) {
        assert(inlineEndMarkers.back().second == callStack.back());
        LLVM_DEBUG(llvm::dbgs()
                   << "- Finished @"
                   << inlineEndMarkers.back().second.getSymName() << "\n");
        inlineEndMarkers.pop_back();
        callStack.pop_back();
      }

      // Handle nested regions.
      for (auto &nestedRegion : op.getRegions())
        if (failed(runOnRegion(nestedRegion, symbolTable, callStack)))
          return failure();

      // We only care about calls.
      auto callOp = dyn_cast<func::CallOp>(op);
      if (!callOp)
        continue;

      // Make sure we're calling a `func.func`.
      auto symbol = callOp.getCalleeAttr();
      auto calledOp = symbolTable.lookup(symbol.getAttr());
      auto funcOp = dyn_cast<func::FuncOp>(calledOp);
      if (!funcOp) {
        auto d = callOp.emitError("function call cannot be inlined: call "
                                  "target is not a regular function");
        d.attachNote(calledOp->getLoc()) << "call target defined here";
        return failure();
      }

      // Ensure that we are not recursively inlining a function, which would
      // just expand infinitely in the IR.
      if (!callStack.insert(funcOp))
        return callOp.emitError("recursive function call cannot be inlined");
      inlineEndMarkers.push_back({op.getNextNode(), funcOp});

      // Inline the function body and remember the call for later removal. The
      // `inlineCall` function will inline the function body *after* the call
      // op, which allows the loop to immediately visit the inlined ops and
      // handling nested calls.
      LLVM_DEBUG(llvm::dbgs() << "- Inlining " << callOp << "\n");
      if (failed(inlineCall(inliner, config.getCloneCallback(), callOp, funcOp,
                            funcOp.getCallableRegion())))
        return callOp.emitError("function call cannot be inlined");
      callsToErase.push_back(callOp);
      ++numInlined;
    }
  }

  // Erase all call ops that were successfully inlined.
  for (auto *callOp : callsToErase)
    callOp->erase();

  return success();
}
