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
#include "mlir/Dialect/UB/IR/UBOps.h"
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

  /// The UB dialect does not implement this inliner hook for its
  /// `ub.unreachable` terminator, which causes the default implementation to
  /// abort. Handle the op here instead: like a branch op, it needs no rewrite
  /// when inlined, since it never transfers control to the continuation block.
  void handleTerminator(Operation *op, Block *newDest) const override {
    if (isa<mlir::ub::UnreachableOp>(op))
      return;
    InlinerInterface::handleTerminator(op, newDest);
  }
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

/// Check whether the given function transitively contains an operation that
/// suspends execution, contains any other LLHD op, or passes LLHD references
/// through a function signature. Calls to such functions must be inlined into
/// their enclosing process and cannot remain as calls.
static bool mustBeInlined(func::FuncOp funcOp, const SymbolTable &symbolTable,
                          SmallPtrSetImpl<Operation *> &visited) {
  if (!visited.insert(funcOp).second)
    return false;
  // LLHD references in the signature tie the function to module state, even
  // if the function itself contains no LLHD ops.
  auto isRefType = [](Type type) { return isa<RefType>(type); };
  if (llvm::any_of(funcOp.getArgumentTypes(), isRefType) ||
      llvm::any_of(funcOp.getResultTypes(), isRefType))
    return true;
  return funcOp
      .walk([&](Operation *op) {
        // Any LLHD op must end up in a procedural region after inlining. This
        // covers ops that suspend execution (llhd.wait, llhd.halt,
        // llhd.call_coroutine) as well as ops that access module state
        // (llhd.prb, llhd.drv, llhd.sig).
        if (isa_and_nonnull<LLHDDialect>(op->getDialect()))
          return WalkResult::interrupt();
        if (auto callOp = dyn_cast<func::CallOp>(op))
          if (auto calledOp = dyn_cast_or_null<func::FuncOp>(
                  symbolTable.lookup(callOp.getCalleeAttr().getAttr())))
            if (mustBeInlined(calledOp, symbolTable, visited))
              return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
}

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

      // Skip extern declarations (e.g. DPI-C imports) — nothing to inline.
      if (funcOp.isDeclaration())
        continue;

      // Recursively inlining a function would just expand infinitely in the
      // IR. If the function never suspends execution and is self-contained
      // (no LLHD ops or LLHD references anywhere in its call graph) it can
      // remain a regular function call; leave it in place. Recursive calls
      // that must be inlined remain an error.
      if (!callStack.insert(funcOp)) {
        SmallPtrSet<Operation *, 4> visited;
        if (mustBeInlined(funcOp, symbolTable, visited)) {
          auto d =
              callOp.emitError("recursive function call cannot be inlined");
          d.attachNote(funcOp.getLoc())
              << "call target suspends execution and must be inlined";
          return failure();
        }
        LLVM_DEBUG(llvm::dbgs() << "- Leaving recursive call un-inlined: "
                                << callOp << "\n");
        continue;
      }
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
