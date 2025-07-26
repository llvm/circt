//===- LockFunctions.cpp - lock functions pass ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the lock functions pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
namespace handshake {
#define GEN_PASS_DEF_HANDSHAKELOCKFUNCTIONS
#include "circt/Dialect/Handshake/HandshakePasses.h.inc"
} // namespace handshake
} // namespace circt

using namespace circt;
using namespace handshake;
using namespace mlir;

/// Adds a locking mechanism around the region.
static LogicalResult lockRegion(Region &r, OpBuilder &rewriter) {
  Block *entry = &r.front();
  Location loc = r.getLoc();

  if (entry->getNumArguments() == 0)
    return r.getParentOp()->emitError("cannot lock a region without arguments");

  auto *ret = r.front().getTerminator();
  if (ret->getNumOperands() == 0)
    return r.getParentOp()->emitError("cannot lock a region without results");

  rewriter.setInsertionPointToStart(entry);
  BackedgeBuilder bebuilder(rewriter, loc);
  auto backEdge = bebuilder.get(rewriter.getNoneType());

  auto buff = handshake::BufferOp::create(rewriter, loc, backEdge, 1,
                                          BufferTypeEnum::seq);

  // Dummy value that causes a buffer initialization, but itself does not have a
  // semantic meaning.
  buff->setAttr("initValues", rewriter.getI64ArrayAttr({0}));

  SmallVector<Value> inSyncOperands =
      llvm::to_vector_of<Value>(entry->getArguments());
  inSyncOperands.push_back(buff);
  auto sync = SyncOp::create(rewriter, loc, inSyncOperands);

  // replace all func arg usages with the synced ones
  // TODO is this UB?
  for (auto &&[arg, synced] :
       llvm::drop_end(llvm::zip(inSyncOperands, sync.getResults())))
    arg.replaceAllUsesExcept(synced, sync);

  rewriter.setInsertionPoint(ret);
  SmallVector<Value> endJoinOperands = llvm::to_vector(ret->getOperands());
  // Add the axilirary control signal output to the end-join
  endJoinOperands.push_back(sync.getResults().back());
  auto endJoin = JoinOp::create(rewriter, loc, endJoinOperands);

  backEdge.setValue(endJoin);
  return success();
}

namespace {

struct HandshakeLockFunctionsPass
    : public circt::handshake::impl::HandshakeLockFunctionsBase<
          HandshakeLockFunctionsPass> {
  void runOnOperation() override {
    handshake::FuncOp op = getOperation();
    if (op.isExternal())
      return;

    OpBuilder builder(op);
    if (failed(lockRegion(op.getRegion(), builder)))
      signalPassFailure();
  };
};
} // namespace

std::unique_ptr<mlir::Pass>
circt::handshake::createHandshakeLockFunctionsPass() {
  return std::make_unique<HandshakeLockFunctionsPass>();
}
