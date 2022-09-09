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

#include "PassDetails.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace handshake;
using namespace mlir;

LogicalResult handshake::lockRegion(Region &r, OpBuilder &rewriter) {
  Block *entry = &r.front();
  Location loc = r.getLoc();

  rewriter.setInsertionPointToStart(entry);
  BackedgeBuilder bebuilder(rewriter, loc);
  auto backEdge = bebuilder.get(rewriter.getNoneType());

  auto buff = rewriter.create<handshake::BufferOp>(
      loc, rewriter.getNoneType(), 1, backEdge,
      /*bufferType=*/BufferTypeEnum::seq);

  // Dummy value that causes a buffer initialization, but itself does not have a
  // semantic meaning.
  buff->setAttr("initValues", rewriter.getI64ArrayAttr({0}));

  auto ctrl = entry->getArguments().back();
  auto join = rewriter.create<JoinOp>(loc, ValueRange({ctrl, buff}));
  ctrl.replaceAllUsesExcept(join, join);

  auto ret = *r.getOps<handshake::ReturnOp>().begin();
  rewriter.setInsertionPoint(ret);

  backEdge.setValue(ret.control());
  return success();
}

namespace {

struct HandshakeLockFunctionsPass
    : public HandshakeLockFunctionsBase<HandshakeLockFunctionsPass> {
  void runOnOperation() override {
    handshake::FuncOp op = getOperation();

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
