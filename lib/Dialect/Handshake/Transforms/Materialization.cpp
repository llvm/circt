//===- Materialization.cpp - Fork/sink materialization pass -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the Fork/sink materialization pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace handshake;
using namespace mlir;

using BlockValues = DenseMap<Block *, std::vector<Value>>;

static void replaceFirstUse(Operation *op, Value oldVal, Value newVal) {
  for (int i = 0, e = op->getNumOperands(); i < e; ++i)
    if (op->getOperand(i) == oldVal) {
      op->setOperand(i, newVal);
      break;
    }
  return;
}

namespace circt {
namespace handshake {

void insertFork(Value result, bool isLazy, OpBuilder &rewriter) {
  // Get successor operations
  std::vector<Operation *> opsToProcess;
  for (auto &u : result.getUses())
    opsToProcess.push_back(u.getOwner());

  // Insert fork after op
  rewriter.setInsertionPointAfterValue(result);
  Operation *newOp;
  if (isLazy)
    newOp = rewriter.create<LazyForkOp>(result.getLoc(), result,
                                        opsToProcess.size());
  else
    newOp =
        rewriter.create<ForkOp>(result.getLoc(), result, opsToProcess.size());

  // Modify operands of successor
  // opsToProcess may have multiple instances of same operand
  // Replace uses one by one to assign different fork outputs to them
  for (int i = 0, e = opsToProcess.size(); i < e; ++i)
    replaceFirstUse(opsToProcess[i], result, newOp->getResult(i));
}

// Insert Fork Operation for every operation and function argument with more
// than one successor.
LogicalResult addForkOps(handshake::FuncOp f, OpBuilder &rewriter) {
  for (Operation &op : f.getOps()) {
    // Ignore terminators, and don't add Forks to Forks.
    if (op.getNumSuccessors() == 0 && !isa<ForkOp>(op)) {
      for (auto result : op.getResults()) {
        // If there is a result and it is used more than once
        if (!result.use_empty() && !result.hasOneUse())
          insertFork(result, false, rewriter);
      }
    }
  }

  for (auto barg : f.front().getArguments())
    if (!barg.use_empty() && !barg.hasOneUse())
      insertFork(barg, false, rewriter);

  return success();
}

// Create sink for every unused result
LogicalResult addSinkOps(handshake::FuncOp f, OpBuilder &rewriter) {
  BlockValues liveOuts;

  for (Block &block : f) {
    for (Operation &op : block) {
      // Do not add sinks for unused MLIR operations which the rewriter will
      // later remove We have already replaced these ops with their handshake
      // equivalents
      // TODO: should we use other indicator for op that has been erased?
      if (isa<mlir::CondBranchOp, mlir::BranchOp, memref::LoadOp,
              mlir::AffineReadOpInterface, mlir::AffineForOp>(op))
        continue;

      if (op.getNumResults() == 0)
        continue;

      for (auto result : op.getResults())
        if (result.use_empty()) {
          rewriter.setInsertionPointAfter(&op);
          auto sinkOp = rewriter.create<SinkOp>(op.getLoc(), result);
          if (result.getType().isa<NoneType>())
            sinkOp->setAttr("control", rewriter.getBoolAttr(true));
        }
    }
  }
  return success();
}

} // namespace handshake
} // namespace circt

namespace {
struct HandshakeMaterializeForksSinksPass
    : public HandshakeMaterializeForksSinksBase<
          HandshakeMaterializeForksSinksPass> {
  void runOnOperation() override {
    handshake::FuncOp op = getOperation();
    OpBuilder builder(op);
    if (addForkOps(op, builder).failed() || addSinkOps(op, builder).failed() ||
        verifyAllValuesHasOneUse(op).failed())
      return signalPassFailure();
  };
};

} // namespace

std::unique_ptr<mlir::Pass>
circt::handshake::createHandshakeMaterializeForksSinksPass() {
  return std::make_unique<HandshakeMaterializeForksSinksPass>();
}
