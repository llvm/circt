//===- DCMaterialization.cpp - Fork/sink materialization pass ---*- C++ -*-===//
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

#include "circt/Dialect/DC/DCOps.h"
#include "circt/Dialect/DC/DCPasses.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace dc {
#define GEN_PASS_DEF_DCMATERIALIZEFORKSSINKS
#define GEN_PASS_DEF_DCDEMATERIALIZEFORKSSINKS
#include "circt/Dialect/DC/DCPasses.h.inc"
} // namespace dc
} // namespace circt

using namespace circt;
using namespace dc;
using namespace mlir;

static bool isDCTyped(Value v) {
  return isa<dc::TokenType, dc::ValueType>(v.getType());
}

static void replaceFirstUse(Operation *op, Value oldVal, Value newVal) {
  for (int i = 0, e = op->getNumOperands(); i < e; ++i)
    if (op->getOperand(i) == oldVal) {
      op->setOperand(i, newVal);
      break;
    }
}

// Adds a sink to the provided token or value-typed Value `v`.
static void insertSink(Value v, OpBuilder &rewriter) {
  rewriter.setInsertionPointAfterValue(v);
  if (isa<ValueType>(v.getType())) {
    // Unpack before sinking
    v = UnpackOp::create(rewriter, v.getLoc(), v).getToken();
  }

  SinkOp::create(rewriter, v.getLoc(), v);
}

// Adds a fork of the provided token or value-typed Value `result`.
static void insertFork(Value result, OpBuilder &rewriter) {
  rewriter.setInsertionPointAfterValue(result);
  // Get successor operations
  std::vector<Operation *> opsToProcess;
  for (auto &u : result.getUses())
    opsToProcess.push_back(u.getOwner());

  bool isValue = isa<ValueType>(result.getType());
  Value token = result;
  Value value;
  if (isValue) {
    auto unpack = UnpackOp::create(rewriter, result.getLoc(), result);
    token = unpack.getToken();
    value = unpack.getOutput();
  }

  // Insert fork after op
  auto forkSize = opsToProcess.size();
  auto newFork = ForkOp::create(rewriter, token.getLoc(), token, forkSize);

  // Modify operands of successor
  // opsToProcess may have multiple instances of same operand
  // Replace uses one by one to assign different fork outputs to them
  for (auto [op, forkOutput] : llvm::zip(opsToProcess, newFork->getResults())) {
    Value forkRes = forkOutput;
    if (isValue)
      forkRes = PackOp::create(rewriter, forkRes.getLoc(), forkRes, value)
                    .getOutput();
    replaceFirstUse(op, result, forkRes);
  }
}

// Insert Fork Operation for every operation and function argument with more
// than one successor.
static LogicalResult addForkOps(Block &block, OpBuilder &rewriter) {
  // Materialization adds operations _after_ their definition, so we can't use
  // llvm::make_early_inc_range. Copy over all of the ops to process.
  llvm::SmallVector<Operation *> opsToProcess;
  for (auto &op : block.getOperations())
    opsToProcess.push_back(&op);

  for (Operation *op : opsToProcess) {
    // Ignore terminators.
    if (!op->hasTrait<OpTrait::IsTerminator>()) {
      for (auto result : op->getResults()) {
        if (!isDCTyped(result))
          continue;
        // If there is a result, it is used more than once, and it is a DC
        // type, fork it!
        if (!result.use_empty() && !result.hasOneUse())
          insertFork(result, rewriter);
      }
    }
  }

  for (auto barg : block.getArguments())
    if (!barg.use_empty() && !barg.hasOneUse())
      if (isDCTyped(barg))
        insertFork(barg, rewriter);

  return success();
}

// Create sink for every unused result
static LogicalResult addSinkOps(Block &block, OpBuilder &rewriter) {
  for (auto arg : block.getArguments()) {
    if (isDCTyped(arg) && arg.use_empty())
      insertSink(arg, rewriter);
  }

  // Materialization adds operations _after_ their definition, so we can't use
  // llvm::make_early_inc_range. Copy over all of the ops to process.
  llvm::SmallVector<Operation *> opsToProcess;
  for (auto &op : block.getOperations())
    opsToProcess.push_back(&op);

  for (Operation *op : opsToProcess) {
    if (op->getNumResults() == 0)
      continue;

    for (auto result : op->getResults()) {
      if (isDCTyped(result) && result.use_empty())
        insertSink(result, rewriter);
    }
  }

  return success();
}

namespace {
struct DCMaterializeForksSinksPass
    : public circt::dc::impl::DCMaterializeForksSinksBase<
          DCMaterializeForksSinksPass> {
  void runOnOperation() override {
    auto *op = getOperation();
    OpBuilder builder(op);

    auto walkRes = op->walk([&](mlir::Block *block) {
      if (addForkOps(*block, builder).failed() ||
          addSinkOps(*block, builder).failed())
        return WalkResult::interrupt();

      return WalkResult::advance();
    });

    if (walkRes.wasInterrupted())
      return signalPassFailure();
  };
};

struct DCDematerializeForksSinksPass
    : public circt::dc::impl::DCDematerializeForksSinksBase<
          DCDematerializeForksSinksPass> {
  void runOnOperation() override {
    auto *op = getOperation();
    op->walk([&](dc::SinkOp sinkOp) { sinkOp.erase(); });
    op->walk([&](dc::ForkOp forkOp) {
      for (auto res : forkOp->getResults())
        res.replaceAllUsesWith(forkOp.getOperand());
      forkOp.erase();
    });
  };
};

} // namespace

std::unique_ptr<mlir::Pass> circt::dc::createDCMaterializeForksSinksPass() {
  return std::make_unique<DCMaterializeForksSinksPass>();
}

std::unique_ptr<mlir::Pass> circt::dc::createDCDematerializeForksSinksPass() {
  return std::make_unique<DCDematerializeForksSinksPass>();
}
