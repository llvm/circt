//===- SimplifyRefs.cpp - moore.concat_ref and queue reference lowering -- ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the SimplifyRefs pass.
// It has two purposes:
// - To disassemble the moore.concat_ref. Which is tricky to lower
// directly. For example, disassemble "{a, b} = c" onto "a = c[7:3]"
// and "b = c[2:0]".
// - To eliminate moore.dyn_queue_ref_element in the case where the reference
// immediately has a value assigned via blocking assignment, replacing it with
// moore.queue.set. Queue element references are tricky to lower into LLHD,
// so it's best to get rid of them.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MoorePasses.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
namespace moore {
#define GEN_PASS_DEF_SIMPLIFYREFS
#include "circt/Dialect/Moore/MoorePasses.h.inc"
} // namespace moore
} // namespace circt

using namespace circt;
using namespace moore;
using namespace mlir;

namespace {

// A helper function for collecting the non-concatRef operands of concatRef.
static void collectOperands(Value operand, SmallVectorImpl<Value> &operands,
                            ConversionPatternRewriter &rewriter) {
  if (auto concatRefOp = operand.getDefiningOp<ConcatRefOp>()) {
    // Assuming the assignment is the only user, erase the op now.
    if (std::distance(concatRefOp->getUsers().begin(),
                      concatRefOp->getUsers().end()) == 1) {
      rewriter.eraseOp(concatRefOp);
    }
    for (auto nestedOperand : concatRefOp.getValues())
      collectOperands(nestedOperand, operands, rewriter);

  } else
    operands.push_back(operand);
}

template <typename OpTy>
struct ConcatRefLowering : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Use to collect the operands of concatRef.
    SmallVector<Value, 4> operands;
    collectOperands(op.getDst(), operands, rewriter);
    auto srcWidth =
        cast<UnpackedType>(op.getSrc().getType()).getBitSize().value();

    // Disassemble assignments with the LHS is concatRef. And create new
    // corresponding assignments using non-concatRef LHS.
    for (auto operand : operands) {
      auto type = cast<RefType>(operand.getType()).getNestedType();
      auto width = type.getBitSize().value();

      rewriter.setInsertionPoint(op);
      // FIXME: Need to estimate whether the bits range is from large to
      // small or vice versa. Like "logic [7:0] or [0:7]".

      // Only able to correctly handle the situation like "[7:0]" now.
      auto extract = ExtractOp::create(rewriter, op.getLoc(), type, op.getSrc(),
                                       srcWidth - width);

      // Update the real bit width of RHS of assignment. Like "c" the above
      // description mentioned.
      srcWidth = srcWidth - width;

      OpTy::create(rewriter, op.getLoc(), operand, extract);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct QueueRefLowering : public OpConversionPattern<DynQueueRefElementOp> {
  using OpConversionPattern<DynQueueRefElementOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DynQueueRefElementOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // For now, we only support using a queue reference in the LHS of a blocking
    // assignment op.
    for (auto *consumer : op->getUsers()) {
      if (isa<BlockingAssignOp>(consumer)) {

        auto assignOp = cast<BlockingAssignOp>(consumer);
        // Replace BlockingAssignOp with a queue.set operation to the index.
        rewriter.setInsertionPoint(consumer);
        moore::QueueSetOp::create(rewriter, op->getLoc(), op.getInput(),
                                  op.getIndex(), assignOp.getSrc());

        rewriter.eraseOp(assignOp);
      } else {
        return mlir::emitError(op.getLoc())
               << "Queue element reference couldn't be reduced to setting the "
                  "value at an index: consuming op "
               << consumer << " is not supported";
      }
    }

    rewriter.eraseOp(op);

    return success();
  }
};

struct SimplifyRefsPass
    : public circt::moore::impl::SimplifyRefsBase<SimplifyRefsPass> {
  void runOnOperation() override;
};

} // namespace

std::unique_ptr<mlir::Pass> circt::moore::createSimplifyRefsPass() {
  return std::make_unique<SimplifyRefsPass>();
}

void SimplifyRefsPass::runOnOperation() {
  MLIRContext &context = getContext();
  ConversionTarget target(context);

  target.addDynamicallyLegalOp<ContinuousAssignOp, BlockingAssignOp,
                               NonBlockingAssignOp>([](auto op) {
    return !op->getOperand(0).template getDefiningOp<ConcatRefOp>();
  });

  target.addLegalDialect<MooreDialect>();
  RewritePatternSet patterns(&context);
  patterns.add<ConcatRefLowering<ContinuousAssignOp>,
               ConcatRefLowering<BlockingAssignOp>,
               ConcatRefLowering<NonBlockingAssignOp>>(&context);

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
    return;
  }

  // Once we have removed ConcatRefOps, attempt to rewrite any queue element
  // references to queue.set
  patterns.clear();
  target.addIllegalOp<DynQueueRefElementOp>();
  patterns.add<QueueRefLowering>(&context);
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
