//===- SimplifyRefs.cpp - moore.concat_ref and queue reference lowering -- ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the SimplifyRefs pass.
// It has three purposes:
// - To disassemble the moore.concat_ref. Which is tricky to lower
// directly. For example, disassemble "{a, b} = c" onto "a = c[7:3]"
// and "b = c[2:0]".
// - To eliminate moore.dyn_queue_ref_element in the case where the reference
// immediately has a value assigned via blocking assignment, replacing it with
// moore.queue.set. Queue element references are tricky to lower into LLHD,
// so it's best to get rid of them.
// - To rewrite assignments to ExtractOp expressions on a packed struct,
// e.g. "s[7:0] = v", into an assignment to a concatenation of the struct's
// (possibly further sliced) fields.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MoorePasses.h"
#include "mlir/IR/IRMapping.h"
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

      // Clone the original op (preserving any extra operand, e.g. a delay on
      // the delayed assign variants) and remap dst/src to the leaf ref and its
      // extracted slice.
      IRMapping mapping;
      mapping.map(op.getDst(), operand);
      mapping.map(op.getSrc(), Value(extract));
      rewriter.clone(*op.getOperation(), mapping);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct FieldInfo {
  Value field = nullptr;
  uint32_t size = 0;
  uint32_t offset = 0;
};

// A helper function that recursively collects the members of a struct.
static LogicalResult collectFields(Value structRef,
                                   SmallVector<FieldInfo> &fields,
                                   ConversionPatternRewriter &rewriter,
                                   uint32_t initialOffset = 0) {
  auto structType =
      cast<StructType>(cast<RefType>(structRef.getType()).getNestedType());
  uint32_t offset = initialOffset;
  // Visit fields in reverse order (declaration order is MSB-first)
  for (auto &member : llvm::reverse(structType.getMembers())) {
    auto fieldRef = StructExtractRefOp::create(rewriter, structRef.getLoc(),
                                               RefType::get(member.type),
                                               member.name, structRef);

    auto fieldSize = member.type.getBitSize();
    if (!fieldSize)
      return mlir::emitError(structRef.getLoc())
             << "unsupported: field with unknown size in struct flattening";

    if (isa<StructType>(member.type)) {
      auto result = collectFields(fieldRef, fields, rewriter, offset);
      if (failed(result))
        return result;
    } else if (isa<UnionType>(member.type)) {
      return mlir::emitError(structRef.getLoc())
             << "unsupported: union member in struct flattening";
    } else {
      fields.push_back({fieldRef, *fieldSize, offset});
    }
    offset += *fieldSize;
  }
  return success();
}

template <typename OpTy>
struct StructExtractLowering : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value dst = op.getDst();

    // We are specifically matching cases in which the LHS of the assignment is
    // an ExtractRef operation on a struct.
    auto extractRefOp = dyn_cast<ExtractRefOp>(dst.getDefiningOp());
    auto baseStructType = dyn_cast_if_present<StructType>(
        cast<RefType>(extractRefOp.getInput().getType()).getNestedType());
    if (!baseStructType)
      return success();

    // Get the boundaries of the ExtractOp.
    uint32_t extractedLow = extractRefOp.getLowBit();
    auto targetWidth =
        cast<RefType>(dst.getType()).getNestedType().getBitSize();
    if (!targetWidth)
      return mlir::emitError(op.getLoc())
             << "unsupported: found field with unknown size in struct "
                "flattening";

    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();

    // Collect all the fields of the struct.
    SmallVector<FieldInfo> fields;
    if (failed(collectFields(extractRefOp.getInput(), fields, rewriter))) {
      return failure();
    }

    // Select only the fields revelant to the ExtractOp
    SmallVector<Value> relevantFields;
    uint32_t offsetInStruct = extractedLow;
    uint32_t remaining = *targetWidth;
    for (auto &field : fields) {
      if (remaining == 0)
        break;

      if (field.offset <= offsetInStruct &&
          field.offset + field.size > offsetInStruct) {
        uint32_t offsetInField = offsetInStruct - field.offset;
        uint32_t remainingInField = field.size - offsetInField;
        uint32_t sizeToExtract = std::min(remaining, remainingInField);

        auto fieldType = cast<RefType>(field.field.getType()).getNestedType();
        auto extractType =
            IntType::get(rewriter.getContext(), sizeToExtract,
                         cast<PackedType>(fieldType).getDomain());
        auto newLhs =
            ExtractRefOp::create(rewriter, loc, RefType::get(extractType),
                                 field.field, offsetInField);
        relevantFields.push_back(newLhs);

        offsetInStruct += sizeToExtract;
        remaining -= sizeToExtract;
      }
    }

    if (relevantFields.size() == 0)
      return mlir::emitError(op.getLoc()) << "Struct extract is out of range";

    // Reverse the fields order to match Concat order
    std::reverse(relevantFields.begin(), relevantFields.end());

    Value finalRef =
        relevantFields.size() == 1
            ? relevantFields.front()
            : Value(ConcatRefOp::create(rewriter, loc, relevantFields));

    IRMapping mapping;
    mapping.map(op.getDst(), finalRef);
    rewriter.clone(*op.getOperation(), mapping);

    rewriter.eraseOp(op);
    rewriter.eraseOp(extractRefOp);
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

struct AssocArrayRefLowering
    : public OpConversionPattern<AssocArrayExtractRefOp> {
  using OpConversionPattern<AssocArrayExtractRefOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AssocArrayExtractRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    for (auto *consumer : op->getUsers()) {
      if (isa<BlockingAssignOp>(consumer)) {
        auto assignOp = cast<BlockingAssignOp>(consumer);
        rewriter.setInsertionPoint(consumer);
        moore::AssocArraySetOp::create(rewriter, op->getLoc(), op.getInput(),
                                       op.getIndex(), assignOp.getSrc());
        rewriter.eraseOp(assignOp);
      } else {
        return mlir::emitError(op.getLoc())
               << "Associative array element reference couldn't be reduced "
                  "to setting the value at an index: consuming op "
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
                               NonBlockingAssignOp, DelayedContinuousAssignOp,
                               DelayedNonBlockingAssignOp>([](auto op) {
    auto extractRefOp =
        op->getOperand(0).template getDefiningOp<ExtractRefOp>();
    if (!extractRefOp)
      return true;
    auto nestedType =
        cast<RefType>(extractRefOp.getInput().getType()).getNestedType();
    return !isa<StructType>(nestedType);
  });

  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  RewritePatternSet extractRefOnStructPatterns(&context);
  extractRefOnStructPatterns
      .add<StructExtractLowering<ContinuousAssignOp>,
           StructExtractLowering<BlockingAssignOp>,
           StructExtractLowering<NonBlockingAssignOp>,
           StructExtractLowering<DelayedContinuousAssignOp>,
           StructExtractLowering<DelayedNonBlockingAssignOp>>(&context);

  if (failed(applyFullConversion(getOperation(), target,
                                 std::move(extractRefOnStructPatterns)))) {
    signalPassFailure();
    return;
  }

  target.addDynamicallyLegalOp<ContinuousAssignOp, BlockingAssignOp,
                               NonBlockingAssignOp, DelayedContinuousAssignOp,
                               DelayedNonBlockingAssignOp>([](auto op) {
    return !op->getOperand(0).template getDefiningOp<ConcatRefOp>();
  });

  RewritePatternSet concatRefPatterns(&context);
  concatRefPatterns.add<ConcatRefLowering<ContinuousAssignOp>,
                        ConcatRefLowering<BlockingAssignOp>,
                        ConcatRefLowering<NonBlockingAssignOp>,
                        ConcatRefLowering<DelayedContinuousAssignOp>,
                        ConcatRefLowering<DelayedNonBlockingAssignOp>>(
      &context);

  if (failed(applyFullConversion(getOperation(), target,
                                 std::move(concatRefPatterns)))) {
    signalPassFailure();
    return;
  }

  // Once we have removed ConcatRefOps, attempt to rewrite any queue element
  // references to queue.set
  RewritePatternSet queueRefPatterns(&context);
  target.addIllegalOp<DynQueueRefElementOp>();
  queueRefPatterns.add<QueueRefLowering>(&context);
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(queueRefPatterns))))
    signalPassFailure();

  // Once we have removed AssocArrayExtractRefOps, attempt to rewrite any
  // associative array element references to assoc_array.set
  RewritePatternSet assocArrayRefPatterns(&context);
  target.addIllegalOp<AssocArrayExtractRefOp>();
  assocArrayRefPatterns.add<AssocArrayRefLowering>(&context);
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(assocArrayRefPatterns))))
    signalPassFailure();
}
