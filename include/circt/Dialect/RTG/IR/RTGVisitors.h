//===- RTGVisitors.h - RTG Dialect Visitors ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines visitors that make it easier to work with RTG IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTG_IR_RTGVISITORS_H
#define CIRCT_DIALECT_RTG_IR_RTGVISITORS_H

#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace rtg {

/// This helps visit TypeOp nodes.
template <typename ConcreteType, typename ResultType = void,
          typename... ExtraArgs>
class RTGOpVisitor {
public:
  ResultType dispatchOpVisitor(Operation *op, ExtraArgs... args) {
    auto *thisCast = static_cast<ConcreteType *>(this);
    return TypeSwitch<Operation *, ResultType>(op)
        .template Case<
            // Constants
            ConstantOp,
            // Bags
            BagCreateOp, BagSelectRandomOp, BagDifferenceOp, BagUnionOp,
            BagUniqueSizeOp, BagConvertToSetOp,
            // Contexts
            OnContextOp, ContextSwitchOp,
            // Labels
            StringToLabelOp, LabelUniqueDeclOp, LabelOp,
            // Registers
            VirtualRegisterOp, RegisterToIndexOp, IndexToRegisterOp,
            // RTG tests
            TestOp, TargetOp, YieldOp, ValidateOp, TestSuccessOp, TestFailureOp,
            // Integers
            RandomNumberInRangeOp,
            // Sequences
            SequenceOp, GetSequenceOp, SubstituteSequenceOp,
            RandomizeSequenceOp, EmbedSequenceOp, InterleaveSequencesOp,
            // Sets
            SetCreateOp, SetSelectRandomOp, SetDifferenceOp, SetUnionOp,
            SetSizeOp, SetCartesianProductOp, SetConvertToBagOp,
            // Arrays
            ArrayCreateOp, ArrayExtractOp, ArrayInjectOp, ArraySizeOp,
            ArrayAppendOp,
            // Tuples
            TupleCreateOp, TupleExtractOp,
            // Immediates
            IntToImmediateOp, ConcatImmediateOp, SliceImmediateOp,
            // Memories
            MemoryAllocOp, MemoryBaseAddressOp, MemorySizeOp,
            // Memory Blocks
            MemoryBlockDeclareOp,
            // Data segment ops
            SpaceOp, StringDataOp, SegmentOp,
            // String ops
            StringConcatOp, IntFormatOp, ImmediateFormatOp, RegisterFormatOp,
            StringToASCIIArrayOp,
            // Misc ops
            CommentOp, ConstraintOp, RandomScopeOp,
            // Effect handler ops
            EffectOp, WithHandlersOp, PerformOp, ResumeOp>(
            [&](auto expr) -> ResultType {
              return thisCast->visitOp(expr, args...);
            })
        .Default([&](auto expr) -> ResultType {
          if (op->getDialect() ==
              op->getContext()->getLoadedDialect<RTGDialect>())
            return visitInvalidTypeOp(op, args...);

          return thisCast->visitExternalOp(op, args...);
        });
  }

  /// This callback is invoked on any RTG operations not handled properly by the
  /// TypeSwitch.
  ResultType visitInvalidTypeOp(Operation *op, ExtraArgs... args) {
    op->emitOpError("Unknown RTG operation: ") << op->getName();
    abort();
  }

  /// This callback is invoked on any operations that are not
  /// handled by the concrete visitor.
  ResultType visitUnhandledOp(Operation *op, ExtraArgs... args);

  ResultType visitExternalOp(Operation *op, ExtraArgs... args) {
    return ResultType();
  }

#define HANDLE(OPTYPE, OPKIND)                                                 \
  ResultType visitOp(OPTYPE op, ExtraArgs... args) {                           \
    return static_cast<ConcreteType *>(this)->visit##OPKIND##Op(op, args...);  \
  }

  HANDLE(ConstantOp, Unhandled);
  HANDLE(SequenceOp, Unhandled);
  HANDLE(GetSequenceOp, Unhandled);
  HANDLE(SubstituteSequenceOp, Unhandled);
  HANDLE(RandomizeSequenceOp, Unhandled);
  HANDLE(InterleaveSequencesOp, Unhandled);
  HANDLE(EmbedSequenceOp, Unhandled);
  HANDLE(RandomNumberInRangeOp, Unhandled);
  HANDLE(OnContextOp, Unhandled);
  HANDLE(ContextSwitchOp, Unhandled);
  HANDLE(SetCreateOp, Unhandled);
  HANDLE(SetSelectRandomOp, Unhandled);
  HANDLE(SetDifferenceOp, Unhandled);
  HANDLE(SetUnionOp, Unhandled);
  HANDLE(SetSizeOp, Unhandled);
  HANDLE(SetCartesianProductOp, Unhandled);
  HANDLE(SetConvertToBagOp, Unhandled);
  HANDLE(BagCreateOp, Unhandled);
  HANDLE(BagSelectRandomOp, Unhandled);
  HANDLE(BagDifferenceOp, Unhandled);
  HANDLE(BagUnionOp, Unhandled);
  HANDLE(BagUniqueSizeOp, Unhandled);
  HANDLE(BagConvertToSetOp, Unhandled);
  HANDLE(ArrayCreateOp, Unhandled);
  HANDLE(ArrayExtractOp, Unhandled);
  HANDLE(ArrayInjectOp, Unhandled);
  HANDLE(ArraySizeOp, Unhandled);
  HANDLE(ArrayAppendOp, Unhandled);
  HANDLE(TupleCreateOp, Unhandled);
  HANDLE(TupleExtractOp, Unhandled);
  HANDLE(CommentOp, Unhandled);
  HANDLE(ConstraintOp, Unhandled);
  HANDLE(LabelUniqueDeclOp, Unhandled);
  HANDLE(LabelOp, Unhandled);
  HANDLE(TestOp, Unhandled);
  HANDLE(TargetOp, Unhandled);
  HANDLE(YieldOp, Unhandled);
  HANDLE(ValidateOp, Unhandled);
  HANDLE(TestSuccessOp, Unhandled);
  HANDLE(TestFailureOp, Unhandled);
  HANDLE(VirtualRegisterOp, Unhandled);
  HANDLE(RegisterToIndexOp, Unhandled);
  HANDLE(IndexToRegisterOp, Unhandled);
  HANDLE(IntToImmediateOp, Unhandled);
  HANDLE(ConcatImmediateOp, Unhandled);
  HANDLE(SliceImmediateOp, Unhandled);
  HANDLE(MemoryBlockDeclareOp, Unhandled);
  HANDLE(MemoryAllocOp, Unhandled);
  HANDLE(MemoryBaseAddressOp, Unhandled);
  HANDLE(MemorySizeOp, Unhandled);
  HANDLE(SpaceOp, Unhandled);
  HANDLE(StringDataOp, Unhandled);
  HANDLE(SegmentOp, Unhandled);
  HANDLE(StringConcatOp, Unhandled);
  HANDLE(IntFormatOp, Unhandled);
  HANDLE(ImmediateFormatOp, Unhandled);
  HANDLE(RegisterFormatOp, Unhandled);
  HANDLE(StringToLabelOp, Unhandled);
  HANDLE(StringToASCIIArrayOp, Unhandled);
  HANDLE(RandomScopeOp, Unhandled);
  HANDLE(EffectOp, Unhandled);
  HANDLE(WithHandlersOp, Unhandled);
  HANDLE(PerformOp, Unhandled);
  HANDLE(ResumeOp, Unhandled);
#undef HANDLE
};

} // namespace rtg
} // namespace circt

#endif // CIRCT_DIALECT_RTG_IR_RTGVISITORS_H
