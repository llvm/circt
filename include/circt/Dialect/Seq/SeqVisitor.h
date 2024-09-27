//===- SeqVisitors.h - Seq Dialect Visitors ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines visitors that make it easier to work with Seq IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SEQ_SEQVISITORS_H
#define CIRCT_DIALECT_SEQ_SEQVISITORS_H

#include "circt/Dialect/Seq/SeqOps.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace seq {

/// This helps visit TypeOp nodes.
template <typename ConcreteType, typename ResultType = void,
          typename... ExtraArgs>
class SeqOpVisitor {
public:
  ResultType dispatchSeqOpVisitor(Operation *op, ExtraArgs... args) {
    auto *thisCast = static_cast<ConcreteType *>(this);
    return TypeSwitch<Operation *, ResultType>(op)
        .template Case<
            // Registers.
            CompRegOp, CompRegClockEnabledOp, ShiftRegOp, FirRegOp, FIFOOp,
            // Memories.
            HLMemOp, ReadPortOp, WritePortOp, FirMemOp, FirMemReadOp,
            FirMemWriteOp, FirMemReadWriteOp,
            // Clock.
            ClockGateOp, ClockMuxOp, ClockDividerOp, ClockInverterOp,
            ConstClockOp, ToClockOp, FromClockOp>([&](auto expr) -> ResultType {
          return thisCast->visitSeq(expr, args...);
        })
        .Default([&](auto expr) -> ResultType {
          return thisCast->visitInvalidSeqOp(op, args...);
        });
  }

  /// This callback is invoked on any non-expression operations.
  ResultType visitInvalidSeqOp(Operation *op, ExtraArgs... args) {
    op->emitOpError("unknown seq op");
    abort();
  }

  /// This callback is invoked on any combinational operations that are not
  /// handled by the concrete visitor.
  ResultType visitUnhandledSeqOp(Operation *op, ExtraArgs... args) {
    return ResultType();
  }

#define HANDLE(OPTYPE, OPKIND)                                                 \
  ResultType visitSeq(OPTYPE op, ExtraArgs... args) {                       \
    return static_cast<ConcreteType *>(this)->visit##OPKIND##SeqOp(op,         \
                                                                   args...);   \
  }

  // Registers.
  HANDLE(CompRegOp, Unhandled);
  HANDLE(CompRegClockEnabledOp, Unhandled);
  HANDLE(ShiftRegOp, Unhandled);

  HANDLE(FirRegOp, Unhandled);
  HANDLE(FIFOOp, Unhandled);

  // Memories.
  HANDLE(HLMemOp, Unhandled);
  HANDLE(ReadPortOp, Unhandled);
  HANDLE(WritePortOp, Unhandled);

  // FIRRTL memory ops.
  HANDLE(FirMemOp, Unhandled);
  HANDLE(FirMemReadOp, Unhandled);
  HANDLE(FirMemWriteOp, Unhandled);
  HANDLE(FirMemReadWriteOp, Unhandled);

  // Clock gate.
  HANDLE(ClockGateOp, Unhandled);
  HANDLE(ClockMuxOp, Unhandled);
  HANDLE(ClockDividerOp, Unhandled);
  HANDLE(ClockInverterOp, Unhandled);

  // Tied-off clock
  HANDLE(ConstClockOp, Unhandled);

  // Clock casts.
  HANDLE(ToClockOp, Unhandled);
  HANDLE(FromClockOp, Unhandled);

#undef HANDLE
};

} // namespace seq
} // namespace circt

#endif // CIRCT_DIALECT_SEQ_SEQVISITORS_H
