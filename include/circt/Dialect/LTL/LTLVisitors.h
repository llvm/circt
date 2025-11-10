//===- LTLVisitors.h - LTL dialect visitors ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LTL_LTLVISITORS_H
#define CIRCT_DIALECT_LTL_LTLVISITORS_H

#include "circt/Dialect/LTL/LTLOps.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace ltl {
template <typename ConcreteType, typename ResultType = void,
          typename... ExtraArgs>
class Visitor {
public:
  ResultType dispatchLTLVisitor(Operation *op, ExtraArgs... args) {
    auto *thisCast = static_cast<ConcreteType *>(this);
    return TypeSwitch<Operation *, ResultType>(op)
        .template Case<AndOp, OrOp, DelayOp, ConcatOp, RepeatOp, NotOp,
                       ImplicationOp, UntilOp, EventuallyOp, ClockOp,
                       ClockedAndOp, ClockedOrOp, ClockedIntersectOp,
                       CreateClockedSequenceOp, ClockedDelayOp, ClockedConcatOp,
                       IntersectOp, ClockedNonConsecutiveRepeatOp,
                       ClockedGoToRepeatOp, ClockedRepeatOp,
                       NonConsecutiveRepeatOp, GoToRepeatOp>(
            [&](auto op) -> ResultType {
              return thisCast->visitLTL(op, args...);
            })
        .Default([&](auto) -> ResultType {
          return thisCast->visitInvalidLTL(op, args...);
        });
  }

  /// This callback is invoked on any non-LTL operations.
  ResultType visitInvalidLTL(Operation *op, ExtraArgs... args) {
    op->emitOpError("is not an LTL operation");
    abort();
  }

  /// This callback is invoked on any LTL operations that were not handled by
  /// their concrete `visitLTL(...)` callback.
  ResultType visitUnhandledLTL(Operation *op, ExtraArgs... args) {
    return ResultType();
  }

#define HANDLE(OPTYPE, OPKIND)                                                 \
  ResultType visitLTL(OPTYPE op, ExtraArgs... args) {                          \
    return static_cast<ConcreteType *>(this)->visit##OPKIND##LTL(op, args...); \
  }

  HANDLE(AndOp, Unhandled);
  HANDLE(OrOp, Unhandled);
  HANDLE(DelayOp, Unhandled);
  HANDLE(ConcatOp, Unhandled);
  HANDLE(RepeatOp, Unhandled);
  HANDLE(NotOp, Unhandled);
  HANDLE(ImplicationOp, Unhandled);
  HANDLE(UntilOp, Unhandled);
  HANDLE(EventuallyOp, Unhandled);
  HANDLE(ClockOp, Unhandled);
  HANDLE(IntersectOp, Unhandled);
  HANDLE(NonConsecutiveRepeatOp, Unhandled);
  HANDLE(GoToRepeatOp, Unhandled);
  HANDLE(ClockedAndOp, Unhandled);
  HANDLE(ClockedConcatOp, Unhandled);
  HANDLE(ClockedDelayOp, Unhandled);
  HANDLE(ClockedGoToRepeatOp, Unhandled);
  HANDLE(ClockedIntersectOp, Unhandled);
  HANDLE(ClockedNonConsecutiveRepeatOp, Unhandled);
  HANDLE(ClockedOrOp, Unhandled);
  HANDLE(ClockedRepeatOp, Unhandled);
  HANDLE(CreateClockedSequenceOp, Unhandled);
#undef HANDLE
};

} // namespace ltl
} // namespace circt

#endif // CIRCT_DIALECT_LTL_LTLVISITORS_H
