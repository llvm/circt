//===- SVVisitors.h - SV Dialect Visitors -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines visitors that make it easier to work with SV IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SV_SVVISITORS_H
#define CIRCT_DIALECT_SV_SVVISITORS_H

#include "circt/Dialect/SV/SVOps.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace sv {

template <typename ConcreteType, typename ResultType = void,
          typename... ExtraArgs>
class Visitor {
public:
  ResultType dispatchSVVisitor(Operation *op, ExtraArgs... args) {
    auto *thisCast = static_cast<ConcreteType *>(this);
    return TypeSwitch<Operation *, ResultType>(op)
        .template Case<TextualValueOp,
                       // Control flow.
                       IfDefOp, IfOp, AlwaysOp, InitialOp,
                       // Other Statements.
                       YieldOp, BPAssignOp, PAssignOp, AliasOp, FWriteOp,
                       FatalOp, FinishOp, VerbatimOp,
                       // Type declarations.
                       InterfaceOp, InterfaceSignalOp, InterfaceModportOp,
                       InterfaceInstanceOp, GetModportOp,
                       // Verification statements.
                       AssertOp, AssumeOp, CoverOp>(
            [&](auto expr) -> ResultType {
              return thisCast->visitSV(expr, args...);
            })
        .Default([&](auto expr) -> ResultType {
          return thisCast->visitInvalidSV(op, args...);
        });
  }

  /// This callback is invoked on any invalid operations.
  ResultType visitInvalidSV(Operation *op, ExtraArgs... args) {
    op->emitOpError("unknown SV node");
    abort();
  }

  /// This callback is invoked on any SV operations that are not handled by the
  /// concrete visitor.
  ResultType visitUnhandledSV(Operation *op, ExtraArgs... args) {
    return ResultType();
  }

#define HANDLE(OPTYPE, OPKIND)                                                 \
  ResultType visitSV(OPTYPE op, ExtraArgs... args) {                           \
    return static_cast<ConcreteType *>(this)->visit##OPKIND##SV(op, args...);  \
  }

  // Expressions
  HANDLE(TextualValueOp, Unhandled)

  // Control flow.
  HANDLE(IfDefOp, Unhandled);
  HANDLE(IfOp, Unhandled);
  HANDLE(AlwaysOp, Unhandled);
  HANDLE(InitialOp, Unhandled);

  // Other Statements.
  HANDLE(YieldOp, Unhandled);
  HANDLE(BPAssignOp, Unhandled);
  HANDLE(PAssignOp, Unhandled);
  HANDLE(AliasOp, Unhandled);
  HANDLE(FWriteOp, Unhandled);
  HANDLE(FatalOp, Unhandled);
  HANDLE(FinishOp, Unhandled);
  HANDLE(VerbatimOp, Unhandled);

  // Type declarations.
  HANDLE(InterfaceOp, Unhandled);
  HANDLE(InterfaceInstanceOp, Unhandled);
  HANDLE(InterfaceSignalOp, Unhandled);
  HANDLE(InterfaceModportOp, Unhandled);
  HANDLE(GetModportOp, Unhandled);

  // Verification statements.
  HANDLE(AssertOp, Unhandled);
  HANDLE(AssumeOp, Unhandled);
  HANDLE(CoverOp, Unhandled);
#undef HANDLE
};

} // namespace sv
} // namespace circt

#endif // CIRCT_DIALECT_SV_SVVISITORS_H
