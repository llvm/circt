//===- SV/Visitors.h - SV Dialect Visitors ----------------------*- C++ -*-===//
//
// This file defines visitors that make it easier to work with SV IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SV_VISITORS_H
#define CIRCT_DIALECT_SV_VISITORS_H

#include "circt/Dialect/SV/Ops.h"
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
                       IfDefOp, IfOp, AlwaysAtPosEdgeOp,
                       // Other Statements.
                       YieldOp, FWriteOp, FatalOp, FinishOp>(
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
  HANDLE(AlwaysAtPosEdgeOp, Unhandled);

  // Other Statements.
  HANDLE(YieldOp, Unhandled);
  HANDLE(FWriteOp, Unhandled);
  HANDLE(FatalOp, Unhandled);
  HANDLE(FinishOp, Unhandled);
#undef HANDLE
};

} // namespace sv
} // namespace circt

#endif // CIRCT_DIALECT_SV_VISITORS_H
