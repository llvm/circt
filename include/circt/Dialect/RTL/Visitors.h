//===- RTL/Visitors.h - RTL Dialect Visitors --------------------*- C++ -*-===//
//
// This file defines visitors that make it easier to work with RTL IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTL_VISITORS_H
#define CIRCT_DIALECT_RTL_VISITORS_H

#include "circt/Dialect/RTL/Ops.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace rtl {

/// This helps visit Combinatorial nodes.
template <typename ConcreteType, typename ResultType = void,
          typename... ExtraArgs>
class CombinatorialVisitor {
public:
  ResultType dispatchCombinatorialVisitor(Operation *op, ExtraArgs... args) {
    auto *thisCast = static_cast<ConcreteType *>(this);
    return TypeSwitch<Operation *, ResultType>(op)
        .template Case<ConstantOp,
                       // Arithmetic and Logical Binary Operations.
                       AddOp, SubOp, MulOp, DivOp, ModOp, AndOp, OrOp, XorOp,
                       // Other operations.
                       SExtOp, ZExtOp, ConcatOp>([&](auto expr) -> ResultType {
          return thisCast->visitComb(expr, args...);
        })
        .Default([&](auto expr) -> ResultType {
          return thisCast->visitInvalidComb(op, args...);
        });
  }

  /// This callback is invoked on any non-expression operations.
  ResultType visitInvalidComb(Operation *op, ExtraArgs... args) {
    op->emitOpError("unknown RTL combinatorial node");
    abort();
  }

  /// This callback is invoked on any combinatorial operations that are not
  /// handled by the concrete visitor.
  ResultType visitUnhandledComb(Operation *op, ExtraArgs... args) {
    return ResultType();
  }

  /// This fallback is invoked on any binary node that isn't explicitly handled.
  /// The default implementation delegates to the 'unhandled' fallback.
  ResultType visitBinaryComb(Operation *op, ExtraArgs... args) {
    return static_cast<ConcreteType *>(this)->visitUnhandledComb(op, args...);
  }

#define HANDLE(OPTYPE, OPKIND)                                                 \
  ResultType visitComb(OPTYPE op, ExtraArgs... args) {                         \
    return static_cast<ConcreteType *>(this)->visit##OPKIND##Comb(op,          \
                                                                  args...);    \
  }

  // Basic nodes.
  HANDLE(ConstantOp, Unhandled)

  // Arithmetic and Logical Binary Operations.
  HANDLE(AddOp, Binary);
  HANDLE(SubOp, Binary);
  HANDLE(MulOp, Binary);
  HANDLE(DivOp, Binary);
  HANDLE(ModOp, Binary);
  HANDLE(AndOp, Binary);
  HANDLE(OrOp, Binary);
  HANDLE(XorOp, Binary);

  // Other operations.
  HANDLE(SExtOp, Unhandled);
  HANDLE(ZExtOp, Unhandled);
  HANDLE(ConcatOp, Unhandled);
#undef HANDLE
};

} // namespace rtl
} // namespace circt

#endif // CIRCT_DIALECT_RTL_VISITORS_H