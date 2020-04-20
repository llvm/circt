//===- FIRRTL/IR/Visitors.h - FIRRTL Dialect Visitors -----------*- C++ -*-===//
//
// This file defines visitors that make it easier to work with FIRRTL IR.
//
//===----------------------------------------------------------------------===//

#ifndef SPT_DIALECT_FIRRTL_IR_VISITORS_H
#define SPT_DIALECT_FIRRTL_IR_VISITORS_H

#include "spt/Dialect/FIRRTL/Ops.h"
#include "llvm/ADT/TypeSwitch.h"

namespace spt {
namespace firrtl {

/// ExprVisitor is a visitor for FIRRTL expression nodes.
template <typename ConcreteType, typename ResultType = void,
          typename... ExtraArgs>
class ExprVisitor {
public:
  ResultType dispatchExprVisitor(Operation *op, ExtraArgs... args) {
    auto *thisCast = static_cast<ConcreteType *>(this);
    return TypeSwitch<Operation *, ResultType>(op)
        // Basic Expressions
        .template Case<
            ConstantOp, SubfieldOp, SubindexOp, SubaccessOp,
            // Arithmetic and Logical Binary Primitives.
            AddPrimOp, SubPrimOp, MulPrimOp, DivPrimOp, RemPrimOp, AndPrimOp,
            OrPrimOp, XorPrimOp,
            // Comparisons.
            LEQPrimOp, LTPrimOp, GEQPrimOp, GTPrimOp, EQPrimOp, NEQPrimOp,
            // Misc Binary Primitives.
            CatPrimOp, DShlPrimOp, DShrPrimOp, ValidIfPrimOp,
            // Unary operators.
            AsSIntPrimOp, AsUIntPrimOp, AsAsyncResetPrimOp, AsClockPrimOp,
            CvtPrimOp, NegPrimOp, NotPrimOp, AndRPrimOp, OrRPrimOp, XorRPrimOp,
            // Miscellaneous.
            BitsPrimOp, HeadPrimOp, MuxPrimOp, PadPrimOp, ShlPrimOp, ShrPrimOp,
            TailPrimOp>([&](auto expr) -> ResultType {
          return thisCast->visitExpr(expr, args...);
        })
        .Default([&](auto expr) -> ResultType {
          return thisCast->visitInvalidExpr(op, args...);
        });
  }

  /// This callback is invoked on any non-expression operations.
  ResultType visitInvalidExpr(Operation *op, ExtraArgs... args) {
    op->emitOpError("unknown firrtl expression");
    abort();
  }

  /// This callback is invoked on any expression operations that are not handled
  /// by the concrete visitor.
  ResultType visitUnhandledExpr(Operation *op, ExtraArgs... args) {
    return ResultType();
  }

  /// This fallback is invoked on any unary expr that isn't explicitly handled.
  /// The default implementation delegates to the unhandled expression fallback.
  ResultType visitUnaryExpr(Operation *op, ExtraArgs... args) {
    return static_cast<ConcreteType *>(this)->visitUnhandledExpr(op, args...);
  }

  /// This fallback is invoked on any binary expr that isn't explicitly handled.
  /// The default implementation delegates to the unhandled expression fallback.
  ResultType visitBinaryExpr(Operation *op, ExtraArgs... args) {
    return static_cast<ConcreteType *>(this)->visitUnhandledExpr(op, args...);
  }

#define HANDLE(OPTYPE, OPKIND)                                                 \
  ResultType visitExpr(OPTYPE op, ExtraArgs... args) {                         \
    return static_cast<ConcreteType *>(this)->visit##OPKIND##Expr(op,          \
                                                                  args...);    \
  }

  // Basic expressions.
  HANDLE(ConstantOp, Unhandled)
  HANDLE(SubfieldOp, Unhandled);
  HANDLE(SubindexOp, Unhandled);
  HANDLE(SubaccessOp, Unhandled);

  // Arithmetic and Logical Binary Primitives.
  HANDLE(AddPrimOp, Binary);
  HANDLE(SubPrimOp, Binary);
  HANDLE(MulPrimOp, Binary);
  HANDLE(DivPrimOp, Binary);
  HANDLE(RemPrimOp, Binary);
  HANDLE(AndPrimOp, Binary);
  HANDLE(OrPrimOp, Binary);
  HANDLE(XorPrimOp, Binary);

  // Comparisons.
  HANDLE(LEQPrimOp, Binary);
  HANDLE(LTPrimOp, Binary);
  HANDLE(GEQPrimOp, Binary);
  HANDLE(GTPrimOp, Binary);
  HANDLE(EQPrimOp, Binary);
  HANDLE(NEQPrimOp, Binary);

  // Misc Binary Primitives.
  HANDLE(CatPrimOp, Binary);
  HANDLE(DShlPrimOp, Binary);
  HANDLE(DShrPrimOp, Binary);
  HANDLE(ValidIfPrimOp, Binary);

  // Unary operators.
  HANDLE(AsSIntPrimOp, Unary);
  HANDLE(AsUIntPrimOp, Unary);
  HANDLE(AsAsyncResetPrimOp, Unary);
  HANDLE(AsClockPrimOp, Unary);
  HANDLE(CvtPrimOp, Unary);
  HANDLE(NegPrimOp, Unary);
  HANDLE(NotPrimOp, Unary);
  HANDLE(AndRPrimOp, Unary);
  HANDLE(OrRPrimOp, Unary);
  HANDLE(XorRPrimOp, Unary);

  // Miscellaneous.
  HANDLE(BitsPrimOp, Unhandled);
  HANDLE(HeadPrimOp, Unhandled);
  HANDLE(MuxPrimOp, Unhandled);
  HANDLE(PadPrimOp, Unhandled);
  HANDLE(ShlPrimOp, Unhandled);
  HANDLE(ShrPrimOp, Unhandled);
  HANDLE(TailPrimOp, Unhandled);
};

} // namespace firrtl
} // namespace spt

#endif // SPT_DIALECT_FIRRTL_IR_VISITORS_H