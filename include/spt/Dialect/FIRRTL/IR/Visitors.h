//===- FIRRTL/IR/Visitors.h - FIRRTL Dialect Visitors -----------*- C++ -*-===//
//
// This file defines visitors that make it easier to work with FIRRTL IR.
//
//===----------------------------------------------------------------------===//

#ifndef SPT_DIALECT_FIRRTL_IR_VISITORS_H
#define SPT_DIALECT_FIRRTL_IR_VISITORS_H

#include "spt/Dialect/FIRRTL/IR/Ops.h"
#include "llvm/ADT/TypeSwitch.h"

namespace spt {
namespace firrtl {

/// ExprVisitor is a visitor for FIRRTL expression nodes.
template <typename ConcreteType, typename ResultType = void>
class ExprVisitor {
public:
  ResultType visitExpr(Operation *op) {
    return TypeSwitch<Operation *, ResultType>(op)
        .template Case<AndRPrimOp, XorRPrimOp>([&](auto expr) -> ResultType {
          return static_cast<ConcreteType *>(this)->visitExpr(expr);
        })
        .Default([&](auto expr) -> ResultType {
          return static_cast<ConcreteType *>(this)->visitInvalidExpr(op);
        });
  }

  /// This callback is invoked on any non-expression operations.
  ResultType visitInvalidExpr(Operation *op) {
    op->emitOpError("unknown firrtl expression");
    abort();
  }

  /// This callback is invoked on any expression operations that are not handled
  /// by the concrete visitor.
  ResultType visitUnhandledExpr(Operation *op) { return ResultType(); }

#define HANDLE(OPTYPE)                                                         \
  ResultType visitExpr(OPTYPE op) {                                            \
    return static_cast<ConcreteType *>(this)->visitUnhandledExpr(op);          \
  }

  // Basic expressions.
  HANDLE(ConstantOp)
  HANDLE(SubfieldOp);
  HANDLE(SubindexOp);
  HANDLE(SubaccessOp);

  // Arithmetic and Logical Binary Primitives.
  HANDLE(AddPrimOp);
  HANDLE(SubPrimOp);
  HANDLE(MulPrimOp);
  HANDLE(DivPrimOp);
  HANDLE(RemPrimOp);
  HANDLE(AndPrimOp);
  HANDLE(OrPrimOp);
  HANDLE(XorPrimOp);

  // Comparisons.
  HANDLE(LEQPrimOp);
  HANDLE(LTPrimOp);
  HANDLE(GEQPrimOp);
  HANDLE(GTPrimOp);
  HANDLE(EQPrimOp);
  HANDLE(NEQPrimOp);

  // Misc Binary Primitives.
  HANDLE(CatPrimOp);
  HANDLE(DShlPrimOp);
  HANDLE(DShrPrimOp);
  HANDLE(ValidIfPrimOp);

  // Unary operators.
  HANDLE(AsSIntPrimOp);
  HANDLE(AsUIntPrimOp);
  HANDLE(AsAsyncResetPrimOp);
  HANDLE(AsClockPrimOp);
  HANDLE(CvtPrimOp);
  HANDLE(NegPrimOp);
  HANDLE(NotPrimOp);
  HANDLE(AndRPrimOp);
  HANDLE(OrRPrimOp);
  HANDLE(XorRPrimOp);

  // Miscellaneous.
  HANDLE(BitsPrimOp);
  HANDLE(HeadPrimOp);
  HANDLE(MuxPrimOp);
  HANDLE(PadPrimOp);
  HANDLE(ShlPrimOp);
  HANDLE(ShrPrimOp);
  HANDLE(TailPrimOp);
};

} // namespace firrtl
} // namespace spt

#endif // SPT_DIALECT_FIRRTL_IR_VISITORS_H