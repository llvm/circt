//===- ArithVisitors.h - Arith Dialect Visitors -----------------*- C++ -*-===//
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

#ifndef CIRCT_DIALECT_RTG_IR_ARITHVISITORS_H
#define CIRCT_DIALECT_RTG_IR_ARITHVISITORS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace arith {

/// This helps visit TypeOp nodes.
template <typename ConcreteType, typename ResultType = void,
          typename... ExtraArgs>
class ArithOpVisitor {
public:
  ResultType dispatchOpVisitor(Operation *op, ExtraArgs... args) {
    auto *thisCast = static_cast<ConcreteType *>(this);
    return TypeSwitch<Operation *, ResultType>(op)
        .template Case<ConstantOp, AddIOp, AddUIExtendedOp, SubIOp, MulIOp,
                       MulSIExtendedOp, MulUIExtendedOp, DivUIOp, DivSIOp,
                       CeilDivUIOp, CeilDivSIOp, FloorDivSIOp, RemUIOp, RemSIOp,
                       AndIOp, OrIOp, XOrIOp, ShLIOp, ShRUIOp, ShRSIOp, NegFOp,
                       AddFOp, SubFOp, MaximumFOp, MaxNumFOp, MaxSIOp, MaxUIOp,
                       MinimumFOp, MinNumFOp, MinSIOp, MinUIOp, MulFOp, DivFOp,
                       RemFOp, ExtUIOp, ExtSIOp, ExtFOp, TruncIOp, TruncFOp,
                       UIToFPOp, SIToFPOp, FPToUIOp, FPToSIOp, IndexCastOp,
                       IndexCastUIOp, BitcastOp, CmpIOp, CmpFOp, SelectOp>(
            [&](auto expr) -> ResultType {
              return thisCast->visitOp(expr, args...);
            })
        .Default([&](auto expr) -> ResultType {
          if (op->getDialect() ==
              op->getContext()->getLoadedDialect<ArithDialect>()) {
            llvm::errs() << "Unknown Arith operation: " << op->getName()
                         << "\n";
            abort();
            return failure();
          }
          return thisCast->visitExternalOp(op, args...);
        });
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
  HANDLE(AddIOp, Unhandled);
  HANDLE(AddUIExtendedOp, Unhandled);
  HANDLE(SubIOp, Unhandled);
  HANDLE(MulIOp, Unhandled);
  HANDLE(MulSIExtendedOp, Unhandled);
  HANDLE(MulUIExtendedOp, Unhandled);
  HANDLE(DivUIOp, Unhandled);
  HANDLE(DivSIOp, Unhandled);
  HANDLE(CeilDivUIOp, Unhandled);
  HANDLE(CeilDivSIOp, Unhandled);
  HANDLE(FloorDivSIOp, Unhandled);
  HANDLE(RemUIOp, Unhandled);
  HANDLE(RemSIOp, Unhandled);
  HANDLE(AndIOp, Unhandled);
  HANDLE(OrIOp, Unhandled);
  HANDLE(XOrIOp, Unhandled);
  HANDLE(ShLIOp, Unhandled);
  HANDLE(ShRUIOp, Unhandled);
  HANDLE(ShRSIOp, Unhandled);
  HANDLE(NegFOp, Unhandled);
  HANDLE(AddFOp, Unhandled);
  HANDLE(SubFOp, Unhandled);
  HANDLE(MaximumFOp, Unhandled);
  HANDLE(MaxNumFOp, Unhandled);
  HANDLE(MaxSIOp, Unhandled);
  HANDLE(MaxUIOp, Unhandled);
  HANDLE(MinimumFOp, Unhandled);
  HANDLE(MinNumFOp, Unhandled);
  HANDLE(MinSIOp, Unhandled);
  HANDLE(MinUIOp, Unhandled);
  HANDLE(MulFOp, Unhandled);
  HANDLE(DivFOp, Unhandled);
  HANDLE(RemFOp, Unhandled);
  HANDLE(ExtUIOp, Unhandled);
  HANDLE(ExtSIOp, Unhandled);
  HANDLE(ExtFOp, Unhandled);
  HANDLE(TruncIOp, Unhandled);
  HANDLE(TruncFOp, Unhandled);
  HANDLE(UIToFPOp, Unhandled);
  HANDLE(SIToFPOp, Unhandled);
  HANDLE(FPToUIOp, Unhandled);
  HANDLE(FPToSIOp, Unhandled);
  HANDLE(IndexCastOp, Unhandled);
  HANDLE(IndexCastUIOp, Unhandled);
  HANDLE(BitcastOp, Unhandled);
  HANDLE(CmpIOp, Unhandled);
  HANDLE(CmpFOp, Unhandled);
  HANDLE(SelectOp, Unhandled);
#undef HANDLE
};

} // namespace arith
} // namespace mlir

#endif // CIRCT_DIALECT_RTG_IR_RTGVISITORS_H
