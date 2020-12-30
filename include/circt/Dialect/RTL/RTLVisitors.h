//===- RTLVisitors.h - RTL Dialect Visitors ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines visitors that make it easier to work with RTL IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTL_RTLVISITORS_H
#define CIRCT_DIALECT_RTL_RTLVISITORS_H

#include "circt/Dialect/RTL/RTLOps.h"
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
                       AddOp, SubOp, MulOp, DivUOp, DivSOp, ModUOp, ModSOp,
                       ShlOp, ShrUOp, ShrSOp,
                       // Bitwise operations
                       AndOp, OrOp, XorOp,
                       // Comparison operations
                       ICmpOp,
                       // Reduction Operators
                       AndROp, OrROp, XorROp,
                       // Other operations.
                       SExtOp, ZExtOp, ConcatOp, ExtractOp, MuxOp,
                       // InOut Expressions
                       ReadInOutOp, ArrayIndexOp>([&](auto expr) -> ResultType {
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

  ResultType visitUnaryComb(Operation *op, ExtraArgs... args) {
    return static_cast<ConcreteType *>(this)->visitUnhandledComb(op, args...);
  }

  ResultType visitVariadicComb(Operation *op, ExtraArgs... args) {
    return static_cast<ConcreteType *>(this)->visitUnhandledComb(op, args...);
  }

#define HANDLE(OPTYPE, OPKIND)                                                 \
  ResultType visitComb(OPTYPE op, ExtraArgs... args) {                         \
    return static_cast<ConcreteType *>(this)->visit##OPKIND##Comb(op,          \
                                                                  args...);    \
  }

  // Basic nodes.
  HANDLE(ConstantOp, Unhandled);

  // Arithmetic and Logical Binary Operations.
  HANDLE(AddOp, Binary);
  HANDLE(SubOp, Binary);
  HANDLE(MulOp, Binary);
  HANDLE(DivUOp, Binary);
  HANDLE(DivSOp, Binary);
  HANDLE(ModUOp, Binary);
  HANDLE(ModSOp, Binary);
  HANDLE(ShlOp, Binary);
  HANDLE(ShrUOp, Binary);
  HANDLE(ShrSOp, Binary);

  HANDLE(AndOp, Variadic);
  HANDLE(OrOp, Variadic);
  HANDLE(XorOp, Variadic);

  HANDLE(AndROp, Unary);
  HANDLE(OrROp, Unary);
  HANDLE(XorROp, Unary);

  HANDLE(ICmpOp, Binary);

  // Other operations.
  HANDLE(SExtOp, Unhandled);
  HANDLE(ZExtOp, Unhandled);
  HANDLE(ConcatOp, Unhandled);
  HANDLE(ExtractOp, Unhandled);
  HANDLE(MuxOp, Unhandled);
  HANDLE(ReadInOutOp, Unhandled);
  HANDLE(ArrayIndexOp, Unhandled);
#undef HANDLE
};

/// This helps visit Combinatorial nodes.
template <typename ConcreteType, typename ResultType = void,
          typename... ExtraArgs>
class StmtVisitor {
public:
  ResultType dispatchStmtVisitor(Operation *op, ExtraArgs... args) {
    auto *thisCast = static_cast<ConcreteType *>(this);
    return TypeSwitch<Operation *, ResultType>(op)
        .template Case<ConnectOp, OutputOp, RegOp, WireOp, InstanceOp>(
            [&](auto expr) -> ResultType {
              return thisCast->visitStmt(expr, args...);
            })
        .Default([&](auto expr) -> ResultType {
          return thisCast->visitInvalidStmt(op, args...);
        });
  }

  /// This callback is invoked on any non-expression operations.
  ResultType visitInvalidStmt(Operation *op, ExtraArgs... args) {
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

  ResultType visitUnaryComb(Operation *op, ExtraArgs... args) {
    return static_cast<ConcreteType *>(this)->visitUnhandledComb(op, args...);
  }

#define HANDLE(OPTYPE, OPKIND)                                                 \
  ResultType visitStmt(OPTYPE op, ExtraArgs... args) {                         \
    return static_cast<ConcreteType *>(this)->visit##OPKIND##Stmt(op,          \
                                                                  args...);    \
  }

  // Basic nodes.
  HANDLE(ConnectOp, Unhandled);
  HANDLE(OutputOp, Unhandled);
  HANDLE(RegOp, Unhandled);
  HANDLE(WireOp, Unhandled);
  HANDLE(InstanceOp, Unhandled);
#undef HANDLE
};

} // namespace rtl
} // namespace circt

#endif // CIRCT_DIALECT_RTL_RTLVISITORS_H
