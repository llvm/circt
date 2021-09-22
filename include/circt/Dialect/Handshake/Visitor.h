//===- Handshake/Visitors.h - Handshake Dialect Visitors --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines visitors that make it easier to work with Handshake IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HANDSHAKE_VISITORS_H
#define CIRCT_DIALECT_HANDSHAKE_VISITORS_H

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace handshake {

/// HandshakeVisitor is a visitor for handshake nodes.
template <typename ConcreteType, typename ResultType = void,
          typename... ExtraArgs>
class HandshakeVisitor {
public:
  ResultType dispatchHandshakeVisitor(Operation *op, ExtraArgs... args) {
    auto *thisCast = static_cast<ConcreteType *>(this);
    return TypeSwitch<Operation *, ResultType>(op)
        .template Case<
            // Handshake nodes.
            BranchOp, BufferOp, ConditionalBranchOp, ConstantOp, ControlMergeOp,
            EndOp, ForkOp, FuncOp, InstanceOp, JoinOp, LazyForkOp, LoadOp,
            MemoryOp, MergeOp, MuxOp, ReturnOp, SinkOp, SourceOp, StartOp,
            StoreOp, TerminatorOp>([&](auto opNode) -> ResultType {
          return thisCast->visitHandshake(opNode, args...);
        })
        .Default([&](auto opNode) -> ResultType {
          return thisCast->visitInvalidOp(op, args...);
        });
  }

  /// This callback is invoked on any invalid operations.
  ResultType visitInvalidOp(Operation *op, ExtraArgs... args) {
    op->emitOpError("is unsupported operation");
    abort();
  }

  /// This callback is invoked on any operations that are not handled by the
  /// concrete visitor.
  ResultType visitUnhandledOp(Operation *op, ExtraArgs... args) {
    return ResultType();
  }

#define HANDLE(OPTYPE)                                                         \
  ResultType visitHandshake(OPTYPE op, ExtraArgs... args) {                    \
    return static_cast<ConcreteType *>(this)->visitUnhandledOp(op, args...);   \
  }

  // Handshake nodes.
  HANDLE(BranchOp);
  HANDLE(BufferOp);
  HANDLE(ConditionalBranchOp);
  HANDLE(ConstantOp);
  HANDLE(ControlMergeOp);
  HANDLE(EndOp);
  HANDLE(ForkOp);
  HANDLE(FuncOp);
  HANDLE(InstanceOp);
  HANDLE(JoinOp);
  HANDLE(LazyForkOp);
  HANDLE(LoadOp);
  HANDLE(MemoryOp);
  HANDLE(MergeOp);
  HANDLE(MuxOp);
  HANDLE(ReturnOp);
  HANDLE(SinkOp);
  HANDLE(SourceOp);
  HANDLE(StartOp);
  HANDLE(StoreOp);
  HANDLE(TerminatorOp);
#undef HANDLE
};

} // namespace handshake
} // namespace circt

namespace mlir {

/// StdExprVisitor is a visitor for standard expression nodes.
template <typename ConcreteType, typename ResultType = void,
          typename... ExtraArgs>
class StdExprVisitor {
public:
  ResultType dispatchStdExprVisitor(Operation *op, ExtraArgs... args) {
    auto *thisCast = static_cast<ConcreteType *>(this);
    return TypeSwitch<Operation *, ResultType>(op)
        .template Case<IndexCastOp, ZeroExtendIOp, TruncateIOp,
                       // Integer binary expressions.
                       CmpIOp, AddIOp, SubIOp, MulIOp, SignedDivIOp,
                       SignedRemIOp, UnsignedDivIOp, UnsignedRemIOp, XOrOp,
                       AndOp, OrOp, ShiftLeftOp, SignedShiftRightOp,
                       UnsignedShiftRightOp>([&](auto opNode) -> ResultType {
          return thisCast->visitStdExpr(opNode, args...);
        })
        .Default([&](auto opNode) -> ResultType {
          return thisCast->visitInvalidOp(op, args...);
        });
  }

  /// This callback is invoked on any invalid operations.
  ResultType visitInvalidOp(Operation *op, ExtraArgs... args) {
    op->emitOpError("is unsupported operation");
    abort();
  }

  /// This callback is invoked on any operations that are not handled by the
  /// concrete visitor.
  ResultType visitUnhandledOp(Operation *op, ExtraArgs... args) {
    return ResultType();
  }

#define HANDLE(OPTYPE)                                                         \
  ResultType visitStdExpr(OPTYPE op, ExtraArgs... args) {                      \
    return static_cast<ConcreteType *>(this)->visitUnhandledOp(op, args...);   \
  }

  HANDLE(IndexCastOp);
  HANDLE(ZeroExtendIOp);
  HANDLE(TruncateIOp);

  // Integer binary expressions.
  HANDLE(CmpIOp);
  HANDLE(AddIOp);
  HANDLE(SubIOp);
  HANDLE(MulIOp);
  HANDLE(SignedDivIOp);
  HANDLE(SignedRemIOp);
  HANDLE(UnsignedDivIOp);
  HANDLE(UnsignedRemIOp);
  HANDLE(XOrOp);
  HANDLE(AndOp);
  HANDLE(OrOp);
  HANDLE(ShiftLeftOp);
  HANDLE(SignedShiftRightOp);
  HANDLE(UnsignedShiftRightOp);
#undef HANDLE
};

} // namespace mlir

#endif // CIRCT_DIALECT_HANDSHAKE_VISITORS_H
