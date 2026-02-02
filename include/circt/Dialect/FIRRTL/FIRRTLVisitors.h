//===- FIRRTLVisitors.h - FIRRTL Dialect Visitors ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines visitors that make it easier to work with FIRRTL IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLVISITORS_H
#define CIRCT_DIALECT_FIRRTL_FIRRTLVISITORS_H

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
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
            ConstantOp, SpecialConstantOp, AggregateConstantOp, InvalidValueOp,
            SubfieldOp, SubindexOp, SubaccessOp, IsTagOp, SubtagOp,
            BundleCreateOp, VectorCreateOp, FEnumCreateOp, MultibitMuxOp,
            TagExtractOp, OpenSubfieldOp, OpenSubindexOp, ObjectSubfieldOp,
            ObjectAnyRefCastOp,
            // Arithmetic and Logical Binary Primitives.
            AddPrimOp, SubPrimOp, MulPrimOp, DivPrimOp, RemPrimOp, AndPrimOp,
            OrPrimOp, XorPrimOp,
            // Elementwise operations,
            ElementwiseOrPrimOp, ElementwiseAndPrimOp, ElementwiseXorPrimOp,
            // Comparisons.
            LEQPrimOp, LTPrimOp, GEQPrimOp, GTPrimOp, EQPrimOp, NEQPrimOp,
            // Misc Binary Primitives.
            CatPrimOp, DShlPrimOp, DShlwPrimOp, DShrPrimOp,
            // Unary operators.
            AsSIntPrimOp, AsUIntPrimOp, AsAsyncResetPrimOp, AsClockPrimOp,
            CvtPrimOp, NegPrimOp, NotPrimOp, AndRPrimOp, OrRPrimOp, XorRPrimOp,
            // Intrinsic Expressions.
            IsXIntrinsicOp, PlusArgsValueIntrinsicOp, PlusArgsTestIntrinsicOp,
            SizeOfIntrinsicOp, ClockGateIntrinsicOp, ClockInverterIntrinsicOp,
            ClockDividerIntrinsicOp, LTLAndIntrinsicOp, LTLOrIntrinsicOp,
            LTLIntersectIntrinsicOp, LTLDelayIntrinsicOp, LTLConcatIntrinsicOp,
            LTLRepeatIntrinsicOp, LTLGoToRepeatIntrinsicOp,
            LTLNonConsecutiveRepeatIntrinsicOp, LTLNotIntrinsicOp,
            LTLImplicationIntrinsicOp, LTLUntilIntrinsicOp,
            LTLEventuallyIntrinsicOp, LTLClockIntrinsicOp, Mux2CellIntrinsicOp,
            Mux4CellIntrinsicOp, HasBeenResetIntrinsicOp,
            ReadDesignConfigIntIntrinsicOp,
            // Miscellaneous.
            BitsPrimOp, HeadPrimOp, MuxPrimOp, PadPrimOp, ShlPrimOp, ShrPrimOp,
            TailPrimOp, VerbatimExprOp, HWStructCastOp, BitCastOp, RefSendOp,
            RefResolveOp, RefSubOp, RWProbeOp, XMRRefOp, XMRDerefOp,
            UnsafeDomainCastOp,
            // Casts to deal with weird stuff
            UninferredResetCastOp, ConstCastOp, RefCastOp,
            // Property expressions.
            StringConstantOp, FIntegerConstantOp, BoolConstantOp,
            DoubleConstantOp, ListCreateOp, ListConcatOp, UnresolvedPathOp,
            PathOp, IntegerAddOp, IntegerMulOp, IntegerShrOp, UnknownValueOp,
            // Format String expressions
            TimeOp, HierarchicalModuleNameOp>([&](auto expr) -> ResultType {
          return thisCast->visitExpr(expr, args...);
        })
        .Default([&](auto expr) -> ResultType {
          return thisCast->visitInvalidExpr(op, args...);
        });
  }

  /// This callback is invoked on any non-expression operations.
  ResultType visitInvalidExpr(Operation *op, ExtraArgs... args) {
    op->emitOpError("unknown FIRRTL expression");
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
  HANDLE(ConstantOp, Unhandled);
  HANDLE(SpecialConstantOp, Unhandled);
  HANDLE(AggregateConstantOp, Unhandled);
  HANDLE(BundleCreateOp, Unhandled);
  HANDLE(VectorCreateOp, Unhandled);
  HANDLE(FEnumCreateOp, Unhandled);
  HANDLE(SubfieldOp, Unhandled);
  HANDLE(SubindexOp, Unhandled);
  HANDLE(SubaccessOp, Unhandled);
  HANDLE(IsTagOp, Unhandled);
  HANDLE(SubtagOp, Unhandled);
  HANDLE(TagExtractOp, Unhandled);
  HANDLE(MultibitMuxOp, Unhandled);
  HANDLE(OpenSubfieldOp, Unhandled);
  HANDLE(OpenSubindexOp, Unhandled);
  HANDLE(ObjectSubfieldOp, Unhandled);
  HANDLE(ObjectAnyRefCastOp, Unhandled);
  HANDLE(CatPrimOp, Unhandled);

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
  HANDLE(DShlPrimOp, Binary);
  HANDLE(DShlwPrimOp, Binary);
  HANDLE(DShrPrimOp, Binary);

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

  HANDLE(ElementwiseOrPrimOp, Unhandled);
  HANDLE(ElementwiseAndPrimOp, Unhandled);
  HANDLE(ElementwiseXorPrimOp, Unhandled);

  // Intrinsic Expr.
  HANDLE(IsXIntrinsicOp, Unhandled);
  HANDLE(PlusArgsValueIntrinsicOp, Unhandled);
  HANDLE(PlusArgsTestIntrinsicOp, Unhandled);
  HANDLE(SizeOfIntrinsicOp, Unhandled);
  HANDLE(ClockGateIntrinsicOp, Unhandled);
  HANDLE(ClockInverterIntrinsicOp, Unhandled);
  HANDLE(ClockDividerIntrinsicOp, Unhandled);
  HANDLE(LTLAndIntrinsicOp, Unhandled);
  HANDLE(LTLOrIntrinsicOp, Unhandled);
  HANDLE(LTLIntersectIntrinsicOp, Unhandled);
  HANDLE(LTLDelayIntrinsicOp, Unhandled);
  HANDLE(LTLConcatIntrinsicOp, Unhandled);
  HANDLE(LTLRepeatIntrinsicOp, Unhandled);
  HANDLE(LTLGoToRepeatIntrinsicOp, Unhandled);
  HANDLE(LTLNonConsecutiveRepeatIntrinsicOp, Unhandled);
  HANDLE(LTLNotIntrinsicOp, Unhandled);
  HANDLE(LTLImplicationIntrinsicOp, Unhandled);
  HANDLE(LTLUntilIntrinsicOp, Unhandled);
  HANDLE(LTLEventuallyIntrinsicOp, Unhandled);
  HANDLE(LTLClockIntrinsicOp, Unhandled);
  HANDLE(Mux4CellIntrinsicOp, Unhandled);
  HANDLE(Mux2CellIntrinsicOp, Unhandled);
  HANDLE(HasBeenResetIntrinsicOp, Unhandled);
  HANDLE(ReadDesignConfigIntIntrinsicOp, Unhandled);

  // Miscellaneous.
  HANDLE(BitsPrimOp, Unhandled);
  HANDLE(HeadPrimOp, Unhandled);
  HANDLE(InvalidValueOp, Unhandled);
  HANDLE(MuxPrimOp, Unhandled);
  HANDLE(PadPrimOp, Unhandled);
  HANDLE(ShlPrimOp, Unhandled);
  HANDLE(ShrPrimOp, Unhandled);
  HANDLE(TailPrimOp, Unhandled);
  HANDLE(VerbatimExprOp, Unhandled);
  HANDLE(RefSendOp, Unhandled);
  HANDLE(RefResolveOp, Unhandled);
  HANDLE(RefSubOp, Unhandled);
  HANDLE(RWProbeOp, Unhandled);
  HANDLE(XMRRefOp, Unhandled);
  HANDLE(XMRDerefOp, Unhandled);
  HANDLE(UnsafeDomainCastOp, Unhandled);

  // Conversions.
  HANDLE(HWStructCastOp, Unhandled);
  HANDLE(UninferredResetCastOp, Unhandled);
  HANDLE(ConstCastOp, Unhandled);
  HANDLE(BitCastOp, Unhandled);
  HANDLE(RefCastOp, Unhandled);

  // Property expressions.
  HANDLE(StringConstantOp, Unhandled);
  HANDLE(FIntegerConstantOp, Unhandled);
  HANDLE(BoolConstantOp, Unhandled);
  HANDLE(DoubleConstantOp, Unhandled);
  HANDLE(ListCreateOp, Unhandled);
  HANDLE(ListConcatOp, Unhandled);
  HANDLE(PathOp, Unhandled);
  HANDLE(UnresolvedPathOp, Unhandled);
  HANDLE(IntegerAddOp, Unhandled);
  HANDLE(IntegerMulOp, Unhandled);
  HANDLE(IntegerShrOp, Unhandled);
  HANDLE(UnknownValueOp, Unhandled);

  // Format string expressions
  HANDLE(TimeOp, Unhandled);
  HANDLE(HierarchicalModuleNameOp, Unhandled);
#undef HANDLE
};

/// ExprVisitor is a visitor for FIRRTL statement nodes.
template <typename ConcreteType, typename ResultType = void,
          typename... ExtraArgs>
class StmtVisitor {
public:
  ResultType dispatchStmtVisitor(Operation *op, ExtraArgs... args) {
    auto *thisCast = static_cast<ConcreteType *>(this);
    return TypeSwitch<Operation *, ResultType>(op)
        .template Case<AttachOp, ConnectOp, MatchingConnectOp, RefDefineOp,
                       DomainDefineOp, ForceOp, PrintFOp, FPrintFOp, FFlushOp,
                       SkipOp, StopOp, WhenOp, AssertOp, AssumeOp, CoverOp,
                       PropAssignOp, RefForceOp, RefForceInitialOp,
                       RefReleaseOp, RefReleaseInitialOp, FPGAProbeIntrinsicOp,
                       VerifAssertIntrinsicOp, VerifAssumeIntrinsicOp,
                       UnclockedAssumeIntrinsicOp, VerifCoverIntrinsicOp,
                       VerifRequireIntrinsicOp, VerifEnsureIntrinsicOp,
                       LayerBlockOp, MatchOp, ViewIntrinsicOp, BindOp>(
            [&](auto opNode) -> ResultType {
              return thisCast->visitStmt(opNode, args...);
            })
        .Default([&](auto expr) -> ResultType {
          return thisCast->visitInvalidStmt(op, args...);
        });
  }

  /// This callback is invoked on any non-Stmt operations.
  ResultType visitInvalidStmt(Operation *op, ExtraArgs... args) {
    op->emitOpError("unknown firrtl stmt");
    abort();
  }

  /// This callback is invoked on any Stmt operations that are not handled
  /// by the concrete visitor.
  ResultType visitUnhandledStmt(Operation *op, ExtraArgs... args) {
    return ResultType();
  }

#define HANDLE(OPTYPE)                                                         \
  ResultType visitStmt(OPTYPE op, ExtraArgs... args) {                         \
    return static_cast<ConcreteType *>(this)->visitUnhandledStmt(op, args...); \
  }

  HANDLE(AttachOp);
  HANDLE(ConnectOp);
  HANDLE(MatchingConnectOp);
  HANDLE(RefDefineOp);
  HANDLE(DomainDefineOp);
  HANDLE(ForceOp);
  HANDLE(PrintFOp);
  HANDLE(FPrintFOp);
  HANDLE(FFlushOp);
  HANDLE(SkipOp);
  HANDLE(StopOp);
  HANDLE(WhenOp);
  HANDLE(AssertOp);
  HANDLE(AssumeOp);
  HANDLE(CoverOp);
  HANDLE(PropAssignOp);
  HANDLE(RefForceOp);
  HANDLE(RefForceInitialOp);
  HANDLE(RefReleaseOp);
  HANDLE(RefReleaseInitialOp);
  HANDLE(FPGAProbeIntrinsicOp);
  HANDLE(VerifAssertIntrinsicOp);
  HANDLE(VerifAssumeIntrinsicOp);
  HANDLE(VerifCoverIntrinsicOp);
  HANDLE(VerifRequireIntrinsicOp);
  HANDLE(VerifEnsureIntrinsicOp);
  HANDLE(UnclockedAssumeIntrinsicOp);
  HANDLE(LayerBlockOp);
  HANDLE(MatchOp);
  HANDLE(ViewIntrinsicOp);
  HANDLE(BindOp);

#undef HANDLE
};

/// ExprVisitor is a visitor for FIRRTL declaration nodes.
template <typename ConcreteType, typename ResultType = void,
          typename... ExtraArgs>
class DeclVisitor {
public:
  ResultType dispatchDeclVisitor(Operation *op, ExtraArgs... args) {
    auto *thisCast = static_cast<ConcreteType *>(this);
    return TypeSwitch<Operation *, ResultType>(op)
        .template Case<InstanceOp, InstanceChoiceOp, ObjectOp, MemOp, NodeOp,
                       RegOp, RegResetOp, WireOp, VerbatimWireOp, ContractOp>(
            [&](auto opNode) -> ResultType {
              return thisCast->visitDecl(opNode, args...);
            })
        .Default([&](auto expr) -> ResultType {
          return thisCast->visitInvalidDecl(op, args...);
        });
  }

  /// This callback is invoked on any non-Decl operations.
  ResultType visitInvalidDecl(Operation *op, ExtraArgs... args) {
    op->emitOpError("unknown firrtl decl");
    abort();
  }

  /// This callback is invoked on any Decl operations that are not handled
  /// by the concrete visitor.
  ResultType visitUnhandledDecl(Operation *op, ExtraArgs... args) {
    return ResultType();
  }

#define HANDLE(OPTYPE)                                                         \
  ResultType visitDecl(OPTYPE op, ExtraArgs... args) {                         \
    return static_cast<ConcreteType *>(this)->visitUnhandledDecl(op, args...); \
  }

  HANDLE(InstanceOp);
  HANDLE(InstanceChoiceOp);
  HANDLE(ObjectOp);
  HANDLE(MemOp);
  HANDLE(NodeOp);
  HANDLE(RegOp);
  HANDLE(RegResetOp);
  HANDLE(WireOp);
  HANDLE(VerbatimWireOp);
  HANDLE(ContractOp);
#undef HANDLE
};

/// StmtExprVisitor is a visitor for FIRRTL operation that has an optional
/// result.
template <typename ConcreteType, typename ResultType = void,
          typename... ExtraArgs>
class StmtExprVisitor {
public:
  ResultType dispatchStmtExprVisitor(Operation *op, ExtraArgs... args) {
    auto *thisCast = static_cast<ConcreteType *>(this);
    return TypeSwitch<Operation *, ResultType>(op)
        .template Case<GenericIntrinsicOp, DPICallIntrinsicOp>(
            [&](auto expr) -> ResultType {
              return thisCast->visitStmtExpr(expr, args...);
            })
        .Default([&](auto expr) -> ResultType {
          return thisCast->visitInvalidStmtExpr(op, args...);
        });
  }

  /// This callback is invoked on any non-StmtExpr operations.
  ResultType visitInvalidStmtExpr(Operation *op, ExtraArgs... args) {
    op->emitOpError("unknown FIRRTL stmt expression");
    abort();
  }

  /// This callback is invoked on any StmtExpr operations that are not handled
  /// by the concrete visitor.
  ResultType visitUnhandledStmtExpr(Operation *op, ExtraArgs... args) {
    return ResultType();
  }

#define HANDLE(OPTYPE, OPKIND)                                                 \
  ResultType visitStmtExpr(OPTYPE op, ExtraArgs... args) {                     \
    return static_cast<ConcreteType *>(this)->visit##OPKIND##StmtExpr(         \
        op, args...);                                                          \
  }

  // Statement expressions.
  HANDLE(DPICallIntrinsicOp, Unhandled);
  HANDLE(GenericIntrinsicOp, Unhandled);
#undef HANDLE
};

/// FIRRTLVisitor allows you to visit all of the expr/stmt/decls with one class
/// declaration.
///
/// Clients call dispatchVisitor to invoke the dispatch, and may implement
/// visitInvalidOp() to get notified about non-FIRRTL dialect nodes and
/// visitUnhandledOp() to get notified about FIRRTL dialect ops that are not
/// handled specifically.
template <typename ConcreteType, typename ResultType = void,
          typename... ExtraArgs>
class FIRRTLVisitor
    : public ExprVisitor<ConcreteType, ResultType, ExtraArgs...>,
      public StmtVisitor<ConcreteType, ResultType, ExtraArgs...>,
      public DeclVisitor<ConcreteType, ResultType, ExtraArgs...>,
      public StmtExprVisitor<ConcreteType, ResultType, ExtraArgs...> {
public:
  /// This is the main entrypoint for the FIRRTLVisitor.
  ResultType dispatchVisitor(Operation *op, ExtraArgs... args) {
    return this->dispatchExprVisitor(op, args...);
  }

  // Chain from each visitor onto the next one.
  ResultType visitInvalidExpr(Operation *op, ExtraArgs... args) {
    return this->dispatchStmtVisitor(op, args...);
  }
  ResultType visitInvalidStmt(Operation *op, ExtraArgs... args) {
    return this->dispatchDeclVisitor(op, args...);
  }
  ResultType visitInvalidDecl(Operation *op, ExtraArgs... args) {
    return this->dispatchStmtExprVisitor(op, args...);
  }
  ResultType visitInvalidStmtExpr(Operation *op, ExtraArgs... args) {
    return static_cast<ConcreteType *>(this)->visitInvalidOp(op, args...);
  }

  // Default to chaining visitUnhandledXXX to visitUnhandledOp.
  ResultType visitUnhandledExpr(Operation *op, ExtraArgs... args) {
    return static_cast<ConcreteType *>(this)->visitUnhandledOp(op, args...);
  }
  ResultType visitUnhandledStmt(Operation *op, ExtraArgs... args) {
    return static_cast<ConcreteType *>(this)->visitUnhandledOp(op, args...);
  }
  ResultType visitUnhandledDecl(Operation *op, ExtraArgs... args) {
    return static_cast<ConcreteType *>(this)->visitUnhandledOp(op, args...);
  }
  ResultType visitUnhandledStmtExpr(Operation *op, ExtraArgs... args) {
    return static_cast<ConcreteType *>(this)->visitUnhandledOp(op, args...);
  }

  /// visitInvalidOp is an override point for non-FIRRTL dialect operations.
  ResultType visitInvalidOp(Operation *op, ExtraArgs... args) {
    return ResultType();
  }

  /// visitUnhandledOp is an override point for FIRRTL dialect ops that the
  /// concrete visitor didn't bother to implement.
  ResultType visitUnhandledOp(Operation *op, ExtraArgs... args) {
    return ResultType();
  }
};
} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLVISITORS_H
