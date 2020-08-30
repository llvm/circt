#ifndef HIR_HIR_H
#define HIR_HIR_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

namespace mlir {
namespace hir {
enum Kinds {
  TimeKind = Type::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
  ValKind,
  ConstKind,
  MemrefKind,
  WireKind
};

class TimeType : public Type::TypeBase<TimeType, Type, DefaultTypeStorage> {
  /**
   * This class defines hir.time type in the dialect.
   */
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == TimeKind; }
  static llvm::StringRef getKeyword() { return "time"; }
  static TimeType get(MLIRContext *context) {
    return Base::get(context, TimeKind);
  }
};

//TODO: Remove this
//class BaseValType :public Type{
//  static bool classof(Type type) {
//    return type.getKind() == hir::ValKind ||
//           type.getKind() == hir::ConstKind;
//  }
//};

class ValType : public Type::TypeBase<ValType, Type, DefaultTypeStorage> {
  /**
   * This class defines hir.val type in the dialect.
   */
public:
  using Base::Base;
  static bool kindof(unsigned kind) { return kind == ValKind; }
  static llvm::StringRef getKeyword() { return "val"; }
  static ValType get(MLIRContext *context) {
    return Base::get(context, ValKind);
  }
};

class ConstType : public Type::TypeBase<ConstType, Type, DefaultTypeStorage> {
  /**
   * This class defines hir.const type in the dialect.
   */

public:
  using Base::Base;
  static bool kindof(unsigned kind) { return kind == ConstKind; }
  static llvm::StringRef getKeyword() { return "const"; }
  static ConstType get(MLIRContext *context) {
    return Base::get(context, ConstKind);
  }
};


class MemrefType
    : public Type::TypeBase<MemrefType, Type, DefaultTypeStorage> {
  /**
   * This class defines hir.memref type in the dialect.
   */
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == MemrefKind; }
  static llvm::StringRef getKeyword() { return "memref"; }
  static MemrefType get(MLIRContext *context) {
    return Base::get(context, MemrefKind);
  }
};

class WireType
    : public Type::TypeBase<WireType, Type, DefaultTypeStorage> {
  /**
   * This class defines hir.wire type in the dialect.
   */
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == WireKind; }
  static llvm::StringRef getKeyword() { return "wire"; }
  static WireType get(MLIRContext *context) {
    return Base::get(context, WireKind);
  }
};

#define GET_OP_CLASSES
#include "circt/Dialect/HIR/HIR.h.inc"

} // namespace hir
} // namespace mlir

#endif // HIR_HIR_H
