#ifndef HIR_HIR_H
#define HIR_HIR_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

namespace mlir {
namespace hir {
enum Kinds {
  TimeKind = Type::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
  MemoryInterfaceKind,
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

class MemoryInterfaceType
    : public Type::TypeBase<MemoryInterfaceType, Type, DefaultTypeStorage> {
  /**
   * This class defines hir.mem_interface type in the dialect.
   */
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == MemoryInterfaceKind; }
  static llvm::StringRef getKeyword() { return "mem_interface"; }
  static MemoryInterfaceType get(MLIRContext *context) {
    return Base::get(context, MemoryInterfaceKind);
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
