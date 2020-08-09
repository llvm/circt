#ifndef HIR_HIR_H
#define HIR_HIR_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
  namespace hir {
    enum Kinds {
      TimeKind = Type::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
      MemoryInterfaceKind 
    };
    class TimeType : public Type::TypeBase<TimeType, Type, DefaultTypeStorage> {
      public:
        using Base::Base;
        // static method definitions
        static bool kindof(unsigned kind) { return kind == TimeKind; }
        static llvm::StringRef getKeyword() { return "time"; }
        static TimeType get(MLIRContext *context) { return Base::get(context, TimeKind); }
    };
  class MemoryInterfaceType : public Type::TypeBase<MemoryInterfaceType, Type, DefaultTypeStorage> {
      public:
        using Base::Base;
        // static method definitions
        static bool kindof(unsigned kind) { return kind == MemoryInterfaceKind; }
        static llvm::StringRef getKeyword() { return "mem_interface"; }
        static MemoryInterfaceType get(MLIRContext *context) { return Base::get(context, MemoryInterfaceKind); }
    };
#define GET_OP_CLASSES
#include "circt/Dialect/HIR/HIR.h.inc"

  } // namespace hir
} // namespace mlir

#endif // HIR_HIR_H
