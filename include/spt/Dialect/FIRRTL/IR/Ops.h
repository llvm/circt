//===- FIRRTL.h - FIRRTL dialect --------------------------------*- C++ -*-===//
//
// This file defines an MLIR dialect for the FIRRTL IR.
//
//===----------------------------------------------------------------------===//

#ifndef SPT_DIALECT_FIRRTL_FIRRTL_H
#define SPT_DIALECT_FIRRTL_FIRRTL_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffects.h"

namespace spt {
namespace firrtl {
  using namespace mlir;

class FIRRTLDialect : public Dialect {
public:
  /// Create the dialect in the given `context`.
  explicit FIRRTLDialect(MLIRContext *context);
  ~FIRRTLDialect();
  
  Type parseType(DialectAsmParser &parser) const override;
  void printType(Type, DialectAsmPrinter &) const override;
    
  static StringRef getDialectNamespace() { return "firrtl"; }
};

namespace FIRRTLTypes {
enum Kind {
  UInt = Type::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
};
} // namespace FIRRTLTypes

// A unsigned integer type with unknown width.
class UIntType : public Type::TypeBase<UIntType, Type> {
public:
  using Base::Base;

  static UIntType get(MLIRContext *context) {
    return Base::get(context, FIRRTLTypes::Kind::UInt);
  }

  /// Support method to enable LLVM-style type casting.
  static bool kindof(unsigned kind) {
    return kind == FIRRTLTypes::Kind::UInt;
  }
};

#define GET_OP_CLASSES
#include "spt/Dialect/FIRRTL/IR/FIRRTL.h.inc"

} // namespace firrtl
} // namespace spt

#endif // SPT_DIALECT_FIRRTL_FIRRTL_H
