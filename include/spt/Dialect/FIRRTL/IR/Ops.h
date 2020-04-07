//===- FIRRTL/IR/Ops.h - FIRRTL dialect -------------------------*- C++ -*-===//
//
// This file defines an MLIR dialect for the FIRRTL IR.
//
//===----------------------------------------------------------------------===//

#ifndef SPT_DIALECT_FIRRTL_IR_OPS_H
#define SPT_DIALECT_FIRRTL_IR_OPS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffects.h"
#include "spt/Dialect/FIRRTL/IR/Types.h"

namespace spt {
namespace firrtl {
using namespace mlir;
class FIRRTLType;

class FIRRTLDialect : public Dialect {
public:
  /// Create the dialect in the given `context`.
  explicit FIRRTLDialect(MLIRContext *context);
  ~FIRRTLDialect();

  Type parseType(DialectAsmParser &parser) const override;
  void printType(Type, DialectAsmPrinter &) const override;

  static StringRef getDialectNamespace() { return "firrtl"; }
};

// Binary primitives.
FIRRTLType getAddSubResult(FIRRTLType lhs, FIRRTLType rhs);
FIRRTLType getMulResult(FIRRTLType lhs, FIRRTLType rhs);
FIRRTLType getDivResult(FIRRTLType lhs, FIRRTLType rhs);
FIRRTLType getRemResult(FIRRTLType lhs, FIRRTLType rhs);
FIRRTLType getCompareResult(FIRRTLType lhs, FIRRTLType rhs);
FIRRTLType getBitwiseBinaryResult(FIRRTLType lhs, FIRRTLType rhs);
FIRRTLType getCatResult(FIRRTLType lhs, FIRRTLType rhs);
FIRRTLType getDShlResult(FIRRTLType lhs, FIRRTLType rhs);
FIRRTLType getDShrResult(FIRRTLType lhs, FIRRTLType rhs);

// Unary primitives.
FIRRTLType getAsClockResult(FIRRTLType input);
FIRRTLType getAsSIntResult(FIRRTLType input);
FIRRTLType getAsUIntResult(FIRRTLType input);
FIRRTLType getCvtResult(FIRRTLType input);
FIRRTLType getNotResult(FIRRTLType input);
FIRRTLType getNegResult(FIRRTLType input);

typedef std::pair<StringAttr, FIRRTLType> ModulePortInfo;

/// Return the function type that corresponds to a module.
FunctionType getModuleType(Operation *op);

/// This function can extract information about ports from a module and an
/// extmodule.
void getModulePortInfo(Operation *op, SmallVectorImpl<ModulePortInfo> &results);

#define GET_OP_CLASSES
#include "spt/Dialect/FIRRTL/IR/FIRRTL.h.inc"

} // namespace firrtl
} // namespace spt

#endif // SPT_DIALECT_FIRRTL_IR_OPS_H
