//===- HWOps.h - Declare HW dialect operations ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the HW dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_OPS_H
#define CIRCT_DIALECT_HW_OPS_H

#include "circt/Dialect/Emit/EmitOpInterfaces.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWEnums.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/BuilderUtils.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/StringExtras.h"

namespace circt {
namespace hw {

class EnumFieldAttr;

/// Flip a port direction.
ModulePort::Direction flip(ModulePort::Direction direction);

/// TODO: Move all these functions to a hw::ModuleLike interface.

// Helpers for working with modules.

/// Return true if isAnyModule or instance.
bool isAnyModuleOrInstance(Operation *module);

/// Return the signature for the specified module as a function type.
FunctionType getModuleType(Operation *module);

/// Returns the verilog module name attribute or symbol name of any module-like
/// operations.
StringAttr getVerilogModuleNameAttr(Operation *module);
inline StringRef getVerilogModuleName(Operation *module) {
  return getVerilogModuleNameAttr(module).getValue();
}

// Index width should be exactly clog2 (size of array), or either 0 or 1 if the
// array is a singleton.
bool isValidIndexBitWidth(Value index, Value array);

/// Return true if the specified operation is a combinational logic op.
bool isCombinational(Operation *op);

/// Return true if the specified attribute tree is made up of nodes that are
/// valid in a parameter expression.
bool isValidParameterExpression(Attribute attr, Operation *module);

/// Check parameter specified by `value` to see if it is valid within the scope
/// of the specified module `module`.  If not, emit an error at the location of
/// `usingOp` and return failure, otherwise return success.
///
/// If `disallowParamRefs` is true, then parameter references are not allowed.
LogicalResult checkParameterInContext(Attribute value, Operation *module,
                                      Operation *usingOp,
                                      bool disallowParamRefs = false);

/// Check parameter specified by `value` to see if it is valid according to the
/// module's parameters.  If not, emit an error to the diagnostic provided as an
/// argument to the lambda 'instanceError' and return failure, otherwise return
/// success.
///
/// If `disallowParamRefs` is true, then parameter references are not allowed.
LogicalResult checkParameterInContext(
    Attribute value, ArrayAttr moduleParameters,
    const std::function<void(std::function<bool(InFlightDiagnostic &)>)>
        &instanceError,
    bool disallowParamRefs = false);

// Check whether an integer value is an offset from a base.
bool isOffset(Value base, Value index, uint64_t offset);

// A class for providing access to the in- and output ports of a module through
// use of the HWModuleBuilder.
class HWModulePortAccessor {

public:
  HWModulePortAccessor(Location loc, const ModulePortInfo &info,
                       Region &bodyRegion);

  // Returns the i'th/named input port of the module.
  Value getInput(unsigned i);
  Value getInput(StringRef name);
  ValueRange getInputs() { return inputArgs; }

  // Assigns the i'th/named output port of the module.
  void setOutput(unsigned i, Value v);
  void setOutput(StringRef name, Value v);

  const ModulePortInfo &getPortList() const { return info; }
  const llvm::SmallVector<Value> &getOutputOperands() const {
    return outputOperands;
  }

private:
  llvm::StringMap<unsigned> inputIdx, outputIdx;
  llvm::SmallVector<Value> inputArgs;
  llvm::SmallVector<Value> outputOperands;
  ModulePortInfo info;
};

using HWModuleBuilder =
    llvm::function_ref<void(OpBuilder &, HWModulePortAccessor &)>;

} // namespace hw
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/HW/HW.h.inc"

#endif // CIRCT_DIALECT_HW_OPS_H
