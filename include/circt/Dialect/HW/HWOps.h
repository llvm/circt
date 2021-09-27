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

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace circt {
namespace hw {

/// A HW module ports direction.
enum PortDirection {
  INPUT = 1,
  OUTPUT = 2,
  INOUT = 3,
};

/// This holds the name, type, direction of a module's ports
struct PortInfo {
  StringAttr name;
  PortDirection direction;
  Type type;
  size_t argNum = ~0U; // Either the argument index or the result index
                       // depending on the direction.

  StringRef getName() const { return name.getValue(); }
  bool isOutput() const { return direction == OUTPUT; }
};

/// This holds a decoded list of input/inout and output ports for a module or
/// instance.
struct ModulePortInfo {
  explicit ModulePortInfo(ArrayRef<PortInfo> inputs, ArrayRef<PortInfo> outputs)
      : inputs(inputs.begin(), inputs.end()),
        outputs(outputs.begin(), outputs.end()) {}

  explicit ModulePortInfo(ArrayRef<PortInfo> mergedPorts) {
    inputs.reserve(mergedPorts.size());
    outputs.reserve(mergedPorts.size());
    for (auto port : mergedPorts) {
      if (port.isOutput())
        outputs.push_back(port);
      else
        inputs.push_back(port);
    }
  }

  /// This contains a list of the input and inout ports.
  SmallVector<PortInfo> inputs;
  /// This is a list of the output ports.
  SmallVector<PortInfo> outputs;
};

/// Return an encapsulated set of information about input and output ports of
/// the specified module or instance.
ModulePortInfo getModulePortInfo(Operation *op);

/// Return an encapsulated set of information about input and output ports of
/// the specified module or instance.  The input ports always come before the
/// output ports in the list.
SmallVector<PortInfo> getAllModulePortInfos(Operation *op);

// Helpers for working with modules.

/// Return true if this is an hw.module, external module, generated module etc.
bool isAnyModule(Operation *module);

/// Return the signature for the specified module as a function type.
FunctionType getModuleType(Operation *module);

/// Returns the verilog module name attribute or symbol name of any module-like
/// operations.
StringAttr getVerilogModuleNameAttr(Operation *module);

/// Return the port name for the specified argument or result.  These can only
/// return a null StringAttr when the IR is invalid.
StringAttr getModuleArgumentNameAttr(Operation *module, size_t argNo);
StringAttr getModuleResultNameAttr(Operation *module, size_t argNo);

static inline StringRef getModuleArgumentName(Operation *module, size_t argNo) {
  auto attr = getModuleArgumentNameAttr(module, argNo);
  return attr ? attr.getValue() : StringRef();
}
static inline StringRef getModuleResultName(Operation *module,
                                            size_t resultNo) {
  auto attr = getModuleResultNameAttr(module, resultNo);
  return attr ? attr.getValue() : StringRef();
}

void setModuleArgumentNames(Operation *module, ArrayRef<Attribute> names);
void setModuleResultNames(Operation *module, ArrayRef<Attribute> names);

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

/// This stores lookup tables to make manipulating and working with the IR more
/// efficient.  There are two phases to this object: the "building" phase in
/// which it is "write only" and then the "using" phase which is read-only (and
/// thus can be used by multiple threads).  The
/// "freeze" method transitions between the two states.
class SymbolCache {
public:
  /// In the building phase, add symbols.
  void addDefinition(StringAttr symbol, Operation *op) {
    assert(!isFrozen && "cannot mutate a frozen cache");
    symbolCache[symbol.getValue()] = op;
  }

  /// Mark the cache as frozen, which allows it to be shared across threads.
  void freeze() { isFrozen = true; }

  Operation *getDefinition(FlatSymbolRefAttr symbol) const {
    assert(isFrozen && "cannot read from this cache until it is frozen");
    auto it = symbolCache.find(symbol.getValue());
    return it != symbolCache.end() ? it->second : nullptr;
  }

private:
  bool isFrozen = false;

  /// This stores a lookup table from symbol attribute to the operation
  /// (hw.module, hw.instance, etc) that defines it.
  /// TODO: It is super annoying that symbols are *defined* as StringAttr, but
  /// are then referenced as FlatSymbolRefAttr.  Why can't we have nice pointer
  /// uniqued things?? :-(
  llvm::StringMap<Operation *> symbolCache;
};

} // namespace hw
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/HW/HW.h.inc"

#endif // CIRCT_DIALECT_HW_OPS_H
