//===- ExportVerilogInternals.h - Shared Internal Impl Details --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRANSLATION_EXPORT_VERILOG_EXPORTVERILOGINTERNAL_H
#define TRANSLATION_EXPORT_VERILOG_EXPORTVERILOGINTERNAL_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"

namespace circt {
struct LoweringOptions;

namespace ExportVerilog {
class GlobalNameResolver;

/// This class keeps track of global names at the module/interface level.
/// It is built in a global pass over the entire design and then frozen to allow
/// concurrent accesses.
/// TODO: right now this just handles module parameter names.
struct GlobalNameTable {
  GlobalNameTable(GlobalNameTable &&) = default;

  /// Return the string to use for the specified parameter name in the specified
  /// module.  Parameters may be renamed for a variety of reasons (e.g.
  /// conflicting with ports or verilog keywords), and this returns the
  /// legalized name to use.
  StringAttr getParameterVerilogName(Operation *module,
                                     StringAttr paramName) const {
    auto it = renamedParams.find(std::make_pair(module, paramName));
    return it != renamedParams.end() ? it->second : paramName;
  }

private:
  friend class GlobalNameResolver;
  GlobalNameTable() {}
  GlobalNameTable(const GlobalNameTable &) = delete;
  void operator=(const GlobalNameTable &) = delete;

  void addRenamedParam(Operation *module, StringAttr oldName,
                       StringAttr newName) {
    renamedParams[{module, oldName}] = newName;
  }

  /// This contains entries for any parameters that got renamed.  The key is a
  /// moduleop/paramName tuple, the value is the name to use.
  DenseMap<std::pair<Operation *, Attribute>, StringAttr> renamedParams;
};

/// This class keeps track of names for values within a module.
struct ModuleNameManager {
  ModuleNameManager() : encounteredError(false) {}

  StringRef addName(Value value, StringRef name) {
    return addName(ValueOrOp(value), name);
  }
  StringRef addName(Operation *op, StringRef name) {
    return addName(ValueOrOp(op), name);
  }
  StringRef addName(Value value, StringAttr name) {
    return addName(ValueOrOp(value), name);
  }
  StringRef addName(Operation *op, StringAttr name) {
    return addName(ValueOrOp(op), name);
  }

  StringRef addLegalName(Value value, StringRef name, Operation *errOp) {
    return addLegalName(ValueOrOp(value), name, errOp);
  }
  StringRef addLegalName(Operation *op, StringRef name, Operation *errOp) {
    return addLegalName(ValueOrOp(op), name, errOp);
  }

  StringRef getName(Value value) { return getName(ValueOrOp(value)); }
  StringRef getName(Operation *op) {
    // If RegOp or WireOp, then result has the name.
    if (isa<sv::WireOp, sv::RegOp>(op))
      return getName(op->getResult(0));
    return getName(ValueOrOp(op));
  }

  bool hasName(Value value) { return nameTable.count(ValueOrOp(value)); }
  bool hasName(Operation *op) {
    // If RegOp or WireOp, then result has the name.
    if (isa<sv::WireOp, sv::RegOp>(op))
      return nameTable.count(op->getResult(0));
    return nameTable.count(ValueOrOp(op));
  }

  void addOutputNames(StringRef name, Operation *module) {
    outputNames.push_back(addLegalName(nullptr, name, module));
  }

  StringRef getOutputName(size_t portNum) { return outputNames[portNum]; }

  bool hadError() { return encounteredError; }

private:
  using ValueOrOp = PointerUnion<Value, Operation *>;

  /// Retrieve a name from the name table.  The name must already have been
  /// added.
  StringRef getName(ValueOrOp valueOrOp) {
    auto entry = nameTable.find(valueOrOp);
    assert(entry != nameTable.end() &&
           "value expected a name but doesn't have one");
    return entry->getSecond();
  }

  /// Add the specified name to the name table, auto-uniquing the name if
  /// required.  If the name is empty, then this creates a unique temp name.
  ///
  /// "valueOrOp" is typically the Value for an intermediate wire etc, but it
  /// can also be an op for an instance, since we want the instances op uniqued
  /// and tracked.  It can also be null for things like outputs which are not
  /// tracked in the nameTable.
  StringRef addName(ValueOrOp valueOrOp, StringRef name);

  StringRef addName(ValueOrOp valueOrOp, StringAttr nameAttr) {
    return addName(valueOrOp, nameAttr ? nameAttr.getValue() : "");
  }

  /// Add the specified name to the name table, emitting an error message if the
  /// name empty or is changed by uniqing.
  StringRef addLegalName(ValueOrOp valueOrOp, StringRef name, Operation *op) {
    auto updatedName = addName(valueOrOp, name);
    if (name.empty())
      emitOpError(op, "should have non-empty name");
    else if (updatedName != name)
      emitOpError(op, "name '") << name << "' changed during emission";
    return updatedName;
  }

  InFlightDiagnostic emitOpError(Operation *op, const Twine &message) {
    encounteredError = true;
    return op->emitOpError(message);
  }

  // Track whether a name error ocurred
  bool encounteredError;

  /// nameTable keeps track of mappings from Value's and operations (for
  /// instances) to their string table entry.
  llvm::DenseMap<ValueOrOp, StringRef> nameTable;

  /// outputNames tracks the uniquified names for output ports, which don't
  /// have
  /// a Value or Op representation.
  SmallVector<StringRef> outputNames;

  llvm::StringSet<> usedNames;

  size_t nextGeneratedNameID = 0;
};

/// Return true for operations that must always be inlined into a containing
/// expression for correctness.
static inline bool isExpressionAlwaysInline(Operation *op) {
  // We need to emit array indexes inline per verilog "lvalue" semantics.
  if (isa<sv::ArrayIndexInOutOp>(op) || isa<sv::ReadInOutOp>(op))
    return true;

  // An SV interface modport is a symbolic name that is always inlined.
  if (isa<sv::GetModportOp>(op) || isa<sv::ReadInterfaceSignalOp>(op))
    return true;

  return false;
}

/// Return whether an operation is a constant.
static inline bool isConstantExpression(Operation *op) {
  return isa<hw::ConstantOp, sv::ConstantXOp, sv::ConstantZOp>(op);
}

/// This predicate returns true if the specified operation is considered a
/// potentially inlinable Verilog expression.  These nodes always have a single
/// result, but may have side effects (e.g. `sv.verbatim.expr.se`).
/// MemoryEffects should be checked if a client cares.
bool isVerilogExpression(Operation *op);

/// For each module we emit, do a prepass over the structure, pre-lowering and
/// otherwise rewriting operations we don't want to emit.
void prepareHWModule(Block &block, ModuleNameManager &names,
                     const LoweringOptions &options);

/// Rewrite module names and interfaces to not conflict with each other or with
/// Verilog keywords.
GlobalNameTable legalizeGlobalNames(ModuleOp topLevel);
} // namespace ExportVerilog

} // namespace circt

#endif // TRANSLATION_EXPORT_VERILOG_EXPORTVERILOGINTERNAL_H
