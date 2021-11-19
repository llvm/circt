//===- ExportVerilogInternals.h - Shared Internal Impl Details --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_EXPORTVERILOG_EXPORTVERILOGINTERNAL_H
#define CONVERSION_EXPORTVERILOG_EXPORTVERILOGINTERNAL_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"

namespace circt {
struct LoweringOptions;

namespace ExportVerilog {
class GlobalNameResolver;

/// If the given `op` is a declaration, return the attribute that dictates its
/// name. For things like wires and registers this will be the `name` attribute,
/// for instances this is `instanceName`, etc.
StringAttr getDeclarationName(Operation *op);

/// This class keeps track of global names at the module/interface level.
/// It is built in a global pass over the entire design and then frozen to allow
/// concurrent accesses.
struct GlobalNameTable {
  GlobalNameTable(GlobalNameTable &&) = default;

  /// Return the string to use for the specified port in the specified module.
  /// Ports may be renamed for a variety of reasons (e.g. conflicting with
  /// Verilog keywords), and this returns the legalized name to use.
  StringRef getPortVerilogName(Operation *module,
                               const hw::PortInfo &port) const {
    auto it = renamedPorts.find(std::make_pair(module, port.getId()));
    return (it != renamedPorts.end() ? it->second : port.name).getValue();
  }

  /// Return the string to use for the specified port in the specified module.
  /// Ports may be renamed for a variety of reasons (e.g. conflicting with
  /// Verilog keywords), and this returns the legalized name to use.
  StringRef getPortVerilogName(Operation *module, ssize_t portArgNum) const {
    auto numInputs = hw::getModuleNumInputs(module);
    // portArgNum is the index into the result of getAllModulePortInfos.
    // We need to translate it into the PortInfo::getId(), which is a unique
    // port identifier. if input port, then (-1 - portArgNum) else numInputs -
    // portArgNum Also ensure the correct index into the input/output list is
    // computed.
    StringAttr nameAttr;
    if (portArgNum < numInputs) {
      nameAttr = module->getAttrOfType<ArrayAttr>("argNames")[portArgNum]
                     .cast<StringAttr>();
      portArgNum = -1 - portArgNum;
    } else {
      portArgNum = numInputs - portArgNum;
      nameAttr = module->getAttrOfType<ArrayAttr>("resultNames")[portArgNum]
                     .cast<StringAttr>();
    }
    auto it = renamedPorts.find(std::make_pair(module, portArgNum));
    return (it != renamedPorts.end() ? it->second : nameAttr).getValue();
  }

  /// Return the string to use for the specified parameter name in the specified
  /// module.  Parameters may be renamed for a variety of reasons (e.g.
  /// conflicting with ports or Verilog keywords), and this returns the
  /// legalized name to use.
  StringRef getParameterVerilogName(Operation *module,
                                    StringAttr paramName) const {
    auto it = renamedParams.find(std::make_pair(module, paramName));
    return (it != renamedParams.end() ? it->second : paramName).getValue();
  }

  /// Return the string to use for the specified declaration. The operation is
  /// commonly a declaration within the module, like `WireOp` and friends, which
  /// have a `name` attribute. Values may be renamed for a variety of reasons
  /// (e.g. conflicting with ports or Verilog keywords), and this returns the
  /// legalized name to use.
  StringRef getDeclarationVerilogName(Operation *op) const {
    auto it = renamedDecls.find(op);
    auto attr = it != renamedDecls.end() ? it->second : getDeclarationName(op);
    return attr ? attr.getValue() : "";
  }

  /// Return the string to use for the specified interface. Interfaces and their
  /// signals may be renamed for a variety of reasons (e.g. conflicting with
  /// Verilog keywords), and this returns the legalized name to use.
  StringRef getInterfaceVerilogName(Operation *op) const {
    auto it = renamedInterfaceOp.find(op);
    auto attr = it != renamedInterfaceOp.end() ? it->second
                                               : SymbolTable::getSymbolName(op);
    return attr.getValue();
  }

private:
  friend class GlobalNameResolver;
  GlobalNameTable() {}
  GlobalNameTable(const GlobalNameTable &) = delete;
  void operator=(const GlobalNameTable &) = delete;

  void addRenamedPort(Operation *module, const hw::PortInfo &port,
                      StringRef newName) {
    renamedPorts[{module, port.getId()}] =
        StringAttr::get(module->getContext(), newName);
  }

  void addRenamedParam(Operation *module, StringAttr oldName,
                       StringRef newName) {
    renamedParams[{module, oldName}] =
        StringAttr::get(oldName.getContext(), newName);
  }

  void addRenamedDeclaration(Operation *declOp, StringRef newName) {
    renamedDecls[declOp] = StringAttr::get(declOp->getContext(), newName);
  }

  void addRenamedInterfaceOp(Operation *interfaceOp, StringRef newName) {
    renamedInterfaceOp[interfaceOp] =
        StringAttr::get(interfaceOp->getContext(), newName);
  }

  /// This contains entries for any ports that got renamed.  The key is a
  /// moduleop/portIdx tuple, the value is the name to use.  The portIdx is the
  /// index of the port in the list returned by getAllModulePortInfos.
  /// Note, PortInfo::getId() returns ssize_t.
  DenseMap<std::pair<Operation *, ssize_t>, StringAttr> renamedPorts;

  /// This contains entries for any parameters that got renamed.  The key is a
  /// moduleop/paramName tuple, the value is the name to use.
  DenseMap<std::pair<Operation *, Attribute>, StringAttr> renamedParams;

  /// This contains entries for any declarations that got renamed.
  DenseMap<Operation *, StringAttr> renamedDecls;

  /// This contains entries for interface operations (sv.interface,
  /// sv.interface.signal, and sv.interface.modport) that need to be renamed
  /// due to conflicts.
  DenseMap<Operation *, StringAttr> renamedInterfaceOp;
};

//===----------------------------------------------------------------------===//
// NameCollisionResolver
//===----------------------------------------------------------------------===//

struct NameCollisionResolver {
  NameCollisionResolver() = default;

  /// Given a name that may have collisions or invalid symbols, return a
  /// replacement name to use, or the original name if it was ok.
  StringRef getLegalName(StringRef originalName);
  StringRef getLegalName(StringAttr originalName) {
    return getLegalName(originalName.getValue());
  }

  /// Insert a string as an already-used name.
  void insertUsedName(StringRef name) { usedNames.insert(name); }

private:
  /// Set of used names, to ensure uniqueness.
  llvm::StringSet<> usedNames;

  /// Numeric suffix used as uniquification agent when resolving conflicts.
  size_t nextGeneratedNameID = 0;

  NameCollisionResolver(const NameCollisionResolver &) = delete;
  void operator=(const NameCollisionResolver &) = delete;
};

//===----------------------------------------------------------------------===//
// Other utilities
//===----------------------------------------------------------------------===//

/// Return true for operations that must always be inlined into a containing
/// expression for correctness.
static inline bool isExpressionAlwaysInline(Operation *op) {
  // We need to emit array indexes inline per verilog "lvalue" semantics.
  if (isa<sv::ArrayIndexInOutOp>(op) || isa<sv::ReadInOutOp>(op))
    return true;

  // An SV interface modport is a symbolic name that is always inlined.
  if (isa<sv::GetModportOp>(op) || isa<sv::ReadInterfaceSignalOp>(op))
    return true;

  // XMRs can't be spilled if they are on the lhs.  Conservatively never spill
  // them.
  if (isa<sv::XMROp>(op))
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
void prepareHWModule(Block &block, const LoweringOptions &options);

/// Rewrite module names and interfaces to not conflict with each other or with
/// Verilog keywords.
GlobalNameTable legalizeGlobalNames(ModuleOp topLevel);
} // namespace ExportVerilog

} // namespace circt

#endif // CONVERSION_EXPORTVERILOG_EXPORTVERILOGINTERNAL_H
