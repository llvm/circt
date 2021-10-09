//===- LegalizeNames.cpp - Name Legalization for ExportVerilog ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This renames modules and variables to avoid conflicts with keywords and other
// declarations.
//
//===----------------------------------------------------------------------===//

#include "ExportVerilogInternals.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"

using namespace circt;
using namespace sv;
using namespace hw;
using namespace ExportVerilog;

//===----------------------------------------------------------------------===//
// NameCollisionResolver
//===----------------------------------------------------------------------===//

namespace {
struct NameCollisionResolver {
  NameCollisionResolver() = default;

  /// Given a name that may have collisions or invalid symbols, return a
  /// replacement name to use, or null if the original name was ok.
  StringRef getLegalName(StringAttr originalName);

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
} // end anonymous namespace

/// Given a name that may have collisions or invalid symbols, return a
/// replacement name to use, or null if the original name was ok.
StringRef NameCollisionResolver::getLegalName(StringAttr originalName) {
  StringRef result =
      legalizeName(originalName.getValue(), usedNames, nextGeneratedNameID);
  return result != originalName.getValue() ? result : StringRef();
}

//===----------------------------------------------------------------------===//
// GlobalNameResolver
//===----------------------------------------------------------------------===//

namespace circt {
namespace ExportVerilog {
/// This class keeps track of modules and interfaces that need to be renamed, as
/// well as module ports and parameters that need to be renamed.  This can
/// happen either due to conflicts between them or due to a conflict with a
/// Verilog keyword.
///
/// Once constructed, this is immutable.
class GlobalNameResolver {
public:
  /// Construct a GlobalNameResolver and do the initial scan to populate and
  /// unique the module/interfaces and port/parameter names.
  GlobalNameResolver(mlir::ModuleOp topLevel);

  GlobalNameTable takeGlobalNameTable() { return std::move(globalNameTable); }

  bool anythingChanged = false;

  /// If the module with the specified name has had a port or parameter renamed,
  /// return the module that defines the name.
  HWModuleOp getModuleWithRenamedInterface(StringAttr name) {
    auto it = modulesWithRenamedPorts.find(name);
    return it != modulesWithRenamedPorts.end() ? it->second : HWModuleOp();
  }

private:
  /// Check to see if the port names of the specified module conflict with
  /// keywords or themselves.  If so, rename them and return true, otherwise
  /// return false.
  bool legalizePortNames(hw::HWModuleOp module);

  void rewriteModuleBody(Block &block, NameCollisionResolver &nameResolver);
  void renameModuleBody(hw::HWModuleOp module);

  /// Set of globally visible names, to ensure uniqueness.
  NameCollisionResolver globalNameResolver;

  /// If a module has a port renamed, then this keeps track of the module it is
  /// associated with so we can update instances.
  DenseMap<Attribute, HWModuleOp> modulesWithRenamedPorts;

  /// This keeps track of globally visible names like module parameters.
  GlobalNameTable globalNameTable;

  GlobalNameResolver(const GlobalNameResolver &) = delete;
  void operator=(const GlobalNameResolver &) = delete;
};
} // namespace ExportVerilog
} // namespace circt

/// Construct a GlobalNameResolver and do the initial scan to populate and
/// unique the module/interfaces and port/parameter names.
GlobalNameResolver::GlobalNameResolver(mlir::ModuleOp topLevel) {
  // This symbol table is lazily constructed when global rewrites of module or
  // interface member names are required.
  mlir::SymbolTableCollection symbolTable;
  Optional<mlir::SymbolUserMap> symbolUsers;

  // Register the names of external modules which we cannot rename. This has to
  // occur in a first pass separate from the modules and interfaces which we are
  // actually allowed to rename, in order to ensure that we don't accidentally
  // rename a module that later collides with an extern module.
  for (auto &op : *topLevel.getBody()) {
    // Note that external modules *often* have name collisions, because they
    // correspond to the same verilog module with different parameters.
    if (isa<HWModuleExternOp>(op) || isa<HWModuleGeneratedOp>(op)) {
      auto name = hw::getVerilogModuleNameAttr(&op).getValue();
      if (!sv::isNameValid(name))
        op.emitError("name \"")
            << name << "\" is not allowed in Verilog output";
      globalNameResolver.insertUsedName(name);
    }
  }

  // If the module's symbol itself conflicts, then rename it and all uses of it.
  auto legalizeSymbolName = [&](Operation *op,
                                NameCollisionResolver &resolver) {
    StringAttr oldName = SymbolTable::getSymbolName(op);
    auto newName = resolver.getLegalName(oldName);
    if (newName.empty())
      return;

    // Lazily construct the symbol table if it hasn't been built yet.
    if (!symbolUsers.hasValue())
      symbolUsers.emplace(symbolTable, topLevel);

    // TODO: This is super inefficient, we should just rename the symbol as part
    // of the other existing walks.
    auto newNameAttr = StringAttr::get(topLevel.getContext(), newName);
    symbolUsers->replaceAllUsesWith(op, newNameAttr);
    SymbolTable::setSymbolName(op, newNameAttr);
    anythingChanged = true;
  };

  // Legalize module and interface names.
  for (auto &op : *topLevel.getBody()) {
    if (auto module = dyn_cast<HWModuleOp>(op)) {
      legalizeSymbolName(module, globalNameResolver);
      if (legalizePortNames(module))
        modulesWithRenamedPorts[module.getNameAttr()] = module;
      continue;
    }

    if (auto interface = dyn_cast<InterfaceOp>(op)) {
      legalizeSymbolName(interface, globalNameResolver);
      continue;
    }
  }

  // Rename individual operations within the bodies.
  for (auto &op : *topLevel.getBody()) {
    if (auto module = dyn_cast<HWModuleOp>(op)) {
      renameModuleBody(module);
      continue;
    }

    if (auto interface = dyn_cast<InterfaceOp>(op)) {
      NameCollisionResolver localNames;

      // Rename signals and modports.
      for (auto &op : *interface.getBodyBlock()) {
        if (isa<InterfaceSignalOp>(op) || isa<InterfaceModportOp>(op))
          legalizeSymbolName(&op, localNames);
      }
    }
  }
}

/// Check to see if the port names of the specified module conflict with
/// keywords or themselves.  If so, rename them and return true, otherwise
/// return false.
bool GlobalNameResolver::legalizePortNames(hw::HWModuleOp module) {
  NameCollisionResolver nameResolver;

  bool changedArgNames = false, changedOutputNames = false;
  SmallVector<Attribute> argNames, outputNames;

  // Legalize the ports.
  for (const PortInfo &port : getAllModulePortInfos(module)) {
    auto newName = nameResolver.getLegalName(port.name);

    auto &namesVector = port.isOutput() ? outputNames : argNames;
    auto &changedBool = port.isOutput() ? changedOutputNames : changedArgNames;

    if (newName.empty()) {
      namesVector.push_back(port.name);
    } else {
      changedBool = true;
      namesVector.push_back(StringAttr::get(module.getContext(), newName));
    }
  }

  if (changedArgNames)
    setModuleArgumentNames(module, argNames);
  if (changedOutputNames)
    setModuleResultNames(module, outputNames);

  // Legalize the parameters.
  for (auto param : module.parameters()) {
    auto paramAttr = param.cast<ParamDeclAttr>();
    auto newName = nameResolver.getLegalName(paramAttr.getName());
    if (!newName.empty())
      globalNameTable.addRenamedParam(module, paramAttr.getName(), newName);
  }

  if (changedArgNames | changedOutputNames) {
    anythingChanged = true;
    return true;
  }

  return false;
}

void GlobalNameResolver::rewriteModuleBody(
    Block &block, NameCollisionResolver &nameResolver) {

  // Rename the instances, regs, and wires.
  for (auto &op : block) {
    if (auto instanceOp = dyn_cast<InstanceOp>(op)) {
      auto newName = nameResolver.getLegalName(instanceOp.getName());
      if (!newName.empty())
        instanceOp.setName(StringAttr::get(instanceOp.getContext(), newName));

      // If this instance is referring to a module with renamed ports or
      // parameter names, update them.
      if (HWModuleOp module = getModuleWithRenamedInterface(
              instanceOp.moduleNameAttr().getAttr())) {
        instanceOp.argNamesAttr(module.argNames());
        instanceOp.resultNamesAttr(module.resultNames());
      }
      continue;
    }

    if (isa<RegOp>(op) || isa<WireOp>(op) || isa<LocalParamOp>(op)) {
      auto oldName = op.getAttrOfType<StringAttr>("name");
      auto newName = nameResolver.getLegalName(oldName);
      if (!newName.empty())
        op.setAttr("name", StringAttr::get(op.getContext(), newName));
    }

    // If this operation has regions, then we recursively process them if they
    // can contain things that need to be renamed.  We don't walk the module
    // in the common case.
    if (op.getNumRegions()) {
      for (auto &region : op.getRegions()) {
        if (!region.empty())
          rewriteModuleBody(region.front(), nameResolver);
      }
    }
  }
}

void GlobalNameResolver::renameModuleBody(hw::HWModuleOp module) {
  // All the ports and parameters are pre-legalized, just add their names to the
  // map so we detect conflicts with them.
  NameCollisionResolver nameResolver;
  for (const PortInfo &port : getAllModulePortInfos(module))
    nameResolver.insertUsedName(port.name.getValue());
  for (auto param : module.parameters())
    nameResolver.insertUsedName(
        param.cast<ParamDeclAttr>().getName().getValue());

  rewriteModuleBody(*module.getBodyBlock(), nameResolver);
}

//===----------------------------------------------------------------------===//
// Public interface
//===----------------------------------------------------------------------===//

/// Rewrite module names and interfaces to not conflict with each other or with
/// Verilog keywords.
GlobalNameTable ExportVerilog::legalizeGlobalNames(ModuleOp topLevel) {
  GlobalNameResolver resolver(topLevel);

  return resolver.takeGlobalNameTable();
}
