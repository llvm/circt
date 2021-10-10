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

private:
  /// Check to see if the port names of the specified module conflict with
  /// keywords or themselves.  If so, add the replacement names to
  /// globalNameTable.
  void legalizePortAndParamNames(hw::HWModuleOp module);

  /// Set of globally visible names, to ensure uniqueness.
  NameCollisionResolver globalNameResolver;

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
      legalizePortAndParamNames(module);
      continue;
    }

    // Legalize the name of the interface itself, as well as any signals and
    // modports within it.
    if (auto interface = dyn_cast<InterfaceOp>(op)) {
      legalizeSymbolName(interface, globalNameResolver);

      NameCollisionResolver localNames;
      // Rename signals and modports.
      for (auto &op : *interface.getBodyBlock()) {
        if (isa<InterfaceSignalOp>(op) || isa<InterfaceModportOp>(op))
          legalizeSymbolName(&op, localNames);
      }
      continue;
    }
  }
}

/// Check to see if the port names of the specified module conflict with
/// keywords or themselves.  If so, add the replacement names to
/// globalNameTable.
void GlobalNameResolver::legalizePortAndParamNames(hw::HWModuleOp module) {
  NameCollisionResolver nameResolver;

  // Legalize the port names.
  size_t portIdx = 0;
  for (const PortInfo &port : getAllModulePortInfos(module)) {
    auto newName = nameResolver.getLegalName(port.name);
    if (!newName.empty())
      globalNameTable.addRenamedPort(module, port, newName);
    ++portIdx;
  }

  // Legalize the parameter names.
  for (auto param : module.parameters()) {
    auto paramAttr = param.cast<ParamDeclAttr>();
    auto newName = nameResolver.getLegalName(paramAttr.getName());
    if (!newName.empty())
      globalNameTable.addRenamedParam(module, paramAttr.getName(), newName);
  }
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
