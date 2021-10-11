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

/// Given a name that may have collisions or invalid symbols, return a
/// replacement name to use, or null if the original name was ok.
StringRef NameCollisionResolver::getLegalName(StringRef originalName) {
  return legalizeName(originalName, usedNames, nextGeneratedNameID);
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

private:
  /// Check to see if the port names of the specified module conflict with
  /// keywords or themselves.  If so, add the replacement names to
  /// globalNameTable.
  void legalizeModuleNames(HWModuleOp module);
  void legalizeInterfaceNames(InterfaceOp interface);

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
  // Register the names of external modules which we cannot rename. This has to
  // occur in a first pass separate from the modules and interfaces which we are
  // actually allowed to rename, in order to ensure that we don't accidentally
  // rename a module that later collides with an extern module.
  for (auto &op : *topLevel.getBody()) {
    // Note that external modules *often* have name collisions, because they
    // correspond to the same verilog module with different parameters.
    if (isa<HWModuleExternOp>(op) || isa<HWModuleGeneratedOp>(op)) {
      auto name = getVerilogModuleNameAttr(&op).getValue();
      if (!sv::isNameValid(name))
        op.emitError("name \"")
            << name << "\" is not allowed in Verilog output";
      globalNameResolver.insertUsedName(name);
    }
  }

  // Legalize module and interface names.
  for (auto &op : *topLevel.getBody()) {
    if (auto module = dyn_cast<HWModuleOp>(op)) {
      legalizeModuleNames(module);
      continue;
    }

    // Legalize the name of the interface itself, as well as any signals and
    // modports within it.
    if (auto interface = dyn_cast<InterfaceOp>(op)) {
      legalizeInterfaceNames(interface);
      continue;
    }
  }
}

/// Check to see if the port names of the specified module conflict with
/// keywords or themselves.  If so, add the replacement names to
/// globalNameTable.
void GlobalNameResolver::legalizeModuleNames(HWModuleOp module) {
  // If the module's symbol itself conflicts, then set a "verilogName" attribute
  // on the module to reflect the name we need to use.
  StringRef oldName = module.getName();
  auto newName = globalNameResolver.getLegalName(oldName);
  if (newName != oldName)
    module->setAttr("verilogName",
                    StringAttr::get(module.getContext(), newName));

  NameCollisionResolver nameResolver;

  // Legalize the port names.
  size_t portIdx = 0;
  for (const PortInfo &port : getAllModulePortInfos(module)) {
    auto newName = nameResolver.getLegalName(port.name);
    if (newName != port.name.getValue())
      globalNameTable.addRenamedPort(module, port, newName);
    ++portIdx;
  }

  // Legalize the parameter names.
  for (auto param : module.parameters()) {
    auto paramAttr = param.cast<ParamDeclAttr>();
    auto newName = nameResolver.getLegalName(paramAttr.getName());
    if (newName != paramAttr.getName().getValue())
      globalNameTable.addRenamedParam(module, paramAttr.getName(), newName);
  }
}

void GlobalNameResolver::legalizeInterfaceNames(InterfaceOp interface) {
  auto newName = globalNameResolver.getLegalName(interface.getName());
  if (newName != interface.getName())
    globalNameTable.addRenamedInterfaceOp(interface, newName);

  NameCollisionResolver localNames;
  // Rename signals and modports.
  for (auto &op : *interface.getBodyBlock()) {
    if (isa<InterfaceSignalOp, InterfaceModportOp>(op)) {
      auto name = SymbolTable::getSymbolName(&op).getValue();
      auto newName = localNames.getLegalName(name);
      if (newName != name)
        globalNameTable.addRenamedInterfaceOp(&op, newName);
    }
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
