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

StringAttr ExportVerilog::getDeclarationName(Operation *op) {
  if (auto attr = op->getAttrOfType<StringAttr>("name"))
    return attr;
  if (auto attr = op->getAttrOfType<StringAttr>("instanceName"))
    return attr;
  if (auto attr =
          op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
    return attr;
  return {};
}

//===----------------------------------------------------------------------===//
// NameCollisionResolver
//===----------------------------------------------------------------------===//

/// Given a name that may have collisions or invalid symbols, return a
/// replacement name to use, or null if the original name was ok.
StringRef NameCollisionResolver::getLegalName(StringRef originalName) {
  return legalizeName(originalName, nextGeneratedNameIDs);
}

//===----------------------------------------------------------------------===//
// FieldNameResolver
//===----------------------------------------------------------------------===//

void FieldNameResolver::setRenamedFieldName(StringAttr fieldName,
                                            StringAttr newFieldName) {
  renamedFieldNames[fieldName] = newFieldName;
  nextGeneratedNameIDs.insert({newFieldName, 0});
}

StringAttr FieldNameResolver::getRenamedFieldName(StringAttr fieldName) {
  auto it = renamedFieldNames.find(fieldName);
  if (it != renamedFieldNames.end())
    return it->second;

  // If a field name is not verilog name or used already, we have to rename it.
  bool hasToBeRenamed = !sv::isNameValid(fieldName.getValue()) ||
                        nextGeneratedNameIDs.count(fieldName.getValue());

  if (!hasToBeRenamed) {
    setRenamedFieldName(fieldName, fieldName);
    return fieldName;
  }

  StringRef newFieldName =
      sv::legalizeName(fieldName.getValue(), nextGeneratedNameIDs);

  auto newFieldNameAttr = StringAttr::get(fieldName.getContext(), newFieldName);

  setRenamedFieldName(fieldName, newFieldNameAttr);
  return newFieldNameAttr;
}

std::string FieldNameResolver::getEnumFieldName(hw::EnumFieldAttr attr) {
  auto aliasType = attr.getType().getValue().dyn_cast<hw::TypeAliasType>();
  if (!aliasType)
    return attr.getField().getValue().str();

  auto fieldStr = attr.getField().getValue().str();
  if (auto prefix = globalNames.getEnumPrefix(aliasType))
    return (prefix.getValue() + "_" + fieldStr).str();

  // No prefix registered, just use the bare field name.
  return fieldStr;
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

  // Gathers prefixes of enum types by inspecting typescopes in the module.
  void gatherEnumPrefixes(mlir::ModuleOp topLevel);

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

  // Gather enum prefixes.
  gatherEnumPrefixes(topLevel);
}

// Gathers prefixes of enum types by investigating typescopes in the module.
void GlobalNameResolver::gatherEnumPrefixes(mlir::ModuleOp topLevel) {
  auto *ctx = topLevel.getContext();
  for (auto typeScope : topLevel.getOps<hw::TypeScopeOp>()) {
    for (auto typeDecl : typeScope.getOps<hw::TypedeclOp>()) {
      auto enumType = typeDecl.getType().dyn_cast<hw::EnumType>();
      if (!enumType)
        continue;

      // Register the enum type as the alias type of the typedecl, since this is
      // how users will request the prefix.
      globalNameTable.enumPrefixes[typeDecl.getAliasType()] =
          StringAttr::get(ctx, typeDecl.getPreferredName());
    }
  }
}

/// Check to see if the port names of the specified module conflict with
/// keywords or themselves.  If so, add the replacement names to
/// globalNameTable.
void GlobalNameResolver::legalizeModuleNames(HWModuleOp module) {
  MLIRContext *ctxt = module.getContext();
  // If the module's symbol itself conflicts, then set a "verilogName" attribute
  // on the module to reflect the name we need to use.
  StringRef oldName = module.getName();
  auto newName = globalNameResolver.getLegalName(oldName);
  if (newName != oldName)
    module->setAttr("verilogName", StringAttr::get(ctxt, newName));

  NameCollisionResolver nameResolver;
  auto verilogNameAttr = StringAttr::get(ctxt, "hw.verilogName");
  // Legalize the port names.
  SmallVector<Attribute, 4> argNames, resultNames;
  for (const PortInfo &port : getAllModulePortInfos(module)) {
    auto newName = nameResolver.getLegalName(port.name);
    if (newName != port.name.getValue()) {
      if (port.isOutput())
        module.setResultAttr(port.argNum, verilogNameAttr,
                             StringAttr::get(ctxt, newName));
      else
        module.setArgAttr(port.argNum, verilogNameAttr,
                          StringAttr::get(ctxt, newName));
    }
  }

  // Legalize the parameter names.
  for (auto param : module.getParameters()) {
    auto paramAttr = param.cast<ParamDeclAttr>();
    auto newName = nameResolver.getLegalName(paramAttr.getName());
    if (newName != paramAttr.getName().getValue())
      globalNameTable.addRenamedParam(module, paramAttr.getName(), newName);
  }

  SmallVector<std::pair<Operation *, StringAttr>> declAndNames;
  // Legalize the value names. We first mark existing hw.verilogName attrs as
  // being used, and then resolve names of declarations.
  module.walk([&](Operation *op) {
    if (!isa<HWModuleOp>(op)) {
      if (auto name = op->getAttrOfType<StringAttr>(verilogNameAttr)) {
        nameResolver.insertUsedName(
            op->getAttrOfType<StringAttr>(verilogNameAttr));
      } else if (auto name = getDeclarationName(op)) {
        declAndNames.push_back({op, name});
      }
    }
  });

  for (auto [op, nameAttr] : declAndNames) {
    auto newName = nameResolver.getLegalName(nameAttr);
    if (newName != nameAttr.getValue())
      op->setAttr(verilogNameAttr, StringAttr::get(ctxt, newName));
  }
}

void GlobalNameResolver::legalizeInterfaceNames(InterfaceOp interface) {
  MLIRContext *ctxt = interface.getContext();
  auto verilogNameAttr = StringAttr::get(ctxt, "hw.verilogName");
  auto newName = globalNameResolver.getLegalName(interface.getName());
  if (newName != interface.getName())
    interface->setAttr(verilogNameAttr, StringAttr::get(ctxt, newName));

  NameCollisionResolver localNames;
  // Rename signals and modports.
  for (auto &op : *interface.getBodyBlock()) {
    if (isa<InterfaceSignalOp, InterfaceModportOp>(op)) {
      auto name = SymbolTable::getSymbolName(&op).getValue();
      auto newName = localNames.getLegalName(name);
      if (newName != name)
        op.setAttr(verilogNameAttr, StringAttr::get(ctxt, newName));
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
