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
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/LoweringOptions.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace sv;
using namespace hw;
using namespace ExportVerilog;

//===----------------------------------------------------------------------===//
// GlobalNameTable
//===----------------------------------------------------------------------===//

void GlobalNameTable::addReservedNames(NameCollisionResolver &resolver) const {
  for (auto &name : reservedNames)
    resolver.insertUsedName(name);
}

//===----------------------------------------------------------------------===//
// NameCollisionResolver
//===----------------------------------------------------------------------===//

/// Given a name that may have collisions or invalid symbols, return a
/// replacement name to use, or null if the original name was ok.
StringRef NameCollisionResolver::getLegalName(StringRef originalName) {
  return legalizeName(originalName, nextGeneratedNameIDs,
                      options.caseInsensitiveKeywords);
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
  bool hasToBeRenamed =
      !sv::isNameValid(fieldName.getValue(), options.caseInsensitiveKeywords) ||
      nextGeneratedNameIDs.contains(fieldName.getValue());

  if (!hasToBeRenamed) {
    setRenamedFieldName(fieldName, fieldName);
    return fieldName;
  }

  StringRef newFieldName =
      sv::legalizeName(fieldName.getValue(), nextGeneratedNameIDs,
                       options.caseInsensitiveKeywords);

  auto newFieldNameAttr = StringAttr::get(fieldName.getContext(), newFieldName);

  setRenamedFieldName(fieldName, newFieldNameAttr);
  return newFieldNameAttr;
}

std::string FieldNameResolver::getEnumFieldName(hw::EnumFieldAttr attr) {
  auto aliasType = dyn_cast<hw::TypeAliasType>(attr.getType().getValue());
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
/// well as module ports, parameters, declarations and verif labels that need to
/// be renamed. This can happen either due to conflicts between them or due to
/// a conflict with a Verilog keyword.
///
/// Once constructed, this is immutable.
class GlobalNameResolver {
public:
  /// Construct a GlobalNameResolver and perform name legalization of the
  /// module/interfaces, port/parameter and declaration names.
  GlobalNameResolver(mlir::ModuleOp topLevel, const LoweringOptions &options);

  GlobalNameTable takeGlobalNameTable() { return std::move(globalNameTable); }

private:
  /// Check to see if the port names of the specified module conflict with
  /// keywords or themselves.  If so, add the replacement names to
  /// globalNameTable.
  void legalizeModuleNames(HWModuleOp module);
  void legalizeInterfaceNames(InterfaceOp interface);
  void legalizeFunctionNames(FuncOp func);

  // Gathers prefixes of enum types by inspecting typescopes in the module.
  void gatherEnumPrefixes(mlir::ModuleOp topLevel);

  /// Set of globally visible names, to ensure uniqueness.
  NameCollisionResolver globalNameResolver;

  /// This keeps track of globally visible names like module parameters.
  GlobalNameTable globalNameTable;

  GlobalNameResolver(const GlobalNameResolver &) = delete;
  void operator=(const GlobalNameResolver &) = delete;

  // Handle to lowering options.
  const LoweringOptions &options;
};
} // namespace ExportVerilog
} // namespace circt

// This function legalizes local names in the given module.
static void legalizeModuleLocalNames(HWEmittableModuleLike module,
                                     const LoweringOptions &options,
                                     const GlobalNameTable &globalNameTable) {
  // A resolver for a local name collison.
  NameCollisionResolver nameResolver(options);
  globalNameTable.addReservedNames(nameResolver);

  // Register names used by parameters.
  if (auto hwModule = dyn_cast<hw::HWModuleOp>(*module))
    for (auto param : hwModule.getParameters())
      nameResolver.insertUsedName(globalNameTable.getParameterVerilogName(
          module, cast<ParamDeclAttr>(param).getName()));

  auto *ctxt = module.getContext();

  auto verilogNameAttr = StringAttr::get(ctxt, "hw.verilogName");
  // Legalize the port names.
  auto ports = module.getPortList();
  SmallVector<Attribute> newNames(ports.size());
  bool updated = false;
  bool isFuncOp = isa<FuncOp>(module);
  for (auto [idx, port] : llvm::enumerate(ports)) {
    auto verilogName = port.attrs.get(verilogNameAttr);
    // A function return value must named the exact same name to its function
    // Verilog name.
    if (isFuncOp && port.attrs.get(FuncOp::getExplicitlyReturnedAttrName())) {
      updated = true;
      newNames[idx] = StringAttr::get(ctxt, getSymOpName(module));
      continue;
    }
    if (verilogName) {
      auto newName = StringAttr::get(
          ctxt, nameResolver.getLegalName(cast<StringAttr>(verilogName)));
      newNames[idx] = newName;
      if (verilogName != newName)
        updated = true;
      continue;
    }
    auto oldName = ports[idx].name;
    auto newName = nameResolver.getLegalName(oldName);
    // Set the verilogName attr only if the name is updated.
    if (newName != oldName) {
      newNames[idx] = StringAttr::get(ctxt, newName);
      updated = true;
    } else
      newNames[idx] = {};
  }
  if (updated)
    module.setPortAttrs(verilogNameAttr, newNames);

  SmallVector<std::pair<Operation *, StringAttr>> nameEntries;
  // Legalize the value names. We first mark existing hw.verilogName attrs as
  // being used, and then resolve names of declarations.
  module.walk([&](Operation *op) {
    if (module != op) {
      // If there is a hw.verilogName attr, mark names as used.
      if (auto name = op->getAttrOfType<StringAttr>(verilogNameAttr)) {
        nameResolver.insertUsedName(
            op->getAttrOfType<StringAttr>(verilogNameAttr));
      } else if (isa<sv::WireOp, hw::WireOp, RegOp, LogicOp, LocalParamOp,
                     hw::InstanceOp, sv::InterfaceInstanceOp, sv::GenerateOp>(
                     op)) {
        // Otherwise, get a verilog name via `getSymOpName`.
        nameEntries.emplace_back(
            op, StringAttr::get(op->getContext(), getSymOpName(op)));
      } else if (auto forOp = dyn_cast<ForOp>(op)) {
        nameEntries.emplace_back(op, forOp.getInductionVarNameAttr());
      } else if (isa<AssertOp, AssumeOp, CoverOp, AssertConcurrentOp,
                     AssumeConcurrentOp, CoverConcurrentOp, AssertPropertyOp,
                     AssumePropertyOp, CoverPropertyOp, verif::AssertOp,
                     verif::CoverOp, verif::AssumeOp>(op)) {
        // Notice and renamify the labels on verification statements.
        if (auto labelAttr = op->getAttrOfType<StringAttr>("label"))
          nameEntries.emplace_back(op, labelAttr);
        else if (options.enforceVerifLabels) {
          // If labels are required for all verif statements, get a default
          // name from verificaiton kinds.
          StringRef defaultName =
              llvm::TypeSwitch<Operation *, StringRef>(op)
                  .Case<AssertOp, AssertConcurrentOp, AssertPropertyOp,
                        verif::AssertOp>([](auto) { return "assert"; })
                  .Case<CoverOp, CoverConcurrentOp, CoverPropertyOp,
                        verif::CoverOp>([](auto) { return "cover"; })
                  .Case<AssumeOp, AssumeConcurrentOp, AssumePropertyOp,
                        verif::AssumeOp>([](auto) { return "assume"; });
          nameEntries.emplace_back(
              op, StringAttr::get(op->getContext(), defaultName));
        }
      }
    }
  });

  for (auto [op, nameAttr] : nameEntries) {
    auto newName = nameResolver.getLegalName(nameAttr);
    assert(!newName.empty() && "must have a valid name");
    // Add a legalized name to "hw.verilogName" attribute.
    op->setAttr(verilogNameAttr, nameAttr.getValue() == newName
                                     ? nameAttr
                                     : StringAttr::get(ctxt, newName));
  }
}

/// Construct a GlobalNameResolver and do the initial scan to populate and
/// unique the module/interfaces and port/parameter names.
GlobalNameResolver::GlobalNameResolver(mlir::ModuleOp topLevel,
                                       const LoweringOptions &options)
    : globalNameResolver(options), options(options) {
  // Register the names of external modules which we cannot rename. This has to
  // occur in a first pass separate from the modules and interfaces which we are
  // actually allowed to rename, in order to ensure that we don't accidentally
  // rename a module that later collides with an extern module.
  for (auto &op : *topLevel.getBody()) {
    // Note that external modules *often* have name collisions, because they
    // correspond to the same verilog module with different parameters.
    if (isa<HWModuleExternOp>(op) || isa<HWModuleGeneratedOp>(op)) {
      auto name = getVerilogModuleNameAttr(&op).getValue();
      if (!sv::isNameValid(name, options.caseInsensitiveKeywords, true))
        op.emitError("name \"")
            << name << "\" is not allowed in Verilog output";
      globalNameResolver.insertUsedName(name);
    } else if (auto reservedNamesOp = dyn_cast<sv::ReserveNamesOp>(op)) {
      for (StringAttr name :
           reservedNamesOp.getReservedNames().getAsRange<StringAttr>()) {
        globalNameTable.reservedNames.insert(name);
        globalNameResolver.insertUsedName(name);
      }
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

  // Legalize names in HW modules parallelly.
  mlir::parallelForEach(
      topLevel.getContext(), topLevel.getOps<HWEmittableModuleLike>(),
      [&](auto module) {
        legalizeModuleLocalNames(module, options, globalNameTable);
      });

  // Gather enum prefixes.
  gatherEnumPrefixes(topLevel);
}

// Gathers prefixes of enum types by investigating typescopes in the module.
void GlobalNameResolver::gatherEnumPrefixes(mlir::ModuleOp topLevel) {
  auto *ctx = topLevel.getContext();
  for (auto typeScope : topLevel.getOps<hw::TypeScopeOp>()) {
    for (auto typeDecl : typeScope.getOps<hw::TypedeclOp>()) {
      auto enumType = dyn_cast<hw::EnumType>(typeDecl.getType());
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

  NameCollisionResolver nameResolver(options);
  // Legalize the parameter names.
  for (auto param : module.getParameters()) {
    auto paramAttr = cast<ParamDeclAttr>(param);
    auto newName = nameResolver.getLegalName(paramAttr.getName());
    if (newName != paramAttr.getName().getValue())
      globalNameTable.addRenamedParam(module, paramAttr.getName(), newName);
  }
}

void GlobalNameResolver::legalizeInterfaceNames(InterfaceOp interface) {
  MLIRContext *ctxt = interface.getContext();
  auto verilogNameAttr = StringAttr::get(ctxt, "hw.verilogName");
  auto newName = globalNameResolver.getLegalName(interface.getName());
  if (newName != interface.getName())
    interface->setAttr(verilogNameAttr, StringAttr::get(ctxt, newName));

  NameCollisionResolver localNames(options);
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

void GlobalNameResolver::legalizeFunctionNames(FuncOp func) {
  MLIRContext *ctxt = func.getContext();
  if (auto verilogName = func.getVerilogName()) {
    globalNameResolver.insertUsedName(*verilogName);
    return;
  }
  auto newName = globalNameResolver.getLegalName(func.getName());
  if (newName != func.getName()) {
    func.setVerilogName(StringAttr::get(ctxt, newName));
  }
}

//===----------------------------------------------------------------------===//
// Public interface
//===----------------------------------------------------------------------===//

/// Rewrite module names and interfaces to not conflict with each other or with
/// Verilog keywords.
GlobalNameTable
ExportVerilog::legalizeGlobalNames(ModuleOp topLevel,
                                   const LoweringOptions &options) {
  GlobalNameResolver resolver(topLevel, options);
  return resolver.takeGlobalNameTable();
}
