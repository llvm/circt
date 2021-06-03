//===- HWLegalizeNames.cpp - HW Name Legalization Pass --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass renames modules and variables to avoid conflicts
// with keywords and other declarations.
//
//===----------------------------------------------------------------------===//

#include "SVPassDetail.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace circt;
using namespace sv;
using namespace hw;

//===----------------------------------------------------------------------===//
// NameCollisionResolver
//===----------------------------------------------------------------------===//

namespace {
struct NameCollisionResolver {
  NameCollisionResolver() = default;

  /// Given a name that may have collisions or invalid symbols, return a
  /// replacement name to use, or null if the original name was ok.
  StringRef getLegalName(StringAttr originalName);

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
// Type declarations
//===----------------------------------------------------------------------===//

/// Given an operation and a type, check if the type is a type alias, and if so,
/// if it needs to have a type declaration generated.
void maybeDeclareTypeAlias(Operation *op, Type type) {
  // Look for only TypeAliasTypes.
  auto alias = type.dyn_cast<TypeAliasType>();
  if (!alias)
    return;

  ModuleOp parentModule = op->getParentOfType<ModuleOp>();
  ImplicitLocOpBuilder builder(op->getLoc(), &parentModule.getBody()->front());

  // If no scope exists, create one in the parent module.
  StringRef scopeName = alias.getRef().getRootReference();
  TypeScopeOp scope = parentModule.lookupSymbol<TypeScopeOp>(scopeName);
  if (!scope) {
    scope = builder.create<TypeScopeOp>(op->getLoc(), scopeName);
    scope.body().emplaceBlock();
  }

  // If no typedecl exists, create one in the scope.
  StringRef symbolName = alias.getRef().getLeafReference();
  TypedeclOp typeDecl = scope.lookupSymbol<TypedeclOp>(symbolName);
  if (!typeDecl) {
    // TODO: The insertion point should generate type aliases of type aliases in
    // an order that respects def-before-use, or fails on mutually recursive
    // type aliases. For now, insert at the end as we go.
    builder.setInsertionPointToEnd(scope.getBodyBlock());
    builder.create<TypedeclOp>(op->getLoc(), symbolName, alias.getInnerType(),
                               StringAttr());
    return;
  }

  // If a typedecl exists with a different type for the same name, emit an
  // error.
  if (typeDecl.type() != alias.getInnerType())
    op->emitOpError("redefining type definition for ") << typeDecl;
}

//===----------------------------------------------------------------------===//
// HWLegalizeNamesPass
//===----------------------------------------------------------------------===//

namespace {
struct HWLegalizeNamesPass
    : public sv::HWLegalizeNamesBase<HWLegalizeNamesPass> {
  void runOnOperation() override;

private:
  bool anythingChanged;

  void runOnModule(hw::HWModuleOp module);
  void runOnModuleExtern(hw::HWModuleExternOp extMod);
  void runOnInterface(sv::InterfaceOp intf, mlir::SymbolUserMap &symbolUsers);
};
} // end anonymous namespace

void HWLegalizeNamesPass::runOnOperation() {
  anythingChanged = false;
  ModuleOp root = getOperation();
  mlir::SymbolTableCollection symbolTable;
  mlir::SymbolUserMap symbolUsers(symbolTable, root);

  // Analyze the legal names for top-level operations in the MLIR module.
  NameCollisionResolver nameResolver;

  // Register the names of external modules which we cannot rename. This has to
  // occur in a first pass separate from the modules and interfaces which we are
  // actually allowed to rename, in order to ensure that we don't accidentally
  // rename a module that later collides with an extern module.
  for (auto &op : *root.getBody()) {
    // Note that external modules *often* have name collisions, because they
    // correspond to the same verilog module with different parameters.
    if (auto extMod = dyn_cast<HWModuleExternOp>(op))
      (void)nameResolver.getLegalName(extMod.getVerilogModuleNameAttr());
  }

  auto symbolAttrName = SymbolTable::getSymbolAttrName();

  // Legalize module and interface names.
  for (auto &op : *root.getBody()) {
    if (!isa<HWModuleOp>(op) && !isa<InterfaceOp>(op))
      continue;

    StringAttr oldName = op.getAttrOfType<StringAttr>(symbolAttrName);
    auto newName = nameResolver.getLegalName(oldName);
    if (newName.empty())
      continue;

    symbolUsers.replaceAllUsesWith(&op, newName);
    SymbolTable::setSymbolName(&op, newName);
    anythingChanged = true;
  }

  // Rename individual operations.
  for (auto &op : *root.getBody()) {
    if (auto module = dyn_cast<HWModuleOp>(op)) {
      runOnModule(module);
    } else if (auto intf = dyn_cast<InterfaceOp>(op)) {
      runOnInterface(intf, symbolUsers);
    } else if (auto extMod = dyn_cast<HWModuleExternOp>(op)) {
      runOnModuleExtern(extMod);
    }
  }

  // If we did not change anything in the graph mark all analysis as
  // preserved.
  if (!anythingChanged)
    markAllAnalysesPreserved();
}

void HWLegalizeNamesPass::runOnModule(hw::HWModuleOp module) {
  NameCollisionResolver nameResolver;

  bool changedArgNames = false, changedOutputNames = false;
  SmallVector<Attribute> argNames, outputNames;

  // Legalize the ports.
  for (const ModulePortInfo &port : getModulePortInfo(module)) {
    auto newName = nameResolver.getLegalName(port.name);

    auto &namesVector = port.isOutput() ? outputNames : argNames;
    auto &changedBool = port.isOutput() ? changedOutputNames : changedArgNames;

    if (newName.empty()) {
      namesVector.push_back(port.name);
    } else {
      changedBool = true;
      namesVector.push_back(StringAttr::get(module.getContext(), newName));
    }

    maybeDeclareTypeAlias(module, port.type);
  }

  if (changedArgNames) {
    setModuleArgumentNames(module, argNames);
    anythingChanged = true;
  }
  if (changedOutputNames) {
    setModuleResultNames(module, outputNames);
    anythingChanged = true;
  }

  // Rename the instances, regs, and wires.
  for (auto &op : *module.getBodyBlock()) {
    if (auto instanceOp = dyn_cast<InstanceOp>(op)) {
      auto newName = nameResolver.getLegalName(instanceOp.getNameAttr());
      if (!newName.empty()) {
        instanceOp.setName(newName);
        anythingChanged = true;
      }
    } else if (isa<RegOp>(op) || isa<WireOp>(op)) {
      auto oldName = op.getAttrOfType<StringAttr>("name");
      auto newName = nameResolver.getLegalName(oldName);
      if (!newName.empty()) {
        op.setAttr("name", StringAttr::get(op.getContext(), newName));
        anythingChanged = true;
      }
    }

    for (auto type : op.getOperandTypes())
      maybeDeclareTypeAlias(&op, type);
    for (auto type : op.getResultTypes())
      maybeDeclareTypeAlias(&op, type);
  }
}

void HWLegalizeNamesPass::runOnModuleExtern(hw::HWModuleExternOp extMod) {
  auto name = extMod.getVerilogModuleName();
  if (!sv::isNameValid(name)) {
    extMod->emitOpError("with invalid name \"" + name + "\"");
  }

  for (const ModulePortInfo &port : getModulePortInfo(extMod))
    maybeDeclareTypeAlias(extMod, port.type);
}

void HWLegalizeNamesPass::runOnInterface(InterfaceOp interface,
                                         mlir::SymbolUserMap &symbolUsers) {
  NameCollisionResolver localNames;
  auto symbolAttrName = SymbolTable::getSymbolAttrName();

  // Rename signals and modports.
  for (auto &op : *interface.getBodyBlock()) {
    if (!isa<InterfaceSignalOp>(op) && !isa<InterfaceModportOp>(op))
      continue;

    for (auto attr : op.getAttrs())
      if (auto typeAttr = attr.second.dyn_cast<TypeAttr>())
        maybeDeclareTypeAlias(&op, typeAttr.getValue());

    StringAttr oldName = op.getAttrOfType<StringAttr>(symbolAttrName);
    auto newName = localNames.getLegalName(oldName);
    if (newName.empty())
      continue;
    symbolUsers.replaceAllUsesWith(&op, newName);
    SymbolTable::setSymbolName(&op, newName);
    anythingChanged = true;
  }
}

std::unique_ptr<Pass> circt::sv::createHWLegalizeNamesPass() {
  return std::make_unique<HWLegalizeNamesPass>();
}
