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

#include "PassDetail.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVPasses.h"

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
// HWLegalizeNamesPass
//===----------------------------------------------------------------------===//

namespace {
struct HWLegalizeNamesPass
    : public sv::HWLegalizeNamesBase<HWLegalizeNamesPass> {
  void runOnOperation() override;

private:
  bool anythingChanged;

  bool legalizePortNames(hw::HWModuleOp module);
  void runOnModule(hw::HWModuleOp module,
                   DenseMap<Attribute, HWModuleOp> &modulesWithRenamedPorts);
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

  DenseMap<Attribute, HWModuleOp> modulesWithRenamedPorts;

  // Legalize module and interface names.
  for (auto &op : *root.getBody()) {
    if (!isa<HWModuleOp>(op) && !isa<InterfaceOp>(op))
      continue;

    // If the module's symbol itself conflicts, then rename it and all uses of
    // it.
    StringAttr oldName = op.getAttrOfType<StringAttr>(symbolAttrName);
    auto newName = nameResolver.getLegalName(oldName);
    if (!newName.empty()) {
      auto newNameAttr = StringAttr::get(&getContext(), newName);
      symbolUsers.replaceAllUsesWith(&op, newNameAttr);
      SymbolTable::setSymbolName(&op, newNameAttr);
      anythingChanged = true;
    }

    if (auto module = dyn_cast<HWModuleOp>(op)) {
      if (legalizePortNames(module))
        modulesWithRenamedPorts[module.getNameAttr()] = module;
    }
  }

  // Rename individual operations.
  for (auto &op : *root.getBody()) {
    if (auto module = dyn_cast<HWModuleOp>(op)) {
      runOnModule(module, modulesWithRenamedPorts);
    } else if (auto intf = dyn_cast<InterfaceOp>(op)) {
      runOnInterface(intf, symbolUsers);
    } else if (auto extMod = dyn_cast<HWModuleExternOp>(op)) {
      auto name = extMod.getVerilogModuleName();
      if (!sv::isNameValid(name)) {
        extMod->emitOpError("with invalid name \"" + name + "\"");
      }
    }
  }

  // If we did not change anything in the graph mark all analysis as
  // preserved.
  if (!anythingChanged)
    markAllAnalysesPreserved();
}

/// Check to see if the port names of the specified module conflict with
/// keywords or themselves.  If so, rename them and return true, otherwise
/// return false.
bool HWLegalizeNamesPass::legalizePortNames(hw::HWModuleOp module) {
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

  if (changedArgNames) {
    setModuleArgumentNames(module, argNames);
    anythingChanged = true;
  }
  if (changedOutputNames) {
    setModuleResultNames(module, outputNames);
    anythingChanged = true;
  }
  return changedArgNames | changedOutputNames;
}

void HWLegalizeNamesPass::runOnModule(
    hw::HWModuleOp module,
    DenseMap<Attribute, HWModuleOp> &modulesWithRenamedPorts) {
  NameCollisionResolver nameResolver;

  // All the ports are pre-legalized, just add their names to the map so we
  // detect conflicts with them.
  for (const PortInfo &port : getAllModulePortInfos(module))
    (void)nameResolver.getLegalName(port.name);

  // Rename the instances, regs, and wires.
  for (auto &op : *module.getBodyBlock()) {
    if (auto instanceOp = dyn_cast<InstanceOp>(op)) {
      auto newName = nameResolver.getLegalName(instanceOp.getName());
      if (!newName.empty()) {
        instanceOp.setName(StringAttr::get(&getContext(), newName));
        anythingChanged = true;
      }

      // If this instance is referring to a module with renamed ports, update
      // them.
      auto it =
          modulesWithRenamedPorts.find(instanceOp.moduleNameAttr().getAttr());
      if (it != modulesWithRenamedPorts.end()) {
        auto mod = it->second;
        instanceOp.setArgumentNames(mod.argNames());
        instanceOp.setResultNames(mod.resultNames());
      }

    } else if (isa<RegOp>(op) || isa<WireOp>(op)) {
      auto oldName = op.getAttrOfType<StringAttr>("name");
      auto newName = nameResolver.getLegalName(oldName);
      if (!newName.empty()) {
        op.setAttr("name", StringAttr::get(&getContext(), newName));
        anythingChanged = true;
      }
    }
  }
}

void HWLegalizeNamesPass::runOnInterface(InterfaceOp interface,
                                         mlir::SymbolUserMap &symbolUsers) {
  NameCollisionResolver localNames;
  auto symbolAttrName = SymbolTable::getSymbolAttrName();

  // Rename signals and modports.
  for (auto &op : *interface.getBodyBlock()) {
    if (!isa<InterfaceSignalOp>(op) && !isa<InterfaceModportOp>(op))
      continue;

    StringAttr oldName = op.getAttrOfType<StringAttr>(symbolAttrName);
    auto newName = localNames.getLegalName(oldName);
    if (newName.empty())
      continue;

    auto newNameAttr = StringAttr::get(&getContext(), newName);
    symbolUsers.replaceAllUsesWith(&op, newNameAttr);
    SymbolTable::setSymbolName(&op, newNameAttr);
    anythingChanged = true;
  }
}

std::unique_ptr<Pass> circt::sv::createHWLegalizeNamesPass() {
  return std::make_unique<HWLegalizeNamesPass>();
}
