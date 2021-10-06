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
#include "circt/Dialect/HW/HWAttributes.h"
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

namespace {
/// This class keeps track of modules and interfaces that need to be renamed, as
/// well as module ports and parameters that need to be renamed.  This can
/// happen either due to conflicts between them or due to a conflict with a
/// Verilog keyword.
///
/// Once constructed, this is immutable.
struct GlobalNameResolver {
  /// Construct a GlobalNameResolver and do the initial scan to populate and
  /// unique the module/interfaces and port/parameter names.
  GlobalNameResolver(mlir::ModuleOp topLevel);

  bool anythingChanged = false;

  /// If the module with the specified name has had a port or parameter renamed,
  /// return the module that defines the name.
  HWModuleOp getModuleWithRenamedInterface(StringAttr name) {
    auto it = modulesWithRenamedPortsOrParams.find(name);
    return it != modulesWithRenamedPortsOrParams.end() ? it->second
                                                       : HWModuleOp();
  }

private:
  /// Check to see if the port names of the specified module conflict with
  /// keywords or themselves.  If so, rename them and return true, otherwise
  /// return false.
  bool legalizePortNames(hw::HWModuleOp module);

  Attribute remapRenamedParameters(Attribute value, HWModuleOp module);
  void updateInstanceForChangedModule(InstanceOp inst, HWModuleOp module);
  void updateInstanceParamDeclRefs(InstanceOp instance);
  void rewriteModuleBody(Block &block, NameCollisionResolver &nameResolver,
                         bool moduleHasRenamedInterface);
  void renameModuleBody(hw::HWModuleOp module);
  void renameInterfaceBody(sv::InterfaceOp intf,
                           mlir::SymbolUserMap &symbolUsers);

  /// Set of globally visible names, to ensure uniqueness.
  NameCollisionResolver globalNames;

  /// If a module has a port or parameter renamed, then this keeps track of the
  /// module it is associated with.
  DenseMap<Attribute, HWModuleOp> modulesWithRenamedPortsOrParams;

  /// This map keeps track of a mapping from <module,parametername> -> newName,
  /// it is populated when a parameter has to be renamed.
  typedef DenseMap<std::pair<Operation *, Attribute>, Attribute>
      RenamedParameterMapTy;

  // This map keeps track of a mapping from <module,parametername> -> newName,
  // it is populated when a parameter has to be renamed.
  RenamedParameterMapTy renamedParameterInfo;

  GlobalNameResolver(const GlobalNameResolver &) = delete;
  void operator=(const GlobalNameResolver &) = delete;
};
} // end anonymous namespace

/// Construct a GlobalNameResolver and do the initial scan to populate and
/// unique the module/interfaces and port/parameter names.
GlobalNameResolver::GlobalNameResolver(mlir::ModuleOp topLevel) {
  // FIXME: Make this lazy.
  mlir::SymbolTableCollection symbolTable;
  mlir::SymbolUserMap symbolUsers(symbolTable, topLevel);

  // Register the names of external modules which we cannot rename. This has to
  // occur in a first pass separate from the modules and interfaces which we are
  // actually allowed to rename, in order to ensure that we don't accidentally
  // rename a module that later collides with an extern module.
  for (auto &op : *topLevel.getBody()) {
    // Note that external modules *often* have name collisions, because they
    // correspond to the same verilog module with different parameters.
    if (auto extMod = dyn_cast<HWModuleExternOp>(op)) {
      auto name = extMod.getVerilogModuleName();
      if (!sv::isNameValid(name))
        extMod->emitOpError("with invalid name \"" + name + "\"");
      globalNames.insertUsedName(name);
    }
  }

  // If the module's symbol itself conflicts, then rename it and all uses of
  // it.
  // TODO: This is super inefficient, we should just rename the symbol as part
  // of the other existing walks.
  auto legalizeSymbolName = [&](Operation *op) {
    StringAttr oldName = SymbolTable::getSymbolName(op);
    auto newName = globalNames.getLegalName(oldName);
    if (!newName.empty()) {
      auto newNameAttr = StringAttr::get(topLevel.getContext(), newName);
      symbolUsers.replaceAllUsesWith(op, newNameAttr);
      SymbolTable::setSymbolName(op, newNameAttr);
      anythingChanged = true;
    }
  };

  // Legalize module and interface names.
  for (auto &op : *topLevel.getBody()) {
    if (auto module = dyn_cast<HWModuleOp>(op)) {
      legalizeSymbolName(module);
      if (legalizePortNames(module))
        modulesWithRenamedPortsOrParams[module.getNameAttr()] = module;
      continue;
    }

    if (auto interface = dyn_cast<InterfaceOp>(op)) {
      legalizeSymbolName(interface);
      continue;
    }
  }

  // Rename individual operations within the bodies.
  for (auto &op : *topLevel.getBody()) {
    if (auto module = dyn_cast<HWModuleOp>(op)) {
      renameModuleBody(module);
    } else if (auto intf = dyn_cast<InterfaceOp>(op)) {
      renameInterfaceBody(intf, symbolUsers);
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
  SmallVector<Attribute> parameters;
  bool changedParameters = false;
  for (auto param : module.parameters()) {
    auto paramAttr = param.cast<ParamDeclAttr>();
    auto newName = nameResolver.getLegalName(paramAttr.getName());
    if (newName.empty())
      parameters.push_back(param);
    else {
      auto newNameAttr = StringAttr::get(paramAttr.getContext(), newName);
      parameters.push_back(ParamDeclAttr::getWithName(paramAttr, newNameAttr));
      changedParameters = true;
      renamedParameterInfo[std::make_pair(module, paramAttr.getName())] =
          newNameAttr;
    }
  }
  if (changedParameters)
    module->setAttr("parameters",
                    ArrayAttr::get(module.getContext(), parameters));

  if (changedArgNames | changedOutputNames | changedParameters) {
    anythingChanged = true;
    return true;
  }

  return false;
}

/// Scan a parameter expression tree, handling any renamed parameters that may
/// occur.
Attribute GlobalNameResolver::remapRenamedParameters(Attribute value,
                                                     HWModuleOp module) {
  // Literals are always fine and never change.
  if (value.isa<IntegerAttr>() || value.isa<FloatAttr>() ||
      value.isa<StringAttr>() || value.isa<ParamVerbatimAttr>())
    return value;

  // Remap leaves of expressions if needed.
  if (auto expr = value.dyn_cast<ParamExprAttr>()) {
    SmallVector<Attribute> newOperands;
    bool anyChanged = false;
    for (auto op : expr.getOperands()) {
      newOperands.push_back(remapRenamedParameters(op, module));
      anyChanged |= newOperands.back() != op;
    }
    // Don't rebuild an attribute if nothing changed.
    if (!anyChanged)
      return value;
    return ParamExprAttr::get(expr.getOpcode(), newOperands);
  }

  // Otherwise this must be a parameter reference.
  auto parameterRef = value.dyn_cast<ParamDeclRefAttr>();
  assert(parameterRef && "Unknown kind of parameter expression");

  // If this parameter is un-renamed, then leave it alone.
  auto nameAttr = parameterRef.getName();
  auto it = renamedParameterInfo.find(std::make_pair(module, nameAttr));
  if (it == renamedParameterInfo.end())
    return value;

  // Okay, it was renamed, return the new name with the right type.
  return ParamDeclRefAttr::get(value.getContext(),
                               it->second.cast<StringAttr>(), value.getType());
}

// If this instance is referring to a module with renamed ports or
// parameter names, update them.
void GlobalNameResolver::updateInstanceForChangedModule(InstanceOp inst,
                                                        HWModuleOp module) {
  inst.argNamesAttr(module.argNames());
  inst.resultNamesAttr(module.resultNames());

  // If any module parameters changed names, take the new name.
  SmallVector<Attribute> newAttrs;
  auto instParameters = inst.parameters();
  auto modParameters = module.parameters();
  for (size_t i = 0, e = instParameters.size(); i != e; ++i) {
    auto instParam = instParameters[i].cast<ParamDeclAttr>();
    auto modParam = modParameters[i].cast<ParamDeclAttr>();
    if (instParam.getName() == modParam.getName())
      newAttrs.push_back(instParam);
    else
      newAttrs.push_back(
          ParamDeclAttr::getWithName(instParam, modParam.getName()));
  }
  inst.parametersAttr(ArrayAttr::get(inst.getContext(), newAttrs));
}

/// Rename any parameter values being specified for an instance if they are
/// referring to parameters that got renamed.
void GlobalNameResolver::updateInstanceParamDeclRefs(InstanceOp instance) {
  auto parameters = instance.parameters();
  if (parameters.empty())
    return;

  auto curModule = instance->getParentOfType<HWModuleOp>();

  SmallVector<Attribute> newParams;
  newParams.reserve(parameters.size());
  bool anyRenamed = false;
  for (Attribute param : parameters) {
    auto paramAttr = param.cast<ParamDeclAttr>();
    auto newValue = remapRenamedParameters(paramAttr.getValue(), curModule);
    if (newValue == paramAttr.getValue()) {
      newParams.push_back(param);
      continue;
    }
    anyRenamed = true;
    newParams.push_back(ParamDeclAttr::get(paramAttr.getName(), newValue));
  }

  instance.parametersAttr(ArrayAttr::get(instance.getContext(), newParams));
}

void GlobalNameResolver::rewriteModuleBody(Block &block,
                                           NameCollisionResolver &nameResolver,
                                           bool moduleHasRenamedInterface) {

  // Rename the instances, regs, and wires.
  for (auto &op : block) {
    if (auto instanceOp = dyn_cast<InstanceOp>(op)) {
      auto newName = nameResolver.getLegalName(instanceOp.getName());
      if (!newName.empty())
        instanceOp.setName(StringAttr::get(instanceOp.getContext(), newName));

      // If this instance is referring to a module with renamed ports or
      // parameter names, update them.
      if (HWModuleOp module = getModuleWithRenamedInterface(
              instanceOp.moduleNameAttr().getAttr()))
        updateInstanceForChangedModule(instanceOp, module);

      if (moduleHasRenamedInterface)
        updateInstanceParamDeclRefs(instanceOp);
      continue;
    }

    if (isa<RegOp>(op) || isa<WireOp>(op)) {
      auto oldName = op.getAttrOfType<StringAttr>("name");
      auto newName = nameResolver.getLegalName(oldName);
      if (!newName.empty())
        op.setAttr("name", StringAttr::get(op.getContext(), newName));
      continue;
    }

    if (auto localParam = dyn_cast<LocalParamOp>(op)) {
      // If the initializer value in the local param was renamed then update it.
      if (moduleHasRenamedInterface) {
        auto curModule = op.getParentOfType<HWModuleOp>();
        localParam.valueAttr(
            remapRenamedParameters(localParam.value(), curModule));
      }
      continue;
    }

    if (auto paramValue = dyn_cast<ParamValueOp>(op)) {
      // If the initializer value in the local param was renamed then update it.
      if (moduleHasRenamedInterface) {
        auto curModule = op.getParentOfType<HWModuleOp>();
        paramValue.valueAttr(
            remapRenamedParameters(paramValue.value(), curModule));
      }
      continue;
    }

    // If this operation has regions, then we recursively process them if they
    // can contain things that need to be renamed.  We don't walk the module
    // in the common case.
    if (op.getNumRegions() && (isa<IfDefOp>(op) || moduleHasRenamedInterface)) {
      for (auto &region : op.getRegions()) {
        if (!region.empty())
          rewriteModuleBody(region.front(), nameResolver,
                            moduleHasRenamedInterface);
      }
    }
  }
}

void GlobalNameResolver::renameModuleBody(hw::HWModuleOp module) {

  // If this module had something about its interface, then a parameter may
  // have been changed.  In that case, we change parameter references to match.
  // This isn't common, so we use this to avoid work.
  bool moduleHasRenamedInterface =
      getModuleWithRenamedInterface(module.getNameAttr()) != HWModuleOp();

  // All the ports are pre-legalized, just add their names to the map so we
  // detect conflicts with them.
  NameCollisionResolver nameResolver;
  for (const PortInfo &port : getAllModulePortInfos(module))
    (void)nameResolver.getLegalName(port.name);

  rewriteModuleBody(*module.getBodyBlock(), nameResolver,
                    moduleHasRenamedInterface);
}

void GlobalNameResolver::renameInterfaceBody(InterfaceOp interface,
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

    auto newNameAttr = StringAttr::get(interface.getContext(), newName);
    symbolUsers.replaceAllUsesWith(&op, newNameAttr);
    SymbolTable::setSymbolName(&op, newNameAttr);
    anythingChanged = true;
  }
}

//===----------------------------------------------------------------------===//
// HWLegalizeNamesPass
//===----------------------------------------------------------------------===//

namespace {
struct HWLegalizeNamesPass
    : public sv::HWLegalizeNamesBase<HWLegalizeNamesPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

void HWLegalizeNamesPass::runOnOperation() {
  GlobalNameResolver nameResolver(getOperation());

  // If we did not change anything in the graph mark all analysis as
  // preserved.
  if (!nameResolver.anythingChanged)
    markAllAnalysesPreserved();
}

std::unique_ptr<Pass> circt::sv::createHWLegalizeNamesPass() {
  return std::make_unique<HWLegalizeNamesPass>();
}
