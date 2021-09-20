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
/// This map keeps track of a mapping from <module,parametername> -> newName,
/// it is populated when a parameter has to be renamed.
typedef DenseMap<std::pair<Operation *, Attribute>, Attribute>
    RenamedParameterMapTy;
} // namespace

namespace {
struct HWLegalizeNamesPass
    : public sv::HWLegalizeNamesBase<HWLegalizeNamesPass> {
  void runOnOperation() override;

private:
  bool anythingChanged;

  bool legalizePortNames(hw::HWModuleOp module,
                         RenamedParameterMapTy &renamedParameterInfo);
  void runOnModule(hw::HWModuleOp module,
                   DenseMap<Attribute, HWModuleOp> &modulesWithRenamedPorts,
                   RenamedParameterMapTy &renamedParameterInfo);
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

  // This map keeps track of a mapping from <module,parametername> -> newName,
  // it is populated when a parameter has to be renamed.
  RenamedParameterMapTy renamedParameterInfo;

  // Legalize module and interface names.
  for (auto &op : *root.getBody()) {
    if (!isa<HWModuleOp>(op) && !isa<InterfaceOp>(op))
      continue;

    // If the module's symbol itself conflicts, then rename it and all uses of
    // it.
    // TODO: This is super inefficient, we should just rename the symbol as part
    // of the other existing walks.
    StringAttr oldName = op.getAttrOfType<StringAttr>(symbolAttrName);
    auto newName = nameResolver.getLegalName(oldName);
    if (!newName.empty()) {
      auto newNameAttr = StringAttr::get(&getContext(), newName);
      symbolUsers.replaceAllUsesWith(&op, newNameAttr);
      SymbolTable::setSymbolName(&op, newNameAttr);
      anythingChanged = true;
    }

    if (auto module = dyn_cast<HWModuleOp>(op)) {
      if (legalizePortNames(module, renamedParameterInfo))
        modulesWithRenamedPorts[module.getNameAttr()] = module;
    }
  }

  // Rename individual operations.
  for (auto &op : *root.getBody()) {
    if (auto module = dyn_cast<HWModuleOp>(op)) {
      runOnModule(module, modulesWithRenamedPorts, renamedParameterInfo);
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

/// Return the specified ParameterAttr with a different name.
static ParameterAttr getParameterWithName(ParameterAttr param,
                                          StringAttr name) {
  return ParameterAttr::get(name, param.type(), param.value(),
                            param.getContext());
}

/// Check to see if the port names of the specified module conflict with
/// keywords or themselves.  If so, rename them and return true, otherwise
/// return false.
bool HWLegalizeNamesPass::legalizePortNames(
    hw::HWModuleOp module, RenamedParameterMapTy &renamedParameterInfo) {
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
    auto paramAttr = param.cast<ParameterAttr>();
    auto newName = nameResolver.getLegalName(paramAttr.name());
    if (newName.empty())
      parameters.push_back(param);
    else {
      auto newNameAttr = StringAttr::get(paramAttr.getContext(), newName);
      parameters.push_back(getParameterWithName(paramAttr, newNameAttr));
      changedParameters = true;
      renamedParameterInfo[std::make_pair(module, paramAttr.name())] =
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
static Attribute
remapRenamedParameters(Attribute value, HWModuleOp module,
                       const RenamedParameterMapTy &renamedParameterInfo) {
  // Literals are always fine and never change.
  if (value.isa<IntegerAttr>() || value.isa<FloatAttr>() ||
      value.isa<StringAttr>() || value.isa<VerbatimParameterValueAttr>())
    return value;

  // TODO: Handle nested expressions when we support them.

  // Otherwise this must be a parameter reference.
  auto parameterRef = value.dyn_cast<ParameterRefAttr>();
  assert(parameterRef && "Unknown kind of parameter expression");

  // If this parameter is un-renamed, then leave it alone.
  auto nameAttr = parameterRef.getName();
  auto it = renamedParameterInfo.find(std::make_pair(module, nameAttr));
  if (it == renamedParameterInfo.end())
    return value;

  // Okay, it was renamed, return the new name with the right type.
  return ParameterRefAttr::get(value.getContext(),
                               it->second.cast<StringAttr>(), value.getType());
}

// If this instance is referring to a module with renamed ports or
// parameter names, update them.
static void updateInstanceForChangedModule(InstanceOp inst, HWModuleOp module) {
  inst.argNamesAttr(module.argNames());
  inst.resultNamesAttr(module.resultNames());

  // If any module parameters changed names, take the new name.
  SmallVector<Attribute> newAttrs;
  auto instParameters = inst.parameters();
  auto modParameters = module.parameters();
  for (size_t i = 0, e = instParameters.size(); i != e; ++i) {
    auto instParam = instParameters[i].cast<ParameterAttr>();
    auto modParam = modParameters[i].cast<ParameterAttr>();
    if (instParam.name() == modParam.name())
      newAttrs.push_back(instParam);
    else
      newAttrs.push_back(getParameterWithName(instParam, modParam.name()));
  }
  inst.parametersAttr(ArrayAttr::get(inst.getContext(), newAttrs));
}

/// Rename any parameter values being specified for an instance if they are
/// referring to parameters that got renamed.
static void
updateInstanceParameterRefs(InstanceOp instance,
                            RenamedParameterMapTy &renamedParameterInfo) {
  auto parameters = instance.parameters();
  if (parameters.empty())
    return;

  auto curModule = instance->getParentOfType<HWModuleOp>();

  SmallVector<Attribute> newParams;
  newParams.reserve(parameters.size());
  bool anyRenamed = false;
  for (Attribute param : parameters) {
    auto paramAttr = param.cast<ParameterAttr>();
    auto newValue = remapRenamedParameters(paramAttr.value(), curModule,
                                           renamedParameterInfo);
    if (newValue == paramAttr.value()) {
      newParams.push_back(param);
      continue;
    }
    anyRenamed = true;
    newParams.push_back(
        getParameterWithValue(paramAttr.name().getValue(), newValue));
  }

  instance.parametersAttr(ArrayAttr::get(instance.getContext(), newParams));
}

static void
rewriteModuleBody(Block &block, NameCollisionResolver &nameResolver,
                  DenseMap<Attribute, HWModuleOp> &modulesWithRenamedPorts,
                  RenamedParameterMapTy &renamedParameterInfo,
                  bool moduleHasRenamedInterface) {

  // Rename the instances, regs, and wires.
  for (auto &op : block) {
    if (auto instanceOp = dyn_cast<InstanceOp>(op)) {
      auto newName = nameResolver.getLegalName(instanceOp.getName());
      if (!newName.empty())
        instanceOp.setName(StringAttr::get(instanceOp.getContext(), newName));

      // If this instance is referring to a module with renamed ports or
      // parameter names, update them.
      auto it =
          modulesWithRenamedPorts.find(instanceOp.moduleNameAttr().getAttr());
      if (it != modulesWithRenamedPorts.end())
        updateInstanceForChangedModule(instanceOp, it->second);

      if (moduleHasRenamedInterface)
        updateInstanceParameterRefs(instanceOp, renamedParameterInfo);
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
        localParam.valueAttr(remapRenamedParameters(
            localParam.value(), curModule, renamedParameterInfo));
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
                            modulesWithRenamedPorts, renamedParameterInfo,
                            moduleHasRenamedInterface);
      }
    }
  }
}

void HWLegalizeNamesPass::runOnModule(
    hw::HWModuleOp module,
    DenseMap<Attribute, HWModuleOp> &modulesWithRenamedPorts,
    RenamedParameterMapTy &renamedParameterInfo) {

  // If this module had something about its interface, then a parameter may
  // have been changed.  In that case, we change parameter references to match.
  // This isn't common, so we use this to avoid work.
  bool moduleHasRenamedInterface =
      modulesWithRenamedPorts.count(module.getNameAttr());

  // All the ports are pre-legalized, just add their names to the map so we
  // detect conflicts with them.
  NameCollisionResolver nameResolver;
  for (const PortInfo &port : getAllModulePortInfos(module))
    (void)nameResolver.getLegalName(port.name);

  rewriteModuleBody(*module.getBodyBlock(), nameResolver,
                    modulesWithRenamedPorts, renamedParameterInfo,
                    moduleHasRenamedInterface);
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
