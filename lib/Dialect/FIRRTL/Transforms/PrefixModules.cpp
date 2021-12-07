//===- PrefixModules.cpp - Prefix module names pass -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the PrefixModules pass.
//
//===----------------------------------------------------------------------===//

#include "../AnnotationDetails.h"
#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/InstanceGraph.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"

using namespace circt;
using namespace firrtl;

/// This maps a FModuleOp to a list of all prefixes that need to be applied.
/// When a module has multiple prefixes, it will be cloned for each one. Usually
/// there is only a single prefix applied to each module, although there could
/// be many.
using PrefixMap = DenseMap<StringRef, llvm::SmallVector<std::string, 1>>;

/// Insert a string into the end of vector if the string is not already present.
static void recordPrefix(PrefixMap &prefixMap, StringRef moduleName,
                         std::string prefix) {
  auto &modulePrefixes = prefixMap[moduleName];
  if (llvm::find(modulePrefixes, prefix) == modulePrefixes.end())
    modulePrefixes.push_back(prefix);
}

namespace {
/// This is the prefix which will be applied to a module.
struct PrefixInfo {

  /// The string to prefix on to the module and all of its children.
  StringRef prefix;

  /// If true, this prefix applies to the module itself.  If false, the prefix
  /// only applies to the module's children.
  bool inclusive;
};
} // end anonymous namespace

/// Get the PrefixInfo for a module from a NestedPrefixModulesAnnotation on a
/// module. If the module is not annotated, the prefix returned will be empty.
static PrefixInfo getPrefixInfo(Operation *module) {
  AnnotationSet annotations(module);

  // Get the annotation from the module.
  auto dict = annotations.getAnnotation(
      "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation");
  if (!dict)
    return {"", false};
  Annotation anno(dict);

  // Get the prefix from the annotation.
  StringRef prefix = "";
  if (auto prefixAttr = anno.getMember<StringAttr>("prefix"))
    prefix = prefixAttr.getValue();

  // Get the inclusive flag from the annotation.
  bool inclusive = false;
  if (auto inclusiveAttr = anno.getMember<BoolAttr>("inclusive"))
    inclusive = inclusiveAttr.getValue();

  return {prefix, inclusive};
}

/// If there is an inclusive prefix attached to the module, return it.
static StringRef getPrefix(Operation *module) {
  auto prefixInfo = getPrefixInfo(module);
  if (prefixInfo.inclusive)
    return prefixInfo.prefix;
  return "";
}

/// This pass finds modules annotated with NestedPrefixAnnotation and prefixes
/// module names using the string stored in the annotation.  This pass prefixes
/// every module instantiated under the annotated root module's hierarchy. If a
/// module is instantiated under two different prefix hierarchies, it will be
/// duplicated and each module will have one prefix applied.
namespace {
class PrefixModulesPass : public PrefixModulesBase<PrefixModulesPass> {
  void renameModuleBody(std::string prefix, FModuleOp module);
  void renameModule(FModuleOp module);
  void renameExtModule(FExtModuleOp extModule);
  void runOnOperation() override;

  /// Mutate Grand Central Interface definitions (an Annotation on the circuit)
  /// with a field "prefix" containing the prefix for that annotation.  This
  /// relies on information built up during renameModule and stored in
  /// interfacePrefixMap.
  void prefixGrandCentralInterfaces();

  /// This is a map from a module name to new prefixes to be applied.
  PrefixMap prefixMap;

  /// A map of Grand Central interface ID to prefix.
  DenseMap<Attribute, std::string> interfacePrefixMap;

  /// Cached instance graph analysis.
  InstanceGraph *instanceGraph = nullptr;

  /// Map of symbol name to GlobalRef op.
  llvm::StringMap<Operation *> glblRefMap;

  /// Boolean keeping track of any name changes.
  bool anythingChanged = false;
};
} // namespace

/// Applies the prefix to the module.  This will update the required prefixes of
/// any referenced module in the prefix map.
void PrefixModulesPass::renameModuleBody(std::string prefix, FModuleOp module) {
  auto *context = module.getContext();

  // If we are renaming the body of this module, we need to mark that we have
  // changed the IR. If we are prefixing with the empty string, then nothing has
  // changed yet.
  if (!prefix.empty())
    anythingChanged = true;

  module.body().walk([&](Operation *op) {
    if (auto memOp = dyn_cast<MemOp>(op)) {
      // Memories will be turned into modules and should be prefixed.
      memOp.nameAttr(StringAttr::get(context, prefix + memOp.name()));
    } else if (auto instanceOp = dyn_cast<InstanceOp>(op)) {
      auto target =
          dyn_cast<FModuleLike>(instanceGraph->getReferencedModule(instanceOp));

      // Skip all external modules, unless one of the following conditions
      // is true:
      //   - This is a Grand Central Data Tap
      //   - This is a Grand Central Mem Tap
      if (auto *extModule = dyn_cast_or_null<FExtModuleOp>(&target)) {
        auto isDataTap = AnnotationSet(*extModule).hasAnnotation(dataTapsClass);
        auto isMemTap =
            AnnotationSet::forPort(*extModule, 0).hasAnnotation(memTapClass);
        if (!isDataTap && !isMemTap)
          return;
      }

      // Record that we must prefix the target module with the current prefix.
      recordPrefix(prefixMap, target.moduleName(), prefix);

      // Fixup this instance op to use the prefixed module name.  Note that the
      // referenced FModuleOp will be renamed later.
      auto newTarget = (prefix + getPrefix(target) + target.moduleName()).str();
      AnnotationSet instAnnos(instanceOp);
      // If the instance has GlobalRef, then update its module name also.
      // There can be multiple GlobalRefs attached to the instance op.

      for (Annotation anno : instAnnos) {
        if (anno.isClass("circt.nonlocal"))
          if (auto glblRef = anno.getMember("circt.nonlocal")) {
            auto glblRefName = glblRef.cast<FlatSymbolRefAttr>().getValue();
            auto f = glblRefMap.find(glblRefName);
            if (f == glblRefMap.end())
              instanceOp.emitError("cannot find GlobalRef :" + glblRefName);
            else {
              auto glblRefOp = dyn_cast<hw::GlobalRef>(f->second);
              // Iterate over the modules of the GlobalRef op, and update
              // it.
              SmallVector<Attribute, 4> newMods;
              for (auto oldMod :
                   glblRefOp.namepath().getAsRange<hw::InnerRefAttr>()) {
                llvm::errs() << "\n instance mod:" << instanceOp.moduleName()
                             << "  this mod::" << oldMod.getModule();
                if (instanceOp.moduleName().equals(
                        oldMod.getModule().getValue()))
                  newMods.push_back(hw::InnerRefAttr::get(
                      StringAttr::get(context, newTarget), oldMod.getName()));
                else
                  newMods.push_back(oldMod);
              }
              glblRefOp->setAttr("namepath", ArrayAttr::get(context, newMods));
            }
          }
      }

      instanceOp.moduleNameAttr(FlatSymbolRefAttr::get(context, newTarget));
    }
  });
}

/// Apply all required renames to the current module.  This will update the
/// prefix map for any referenced module.
void PrefixModulesPass::renameModule(FModuleOp module) {
  // If the module is annotated to have a prefix, it will be applied after the
  // parent's prefix.
  auto prefixInfo = getPrefixInfo(module);
  auto innerPrefix = prefixInfo.prefix;

  // Remove the annotation from the module.
  AnnotationSet::removeAnnotations(
      module, "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation");

  // We only add the annotated prefix to the module name if it is inclusive.
  auto moduleName = module.getName().str();
  if (prefixInfo.inclusive)
    moduleName = (innerPrefix + moduleName).str();

  auto &prefixes = prefixMap[module.getName()];

  // If there are no required prefixes of this module, then this module is a
  // top-level module, and there is an implicit requirement that it has an empty
  // prefix. This empty prefix will be applied to to all modules instantiated by
  // this module.
  if (prefixes.empty())
    prefixes.push_back("");

  // Rename the module for each required prefix. This will clone the module
  // once for each prefix but the first.
  OpBuilder builder(module);
  builder.setInsertionPointAfter(module);
  for (auto &outerPrefix : drop_begin(prefixes)) {
    auto moduleClone = cast<FModuleOp>(builder.clone(*module));
    moduleClone.setName(outerPrefix + moduleName);
    renameModuleBody((outerPrefix + innerPrefix).str(), moduleClone);
  }

  // The first prefix renames the module in place. There is always at least 1
  // prefix.
  auto &outerPrefix = prefixes.front();
  module.setName(outerPrefix + moduleName);
  renameModuleBody((outerPrefix + innerPrefix).str(), module);

  // If this module contains a Grand Central interface, then also apply renames
  // to that, but only if there are prefixes to apply.
  if (prefixes.empty())
    return;
  AnnotationSet annotations(module);
  if (!annotations.hasAnnotation(
          "sifive.enterprise.grandcentral.ViewAnnotation"))
    return;
  auto prefixFull = (outerPrefix + innerPrefix).str();
  SmallVector<Attribute> newAnnotations;
  for (auto anno : annotations) {
    if (!anno.isClass("sifive.enterprise.grandcentral.ViewAnnotation")) {
      newAnnotations.push_back(anno.getDict());
      continue;
    }

    NamedAttrList newAnno;
    for (auto attr : anno.getDict()) {
      if (attr.getName() == "name") {
        newAnno.append(attr.getName(),
                       (Attribute)builder.getStringAttr(
                           Twine(prefixFull) +
                           attr.getValue().cast<StringAttr>().getValue()));
        continue;
      }
      newAnno.append(attr.getName(), attr.getValue());
    }
    newAnnotations.push_back(
        DictionaryAttr::getWithSorted(builder.getContext(), newAnno));

    // Record that we need to apply this prefix to the interface definition.
    if (anno.getMember<StringAttr>("type").getValue() == "parent")
      interfacePrefixMap[anno.getMember<IntegerAttr>("id")] = prefixFull;
  }
  AnnotationSet(newAnnotations, builder.getContext()).applyToOperation(module);
}

/// Apply prefixes from the `prefixMap` to an external module.  No modifications
/// are made if there are no prefixes for this external module.  If one prefix
/// exists, then the external module will be updated in place.  If multiple
/// prefixes exist, then the original external module will be updated in place
/// and prefixes _after_ the first will cause the module to be cloned
/// ("duplicated" in Scala FIRRTL Compiler terminology).  The logic of this
/// member function is the same as `renameModule` except that there is no module
/// body to recursively update.
void PrefixModulesPass::renameExtModule(FExtModuleOp extModule) {
  // Lookup prefixes for this module.  If none exist, bail out.
  auto &prefixes = prefixMap[extModule.getName()];
  if (prefixes.empty())
    return;

  OpBuilder builder(extModule);
  builder.setInsertionPointAfter(extModule);

  // Function to apply an outer prefix to an external module.  If the module has
  // an optional "defname" (a name that will be used to generate Verilog), also
  // update the defname.
  auto applyPrefixToNameAndDefName = [&](FExtModuleOp &extModule,
                                         StringRef prefix) {
    extModule.setName((prefix + extModule.getName()).str());
    if (auto defname = extModule.defname())
      extModule->setAttr("defname",
                         builder.getStringAttr(prefix + defname.getValue()));
  };

  // Duplicate the external module if there is more than one prefix.
  for (auto &prefix : drop_begin(prefixes)) {
    auto duplicate = cast<FExtModuleOp>(builder.clone(*extModule));
    applyPrefixToNameAndDefName(duplicate, prefix);
  }

  // Update the original module with a new prefix.
  applyPrefixToNameAndDefName(extModule, prefixes.front());
}

void PrefixModulesPass::runOnOperation() {
  auto *context = &getContext();
  instanceGraph = &getAnalysis<InstanceGraph>();
  auto circuitOp = getOperation();

  // Record all the glblRef ops in the circt.
  for (auto glblRef : circuitOp.body().getOps<hw::GlobalRef>())
    glblRefMap[glblRef.sym_name()] = glblRef;

  // If the main module is prefixed, we have to update the CircuitOp.
  auto mainModule = instanceGraph->getTopLevelModule();
  auto prefix = getPrefix(mainModule);
  if (!prefix.empty()) {
    auto newMainModuleName = ((prefix + circuitOp.name()).str());
    circuitOp.nameAttr(StringAttr::get(context, newMainModuleName));
    // Now update all the glblRefs that have the top level module symbol.
    for (auto &n : glblRefMap) {
      auto glblRef = cast<hw::GlobalRef>(n.second);
      auto oldMods = glblRef.namepath();
      if (oldMods.empty())
        continue;
      SmallVector<Attribute, 4> newMods(oldMods.begin(), oldMods.end());
      auto iref = oldMods[0].cast<hw::InnerRefAttr>();
      if (iref.getModule().getValue().equals(mainModule.moduleName()))
        newMods[0] = hw::InnerRefAttr::get(
            StringAttr::get(context, newMainModuleName), iref.getName());
      glblRef->setAttr("namepath", ArrayAttr::get(context, newMods));
    }
  }

  // Walk all Modules in a top-down order.  For each module, look at the list of
  // required prefixes to be applied.
  DenseSet<InstanceGraphNode *> visited;
  for (auto *current : *instanceGraph) {
    for (auto &node : llvm::inverse_post_order_ext(current, visited)) {
      if (auto module = dyn_cast<FModuleOp>(node->getModule()))
        renameModule(module);
      if (auto extModule = dyn_cast<FExtModuleOp>(node->getModule()))
        renameExtModule(extModule);
    }
  }

  // Update any interface definitions if needed.
  prefixGrandCentralInterfaces();

  prefixMap.clear();
  interfacePrefixMap.clear();
  if (!anythingChanged)
    markAllAnalysesPreserved();
}

/// Mutate circuit-level annotations to add prefix information to Grand Central
/// (SystemVerilog) interfaces.  Add a "prefix" field to each interface
/// definition (an annotation with class "AugmentedBundleType") that holds the
/// prefix that was determined during runOnModule.  It is assumed that this
/// field did not exist before.
void PrefixModulesPass::prefixGrandCentralInterfaces() {
  // Early exit if no interfaces need prefixes.
  if (interfacePrefixMap.empty())
    return;

  auto circuit = getOperation();
  OpBuilder builder(circuit);

  SmallVector<Attribute> newCircuitAnnotations;
  for (auto anno : AnnotationSet(circuit)) {
    // Only mutate this annotation if it is an AugmentedBundleType and
    // interfacePrefixMap has prefix information for it.
    StringRef prefix;
    if (anno.isClass("sifive.enterprise.grandcentral.AugmentedBundleType")) {
      if (auto id = anno.getMember<IntegerAttr>("id"))
        prefix = interfacePrefixMap[id];
    }

    // Nothing to do.  Copy the annotation.
    if (prefix.empty()) {
      newCircuitAnnotations.push_back(anno.getDict());
      continue;
    }

    // Add a "prefix" field with the prefix for this interface.  This is safe to
    // put at the back and do a `getWithSorted` because the last field is
    // conveniently called "name".
    NamedAttrList newAnno(anno.getDict().getValue());
    newAnno.append("prefix", builder.getStringAttr(prefix));
    newCircuitAnnotations.push_back(
        DictionaryAttr::getWithSorted(builder.getContext(), newAnno));
  }

  // Overwrite the old circuit annotation with the new one created here.
  AnnotationSet(newCircuitAnnotations, builder.getContext())
      .applyToOperation(circuit);
}

std::unique_ptr<mlir::Pass> circt::firrtl::createPrefixModulesPass() {
  return std::make_unique<PrefixModulesPass>();
}
