//===- OMLinkModules.cpp - OM Linker pass -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the OM Linker pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/OM/OMOps.h"
#include "circt/Dialect/OM/OMPasses.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Threading.h"

#include <memory>

using namespace circt;
using namespace om;
using namespace hw;
using namespace mlir;
using namespace llvm;

namespace {
// A map from a pair of enclosing module op and old symbol to a new symbol.
using SymMappingTy =
    llvm::DenseMap<std::pair<ModuleOp, StringAttr>, StringAttr>;

struct ModuleInfo {
  ModuleInfo(mlir::ModuleOp module) : module(module) {}

  // Populate `symbolToClasses`.
  LogicalResult initialize();

  // Update symbols based on the mapping and erase external classes.
  void postProcess(const SymMappingTy &symMapping);

  // A map from symbols to classes.
  llvm::DenseMap<StringAttr, ClassLike> symbolToClasses;
  // A map from symbols to HWModules.
  llvm::DenseMap<StringAttr, HWModuleLike> symbolToHWModules;
  // A map from symbols to all symbol ops.
  llvm::DenseMap<StringAttr, Operation *> symbolToOps;

  // The symbol attribute name.
  const StringRef symAttrName = "sym_name";

  // A target module.
  ModuleOp module;
};

struct LinkModulesPass : public LinkModulesBase<LinkModulesPass> {
  void runOnOperation() override;
};

} // namespace

LogicalResult ModuleInfo::initialize() {
  for (auto &op : llvm::make_early_inc_range(module.getOps())) {
    // If the op delares a symbol.
    if (op.hasAttr(symAttrName)) {
      auto sym = op.getAttrOfType<StringAttr>(symAttrName);
      symbolToOps.insert({sym, &op});
      if (auto classLike = dyn_cast<ClassLike>(op))
        symbolToClasses.insert({sym, classLike});
      if (auto hwMod = dyn_cast<HWModuleLike>(op))
        symbolToHWModules.insert({sym, hwMod});
    }
  }
  return success();
}

void ModuleInfo::postProcess(const SymMappingTy &symMapping) {
  AttrTypeReplacer replacer;
  replacer.addReplacement([&](FlatSymbolRefAttr oldSym)
                              -> std::pair<FlatSymbolRefAttr, WalkResult> {
    // Update all renamed symbol references.
    auto it = symMapping.find({module, oldSym.getAttr()});
    if (it == symMapping.end())
      return {oldSym, WalkResult::skip()};
    return {FlatSymbolRefAttr::get(it->getSecond()), WalkResult::skip()};
  });
  replacer.addReplacement(
      // Update class types when their symbols were renamed.
      [&](om::ClassType classType) -> std::pair<mlir::Type, WalkResult> {
        auto it = symMapping.find({module, classType.getClassName().getAttr()});
        // No change.
        if (it == symMapping.end())
          return {classType, WalkResult::skip()};
        return {om::ClassType::get(classType.getContext(),
                                   FlatSymbolRefAttr::get(it->second)),
                WalkResult::skip()};
      });

  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    // Extern classes must be deleted.
    if (isa<ClassExternOp>(op)) {
      op->erase();
      // ClassExternFieldOp will be deleted as well.
      return WalkResult::skip();
    }
    // External modules must be erased.
    if (isa<HWModuleExternOp>(op)) {
      op->erase();
      return WalkResult::skip();
    }

    if (op->hasAttr(symAttrName)) {
      // If the symbol for this op is renamed, update it.
      auto it =
          symMapping.find({module, op->getAttrOfType<StringAttr>(symAttrName)});
      if (it != symMapping.end())
        op->setAttr(symAttrName, it->second);
    }
    if (auto objectOp = dyn_cast<ObjectOp>(op)) {
      // Update its class name if changed..
      auto it = symMapping.find({module, objectOp.getClassNameAttr()});
      if (it != symMapping.end())
        objectOp.setClassNameAttr(it->second);
    }

    replacer.replaceElementsIn(op,
                               /*replaceAttrs=*/true,
                               /*replaceLocs=*/false,
                               /*replaceTypes=*/true);
    return WalkResult::advance();
  });
}

// Return a failure if modules cannot be resolved. Return true if
// it's necessary to rename symbols.
static FailureOr<bool> resolveHWModules(StringAttr name,
                                        ArrayRef<HWModuleLike> modOps) {
  bool existsExternalModule = false;
  size_t countDefinition = 0;
  HWModuleOp hwModule;

  bool isPrivate = false;
  SmallVector<Location> publicModules;
  SmallVector<HWModuleLike> hwModOps;
  for (auto mod : modOps) {

    if (isa<HWModuleExternOp>(mod))
      existsExternalModule = true;
    else if (!countDefinition) {
      hwModule = cast<HWModuleOp>(mod);
      isPrivate = hwModule.isPrivate();
      ++countDefinition;
    } else if (hwModule.isPrivate() && isPrivate) {
      ++countDefinition;
    } else {
      publicModules.push_back(mod->getLoc());
    }
  }

  if (!publicModules.empty()) {
    auto diag =
        emitError(hwModule.getLoc())
        << "module " << name << " is declared as a public module but "
        << "there are multiple public modules defined with the same name";
    for (auto loc : publicModules)

      diag.attachNote(loc) << "module " << name
                           << " is declared here as public";

    return failure();
  }
  // There must be exactly one definition if the symbol was referred by an
  // external module.
  if (existsExternalModule && countDefinition != 1) {
    SmallVector<Location> modExternLocs;
    SmallVector<Location> modLocs;
    for (auto op : modOps)

      (isa<HWModuleExternOp>(op) ? modExternLocs : modLocs)
          .push_back(op.getLoc());

    auto diag = emitError(modExternLocs.front())
                << "module " << name
                << " is declared as an external module but "
                << (countDefinition == 0 ? "there is no definition"
                                         : "there are multiple definitions");
    for (auto loc : ArrayRef(modExternLocs).drop_front())
      diag.attachNote(loc) << "module " << name << " is declared here as well";

    if (countDefinition != 0) {
      // There are multiple definitions.
      for (auto loc : modLocs)
        diag.attachNote(loc) << "module " << name << " is defined here";
    }
    return failure();
  }

  if (!existsExternalModule)
    return countDefinition != 1;

  assert(hwModule && countDefinition == 1);

  // Raise errors if linked external modules are not compatible with the
  // definition.
  auto emitError = [&](Operation *op) {
    auto diag = op->emitError()
                << "failed to link module " << name
                << " since declaration doesn't match the definition: ";
    diag.attachNote(hwModule.getLoc()) << "definition is here";
    return diag;
  };

  SmallVector<PortInfo> ports = hwModule.getPortList();

  for (auto mod : modOps) {
    if (mod == hwModule)
      continue;
    auto modPorts = mod.getPortList();
    if (modPorts == ports)
      continue;
    return emitError(mod);
  }
  return false;
}

// Return a failure if classes cannot be resolved. Return true if
// it's necessary to rename symbols.
static FailureOr<bool> resolveClasses(StringAttr name,
                                      ArrayRef<ClassLike> classes) {
  bool existExternalClass = false;
  size_t countDefinition = 0;
  ClassOp classOp;

  for (auto op : classes) {
    if (isa<ClassExternOp>(op))
      existExternalClass = true;
    else {
      classOp = cast<ClassOp>(op);
      ++countDefinition;
    }
  }

  // There must be exactly one definition if the symbol was referred by an
  // external class.
  if (existExternalClass && countDefinition != 1) {
    SmallVector<Location> classExternLocs;
    SmallVector<Location> classLocs;
    for (auto op : classes)
      (isa<ClassExternOp>(op) ? classExternLocs : classLocs)
          .push_back(op.getLoc());

    auto diag = emitError(classExternLocs.front())
                << "class " << name << " is declared as an external class but "
                << (countDefinition == 0 ? "there is no definition"
                                         : "there are multiple definitions");
    for (auto loc : ArrayRef(classExternLocs).drop_front())
      diag.attachNote(loc) << "class " << name << " is declared here as well";

    if (countDefinition != 0) {
      // There are multiple definitions.
      for (auto loc : classLocs)
        diag.attachNote(loc) << "class " << name << " is defined here";
    }
    return failure();
  }

  if (!existExternalClass)
    return countDefinition != 1;

  assert(classOp && countDefinition == 1);

  // Raise errors if linked external modules are not compatible with the
  // definition.
  auto emitError = [&](Operation *op) {
    auto diag = op->emitError()
                << "failed to link class " << name
                << " since declaration doesn't match the definition: ";
    diag.attachNote(classOp.getLoc()) << "definition is here";
    return diag;
  };

  llvm::MapVector<StringAttr, Type> classFields;
  for (auto fieldOp : classOp.getOps<om::ClassFieldOp>())
    classFields.insert({fieldOp.getNameAttr(), fieldOp.getType()});

  for (auto op : classes) {
    if (op == classOp)
      continue;

    if (classOp.getBodyBlock()->getNumArguments() !=
        op.getBodyBlock()->getNumArguments())
      return emitError(op) << "the number of arguments is not equal, "
                           << classOp.getBodyBlock()->getNumArguments()
                           << " vs " << op.getBodyBlock()->getNumArguments();
    unsigned index = 0;
    for (auto [l, r] : llvm::zip(classOp.getBodyBlock()->getArgumentTypes(),
                                 op.getBodyBlock()->getArgumentTypes())) {
      if (l != r)
        return emitError(op) << index << "-th argument type is not equal, " << l
                             << " vs " << r;
      index++;
    }
    // Check declared fields.
    llvm::DenseSet<StringAttr> declaredFields;
    for (auto fieldOp : op.getBodyBlock()->getOps<om::ClassExternFieldOp>()) {
      auto it = classFields.find(fieldOp.getNameAttr());

      // Field not found in its definition.
      if (it == classFields.end())
        return emitError(op)
               << "declaration has a field " << fieldOp.getNameAttr()
               << " but not found in its definition";

      if (it->second != fieldOp.getType())
        return emitError(op)
               << "declaration has a field " << fieldOp.getNameAttr()
               << " but types don't match, " << it->second << " vs "
               << fieldOp.getType();
      declaredFields.insert(fieldOp.getNameAttr());
    }

    for (auto [fieldName, _] : classFields)
      if (!declaredFields.count(fieldName))
        return emitError(op) << "definition has a field " << fieldName
                             << " but not found in this declaration";
  }
  return false;
}

void LinkModulesPass::runOnOperation() {
  auto toplevelModule = getOperation();
  // 1. Initialize ModuleInfo.
  SmallVector<ModuleInfo> modules;
  size_t counter = 0;
  for (auto module : toplevelModule.getOps<ModuleOp>()) {
    auto name = module->getAttrOfType<StringAttr>("om.namespace");
    // Use `counter` if the namespace is not specified beforehand.
    if (!name) {
      name = StringAttr::get(module.getContext(), "module_" + Twine(counter++));
      module->setAttr("om.namespace", name);
    }
    modules.emplace_back(module);
  }

  if (failed(failableParallelForEach(&getContext(), modules, [](auto &info) {
        // Collect local information.
        return info.initialize();
      })))
    return signalPassFailure();

  // 2. Symbol resolution. Check that there is exactly single definition for
  //    public symbols and rename private symbols if necessary.

  // Global namespace to get unique names to symbols.
  Namespace nameSpace;
  // A map from a pair of enclosing module op and old symbol to a new symbol.
  SymMappingTy symMapping;

  // Construct a global map from symbols to class operations.
  llvm::MapVector<StringAttr, SmallVector<ClassLike>> symbolToClasses;
  llvm::MapVector<StringAttr, SmallVector<HWModuleLike>> symbolToHWModules;
  llvm::MapVector<StringAttr, SmallVector<Operation *>> symbolToOps;
  for (const auto &info : modules) {

    for (auto &[symName, symOp] : info.symbolToOps) {
      // Add names to avoid collision.
      if (!symbolToOps.contains(symName))
        (void)nameSpace.newName(symName.getValue());
      if (auto classOp = dyn_cast<ClassLike>(symOp))
        symbolToClasses[symName].push_back(classOp);
      else if (auto modOp = dyn_cast<HWModuleLike>(symOp))
        symbolToHWModules[symName].push_back(modOp);
      symbolToOps[symName].push_back(symOp);
    }
  }

  for (auto &[name, hwModules] : symbolToHWModules) {
    // Check if it's legal to link modules. `resolveHWModules` returns true if
    // it's necessary to rename symbols.
    auto result = resolveHWModules(name, hwModules);
    if (failed(result))
      return signalPassFailure();

    // We can resolve symbol collision for private modules.
    if (*result)
      for (auto op : hwModules) {
        auto enclosingModule = cast<mlir::ModuleOp>(op->getParentOp());
        symMapping[{enclosingModule, name}] =
            StringAttr::get(&getContext(), nameSpace.newName(name.getValue()));
      }
  }
  // Resolve symbols. We consider a symbol used as an external module to be
  // "public" thus we cannot rename such symbols when there is collision. We
  // require a public symbol to have exactly one definition so otherwise raise
  // an error.
  for (auto &[name, classes] : symbolToClasses) {
    // Check if it's legal to link classes. `resolveClasses` returns true if
    // it's necessary to rename symbols.
    auto result = resolveClasses(name, classes);
    if (failed(result))
      return signalPassFailure();

    // We can resolve symbol collision for symbols not referred by external
    // classes. Create a new name using `om.namespace` attributes as a
    // suffix.
    if (*result || (symbolToHWModules.contains(name)))
      for (auto op : classes) {
        auto enclosingModule = cast<mlir::ModuleOp>(op->getParentOp());
        auto nameSpaceId =
            enclosingModule->getAttrOfType<StringAttr>("om.namespace");
        symMapping[{enclosingModule, name}] = StringAttr::get(
            &getContext(),
            nameSpace.newName(name.getValue(), nameSpaceId.getValue()));
      }
  }

  for (auto &[name, symOps] : symbolToOps) {

    if (symOps.size() > 1)
      for (auto *op : symOps) {
        if (isa<HWModuleLike, ClassLike>(op))
          continue;
        auto enclosingModule = cast<mlir::ModuleOp>(op->getParentOp());
        symMapping[{enclosingModule, name}] =
            StringAttr::get(&getContext(), nameSpace.newName(name.getValue()));
      }
  }

  // 3. Post-processing. Update class names and erase external classes.

  // Rename private symbols and remove external classes.
  parallelForEach(&getContext(), modules,
                  [&](auto &info) { info.postProcess(symMapping); });

  // Finally move operations to the toplevel module.
  auto *block = toplevelModule.getBody();
  for (auto info : modules) {
    block->getOperations().splice(block->end(),
                                  info.module.getBody()->getOperations());
    // Erase the module.
    info.module.erase();
  }
}

std::unique_ptr<mlir::Pass> circt::om::createOMLinkModulesPass() {
  return std::make_unique<LinkModulesPass>();
}
