//===- HWOpInterfaces.td - Operation Interfaces ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the HW operation interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWOPINTERFACES
#define CIRCT_DIALECT_HW_HWOPINTERFACES

include "mlir/IR/OpBase.td"

def HWModuleLike : OpInterface<"HWModuleLike"> {
  let cppNamespace = "circt::hw";
  let description = "Provide common  module information.";

  let methods = [
    InterfaceMethod<"Check whether the module is publicly visible",
    "bool", "isPublic", (ins),
    /*methodBody=*/[{
      return ::mlir::SymbolTable::getSymbolVisibility($_op) ==
             ::mlir::SymbolTable::Visibility::Public;
    }]>,

    InterfaceMethod<"Get the module name",
    "::llvm::StringRef", "moduleName", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{ return $_op.getName(); }]>,

    InterfaceMethod<"Get the module name",
    "::mlir::StringAttr", "moduleNameAttr", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{ return $_op.getNameAttr(); }]>,

    InterfaceMethod<"Get the InstanceLike with symbol name",
    "circt::hw::HWInstanceLike", "getInstance", (ins "mlir::StringAttr": $name),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{
    for (auto i : $_op.template getOps<HWInstanceLike>())
      if (i.instanceName() == name)
        return i;
    return {};
    }]>,
  ];

  let verify = [{
    static_assert(
        ConcreteOp::template hasTrait<::mlir::SymbolOpInterface::Trait>(),
        "expected operation to be a symbol");
    return success();
  }];
}


def HWInstanceLike : OpInterface<"HWInstanceLike"> {
  let cppNamespace = "circt::hw";
  let description = "Provide common  module information.";

  let methods = [
    InterfaceMethod<"Get the name of the instance",
    "::llvm::StringRef", "instanceName", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{ return $_op.instanceName(); }]>,

    InterfaceMethod<"Get the name of the instantiated module",
    "::llvm::StringRef", "referencedModuleName", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{ return $_op.moduleName(); }]>,

    InterfaceMethod<"Get the name of the instantiated module",
    "::mlir::StringAttr", "referencedModuleNameAttr", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{ return $_op.moduleNameAttr().getAttr(); }]>,
  ];
}

def HierarchicalPathLike : OpInterface<"HierarchicalPathLike"> {
  let cppNamespace = "circt::hw";
  let description = [{
    Represents a path through the hierarchy.
  }];
  let methods = [
    InterfaceMethod<
      "Drop the module from the namepath. If its a InnerNameRef, then drop the Module-Instance pair, else drop the final module from the namepath. Return true if any update is made.",
      "bool",
      "dropModule",
      (ins
        "::mlir::StringAttr":$moduleToDrop),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        SmallVector<Attribute, 4> newPath;
        bool updateMade = false;
        for (auto nameRef : $_op.namepath()) {
          // nameRef is either an InnerRefAttr or a FlatSymbolRefAttr.
          if (auto ref = nameRef.template dyn_cast<hw::InnerRefAttr>()) {
            if (ref.getModule() == moduleToDrop)
              updateMade = true;
            else
              newPath.push_back(ref);
          } else {
            if (nameRef.template cast<FlatSymbolRefAttr>().getAttr() == moduleToDrop)
              updateMade = true;
            else
              newPath.push_back(nameRef);
          }
        }
        if (updateMade)
          $_op.namepathAttr(ArrayAttr::get($_op.getContext(), newPath));
        return updateMade;
      }]
    >,

    InterfaceMethod<
      "Inline the module in the namepath. Update the symbol name for the inlined module instance, by prepending the symbol name of the instance at which the inlining was done. Return true if any update is made.",
      "bool",
      "inlineModule",
      (ins
        "::mlir::StringAttr":$moduleToDrop),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        SmallVector<Attribute, 4> newPath;
        bool updateMade = false;
        StringRef inlinedInstanceName = "";
        for (auto nameRef : $_op.namepath()) {
          // nameRef is either an InnerRefAttr or a FlatSymbolRefAttr.
          if (auto ref = nameRef.template dyn_cast<hw::InnerRefAttr>()) {
            if (ref.getModule() == moduleToDrop) {
              inlinedInstanceName = ref.getName().getValue();
              updateMade = true;
            } else if (!inlinedInstanceName.empty()) {
              newPath.push_back(hw::InnerRefAttr::get(
                  ref.getModule(),
                  StringAttr::get($_op.getContext(), inlinedInstanceName + "_" +
                                                    ref.getName().getValue())));
              inlinedInstanceName = "";
            } else
              newPath.push_back(ref);
          } else {
            if (nameRef.template cast<FlatSymbolRefAttr>().getAttr() == moduleToDrop)
              updateMade = true;
            else
              newPath.push_back(nameRef);
          }
        }
        if (updateMade)
          $_op.namepathAttr(ArrayAttr::get($_op.getContext(), newPath));
        return updateMade;
      }]
    >,

    InterfaceMethod<
      "Replace the oldMod module with newMod module in the namepath of the NLA. Return true if any update is made.",
      "bool",
      "updateModule",
      (ins
        "::mlir::StringAttr":$oldMod,
        "::mlir::StringAttr":$newMod),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        SmallVector<Attribute, 4> newPath;
        bool updateMade = false;
        for (auto nameRef : $_op.namepath()) {
          // nameRef is either an InnerRefAttr or a FlatSymbolRefAttr.
          if (auto ref = nameRef.template dyn_cast<hw::InnerRefAttr>()) {
            if (ref.getModule() == oldMod) {
              newPath.push_back(hw::InnerRefAttr::get(newMod, ref.getName()));
              updateMade = true;
            } else
              newPath.push_back(ref);
          } else {
            if (nameRef.template cast<FlatSymbolRefAttr>().getAttr() == oldMod) {
              newPath.push_back(FlatSymbolRefAttr::get(newMod));
              updateMade = true;
            } else
              newPath.push_back(nameRef);
          }
        }
        if (updateMade)
          $_op.namepathAttr(ArrayAttr::get($_op.getContext(), newPath));
        return updateMade;
      }]
    >,

    InterfaceMethod<
      "Replace the oldMod module with newMod module in the namepath of the NLA. Since the module is being updated, the symbols inside the module should also be renamed. Use the rename Map to update the corresponding inner_sym names in the namepath. Return true if any update is made.",
      "bool",
      "updateModuleAndInnerRef",
      (ins
        "::mlir::StringAttr":$oldMod,
        "::mlir::StringAttr":$newMod,
        "const llvm::DenseMap<mlir::StringAttr, mlir::StringAttr> &":$innerSymRenameMap),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        auto fromRef = FlatSymbolRefAttr::get(oldMod);
        if (oldMod == newMod)
          return false;
        auto namepathNew = $_op.namepath().getValue().vec();
        bool updateMade = false;
        // Break from the loop if the module is found, since it can occur only once.
        for (auto &element : namepathNew) {
          if (auto innerRef = element.template dyn_cast<hw::InnerRefAttr>()) {
            if (innerRef.getModule() != oldMod)
              continue;
            auto symName = innerRef.getName();
            // Since the module got updated, the old innerRef symbol inside oldMod
            // should also be updated to the new symbol inside the newMod.
            auto to = innerSymRenameMap.find(symName);
            if (to != innerSymRenameMap.end())
              symName = to->second;
            updateMade = true;
            element = hw::InnerRefAttr::get(newMod, symName);
            break;
          }
          if (element != fromRef)
            continue;
          updateMade = true;
          element = FlatSymbolRefAttr::get(newMod);
          break;
        }
        if (updateMade)
          $_op.namepathAttr(ArrayAttr::get($_op.getContext(), namepathNew));
        return updateMade;
      }]
    >,

    InterfaceMethod<
      "Truncate the namepath for this NLA, at atMod module. If includeMod is false, drop atMod and beyond, else include it and drop everything after it. Return true if any update is made.",
      "bool",
      "truncateAtModule",
      (ins
        "::mlir::StringAttr":$atMod,
        "bool":$includeMod),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        SmallVector<Attribute, 4> newPath;
        bool updateMade = false;
        for (auto nameRef : $_op.namepath()) {
          // nameRef is either an InnerRefAttr or a FlatSymbolRefAttr.
          if (auto ref = nameRef.template dyn_cast<hw::InnerRefAttr>()) {
            if (ref.getModule() == atMod) {
              updateMade = true;
              if (includeMod)
                newPath.push_back(ref);
            } else
              newPath.push_back(ref);
          } else {
            if (nameRef.template cast<FlatSymbolRefAttr>().getAttr() == atMod && !includeMod)
              updateMade = true;
            else
              newPath.push_back(nameRef);
          }
          if (updateMade)
            break;
        }
        if (updateMade)
          $_op.namepathAttr(ArrayAttr::get($_op.getContext(), newPath));
        return updateMade;
      }]
    >,

    InterfaceMethod<
      "Return just the module part of the namepath at a specific index.",
      "::mlir::StringAttr",
      "modPart",
      (ins
        "unsigned":$i),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return TypeSwitch<Attribute, StringAttr>($_op.namepath()[i])
            .template Case<FlatSymbolRefAttr>([](auto a) { return a.getAttr(); })
            .template Case<hw::InnerRefAttr>([](auto a) { return a.getModule(); });
      }]
    >,

    InterfaceMethod<
      "Return the root module.",
      "::mlir::StringAttr",
      "root",
      (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        assert(!$_op.namepath().empty());
        return modPart(0);
      }]
    >,
    InterfaceMethod<
      "Return just the reference part of the namepath at a specific index. This will return an empty attribute if this is the leaf and the leaf is a module.",
      "::mlir::StringAttr",
      "refPart",
      (ins
        "unsigned":$i),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return TypeSwitch<Attribute, StringAttr>($_op.namepath()[i])
            .template Case<FlatSymbolRefAttr>([](auto a) { return StringAttr({}); })
            .template Case<hw::InnerRefAttr>([](auto a) { return a.getName(); });
      }]
    >,

    InterfaceMethod<
      "Return the leaf reference.  This returns an empty attribute if the leaf reference is a module.",
      "::mlir::StringAttr",
      "ref",
      (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        assert(!$_op.namepath().empty());
        return refPart($_op.namepath().size() - 1);
      }]
    >,

    InterfaceMethod<
      "Return the leaf Module.",
      "::mlir::StringAttr",
      "leafMod",
      (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        assert(!$_op.namepath().empty());
        return modPart($_op.namepath().size() - 1);
      }]
    >,

    InterfaceMethod<
      "Returns true, if the NLA path contains the module.",
      "bool",
      "hasModule",
      (ins
        "::mlir::StringAttr":$modName),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        for (auto nameRef : $_op.namepath()) {
          // nameRef is either an InnerRefAttr or a FlatSymbolRefAttr.
          if (auto ref = nameRef.template dyn_cast<hw::InnerRefAttr>()) {
            if (ref.getModule() == modName)
              return true;
          } else {
            if (nameRef.template cast<FlatSymbolRefAttr>().getAttr() == modName)
              return true;
          }
        }
        return false;
      }]
    >,

    InterfaceMethod<
      "Returns true, if the NLA path contains the InnerSym {modName, symName}.",
      "bool",
      "hasInnerSym",
      (ins "::mlir::StringAttr":$modName, "::mlir::StringAttr":$symName),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        for (auto nameRef : $_op.namepath())
          if (auto ref = nameRef.template dyn_cast<hw::InnerRefAttr>())
            if (ref.getName() == symName && ref.getModule() == modName)
              return true;
        return false;
      }]
    >,

    InterfaceMethod<
      "Returns true if this NLA targets a module or instance of a module (as opposed to an instance's port or something inside an instance).",
      "bool",
      "isModule",
      (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return !$_op.ref();
      }]
    >,

    InterfaceMethod<
      "Returns true if this NLA targets something inside a module (as opposed to a module or an instance of a module)",
      "bool",
      "isComponent",
      (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return (bool)$_op.ref();
      }]
    >,

    ];
}


#endif
