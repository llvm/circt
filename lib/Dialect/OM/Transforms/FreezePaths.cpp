//===- FreezePaths.cpp - Freeze Paths pass ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the OM freeze paths pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/InnerSymbolTable.h"
#include "circt/Dialect/OM/OMAttributes.h"
#include "circt/Dialect/OM/OMOps.h"
#include "circt/Dialect/OM/OMPasses.h"

using namespace circt;
using namespace om;

namespace {
struct PathVisitor {
  PathVisitor(hw::InstanceGraph &instanceGraph, hw::InnerRefNamespace &irn)
      : instanceGraph(instanceGraph), irn(irn) {}
  LogicalResult processPath(PathOp path);
  LogicalResult run(Operation *op);
  hw::InstanceGraph &instanceGraph;
  hw::InnerRefNamespace &irn;
};
} // namespace

static LogicalResult getAccessPath(Location loc, Type type, size_t fieldId,
                                   SmallVectorImpl<char> &result) {
  while (fieldId) {
    if (auto aliasType = dyn_cast<hw::TypeAliasType>(type))
      type = aliasType.getCanonicalType();
    if (auto structType = dyn_cast<hw::StructType>(type)) {
      auto index = structType.getIndexForFieldID(fieldId);
      auto &element = structType.getElements()[index];
      result.push_back('.');
      llvm::append_range(result, element.name.getValue());
      type = element.type;
      fieldId -= structType.getFieldID(index);
    } else if (auto arrayType = dyn_cast<hw::ArrayType>(type)) {
      auto index = arrayType.getIndexForFieldID(fieldId);
      result.push_back('[');
      Twine(index).toVector(result);
      result.push_back(']');
      type = arrayType.getElementType();
      fieldId -= arrayType.getFieldID(index);
    } else {
      return emitError(loc) << "can't create access path with fieldID "
                            << fieldId << " in type " << type;
    }
  }
  return success();
}

LogicalResult PathVisitor::processPath(PathOp path) {
  auto *context = path->getContext();
  auto &symbolTable = irn.symTable;

  StringRef targetKind;
  switch (path.getTargetKind()) {
  case TargetKind::DontTouch:
    targetKind = "OMDontTouchedReferenceTarget";
    break;
  case TargetKind::Instance:
    targetKind = "OMInstanceTarget";
    break;
  case TargetKind::Reference:
    targetKind = "OMReferenceTarget";
    break;
  case TargetKind::MemberReference:
    targetKind = "OMMemberReferenceTarget";
    break;
  case TargetKind::MemberInstance:
    targetKind = "OMMemberInstanceTarget";
    break;
  }

  // Look up the associated HierPathOp.
  auto hierPathOp =
      symbolTable.lookup<hw::HierPathOp>(path.getTargetAttr().getAttr());
  auto namepath = hierPathOp.getNamepathAttr().getValue();

  // The name of the module which starts off the path.
  StringAttr topModule;
  // The path from the top module to the target. Represents a pair of instance
  // and its target module's name.
  SmallVector<std::pair<StringAttr, StringAttr>> modules;
  // If we're targeting a component or port of the target module, this will hold
  // its name.
  StringAttr component;
  // If we're indexing in to the component, this will be the access path.
  SmallString<64> field;

  // Process the final target first.
  auto &end = namepath.back();
  if (auto innerRef = dyn_cast<hw::InnerRefAttr>(end)) {
    auto target = irn.lookup(innerRef);
    if (target.isPort()) {
      // We are targeting the port of a module.
      auto module = cast<hw::HWModuleLike>(target.getOp());
      auto index = target.getPort();
      component = StringAttr::get(context, module.getPortName(index));
      auto loc = module.getPortLoc(index);
      auto type = module.getPortTypes()[index];
      if (failed(getAccessPath(loc, type, target.getField(), field)))
        return failure();
      topModule = module.getModuleNameAttr();
    } else {
      auto *op = target.getOp();
      assert(op && "innerRef should be targeting something");
      // Get the current module.
      topModule = innerRef.getModule();
      // Get the verilog name of the target.
      auto verilogName = op->getAttrOfType<StringAttr>("hw.verilogName");
      if (!verilogName) {
        auto diag = path->emitError("component does not have verilog name");
        diag.attachNote(op->getLoc()) << "component here";
        return diag;
      }
      if (auto inst = dyn_cast<hw::HWInstanceLike>(op)) {
        // We are targeting an instance.
        modules.emplace_back(verilogName, inst.getReferencedModuleNameAttr());
      } else {
        // We are targeting a regular component.
        component = verilogName;
        auto innerSym = cast<hw::InnerSymbolOpInterface>(op);
        auto value = innerSym.getTargetResult();
        if (failed(getAccessPath(value.getLoc(), value.getType(),
                                 target.getField(), field)))
          return failure();
      }
    }
  } else {
    // We are targeting a module.
    auto symbolRef = cast<FlatSymbolRefAttr>(end);
    topModule = symbolRef.getAttr();
  }

  // Process the rest of the hierarchical path.
  for (auto attr : llvm::reverse(namepath.drop_back())) {
    auto innerRef = cast<hw::InnerRefAttr>(attr);
    modules.emplace_back(innerRef.getName(), topModule);
    topModule = innerRef.getModule();
  }

  // Handle the modules not present in the path.
  assert(topModule && "must have something?");
  auto *node = instanceGraph.lookup(topModule);
  while (!node->noUses()) {
    auto module = node->getModule<hw::HWModuleLike>();

    // Dirty hack--scalarized modules are given public visibility to identify units of interest.
    if (module.getVisibility() == SymbolTable::Visibility::Public)
      break;

    if (!node->hasOneUse()) {
      auto diag = path->emitError() << "unable to uniquely resolve target "
                                       "due to multiple instantiation";
      diag.attachNote(path->getLoc()) << "the path is " << namepath << '\n';
      for (auto *use : node->uses()) {
        if (auto *op = use->getInstance<Operation *>())
          diag.attachNote(op->getLoc()) << "instance here";
        else
          diag.attachNote(module->getLoc()) << "module marked public " << module.getModuleName();
      }
      return diag;
    }
    auto *record = *node->usesBegin();

    // Get the verilog name of the instance.
    auto *inst = record->getInstance<Operation *>();
    // If the instance is external, just break here.
    if (!inst)
      break;
    auto verilogName = inst->getAttrOfType<StringAttr>("hw.verilogName");
    if (!verilogName)
      return inst->emitError("component does not have verilog name");
    modules.emplace_back(verilogName, module.getModuleNameAttr());
    node = record->getParent();
  }

  // We are finally at the top of the instance graph.
  topModule = node->getModule().getModuleNameAttr();

  // Create the target string.
  SmallString<128> targetString;
  targetString.append(targetKind);
  targetString.append(":");
  targetString.append(topModule);
  if (!modules.empty())
    targetString.append("/");
  for (auto [instance, module] : llvm::reverse(modules)) {
    targetString.append(instance);
    targetString.append(":");
    targetString.append(module);
  }
  if (component) {
    targetString.append(">");
    targetString.append(component);
  }
  targetString.append(field);
  auto targetPath = PathAttr::get(StringAttr::get(context, targetString));

  // Replace the old path operation.
  OpBuilder builder(path);
  auto constantOp = builder.create<ConstantOp>(path.getLoc(), targetPath);
  path.replaceAllUsesWith(constantOp.getResult());
  path->erase();
  return success();
}

LogicalResult PathVisitor::run(Operation *op) {
  auto result = op->walk([&](PathOp path) -> WalkResult {
    if (failed(processPath(path)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  return success();
}

namespace {
struct FreezePathsPass : public FreezePathsBase<FreezePathsPass> {
  void runOnOperation() override;
};
} // namespace

void FreezePathsPass::runOnOperation() {
  auto module = getOperation();
  auto &instanceGraph = getAnalysis<hw::InstanceGraph>();
  auto &symbolTable = getAnalysis<SymbolTable>();
  hw::InnerSymbolTableCollection collection(module);
  hw::InnerRefNamespace irn{symbolTable, collection};
  if (failed(PathVisitor(instanceGraph, irn).run(module)))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> circt::om::createFreezePathsPass() {
  return std::make_unique<FreezePathsPass>();
}
