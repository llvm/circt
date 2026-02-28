//===- CreateVTables.cpp - Create VTables from ClassDeclOps --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CreateVTables pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MoorePasses.h"

namespace circt {
namespace moore {
#define GEN_PASS_DEF_CREATEVTABLES
#include "circt/Dialect/Moore/MoorePasses.h.inc"
} // namespace moore
} // namespace circt

using namespace circt;
using namespace moore;
using namespace llvm;
using namespace mlir;

namespace {
using MethodMap = llvm::DenseMap<StringRef, std::optional<SymbolRefAttr>>;
using ClassMap = llvm::DenseMap<ClassDeclOp, MethodMap>;

struct CreateVTablesPass
    : public circt::moore::impl::CreateVTablesBase<CreateVTablesPass> {
  void runOnOperation() override;

private:
  /// Cache that stores the most derived virtual method for each class
  ClassMap classToMethodMap;
  /// Helper function to collect vtable info for every top-level classdecl
  void collectClasses(ModuleOp mod, SymbolTable &symTab);
  /// Recursive helper function to determine most derived virtual method impl
  /// per class
  void collectClassDependencies(ModuleOp mod, SymbolTable &symTab,
                                ClassDeclOp &clsDecl,
                                SymbolRefAttr dependencyName);

  /// Function to emit VTable computed in classToMethodMap
  void emitVTablePerClass(ModuleOp mod, SymbolTable &symTab,
                          OpBuilder &builder);
  void emitVTablePerDependencyClass(ModuleOp mod, SymbolTable &symTab,
                                    OpBuilder &builder, ClassDeclOp &clsDecl,
                                    SymbolRefAttr dependencyName);
};
} // namespace

std::unique_ptr<mlir::Pass> circt::moore::createVTablesPass() {
  return std::make_unique<CreateVTablesPass>();
}

void CreateVTablesPass::collectClassDependencies(ModuleOp mod,
                                                 SymbolTable &symTab,
                                                 ClassDeclOp &clsDecl,
                                                 SymbolRefAttr dependencyName) {
  auto dependencyDecl =
      symTab.lookupNearestSymbolFrom<ClassDeclOp>(mod, dependencyName);

  auto &clsMap = classToMethodMap[clsDecl];
  for (auto methodDecl : dependencyDecl.getBody().getOps<ClassMethodDeclOp>()) {

    // If a derived class already maps this method, continue
    auto &mapEntry = clsMap[methodDecl.getSymName()];
    if (mapEntry)
      continue;

    std::optional<SymbolRefAttr> impl;
    if (methodDecl.getImpl().has_value()) {
      impl = methodDecl.getImpl().value();
    } else {
      impl = std::nullopt;
    }

    mapEntry = impl;
  }
  if (dependencyDecl.getBase().has_value())
    collectClassDependencies(mod, symTab, clsDecl,
                             dependencyDecl.getBase().value());
  if (dependencyDecl.getImplementedInterfaces().has_value())
    for (auto intf : dependencyDecl.getImplementedInterfacesAttr())
      collectClassDependencies(mod, symTab, clsDecl, cast<SymbolRefAttr>(intf));
}

void CreateVTablesPass::collectClasses(ModuleOp mod, SymbolTable &symTab) {
  for (auto clsDecl : mod.getBodyRegion().getOps<ClassDeclOp>()) {
    auto &clsMap = classToMethodMap[clsDecl];
    // Don't override if already filled
    if (!clsMap.empty())
      continue;
    collectClassDependencies(mod, symTab, clsDecl, SymbolRefAttr::get(clsDecl));
  }
}

static bool noneHaveImpl(const MethodMap &methods) {
  return llvm::all_of(methods,
                      [](const auto &kv) { return !kv.second.has_value(); });
}

static bool allHaveImpl(const MethodMap &methods) {
  return llvm::all_of(methods,
                      [](const auto &kv) { return kv.second.has_value(); });
}

static inline SymbolRefAttr getVTableName(ClassDeclOp &clsDecl) {

  auto base = SymbolRefAttr::get(clsDecl); // e.g. @MyClass
  auto suffix = mlir::FlatSymbolRefAttr::get(clsDecl.getContext(), "vtable");
  auto vTableName = mlir::SymbolRefAttr::get(base.getRootReference(), {suffix});
  return vTableName;
}

void CreateVTablesPass::emitVTablePerDependencyClass(
    ModuleOp mod, SymbolTable &symTab, OpBuilder &builder, ClassDeclOp &clsDecl,
    SymbolRefAttr dependencyName) {

  auto dependencyDecl =
      symTab.lookupNearestSymbolFrom<ClassDeclOp>(mod, dependencyName);

  auto clsMethodMap = classToMethodMap[clsDecl];
  auto depMethodMap = classToMethodMap[dependencyDecl];

  // If the VTable would be empty, don't emit it.
  if (depMethodMap.empty())
    return;

  auto vTableName = getVTableName(dependencyDecl);

  auto clsVTable =
      VTableOp::create(builder, dependencyDecl.getLoc(), vTableName);
  auto &region = clsVTable.getRegion();
  auto &block = region.emplaceBlock();

  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToEnd(&block);

  // First emit base class if available
  if (dependencyDecl.getBase().has_value())
    emitVTablePerDependencyClass(mod, symTab, builder, clsDecl,
                                 dependencyDecl.getBase().value());

  // Next emit interface classes if available
  if (dependencyDecl.getImplementedInterfaces().has_value())
    for (auto intf : dependencyDecl.getImplementedInterfacesAttr())
      emitVTablePerDependencyClass(mod, symTab, builder, clsDecl,
                                   cast<SymbolRefAttr>(intf));

  // Last, emit any own method symbol entries
  for (auto methodDecl : dependencyDecl.getBody().getOps<ClassMethodDeclOp>()) {
    auto methodName = methodDecl.getSymName();
    VTableEntryOp::create(builder, methodDecl.getLoc(), methodName,
                          clsMethodMap[methodName].value());
  }
}

void CreateVTablesPass::emitVTablePerClass(ModuleOp mod, SymbolTable &symTab,
                                           OpBuilder &builder) {
  // Check emission for every top-level class decl
  for (auto [clsDecl, methodMap] : classToMethodMap) {

    // Sanity check that either all methods are implemented or none are.
    if (!(allHaveImpl(methodMap) || noneHaveImpl(methodMap))) {
      clsDecl.emitError()
          << "Class declaration " << clsDecl.getSymName()
          << " is malformed; some methods are abstract and some are concrete, "
             "which is not legal in System Verilog.";
      return;
    }

    // Skip abstract classes
    if (noneHaveImpl(methodMap))
      continue;

    auto vTableName = getVTableName(clsDecl);
    // Don't try to emit a vtable if it already exists.
    if (symTab.lookupNearestSymbolFrom<VTableOp>(mod, vTableName))
      continue;

    builder.setInsertionPointAfter(clsDecl);

    emitVTablePerDependencyClass(mod, symTab, builder, clsDecl,
                                 SymbolRefAttr::get(clsDecl));
  }
}

void CreateVTablesPass::runOnOperation() {
  ModuleOp mod = getOperation();
  SymbolTable symTab(mod);
  collectClasses(mod, symTab);
  mlir::OpBuilder builder = mlir::OpBuilder::atBlockBegin(mod.getBody());
  emitVTablePerClass(mod, symTab, builder);
}
