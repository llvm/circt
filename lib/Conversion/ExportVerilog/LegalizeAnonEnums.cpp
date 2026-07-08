//===- LegalizeAnonEnums.cpp - Legalizes anonymous enumerations -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass replaces all anonymous enumeration with typedecls in the output
// Verilog.
//
//===----------------------------------------------------------------------===//

#include "ExportVerilogInternals.h"
#include "circt/Conversion/ExportVerilog.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/WalkResult.h"
#include "llvm/ADT/DenseSet.h"

namespace circt {
#define GEN_PASS_DEF_LEGALIZEANONENUMS
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace circt;
using namespace hw;
using namespace sv;
using namespace mlir;

namespace {
struct LegalizeAnonEnums
    : public circt::impl::LegalizeAnonEnumsBase<LegalizeAnonEnums> {
  /// Creates a TypeScope on demand for anonymous enumerations.
  TypeScopeOp getTypeScope() {
    auto topLevel = getOperation();
    if (!typeScope) {
      auto builder = OpBuilder::atBlockBegin(&topLevel.getRegion().front());
      typeScope = TypeScopeOp::create(builder, topLevel.getLoc(), "Enums");
      typeScope.getBodyRegion().push_back(new Block());
      mlir::SymbolTable symbolTable(topLevel);
      symbolTable.insert(typeScope);
    }
    return typeScope;
  }

  /// Helper to create TypeDecls and TypeAliases for EnumTypes;
  Type getEnumTypeDecl(EnumType type) {
    auto &typeAlias = enumTypeAliases[type];
    if (typeAlias)
      return typeAlias;
    auto *context = &getContext();
    auto loc = UnknownLoc::get(context);
    auto typeScope = getTypeScope();
    auto builder = OpBuilder::atBlockEnd(&typeScope.getRegion().front());
    auto declName = StringAttr::get(context, "enum" + Twine(enumCount++));
    TypedeclOp::create(builder, loc, declName, TypeAttr::get(type), nullptr);
    auto symRef = SymbolRefAttr::get(typeScope.getSymNameAttr(),
                                     FlatSymbolRefAttr::get(declName));
    typeAlias = TypeAliasType::get(symRef, type);
    return typeAlias;
  }

  AttrTypeReplacer &addReplacementFns(AttrTypeReplacer &replacer) {
    // We globally skip all TypeAttr so that existing TypedeclOps keep their
    // direct !hw.enum defining type rather than having it replaced with a
    // TypeAliasType.
    //
    // However, this global skip has a side effect, which means it also skips
    // e.g. EnumFieldAttr's Type and hw.module's ModuleType, preventing their
    // inner types from being replaced with TypeAliasType. So we add two manual
    // compensations below.
    //
    // TODO: Once the upstream AttrTypeReplacer has per-op replacement
    // filtering, we can simplify this by adding a per-op replacement filter
    // that skips only TypedeclOp's TypeAttr. This would eliminate the need for
    // the two manual compensations below.
    replacer.addReplacement(
        [&](Attribute attr) -> AttrTypeReplacer::ReplaceFnResult<Attribute> {
          // Side-effect compensation: manually update EnumFieldAttr's inner
          // type, which would otherwise be skipped by the TypeAttr skip.
          if (auto fieldAttr = dyn_cast<EnumFieldAttr>(attr)) {
            auto innerType = fieldAttr.getType().getValue();
            if (auto enumType = dyn_cast<EnumType>(innerType)) {
              Type newType = getEnumTypeDecl(enumType);
              if (newType != innerType)
                return std::make_pair(Attribute(EnumFieldAttr::get(
                                          UnknownLoc::get(&getContext()),
                                          fieldAttr.getField(), newType)),
                                      WalkResult::skip());
            }
          }
          // Skip TypeAttr instances inside TypedeclOps.
          if (isa<TypeAttr>(attr))
            return std::make_pair(attr, WalkResult::skip());
          return std::nullopt;
        });
    // Type replacement: skip existing TypeAliasType sub-elements to prevent
    // nested aliases, and replace bare EnumType with a TypeAliasType.
    replacer.addReplacement(
        [&](Type type) -> AttrTypeReplacer::ReplaceFnResult<Type> {
          if (isa<TypeAliasType>(type))
            return std::make_pair(type, WalkResult::skip());
          if (auto enumType = dyn_cast<EnumType>(type))
            return std::make_pair(getEnumTypeDecl(enumType),
                                  WalkResult::advance());
          return std::nullopt;
        });
    return replacer;
  }

  void runOnOperation() override {
    enumCount = 0;
    typeScope = {};

    AttrTypeReplacer replacer;
    addReplacementFns(replacer).recursivelyReplaceElementsIn(
        getOperation(), /*replaceAttrs=*/true,
        /*replaceLocs=*/true,
        /*replaceTypes=*/true);

    // Side-effect compensation: explicitly update HWModuleLike types to keep
    // them in sync with block arguments, which would otherwise be skipped by
    // the TypeAttr skip.
    getOperation()->walk([&](HWModuleLike modLike) {
      auto modType = modLike.getHWModuleType();
      bool changed = false;
      SmallVector<ModulePort> ports(modType.getPorts());
      for (auto &p : ports) {
        if (Type newType = replacer.replace(p.type)) {
          if (newType != p.type) {
            p.type = newType;
            changed = true;
          }
        }
      }
      if (changed)
        modLike.setHWModuleType(ModuleType::get(modLike.getContext(), ports));
    });

    enumTypeAliases.clear();
  }

private:
  TypeScopeOp typeScope;
  unsigned enumCount;
  DenseMap<Type, Type> enumTypeAliases;
};

} // end anonymous namespace
