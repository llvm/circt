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

/// Helper class extending AttrTypeReplacer to skip specific operation subtrees.
/// Registered operation types will be ignored during recursive replacement.
class AttrTypeReplacerWithSkippedOps : public AttrTypeReplacer {
public:
  /// Register operation types to be skipped.
  template <typename... SkippedOps>
  void addSkippedOps() {
    (addSkippedOpImpl<SkippedOps>(), ...);
  }

  /// Recursively replace elements, skipping subtrees of registered ops.
  void recursivelyReplaceElementsIn(Operation *op, bool replaceAttrs = true,
                                    bool replaceLocs = false,
                                    bool replaceTypes = false) {
    op->walk([&](Operation *nestedOp) -> WalkResult {
      if (!contains(nestedOp)) {
        replaceElementsIn(nestedOp, replaceAttrs, replaceLocs, replaceTypes);
        return WalkResult::advance();
      }
      return WalkResult::skip();
    });
  };

  /// Checks whether the given operation is in the skip list.
  bool contains(Operation *op) const {
    if (auto info = op->getRegisteredInfo())
      return skippedOps.contains(info->getTypeID());
    return false;
  }

private:
  /// Insert the TypeID of a single operation type into the skip set.
  template <typename T>
  void addSkippedOpImpl() {
    static_assert(std::is_base_of_v<::mlir::OpState, T>,
                  "Unexpected type of inserted operation");
    skippedOps.insert(TypeID::get<T>());
  }

  /// Storage for type IDs of operations to skip.
  DenseSet<TypeID> skippedOps;
};

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

  /// Replace bare EnumType with TypeAliasType, and skip TypedeclOp subtrees to
  /// keep their property-backed raw EnumType intact.
  AttrTypeReplacerWithSkippedOps &
  addReplacementFns(AttrTypeReplacerWithSkippedOps &replacer) {
    replacer.addReplacement(
        [&](Type type) -> AttrTypeReplacer::ReplaceFnResult<Type> {
          if (isa<TypeAliasType>(type))
            return std::make_pair(type, WalkResult::skip());
          if (auto enumType = dyn_cast<EnumType>(type))
            return std::make_pair(getEnumTypeDecl(enumType),
                                  WalkResult::advance());
          return std::nullopt;
        });
    replacer.addSkippedOps<TypedeclOp>();
    return replacer;
  }

  void runOnOperation() override {
    enumCount = 0;
    typeScope = {};

    AttrTypeReplacerWithSkippedOps replacer;
    addReplacementFns(replacer).recursivelyReplaceElementsIn(
        getOperation(), /*replaceAttrs=*/true,
        /*replaceLocs=*/true,
        /*replaceTypes=*/true);

    // hw.module stores its module_type as a property, not a discardable attr,
    // so the replacer can't touch it. Manually sync it with the already-updated
    // block argument types.
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
