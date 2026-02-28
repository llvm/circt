//===- ESILowerTypes.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower high-level ESI types to HW conversions and pass.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/HW/ConversionPatterns.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWTypes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
namespace esi {
#define GEN_PASS_DEF_LOWERESITYPES
#include "circt/Dialect/ESI/ESIPasses.h.inc"
} // namespace esi
} // namespace circt

using namespace circt;
using namespace circt::esi;

namespace {
/// Lower all "high-level" ESI types on modules to some lower construct.
struct ESILowerTypesPass
    : public circt::esi::impl::LowerESITypesBase<ESILowerTypesPass> {
  void runOnOperation() override;
};
} // anonymous namespace

namespace {
/// Materializations and type conversions to lower ESI data windows.
class LowerTypesConverter : public TypeConverter {
public:
  LowerTypesConverter() {
    addConversion([](Type t) { return t; });
    addConversion([](WindowType window) { return window.getLoweredType(); });
    addConversion([&](hw::ArrayType array) -> Type {
      Type element = convertType(array.getElementType());
      if (!element)
        return Type();
      if (element == array.getElementType())
        return array;
      return hw::ArrayType::get(element, array.getNumElements());
    });
    addConversion([&](hw::StructType structType) -> Type {
      SmallVector<hw::StructType::FieldInfo> fields;
      fields.reserve(structType.getElements().size());
      bool changed = false;
      for (auto field : structType.getElements()) {
        Type lowered = convertType(field.type);
        if (!lowered)
          return Type();
        changed |= lowered != field.type;
        fields.push_back({field.name, lowered});
      }
      if (!changed)
        return structType;
      return hw::StructType::get(structType.getContext(), fields);
    });
    addConversion([&](hw::UnionType unionType) -> Type {
      SmallVector<hw::UnionType::FieldInfo> fields;
      fields.reserve(unionType.getElements().size());
      bool changed = false;
      for (auto field : unionType.getElements()) {
        Type lowered = convertType(field.type);
        if (!lowered)
          return Type();
        changed |= lowered != field.type;
        fields.push_back({field.name, lowered, field.offset});
      }
      if (!changed)
        return unionType;
      return hw::UnionType::get(unionType.getContext(), fields);
    });
    addConversion([&](esi::ListType listType) -> Type {
      Type element = convertType(listType.getElementType());
      if (!element)
        return Type();
      if (element == listType.getElementType())
        return listType;
      return esi::ListType::get(listType.getContext(), element);
    });
    addConversion([&](hw::TypeAliasType alias) -> Type {
      Type lowered = convertType(alias.getInnerType());
      if (!lowered)
        return Type();
      if (lowered == alias.getInnerType())
        return alias;
      return hw::TypeAliasType::get(alias.getRef(), lowered);
    });
    addSourceMaterialization(wrapMaterialization);
    addTargetMaterialization(unwrapMaterialization);
  }

private:
  static mlir::Value wrapMaterialization(OpBuilder &b, WindowType resultType,
                                         ValueRange inputs, Location loc) {
    if (inputs.size() != 1)
      return mlir::Value();
    return b.createOrFold<WrapWindow>(loc, resultType, inputs[0]);
  }

  static mlir::Value unwrapMaterialization(OpBuilder &b, Type resultType,
                                           ValueRange inputs, Location loc) {
    if (inputs.size() != 1 || !isa<WindowType>(inputs[0].getType()))
      return mlir::Value();
    return b.createOrFold<UnwrapWindow>(loc, resultType, inputs[0]);
  }
};
} // namespace

static bool containsWindowType(Type type) {
  return TypeSwitch<Type, bool>(type)
      .Case([](WindowType) { return true; })
      .Case<hw::ArrayType>([](hw::ArrayType array) {
        return containsWindowType(array.getElementType());
      })
      .Case<hw::StructType>([](hw::StructType structType) {
        for (auto field : structType.getElements())
          if (containsWindowType(field.type))
            return true;
        return false;
      })
      .Case<hw::UnionType>([](hw::UnionType unionType) {
        for (auto field : unionType.getElements())
          if (containsWindowType(field.type))
            return true;
        return false;
      })
      .Case<esi::ListType>([](esi::ListType listType) {
        return containsWindowType(listType.getElementType());
      })
      .Case<hw::TypeAliasType>([](hw::TypeAliasType aliasType) {
        return containsWindowType(aliasType.getInnerType());
      })
      .Default([](Type) { return false; });
}

void ESILowerTypesPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalOp<WrapWindow, UnwrapWindow>();

  // We need to lower instances, modules, and outputs with data windows.
  target.markUnknownOpDynamicallyLegal([](Operation *op) {
    return TypeSwitch<Operation *, bool>(op)
        .Case([](igraph::InstanceOpInterface inst) {
          auto hasWindow = [](Type type) { return containsWindowType(type); };
          return !(llvm::any_of(inst->getOperandTypes(), hasWindow) ||
                   llvm::any_of(inst->getResultTypes(), hasWindow));
        })
        .Case([](hw::HWMutableModuleLike mod) {
          auto isWindowPort = [](hw::PortInfo p) {
            return containsWindowType(p.type);
          };
          return !(llvm::any_of(mod.getPortList(), isWindowPort));
        })
        .Default([](Operation *op) {
          if (llvm::any_of(op->getOperandTypes(), containsWindowType) ||
              llvm::any_of(op->getResultTypes(), containsWindowType))
            return false;
          return true;
        });
  });

  LowerTypesConverter types;
  RewritePatternSet patterns(&getContext());
  patterns.add<TypeConversionPattern>(types, &getContext());
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();

  // Now do a canonicalization pass to clean up any unnecessary wrap-unwrap
  // pairs.
  mlir::ConversionConfig config;
  config.foldingMode = mlir::DialectConversionFoldingMode::BeforePatterns;
  ConversionTarget partialCanonicalizedTarget(getContext());
  RewritePatternSet partialPatterns(&getContext());
  partialCanonicalizedTarget.addIllegalOp<WrapWindow, UnwrapWindow>();
  WrapWindow::getCanonicalizationPatterns(partialPatterns, &getContext());
  UnwrapWindow::getCanonicalizationPatterns(partialPatterns, &getContext());
  if (failed(mlir::applyPartialConversion(getOperation(),
                                          partialCanonicalizedTarget,
                                          std::move(partialPatterns), config)))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
circt::esi::createESITypeLoweringPass() {
  return std::make_unique<ESILowerTypesPass>();
}
