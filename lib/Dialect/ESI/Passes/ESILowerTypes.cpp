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
#include "circt/Dialect/HW/HWOpInterfaces.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace circt::esi;

namespace {
/// Lower all "high-level" ESI types on modules to some lower construct.
struct ESILowerTypesPass : public LowerESITypesBase<ESILowerTypesPass> {
  void runOnOperation() override;
};
} // anonymous namespace

// Converts a function type wrt. the given type converter.
static FunctionType convertFunctionType(TypeConverter &typeConverter,
                                        FunctionType type) {
  // Convert the original function types.
  llvm::SmallVector<Type> res, arg;
  llvm::transform(type.getResults(), std::back_inserter(res),
                  [&](Type t) { return typeConverter.convertType(t); });
  llvm::transform(type.getInputs(), std::back_inserter(arg),
                  [&](Type t) { return typeConverter.convertType(t); });

  return FunctionType::get(type.getContext(), arg, res);
}

namespace {
/// Generic pattern which replaces an operation by one of the same operation
/// name, but with converted attributes, operands, and result types to eliminate
/// illegal types. Uses generic builders based on OperationState to make sure
/// that this pattern can apply to _any_ operation.
struct TypeConversionPattern : public ConversionPattern {
public:
  TypeConversionPattern(TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, context) {}
  using ConversionPattern::ConversionPattern;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult TypeConversionPattern::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {

  // Convert the TypeAttrs.
  llvm::SmallVector<NamedAttribute, 4> newAttrs;
  newAttrs.reserve(op->getAttrs().size());
  for (auto attr : op->getAttrs()) {
    if (auto typeAttr = attr.getValue().dyn_cast<TypeAttr>()) {
      auto innerType = typeAttr.getValue();
      // TypeConvert::convertType doesn't handle function types, so we need to
      // handle them manually.
      if (auto funcType = innerType.dyn_cast<FunctionType>(); innerType) {
        innerType = convertFunctionType(*getTypeConverter(), funcType);
      } else {
        innerType = getTypeConverter()->convertType(innerType);
      }
      newAttrs.emplace_back(attr.getName(), TypeAttr::get(innerType));
    } else {
      newAttrs.push_back(attr);
    }
  }

  // Convert the result types.
  llvm::SmallVector<Type, 4> newResults;
  (void)getTypeConverter()->convertTypes(op->getResultTypes(), newResults);

  // Build the state for the edited clone.
  OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                       newResults, newAttrs, op->getSuccessors());
  for (size_t i = 0, e = op->getNumRegions(); i < e; ++i)
    state.addRegion();

  // Must create the op before running any modifications on the regions so that
  // we don't crash with '-debug' and so we have something to 'root update'.
  Operation *newOp = rewriter.create(state);

  // Move the regions over, converting the signatures as we go.
  rewriter.startRootUpdate(newOp);
  for (size_t i = 0, e = op->getNumRegions(); i < e; ++i) {
    Region &region = op->getRegion(i);
    Region *newRegion = &newOp->getRegion(i);

    // TypeConverter::SignatureConversion drops argument locations, so we need
    // to manually copy them over (a verifier in e.g. HWModule checks this).
    llvm::SmallVector<Location, 4> argLocs;
    for (auto arg : region.getArguments())
      argLocs.push_back(arg.getLoc());

    rewriter.inlineRegionBefore(region, *newRegion, newRegion->begin());
    TypeConverter::SignatureConversion result(newRegion->getNumArguments());
    (void)getTypeConverter()->convertSignatureArgs(
        newRegion->getArgumentTypes(), result);
    rewriter.applySignatureConversion(newRegion, result, getTypeConverter());

    // Apply the argument locations.
    for (auto [arg, loc] : llvm::zip(newRegion->getArguments(), argLocs))
      arg.setLoc(loc);
  }
  rewriter.finalizeRootUpdate(newOp);

  rewriter.replaceOp(op, newOp->getResults());
  return success();
}

namespace {
/// Materializations and type conversions to lower ESI data windows.
class LowerTypesConverter : public TypeConverter {
public:
  LowerTypesConverter() {
    addConversion([](Type t) { return t; });
    addConversion([](WindowType window) { return window.getLoweredType(); });
    addSourceMaterialization(wrapMaterialization);
    addArgumentMaterialization(wrapMaterialization);
    addTargetMaterialization(unwrapMaterialization);
  }

private:
  static std::optional<mlir::Value> wrapMaterialization(OpBuilder &b,
                                                        WindowType resultType,
                                                        ValueRange inputs,
                                                        Location loc) {
    if (inputs.size() != 1)
      return std::nullopt;
    auto wrap = b.create<WrapWindow>(loc, resultType, inputs[0]);
    return wrap.getWindow();
  }

  static std::optional<mlir::Value>
  unwrapMaterialization(OpBuilder &b, hw::UnionType resultType,
                        ValueRange inputs, Location loc) {
    if (inputs.size() != 1 || !isa<WindowType>(inputs[0].getType()))
      return std::nullopt;
    auto unwrap = b.create<UnwrapWindow>(loc, resultType, inputs[0]);
    return unwrap.getFrame();
  }
};
} // namespace

void ESILowerTypesPass::runOnOperation() {
  ConversionTarget target(getContext());

  // We need to lower instances, modules, and outputs with data windows.
  target.markUnknownOpDynamicallyLegal([](Operation *op) {
    return TypeSwitch<Operation *, bool>(op)
        .Case([](hw::HWInstanceLike inst) {
          return !(
              llvm::any_of(inst->getOperandTypes(), hw::type_isa<WindowType>) ||
              llvm::any_of(inst->getResultTypes(), hw::type_isa<WindowType>));
        })
        .Case([](hw::HWMutableModuleLike mod) {
          auto isWindowPort = [](hw::PortInfo p) {
            return hw::type_isa<WindowType>(p.type);
          };
          return !(llvm::any_of(mod.getPorts().inputs, isWindowPort) ||
                   llvm::any_of(mod.getPorts().outputs, isWindowPort));
        })
        .Default([](Operation *op) {
          if (op->hasTrait<OpTrait::ReturnLike>())
            return !llvm::any_of(op->getOperandTypes(),
                                 hw::type_isa<WindowType>);
          return true;
        });
  });

  LowerTypesConverter types;
  RewritePatternSet patterns(&getContext());
  patterns.add<TypeConversionPattern>(types, &getContext());
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
circt::esi::createESITypeLoweringPass() {
  return std::make_unique<ESILowerTypesPass>();
}
