//===- HWArithToHW.cpp - HWArith to HW Lowering pass ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main HWArith to HW Lowering Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/HWArithToHW.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/ConversionPatterns.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HWArith/HWArithOps.h"
#include "circt/Dialect/MSFT/MSFTOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
#define GEN_PASS_DEF_HWARITHTOHW
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace hwarith;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

// Function for setting the 'sv.namehint' attribute on 'newOp' based on any
// currently existing 'sv.namehint' attached to the source operation of 'value'.
// The user provides a callback which returns a new namehint based on the old
// namehint.
static void
improveNamehint(Value oldValue, Operation *newOp,
                llvm::function_ref<std::string(StringRef)> namehintCallback) {
  if (auto *sourceOp = oldValue.getDefiningOp()) {
    if (auto namehint =
            sourceOp->getAttrOfType<mlir::StringAttr>("sv.namehint")) {
      auto newNamehint = namehintCallback(namehint.strref());
      newOp->setAttr("sv.namehint",
                     StringAttr::get(oldValue.getContext(), newNamehint));
    }
  }
}

// Extract a bit range, specified via start bit and width, from a given value.
static Value extractBits(OpBuilder &builder, Location loc, Value value,
                         unsigned startBit, unsigned bitWidth) {
  Value extractedValue =
      builder.createOrFold<comb::ExtractOp>(loc, value, startBit, bitWidth);
  Operation *definingOp = extractedValue.getDefiningOp();
  if (extractedValue != value && definingOp) {
    // only change namehint if a new operation was created.
    improveNamehint(value, definingOp, [&](StringRef oldNamehint) {
      return (oldNamehint + "_" + std::to_string(startBit) + "_to_" +
              std::to_string(startBit + bitWidth))
          .str();
    });
  }
  return extractedValue;
}

// Perform the specified bit-extension (either sign- or zero-extension) for a
// given value to a desired target width.
static Value extendTypeWidth(OpBuilder &builder, Location loc, Value value,
                             unsigned targetWidth, bool signExtension) {
  unsigned sourceWidth = value.getType().getIntOrFloatBitWidth();
  unsigned extensionLength = targetWidth - sourceWidth;

  if (extensionLength == 0)
    return value;

  Value extensionBits;
  // https://circt.llvm.org/docs/Dialects/Comb/RationaleComb/#no-complement-negate-zext-sext-operators
  if (signExtension) {
    // Sign extension
    Value highBit = extractBits(builder, loc, value,
                                /*startBit=*/sourceWidth - 1, /*bitWidth=*/1);
    extensionBits =
        builder.createOrFold<comb::ReplicateOp>(loc, highBit, extensionLength);
  } else {
    // Zero extension
    extensionBits =
        hw::ConstantOp::create(builder, loc,
                               builder.getIntegerType(extensionLength), 0)
            ->getOpResult(0);
  }

  auto extOp = comb::ConcatOp::create(builder, loc, extensionBits, value);
  improveNamehint(value, extOp, [&](StringRef oldNamehint) {
    return (oldNamehint + "_" + (signExtension ? "sext_" : "zext_") +
            std::to_string(targetWidth))
        .str();
  });

  return extOp->getOpResult(0);
}

static bool isSignednessType(Type type) {
  auto match =
      llvm::TypeSwitch<Type, bool>(type)
          .Case<IntegerType>([](auto type) { return !type.isSignless(); })
          .Case<hw::ArrayType>(
              [](auto type) { return isSignednessType(type.getElementType()); })
          .Case<hw::UnpackedArrayType>(
              [](auto type) { return isSignednessType(type.getElementType()); })
          .Case<hw::StructType>([](auto type) {
            return llvm::any_of(type.getElements(), [](auto element) {
              return isSignednessType(element.type);
            });
          })
          .Case<hw::InOutType>(
              [](auto type) { return isSignednessType(type.getElementType()); })
          .Case<hw::TypeAliasType>(
              [](auto type) { return isSignednessType(type.getInnerType()); })
          .Default([](auto type) { return false; });

  return match;
}

static bool isSignednessAttr(Attribute attr) {
  if (auto typeAttr = dyn_cast<TypeAttr>(attr))
    return isSignednessType(typeAttr.getValue());
  return false;
}

/// Returns true if the given `op` is considered as legal for HWArith
/// conversion.
static bool isLegalOp(Operation *op) {
  if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
    return llvm::none_of(funcOp.getArgumentTypes(), isSignednessType) &&
           llvm::none_of(funcOp.getResultTypes(), isSignednessType) &&
           llvm::none_of(funcOp.getFunctionBody().getArgumentTypes(),
                         isSignednessType);
  }

  if (auto modOp = dyn_cast<hw::HWModuleLike>(op)) {
    return llvm::none_of(modOp.getPortTypes(), isSignednessType) &&
           llvm::none_of(modOp.getModuleBody().getArgumentTypes(),
                         isSignednessType);
  }

  auto attrs = llvm::map_range(op->getAttrs(), [](const NamedAttribute &attr) {
    return attr.getValue();
  });

  bool operandsOK = llvm::none_of(op->getOperandTypes(), isSignednessType);
  bool resultsOK = llvm::none_of(op->getResultTypes(), isSignednessType);
  bool attrsOK = llvm::none_of(attrs, isSignednessAttr);
  return operandsOK && resultsOK && attrsOK;
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
struct ConstantOpLowering : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp constOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(constOp,
                                                constOp.getConstantValue());
    return success();
  }
};
struct DivOpLowering : public OpConversionPattern<DivOp> {
  using OpConversionPattern<DivOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto isLhsTypeSigned =
        cast<IntegerType>(op.getOperand(0).getType()).isSigned();
    auto rhsType = cast<IntegerType>(op.getOperand(1).getType());
    auto targetType = cast<IntegerType>(op.getResult().getType());

    // comb.div* needs identical bitwidths for its operands and its result.
    // Hence, we need to calculate the minimal bitwidth that can be used to
    // represent the result as well as the operands without precision or sign
    // loss. The target size only depends on LHS and already handles the edge
    // cases where the bitwidth needs to be increased by 1. Thus, the targetType
    // is good enough for both the result as well as LHS.
    // The bitwidth for RHS is bit tricky. If the RHS is unsigned and we are
    // about to perform a signed division, then we need one additional bit to
    // avoid misinterpretation of RHS as a signed value!
    bool signedDivision = targetType.isSigned();
    unsigned extendSize = std::max(
        targetType.getWidth(),
        rhsType.getWidth() + (signedDivision && !rhsType.isSigned() ? 1 : 0));

    // Extend the operands
    Value lhsValue = extendTypeWidth(rewriter, loc, adaptor.getInputs()[0],
                                     extendSize, isLhsTypeSigned);
    Value rhsValue = extendTypeWidth(rewriter, loc, adaptor.getInputs()[1],
                                     extendSize, rhsType.isSigned());

    Value divResult;
    if (signedDivision)
      divResult = comb::DivSOp::create(rewriter, loc, lhsValue, rhsValue, false)
                      ->getOpResult(0);
    else
      divResult = comb::DivUOp::create(rewriter, loc, lhsValue, rhsValue, false)
                      ->getOpResult(0);

    // Carry over any attributes from the original div op.
    auto *divOp = divResult.getDefiningOp();
    rewriter.modifyOpInPlace(
        divOp, [&]() { divOp->setDialectAttrs(op->getDialectAttrs()); });

    // finally truncate back to the expected result size!
    Value truncateResult = extractBits(rewriter, loc, divResult, /*startBit=*/0,
                                       /*bitWidth=*/targetType.getWidth());
    rewriter.replaceOp(op, truncateResult);

    return success();
  }
};
} // namespace

namespace {
struct CastOpLowering : public OpConversionPattern<CastOp> {
  using OpConversionPattern<CastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceType = cast<IntegerType>(op.getIn().getType());
    auto sourceWidth = sourceType.getWidth();
    bool isSourceTypeSigned = sourceType.isSigned();
    auto targetWidth = cast<IntegerType>(op.getOut().getType()).getWidth();

    Value replaceValue;
    if (sourceWidth == targetWidth) {
      // the width does not change, we are done here and can directly use the
      // lowering input value
      replaceValue = adaptor.getIn();
    } else if (sourceWidth < targetWidth) {
      // bit extensions needed, the type of extension required is determined by
      // the source type only!
      replaceValue = extendTypeWidth(rewriter, op.getLoc(), adaptor.getIn(),
                                     targetWidth, isSourceTypeSigned);
    } else {
      // bit truncation needed
      replaceValue = extractBits(rewriter, op.getLoc(), adaptor.getIn(),
                                 /*startBit=*/0, /*bitWidth=*/targetWidth);
    }
    rewriter.replaceOp(op, replaceValue);

    return success();
  }
};
} // namespace

namespace {

// Utility lowering function that maps a hwarith::ICmpPredicate predicate and
// the information whether the comparison contains signed values to the
// corresponding comb::ICmpPredicate.
static comb::ICmpPredicate lowerPredicate(ICmpPredicate pred, bool isSigned) {
  switch (pred) {
  case ICmpPredicate::eq:
    return comb::ICmpPredicate::eq;
  case ICmpPredicate::ne:
    return comb::ICmpPredicate::ne;
  case ICmpPredicate::lt:
    return isSigned ? comb::ICmpPredicate::slt : comb::ICmpPredicate::ult;
  case ICmpPredicate::ge:
    return isSigned ? comb::ICmpPredicate::sge : comb::ICmpPredicate::uge;
  case ICmpPredicate::le:
    return isSigned ? comb::ICmpPredicate::sle : comb::ICmpPredicate::ule;
  case ICmpPredicate::gt:
    return isSigned ? comb::ICmpPredicate::sgt : comb::ICmpPredicate::ugt;
  }

  llvm_unreachable(
      "Missing hwarith::ICmpPredicate to comb::ICmpPredicate lowering");
  return comb::ICmpPredicate::eq;
}

struct ICmpOpLowering : public OpConversionPattern<ICmpOp> {
  using OpConversionPattern<ICmpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ICmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto lhsType = cast<IntegerType>(op.getLhs().getType());
    auto rhsType = cast<IntegerType>(op.getRhs().getType());
    IntegerType::SignednessSemantics cmpSignedness;
    const unsigned cmpWidth =
        inferAddResultType(cmpSignedness, lhsType, rhsType) - 1;

    ICmpPredicate pred = op.getPredicate();
    comb::ICmpPredicate combPred = lowerPredicate(
        pred, cmpSignedness == IntegerType::SignednessSemantics::Signed);

    const auto loc = op.getLoc();
    Value lhsValue = extendTypeWidth(rewriter, loc, adaptor.getLhs(), cmpWidth,
                                     lhsType.isSigned());
    Value rhsValue = extendTypeWidth(rewriter, loc, adaptor.getRhs(), cmpWidth,
                                     rhsType.isSigned());

    auto newOp = comb::ICmpOp::create(rewriter, op->getLoc(), combPred,
                                      lhsValue, rhsValue, false);
    rewriter.modifyOpInPlace(
        newOp, [&]() { newOp->setDialectAttrs(op->getDialectAttrs()); });
    rewriter.replaceOp(op, newOp);

    return success();
  }
};

template <class BinOp, class ReplaceOp>
struct BinaryOpLowering : public OpConversionPattern<BinOp> {
  using OpConversionPattern<BinOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<BinOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(BinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto isLhsTypeSigned =
        cast<IntegerType>(op.getOperand(0).getType()).isSigned();
    auto isRhsTypeSigned =
        cast<IntegerType>(op.getOperand(1).getType()).isSigned();
    auto targetWidth = cast<IntegerType>(op.getResult().getType()).getWidth();

    Value lhsValue = extendTypeWidth(rewriter, loc, adaptor.getInputs()[0],
                                     targetWidth, isLhsTypeSigned);
    Value rhsValue = extendTypeWidth(rewriter, loc, adaptor.getInputs()[1],
                                     targetWidth, isRhsTypeSigned);
    auto newOp =
        ReplaceOp::create(rewriter, op.getLoc(), lhsValue, rhsValue, false);
    rewriter.modifyOpInPlace(
        newOp, [&]() { newOp->setDialectAttrs(op->getDialectAttrs()); });
    rewriter.replaceOp(op, newOp);

    return success();
  }
};

} // namespace

Type HWArithToHWTypeConverter::removeSignedness(Type type) {
  auto it = conversionCache.find(type);
  if (it != conversionCache.end())
    return it->second.type;

  auto convertedType =
      llvm::TypeSwitch<Type, Type>(type)
          .Case<IntegerType>([](auto type) {
            if (type.isSignless())
              return type;
            return IntegerType::get(type.getContext(), type.getWidth());
          })
          .Case<hw::ArrayType>([this](auto type) {
            return hw::ArrayType::get(removeSignedness(type.getElementType()),
                                      type.getNumElements());
          })
          .Case<hw::UnpackedArrayType>([this](auto type) {
            return hw::UnpackedArrayType::get(
                removeSignedness(type.getElementType()), type.getNumElements());
          })
          .Case<hw::StructType>([this](auto type) {
            // Recursively convert each element.
            llvm::SmallVector<hw::StructType::FieldInfo> convertedElements;
            for (auto element : type.getElements()) {
              convertedElements.push_back(
                  {element.name, removeSignedness(element.type)});
            }
            return hw::StructType::get(type.getContext(), convertedElements);
          })
          .Case<hw::InOutType>([this](auto type) {
            return hw::InOutType::get(removeSignedness(type.getElementType()));
          })
          .Case<hw::TypeAliasType>([this](auto type) {
            return hw::TypeAliasType::get(
                type.getRef(), removeSignedness(type.getInnerType()));
          })
          .Default([](auto type) { return type; });

  return convertedType;
}

HWArithToHWTypeConverter::HWArithToHWTypeConverter() {
  // Pass any type through the signedness remover.
  addConversion([this](Type type) { return removeSignedness(type); });

  addTargetMaterialization([&](mlir::OpBuilder &builder, mlir::Type resultType,
                               mlir::ValueRange inputs,
                               mlir::Location loc) -> mlir::Value {
    if (inputs.size() != 1)
      return Value();
    return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                              inputs[0])
        ->getResult(0);
  });

  addSourceMaterialization([&](mlir::OpBuilder &builder, mlir::Type resultType,
                               mlir::ValueRange inputs,
                               mlir::Location loc) -> mlir::Value {
    if (inputs.size() != 1)
      return Value();
    return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                              inputs[0])
        ->getResult(0);
  });
}

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

void circt::populateHWArithToHWConversionPatterns(
    HWArithToHWTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<ConstantOpLowering, CastOpLowering, ICmpOpLowering,
               BinaryOpLowering<AddOp, comb::AddOp>,
               BinaryOpLowering<SubOp, comb::SubOp>,
               BinaryOpLowering<MulOp, comb::MulOp>, DivOpLowering>(
      typeConverter, patterns.getContext());
}

namespace {

class HWArithToHWPass : public circt::impl::HWArithToHWBase<HWArithToHWPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();

    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal(isLegalOp);
    RewritePatternSet patterns(&getContext());
    HWArithToHWTypeConverter typeConverter;
    target.addIllegalDialect<HWArithDialect>();

    // Add HWArith-specific conversion patterns.
    populateHWArithToHWConversionPatterns(typeConverter, patterns);

    // ALL other operations are converted via the TypeConversionPattern which
    // will replace an operation to an identical operation with replaced
    // result types and operands.
    patterns.add<TypeConversionPattern>(typeConverter, patterns.getContext());

    // Apply a full conversion - all operations must either be legal, be caught
    // by one of the HWArith patterns or be converted by the
    // TypeConversionPattern.
    if (failed(applyFullConversion(module, target, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass initialization
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> circt::createHWArithToHWPass() {
  return std::make_unique<HWArithToHWPass>();
}
