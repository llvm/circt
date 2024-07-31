//===- FIRRTLIntrinsics.cpp - Lower Intrinsics ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLIntrinsics.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// GenericIntrinsic
//===----------------------------------------------------------------------===//

// Checks for a number of operands between n and n+c (allows for c optional
// inputs)
ParseResult GenericIntrinsic::hasNInputs(unsigned n, unsigned c) {
  auto numOps = op.getNumOperands();
  unsigned m = n + c;
  if (numOps < n || numOps > m) {
    auto err = emitError() << " has " << numOps << " inputs instead of ";
    if (c == 0)
      err << n;
    else
      err << " between " << n << " and " << m;
    return failure();
  }
  return success();
}

// Accessor method for the number of inputs
unsigned GenericIntrinsic::getNumInputs() { return op.getNumOperands(); }

ParseResult GenericIntrinsic::hasNOutputElements(unsigned n) {
  auto b = getOutputBundle();
  if (!b)
    return emitError() << " missing output bundle";
  if (b.getType().getNumElements() != n)
    return emitError() << " has " << b.getType().getNumElements()
                       << " output elements instead of " << n;
  return success();
}

ParseResult GenericIntrinsic::hasNParam(unsigned n, unsigned c) {
  unsigned num = 0;
  if (op.getParameters())
    num = op.getParameters().size();
  if (num < n || num > n + c) {
    auto d = emitError() << " has " << num << " parameters instead of ";
    if (c == 0)
      d << n;
    else
      d << " between " << n << " and " << (n + c);
    return failure();
  }
  return success();
}

ParseResult GenericIntrinsic::namedParam(StringRef paramName, bool optional) {
  for (auto a : op.getParameters()) {
    auto param = cast<ParamDeclAttr>(a);
    if (param.getName().getValue() == paramName) {
      if (isa<StringAttr>(param.getValue()))
        return success();

      return emitError() << " has parameter '" << param.getName()
                         << "' which should be a string but is not";
    }
  }
  if (optional)
    return success();
  return emitError() << " is missing parameter " << paramName;
}

ParseResult GenericIntrinsic::namedIntParam(StringRef paramName,
                                            bool optional) {
  for (auto a : op.getParameters()) {
    auto param = cast<ParamDeclAttr>(a);
    if (param.getName().getValue() == paramName) {
      if (isa<IntegerAttr>(param.getValue()))
        return success();

      return emitError() << " has parameter '" << param.getName()
                         << "' which should be an integer but is not";
    }
  }
  if (optional)
    return success();
  return emitError() << " is missing parameter " << paramName;
}

//===----------------------------------------------------------------------===//
// IntrinsicOpConversion
//===----------------------------------------------------------------------===//

/// Conversion pattern adaptor dispatching via generic intrinsic name.
namespace {
class IntrinsicOpConversion final
    : public OpConversionPattern<GenericIntrinsicOp> {
public:
  using ConversionMapTy = IntrinsicLowerings::ConversionMapTy;

  IntrinsicOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                        const ConversionMapTy &conversions,
                        size_t &numConversions,
                        bool allowUnknownIntrinsics = false)
      : OpConversionPattern(typeConverter, context), conversions(conversions),
        numConversions(numConversions),
        allowUnknownIntrinsics(allowUnknownIntrinsics) {}

  LogicalResult
  matchAndRewrite(GenericIntrinsicOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto it = conversions.find(op.getIntrinsicAttr());
    if (it == conversions.end()) {
      if (!allowUnknownIntrinsics)
        return op.emitError("unknown intrinsic ") << op.getIntrinsicAttr();
      return failure();
    }

    auto &conv = *it->second;
    if (conv.check(GenericIntrinsic(op)))
      return failure();
    conv.convert(GenericIntrinsic(op), adaptor, rewriter);
    ++numConversions;
    return success();
  }

private:
  const ConversionMapTy &conversions;
  size_t &numConversions;
  const bool allowUnknownIntrinsics;
};
} // namespace

//===----------------------------------------------------------------------===//
// IntrinsicLowerings
//===----------------------------------------------------------------------===//

FailureOr<size_t> IntrinsicLowerings::lower(FModuleOp mod,
                                            bool allowUnknownIntrinsics) {

  ConversionTarget target(*context);

  target.markUnknownOpDynamicallyLegal([](Operation *op) { return true; });
  if (allowUnknownIntrinsics)
    target.addDynamicallyLegalOp<GenericIntrinsicOp>(
        [this](GenericIntrinsicOp op) {
          return !conversions.contains(op.getIntrinsicAttr());
        });
  else
    target.addIllegalOp<GenericIntrinsicOp>();

  // Automatically insert wires + connect for compatible FIRRTL base types.
  // For now, this is not customizable/extendable.
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });
  auto firrtlBaseTypeMaterialization =
      [](OpBuilder &builder, FIRRTLBaseType resultType, ValueRange inputs,
         Location loc) -> Value {
    if (inputs.size() != 1)
      return {};
    auto inputType = type_dyn_cast<FIRRTLBaseType>(inputs.front().getType());
    if (!inputType)
      return {};

    if (!areTypesEquivalent(resultType, inputType) ||
        !isTypeLarger(resultType, inputType))
      return {};

    auto w = builder.create<WireOp>(loc, resultType).getResult();
    emitConnect(builder, loc, w, inputs.front());
    return w;
  };
  // New result doesn't match? Add wire + connect.
  typeConverter.addSourceMaterialization(firrtlBaseTypeMaterialization);
  // New operand doesn't match? Add wire + connect.
  typeConverter.addTargetMaterialization(firrtlBaseTypeMaterialization);

  RewritePatternSet patterns(context);
  size_t count = 0;
  patterns.add<IntrinsicOpConversion>(typeConverter, context, conversions,
                                      count, allowUnknownIntrinsics);

  if (failed(mlir::applyPartialConversion(mod, target, std::move(patterns))))
    return failure();

  return count;
}

//===----------------------------------------------------------------------===//
// IntrinsicLoweringInterfaceCollection
//===----------------------------------------------------------------------===//

void IntrinsicLoweringInterfaceCollection::populateIntrinsicLowerings(
    IntrinsicLowerings &lowering) const {
  for (const IntrinsicLoweringDialectInterface &interface : *this)
    interface.populateIntrinsicLowerings(lowering);
}

//===----------------------------------------------------------------------===//
// FIRRTL intrinsic lowering converters
//===----------------------------------------------------------------------===//

namespace {

class CirctSizeofConverter : public IntrinsicOpConverter<SizeOfIntrinsicOp> {
public:
  using IntrinsicOpConverter::IntrinsicOpConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(1) || gi.sizedOutput<UIntType>(32) || gi.hasNParam(0);
  }
};

class CirctIsXConverter : public IntrinsicOpConverter<IsXIntrinsicOp> {
public:
  using IntrinsicOpConverter::IntrinsicOpConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(1) || gi.sizedOutput<UIntType>(1) || gi.hasNParam(0);
  }
};

class CirctPlusArgTestConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(0) || gi.sizedOutput<UIntType>(1) ||
           gi.namedParam("FORMAT") || gi.hasNParam(1);
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    rewriter.replaceOpWithNewOp<PlusArgsTestIntrinsicOp>(
        gi.op, gi.getParamValue<StringAttr>("FORMAT"));
  }
};

class CirctPlusArgValueConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNOutputElements(2) ||
           gi.sizedOutputElement<UIntType>(0, "found", 1) ||
           gi.hasOutputElement(1, "result") || gi.namedParam("FORMAT") ||
           gi.hasNParam(1);
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto bty = gi.getOutputBundle().getType();
    auto newop = rewriter.create<PlusArgsValueIntrinsicOp>(
        gi.op.getLoc(), bty.getElementTypePreservingConst(0),
        bty.getElementTypePreservingConst(1),
        gi.getParamValue<StringAttr>("FORMAT"));
    rewriter.replaceOpWithNewOp<BundleCreateOp>(
        gi.op, bty, ValueRange({newop.getFound(), newop.getResult()}));
  }
};

class CirctClockGateConverter
    : public IntrinsicOpConverter<ClockGateIntrinsicOp> {
public:
  using IntrinsicOpConverter::IntrinsicOpConverter;

  bool check(GenericIntrinsic gi) override {
    if (gi.op.getNumOperands() == 3) {
      return gi.typedInput<ClockType>(0) || gi.sizedInput<UIntType>(1, 1) ||
             gi.sizedInput<UIntType>(2, 1) || gi.typedOutput<ClockType>() ||
             gi.hasNParam(0);
    }
    if (gi.op.getNumOperands() == 2) {
      return gi.typedInput<ClockType>(0) || gi.sizedInput<UIntType>(1, 1) ||
             gi.typedOutput<ClockType>() || gi.hasNParam(0);
    }
    gi.emitError() << " has " << gi.op.getNumOperands()
                   << " ports instead of 3 or 4";
    return true;
  }
};

class CirctClockInverterConverter
    : public IntrinsicOpConverter<ClockInverterIntrinsicOp> {
public:
  using IntrinsicOpConverter::IntrinsicOpConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(1) || gi.typedInput<ClockType>(0) ||
           gi.typedOutput<ClockType>() || gi.hasNParam(0);
  }
};

class CirctClockDividerConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(1) || gi.typedInput<ClockType>(0) ||
           gi.typedOutput<ClockType>() || gi.namedIntParam("POW_2") ||
           gi.hasNParam(1);
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto pow2 =
        gi.getParamValue<IntegerAttr>("POW_2").getValue().getZExtValue();

    auto pow2Attr = rewriter.getI64IntegerAttr(pow2);

    rewriter.replaceOpWithNewOp<ClockDividerIntrinsicOp>(
        gi.op, adaptor.getOperands()[0], pow2Attr);
  }
};

template <typename OpTy>
class CirctLTLBinaryConverter : public IntrinsicOpConverter<OpTy> {
public:
  using IntrinsicOpConverter<OpTy>::IntrinsicOpConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(2) || gi.sizedInput<UIntType>(0, 1) ||
           gi.sizedInput<UIntType>(1, 1) || gi.sizedOutput<UIntType>(1) ||
           gi.hasNParam(0);
  }
};

template <typename OpTy>
class CirctLTLUnaryConverter : public IntrinsicOpConverter<OpTy> {
public:
  using IntrinsicOpConverter<OpTy>::IntrinsicOpConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(1) || gi.sizedInput<UIntType>(0, 1) ||
           gi.sizedOutput<UIntType>(1) || gi.hasNParam(0);
  }
};

class CirctLTLDelayConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(1) || gi.sizedInput<UIntType>(0, 1) ||
           gi.sizedOutput<UIntType>(1) || gi.namedIntParam("delay") ||
           gi.namedIntParam("length", true) || gi.hasNParam(1, 1);
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto getI64Attr = [&](IntegerAttr val) {
      if (!val)
        return IntegerAttr();
      return rewriter.getI64IntegerAttr(val.getValue().getZExtValue());
    };
    auto delay = getI64Attr(gi.getParamValue<IntegerAttr>("delay"));
    auto length = getI64Attr(gi.getParamValue<IntegerAttr>("length"));
    rewriter.replaceOpWithNewOp<LTLDelayIntrinsicOp>(
        gi.op, gi.op.getResultTypes(), adaptor.getOperands()[0], delay, length);
  }
};

class CirctLTLClockConverter
    : public IntrinsicOpConverter<LTLClockIntrinsicOp> {
public:
  using IntrinsicOpConverter::IntrinsicOpConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(2) || gi.sizedInput<UIntType>(0, 1) ||
           gi.typedInput<ClockType>(1) || gi.sizedOutput<UIntType>(1) ||
           gi.hasNParam(0);
  }
};

class CirctLTLRepeatConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(1) || gi.sizedInput<UIntType>(0, 1) ||
           gi.sizedOutput<UIntType>(1) || gi.namedIntParam("base") ||
           gi.namedIntParam("more", true) || gi.hasNParam(1, 1);
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto getI64Attr = [&](IntegerAttr val) {
      if (!val)
        return IntegerAttr();
      return rewriter.getI64IntegerAttr(val.getValue().getZExtValue());
    };
    auto base = getI64Attr(gi.getParamValue<IntegerAttr>("base"));
    auto more = getI64Attr(gi.getParamValue<IntegerAttr>("more"));
    rewriter.replaceOpWithNewOp<LTLRepeatIntrinsicOp>(
        gi.op, gi.op.getResultTypes(), adaptor.getOperands()[0], base, more);
  }
};

class CirctLTLGoToRepeatConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(1) || gi.sizedInput<UIntType>(0, 1) ||
           gi.sizedOutput<UIntType>(1) || gi.namedIntParam("base") ||
           gi.namedIntParam("more") || gi.hasNParam(1, 1);
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto getI64Attr = [&](IntegerAttr val) {
      if (!val)
        return IntegerAttr();
      return rewriter.getI64IntegerAttr(val.getValue().getZExtValue());
    };
    auto base = getI64Attr(gi.getParamValue<IntegerAttr>("base"));
    auto more = getI64Attr(gi.getParamValue<IntegerAttr>("more"));
    rewriter.replaceOpWithNewOp<LTLGoToRepeatIntrinsicOp>(
        gi.op, gi.op.getResultTypes(), adaptor.getOperands()[0], base, more);
  }
};

class CirctLTLNonConsecutiveRepeatConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(1) || gi.sizedInput<UIntType>(0, 1) ||
           gi.sizedOutput<UIntType>(1) || gi.namedIntParam("base") ||
           gi.namedIntParam("more") || gi.hasNParam(1, 1);
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto getI64Attr = [&](IntegerAttr val) {
      if (!val)
        return IntegerAttr();
      return rewriter.getI64IntegerAttr(val.getValue().getZExtValue());
    };
    auto base = getI64Attr(gi.getParamValue<IntegerAttr>("base"));
    auto more = getI64Attr(gi.getParamValue<IntegerAttr>("more"));
    rewriter.replaceOpWithNewOp<LTLNonConsecutiveRepeatIntrinsicOp>(
        gi.op, gi.op.getResultTypes(), adaptor.getOperands()[0], base, more);
  }
};

template <class Op>
class CirctVerifConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(1, 2) || gi.sizedInput<UIntType>(0, 1) ||
           gi.namedParam("label", true) || gi.hasNParam(0, 1) ||
           gi.hasNoOutput();
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto label = gi.getParamValue<StringAttr>("label");
    auto operands = adaptor.getOperands();

    // Check if an enable was provided
    Value enable;
    if (gi.getNumInputs() == 2)
      enable = operands[1];

    rewriter.replaceOpWithNewOp<Op>(gi.op, operands[0], enable, label);
  }
};

class CirctMux2CellConverter : public IntrinsicConverter {
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(3) || gi.typedInput<UIntType>(0) || gi.hasNParam(0) ||
           gi.hasOutput();
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto operands = adaptor.getOperands();
    rewriter.replaceOpWithNewOp<Mux2CellIntrinsicOp>(gi.op, operands[0],
                                                     operands[1], operands[2]);
  }
};

class CirctMux4CellConverter : public IntrinsicConverter {
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(5) || gi.typedInput<UIntType>(0) || gi.hasNParam(0) ||
           gi.hasOutput();
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto operands = adaptor.getOperands();
    rewriter.replaceOpWithNewOp<Mux4CellIntrinsicOp>(
        gi.op, operands[0], operands[1], operands[2], operands[3], operands[4]);
  }
};

class CirctHasBeenResetConverter
    : public IntrinsicOpConverter<HasBeenResetIntrinsicOp> {
public:
  using IntrinsicOpConverter::IntrinsicOpConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(2) || gi.typedInput<ClockType>(0) ||
           gi.hasResetInput(1) || gi.sizedOutput<UIntType>(1) ||
           gi.hasNParam(0);
  }
};

class CirctProbeConverter : public IntrinsicOpConverter<FPGAProbeIntrinsicOp> {
public:
  using IntrinsicOpConverter::IntrinsicOpConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(2) || gi.typedInput<ClockType>(1) || gi.hasNParam(0) ||
           gi.hasNoOutput();
  }
};

template <class OpTy, bool ifElseFatal = false>
class CirctAssertConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.typedInput<ClockType>(0) || gi.sizedInput<UIntType>(1, 1) ||
           gi.sizedInput<UIntType>(2, 1) ||
           gi.namedParam("format", /*optional=*/true) ||
           gi.namedParam("label", /*optional=*/true) ||
           gi.namedParam("guards", /*optional=*/true) || gi.hasNParam(0, 3) ||
           gi.hasNoOutput();
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto format = gi.getParamValue<StringAttr>("format");
    auto label = gi.getParamValue<StringAttr>("label");
    auto guards = gi.getParamValue<StringAttr>("guards");

    auto clock = adaptor.getOperands()[0];
    auto predicate = adaptor.getOperands()[1];
    auto enable = adaptor.getOperands()[2];

    auto substitutions = adaptor.getOperands().drop_front(3);
    auto name = label ? label.strref() : "";
    // Message is not optional, so provide empty string if not present.
    auto message = format ? format : rewriter.getStringAttr("");
    auto op = rewriter.template replaceOpWithNewOp<OpTy>(
        gi.op, clock, predicate, enable, message, substitutions, name,
        /*isConcurrent=*/true);
    if (guards) {
      SmallVector<StringRef> guardStrings;
      guards.strref().split(guardStrings, ';', /*MaxSplit=*/-1,
                            /*KeepEmpty=*/false);
      rewriter.startOpModification(op);
      op->setAttr("guards", rewriter.getStrArrayAttr(guardStrings));
      rewriter.finalizeOpModification(op);
    }

    if constexpr (ifElseFatal) {
      rewriter.startOpModification(op);
      op->setAttr("format", rewriter.getStringAttr("ifElseFatal"));
      rewriter.finalizeOpModification(op);
    }
  }
};

class CirctCoverConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(3) || gi.hasNoOutput() ||
           gi.typedInput<ClockType>(0) || gi.sizedInput<UIntType>(1, 1) ||
           gi.sizedInput<UIntType>(2, 1) ||
           gi.namedParam("label", /*optional=*/true) ||
           gi.namedParam("guards", /*optional=*/true) || gi.hasNParam(0, 2);
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto label = gi.getParamValue<StringAttr>("label");
    auto guards = gi.getParamValue<StringAttr>("guards");

    auto clock = adaptor.getOperands()[0];
    auto predicate = adaptor.getOperands()[1];
    auto enable = adaptor.getOperands()[2];

    auto name = label ? label.strref() : "";
    // Empty message string for cover, only 'name' / label.
    auto message = rewriter.getStringAttr("");
    auto op = rewriter.replaceOpWithNewOp<CoverOp>(
        gi.op, clock, predicate, enable, message, ValueRange{}, name,
        /*isConcurrent=*/true);
    if (guards) {
      SmallVector<StringRef> guardStrings;
      guards.strref().split(guardStrings, ';', /*MaxSplit=*/-1,
                            /*KeepEmpty=*/false);
      rewriter.startOpModification(op);
      op->setAttr("guards", rewriter.getStrArrayAttr(guardStrings));
      rewriter.finalizeOpModification(op);
    }
  }
};

class CirctUnclockedAssumeConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.sizedInput<UIntType>(0, 1) || gi.sizedInput<UIntType>(1, 1) ||
           gi.namedParam("format", /*optional=*/true) ||
           gi.namedParam("label", /*optional=*/true) ||
           gi.namedParam("guards", /*optional=*/true) || gi.hasNParam(0, 3) ||
           gi.hasNoOutput();
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto format = gi.getParamValue<StringAttr>("format");
    auto label = gi.getParamValue<StringAttr>("label");
    auto guards = gi.getParamValue<StringAttr>("guards");

    auto predicate = adaptor.getOperands()[0];
    auto enable = adaptor.getOperands()[1];

    auto substitutions = adaptor.getOperands().drop_front(2);
    auto name = label ? label.strref() : "";
    // Message is not optional, so provide empty string if not present.
    auto message = format ? format : rewriter.getStringAttr("");
    auto op = rewriter.template replaceOpWithNewOp<UnclockedAssumeIntrinsicOp>(
        gi.op, predicate, enable, message, substitutions, name);
    if (guards) {
      SmallVector<StringRef> guardStrings;
      guards.strref().split(guardStrings, ';', /*MaxSplit=*/-1,
                            /*KeepEmpty=*/false);
      rewriter.startOpModification(op);
      op->setAttr("guards", rewriter.getStrArrayAttr(guardStrings));
      rewriter.finalizeOpModification(op);
    }
  }
};

class CirctDPICallConverter : public IntrinsicConverter {
  static bool getIsClocked(GenericIntrinsic gi) {
    return !gi.getParamValue<IntegerAttr>("isClocked").getValue().isZero();
  }

public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    if (gi.hasNParam(2, 2) || gi.namedIntParam("isClocked") ||
        gi.namedParam("functionName") ||
        gi.namedParam("inputNames", /*optional=*/true) ||
        gi.namedParam("outputName", /*optional=*/true))
      return true;
    auto isClocked = getIsClocked(gi);
    // If clocked, the first operand must be a clock.
    if (isClocked && gi.typedInput<ClockType>(0))
      return true;
    // Enable must be UInt<1>.
    if (gi.sizedInput<UIntType>(isClocked, 1))
      return true;

    return false;
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto isClocked = getIsClocked(gi);
    auto functionName = gi.getParamValue<StringAttr>("functionName");
    ArrayAttr inputNamesStrArray;
    StringAttr outputStr = gi.getParamValue<StringAttr>("outputName");
    if (auto inputNames = gi.getParamValue<StringAttr>("inputNames")) {
      SmallVector<StringRef> inputNamesTemporary;
      inputNames.strref().split(inputNamesTemporary, ';', /*MaxSplit=*/-1,
                                /*KeepEmpty=*/false);
      inputNamesStrArray = rewriter.getStrArrayAttr(inputNamesTemporary);
    }
    // Clock and enable are optional.
    Value clock = isClocked ? adaptor.getOperands()[0] : Value();
    Value enable = adaptor.getOperands()[static_cast<size_t>(isClocked)];

    auto inputs =
        adaptor.getOperands().drop_front(static_cast<size_t>(isClocked) + 1);

    rewriter.replaceOpWithNewOp<DPICallIntrinsicOp>(
        gi.op, gi.op.getResultTypes(), functionName, inputNamesStrArray,
        outputStr, clock, enable, inputs);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// FIRRTL intrinsic lowering dialect interface
//===----------------------------------------------------------------------===//

void FIRRTLIntrinsicLoweringDialectInterface::populateIntrinsicLowerings(
    IntrinsicLowerings &lowering) const {
  lowering.add<CirctSizeofConverter>("circt.sizeof", "circt_sizeof");
  lowering.add<CirctIsXConverter>("circt.isX", "circt_isX");
  lowering.add<CirctPlusArgTestConverter>("circt.plusargs.test",
                                          "circt_plusargs_test");
  lowering.add<CirctPlusArgValueConverter>("circt.plusargs.value",
                                           "circt_plusargs_value");
  lowering.add<CirctClockGateConverter>("circt.clock_gate", "circt_clock_gate");
  lowering.add<CirctClockInverterConverter>("circt.clock_inv",
                                            "circt_clock_inv");
  lowering.add<CirctClockDividerConverter>("circt.clock_div",
                                           "circt_clock_div");
  lowering.add<CirctLTLBinaryConverter<LTLAndIntrinsicOp>>("circt.ltl.and",
                                                           "circt_ltl_and");
  lowering.add<CirctLTLBinaryConverter<LTLOrIntrinsicOp>>("circt.ltl.or",
                                                          "circt_ltl_or");
  lowering.add<CirctLTLBinaryConverter<LTLIntersectIntrinsicOp>>(
      "circt.ltl.intersect", "circt_ltl_intersect");
  lowering.add<CirctLTLBinaryConverter<LTLConcatIntrinsicOp>>(
      "circt.ltl.concat", "circt_ltl_concat");
  lowering.add<CirctLTLBinaryConverter<LTLImplicationIntrinsicOp>>(
      "circt.ltl.implication", "circt_ltl_implication");
  lowering.add<CirctLTLBinaryConverter<LTLUntilIntrinsicOp>>("circt.ltl.until",
                                                             "circt_ltl_until");
  lowering.add<CirctLTLUnaryConverter<LTLNotIntrinsicOp>>("circt.ltl.not",
                                                          "circt_ltl_not");
  lowering.add<CirctLTLUnaryConverter<LTLEventuallyIntrinsicOp>>(
      "circt.ltl.eventually", "circt_ltl_eventually");

  lowering.add<CirctLTLDelayConverter>("circt.ltl.delay", "circt_ltl_delay");
  lowering.add<CirctLTLRepeatConverter>("circt.ltl.repeat", "circt_ltl_repeat");
  lowering.add<CirctLTLGoToRepeatConverter>("circt.ltl.goto_repeat",
                                            "circt_ltl_goto_repeat");
  lowering.add<CirctLTLNonConsecutiveRepeatConverter>(
      "circt.ltl.non_consecutive_repeat", "circt_ltl_non_consecutive_repeat");
  lowering.add<CirctLTLClockConverter>("circt.ltl.clock", "circt_ltl_clock");

  lowering.add<CirctVerifConverter<VerifAssertIntrinsicOp>>(
      "circt.verif.assert", "circt_verif_assert");
  lowering.add<CirctVerifConverter<VerifAssumeIntrinsicOp>>(
      "circt.verif.assume", "circt_verif_assume");
  lowering.add<CirctVerifConverter<VerifCoverIntrinsicOp>>("circt.verif.cover",
                                                           "circt_verif_cover");
  lowering.add<CirctMux2CellConverter>("circt.mux2cell", "circt_mux2cell");
  lowering.add<CirctMux4CellConverter>("circt.mux4cell", "circt_mux4cell");
  lowering.add<CirctHasBeenResetConverter>("circt.has_been_reset",
                                           "circt_has_been_reset");
  lowering.add<CirctProbeConverter>("circt.fpga_probe", "circt_fpga_probe");
  lowering.add<CirctAssertConverter<AssertOp>>("circt.chisel_assert",
                                               "circt_chisel_assert");
  lowering.add<CirctAssertConverter<AssertOp, /*ifElseFatal=*/true>>(
      "circt.chisel_ifelsefatal", "circt_chisel_ifelsefatal");
  lowering.add<CirctAssertConverter<AssumeOp>>("circt.chisel_assume",
                                               "circt_chisel_assume");
  lowering.add<CirctCoverConverter>("circt.chisel_cover", "circt_chisel_cover");
  lowering.add<CirctUnclockedAssumeConverter>("circt.unclocked_assume",
                                              "circt_unclocked_assume");
  lowering.add<CirctDPICallConverter>("circt.dpi_call", "circt_dpi_call");
}
