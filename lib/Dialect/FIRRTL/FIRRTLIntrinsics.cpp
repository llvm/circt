//===- FIRRTLIntrinsics.cpp - Lower Intrinsics ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLIntrinsics.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace firrtl;

ParseResult GenericIntrinsic::hasNInputs(unsigned n) {
  if (op.getNumOperands() != n)
    return emitError() << " has " << op.getNumOperands()
                       << " inputs instead of " << n;
  return success();
}

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
    if (param.getName().getValue().equals(paramName)) {
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
    if (param.getName().getValue().equals(paramName)) {
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

/// Conversion pattern adaptor dispatching via generic intrinsic name.
class IntrinsicOpConversion final
    : public OpConversionPattern<GenericIntrinsicOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  using ConversionMapTy = IntrinsicLowerings::ConversionMapTy;

  IntrinsicOpConversion(MLIRContext *context,
                        const ConversionMapTy &conversions)
      : OpConversionPattern(context), conversions(conversions) {}

  LogicalResult
  matchAndRewrite(GenericIntrinsicOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto it = conversions.find(op.getIntrinsicAttr());
    if (it == conversions.end())
      return failure();

    auto &conv = *it->second;
    if (conv.check(GenericIntrinsic(op)))
      return failure();
    conv.convert(GenericIntrinsic(op), adaptor, rewriter);
    return success();
  }

private:
  const ConversionMapTy &conversions;
};

LogicalResult IntrinsicLowerings::lower(FModuleOp mod,
                                        bool allowUnknownIntrinsics) {

  ConversionTarget target(*context);

  target.addLegalDialect<FIRRTLDialect>();
  if (allowUnknownIntrinsics)
    target.addDynamicallyLegalOp<GenericIntrinsicOp>(
        [this](GenericIntrinsicOp op) {
          return !conversions.count(op.getIntrinsicAttr());
        });
  else
    target.addIllegalOp<GenericIntrinsicOp>();

  RewritePatternSet patterns(context);
  patterns.add<IntrinsicOpConversion>(context, conversions);

  return mlir::applyPartialConversion(mod, target, std::move(patterns));
}
