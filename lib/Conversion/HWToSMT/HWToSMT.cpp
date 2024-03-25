//===- HWToSMT.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/HWToSMT.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SMT/SMTOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTHWTOSMT
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace circt;
using namespace hw;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
/// Lower a hw::ConstantOp operation to smt::BVConstantOp
struct HWConstantOpConversion : OpConversionPattern<ConstantOp> {
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<smt::BVConstantOp>(op, adaptor.getValue());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert HW to SMT pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertHWToSMTPass
    : public impl::ConvertHWToSMTBase<ConvertHWToSMTPass> {
  void runOnOperation() override;
};
} // namespace

void circt::populateHWToSMTTypeConverter(TypeConverter &converter) {
  // The semantics of the builtin integer at the CIRCT core level is currently
  // not very well defined. It is used for two-valued, four-valued, and possible
  // other multi-valued logic. Here, we interpret it as two-valued for now.
  // From a formal perspective, CIRCT would ideally define its own types for
  // two-valued, four-valued, nine-valued (etc.) logic each. In MLIR upstream
  // the integer type also carries poison information (which we don't have in
  // CIRCT?).
  converter.addConversion([](IntegerType type) -> std::optional<Type> {
    if (type.getWidth() <= 0)
      return std::nullopt;
    return smt::BitVectorType::get(type.getContext(), type.getWidth());
  });

  // Default target materialization to convert from illegal types to legal
  // types, e.g., at the boundary of an inlined child block.
  converter.addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                                         ValueRange inputs,
                                         Location loc) -> std::optional<Value> {
    return builder
        .create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs)
        ->getResult(0);
  });

  // Convert a 'smt.bool'-typed value to a 'smt.bv<N>'-typed value
  converter.addTargetMaterialization(
      [&](OpBuilder &builder, smt::BitVectorType resultType, ValueRange inputs,
          Location loc) -> std::optional<Value> {
        if (inputs.size() != 1)
          return std::nullopt;

        if (!isa<smt::BoolType>(inputs[0].getType()))
          return std::nullopt;

        unsigned width = resultType.getWidth();
        Value constZero = builder.create<smt::BVConstantOp>(loc, 0, width);
        Value constOne = builder.create<smt::BVConstantOp>(loc, 1, width);
        return builder.create<smt::IteOp>(loc, inputs[0], constOne, constZero);
      });

  // Convert a 'smt.bv<1>'-typed value to a 'smt.bool'-typed value
  converter.addTargetMaterialization(
      [&](OpBuilder &builder, smt::BoolType resultType, ValueRange inputs,
          Location loc) -> std::optional<Value> {
        if (inputs.size() != 1)
          return std::nullopt;

        auto bvType = dyn_cast<smt::BitVectorType>(inputs[0].getType());
        if (!bvType || bvType.getWidth() != 1)
          return std::nullopt;

        Value constOne = builder.create<smt::BVConstantOp>(loc, 1, 1);
        return builder.create<smt::EqOp>(loc, inputs[0], constOne);
      });

  // Default source materialization to convert from illegal types to legal
  // types, e.g., at the boundary of an inlined child block.
  converter.addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                                         ValueRange inputs,
                                         Location loc) -> std::optional<Value> {
    return builder
        .create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs)
        ->getResult(0);
  });
}

void circt::populateHWToSMTConversionPatterns(TypeConverter &converter,
                                              RewritePatternSet &patterns) {
  patterns.add<HWConstantOpConversion>(converter, patterns.getContext());
}

void ConvertHWToSMTPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addIllegalDialect<hw::HWDialect>();
  target.addLegalDialect<smt::SMTDialect>();

  RewritePatternSet patterns(&getContext());
  TypeConverter converter;
  populateHWToSMTTypeConverter(converter);
  populateHWToSMTConversionPatterns(converter, patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}
