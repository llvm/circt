//===- CombToSystemC.cpp - Comb To SystemC Conversion Pass ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main Comb to SystemC Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CombToSystemC.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SystemC/SystemCOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace comb;
using namespace systemc;

//===----------------------------------------------------------------------===//
// Operation Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {

/// This works on each HW module, creates corresponding SystemC modules, moves
/// the body of the module into the new SystemC module by splitting up the body
/// into field declarations, initializations done in a newly added systemc.ctor,
/// and internal methods to be registered in the constructor.
struct ConvertAdd : public OpConversionPattern<comb::AddOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comb::AddOp addOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (addOp.getInputs().size() < 2)
      return failure();

    Value sum = rewriter.create<systemc::AddOp>(
        addOp->getLoc(), adaptor.getInputs()[0], adaptor.getInputs()[1]);
    for (size_t i = 2, e = adaptor.getInputs().size(); i < e; ++i) {
      sum = rewriter.create<systemc::AddOp>(addOp->getLoc(), sum,
                                            adaptor.getInputs()[i]);
    }

    rewriter.replaceOp(addOp, sum);

    return success();
  }
};

struct ConvertSub : public OpConversionPattern<comb::SubOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comb::SubOp addOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value sum = rewriter.create<systemc::SubOp>(
        addOp->getLoc(), adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(addOp, sum);

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Conversion Infrastructure
//===----------------------------------------------------------------------===//

static void populateLegality(ConversionTarget &target) {
  target.addIllegalDialect<hw::HWDialect>();
  target.addIllegalDialect<comb::CombDialect>();
  target.addLegalDialect<mlir::BuiltinDialect>();
  target.addLegalDialect<systemc::SystemCDialect>();
  target.addLegalOp<hw::ConstantOp>();
}

static void populateOpConversion(RewritePatternSet &patterns,
                                 TypeConverter &typeConverter) {
  patterns.add<ConvertAdd, ConvertSub>(typeConverter, patterns.getContext());
}

static void populateTypeConversion(TypeConverter &converter) {
  converter.addConversion([](IntegerType type) -> Type {
    auto bw = type.getIntOrFloatBitWidth();
    if (bw == 1)
      return type;
    if (bw <= 64 && type.isSigned())
      return systemc::IntType::get(type.getContext(), bw);
    if (bw <= 64)
      return UIntType::get(type.getContext(), bw);
    if (bw <= 512 && type.isSigned())
      return BigIntType::get(type.getContext(), bw);
    if (bw <= 512)
      return BigUIntType::get(type.getContext(), bw);

    return BitVectorType::get(type.getContext(), bw);
  });
  converter.addConversion([](UIntType type) { return type; });
  converter.addSourceMaterialization(
      [](OpBuilder &builder, Type type, ValueRange values, Location loc) {
        assert(values.size() == 1);
        llvm::errs() << "derived:\n";
        if (type.isa<UIntType>())
          llvm::errs() << type << "\n";
        llvm::errs() << "base:\n";
        if (type.isa<UIntBaseType>())
          llvm::errs() << type << "\n";

        auto op = builder.create<ConvertOp>(loc, type, values[0]);
        return op.getResult();
      });
  converter.addTargetMaterialization(
      [](OpBuilder &builder, Type type, ValueRange values, Location loc) {
        assert(values.size() == 1);
        llvm::errs() << "derived:\n";
        if (type.isa<UIntType>())
          llvm::errs() << type << "\n";
        llvm::errs() << "base:\n";
        if (type.isa<UIntBaseType>())
          llvm::errs() << type << "\n";

        auto op = builder.create<ConvertOp>(loc, type, values[0]);
        return op.getResult();
      });
}

//===----------------------------------------------------------------------===//
// Comb to SystemC Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct CombToSystemCPass : public ConvertCombToSystemCBase<CombToSystemCPass> {
  void runOnOperation() override;
};
} // namespace

/// Create a Comb to SystemC dialects conversion pass.
std::unique_ptr<OperationPass<ModuleOp>>
circt::createConvertCombToSystemCPass() {
  return std::make_unique<CombToSystemCPass>();
}

/// This is the main entrypoint for the Comb to SystemC conversion pass.
void CombToSystemCPass::runOnOperation() {
  MLIRContext &context = getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(context);
  TypeConverter typeConverter;
  RewritePatternSet patterns(&context);
  populateLegality(target);
  populateTypeConversion(typeConverter);
  populateOpConversion(patterns, typeConverter);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
