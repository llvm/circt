//===- VerifToSMT.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/VerifToSMT.h"
#include "circt/Conversion/HWToSMT.h"
#include "circt/Dialect/SMT/SMTOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTVERIFTOSMT
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace hw;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
/// Lower a verif::AssertOp operation with an i1 operand to a smt::AssertOp,
/// negated to check for unsatisfiability.
struct VerifAssertOpConversion : OpConversionPattern<verif::AssertOp> {
  using OpConversionPattern<verif::AssertOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(verif::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value cond = typeConverter->materializeTargetConversion(
        rewriter, op.getLoc(), smt::BoolType::get(getContext()),
        adaptor.getProperty());
    Value notCond = rewriter.create<smt::NotOp>(op.getLoc(), cond);
    rewriter.replaceOpWithNewOp<smt::AssertOp>(op, notCond);
    return success();
  }
};

/// Lower a verif::AssumeOp operation with an i1 operand to a smt::AssertOp
struct VerifAssumeOpConversion : OpConversionPattern<verif::AssumeOp> {
  using OpConversionPattern<verif::AssumeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(verif::AssumeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value cond = typeConverter->materializeTargetConversion(
        rewriter, op.getLoc(), smt::BoolType::get(getContext()),
        adaptor.getProperty());
    rewriter.replaceOpWithNewOp<smt::AssertOp>(op, cond);
    return success();
  }
};

// Fail to convert unsupported verif ops to avoid silent failure
struct VerifCoverOpConversion : OpConversionPattern<verif::CoverOp> {
  using OpConversionPattern<verif::CoverOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(verif::CoverOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return op.emitError("Conversion of CoverOps to SMT not yet supported");
  };
};

struct VerifClockedAssertOpConversion
    : OpConversionPattern<verif::ClockedAssertOp> {
  using OpConversionPattern<verif::ClockedAssertOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(verif::ClockedAssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return op.emitError(
        "Conversion of ClockedAssertOps to SMT not yet supported");
  };
};

struct VerifClockedCoverOpConversion
    : OpConversionPattern<verif::ClockedCoverOp> {
  using OpConversionPattern<verif::ClockedCoverOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(verif::ClockedCoverOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return op.emitError(
        "Conversion of ClockedCoverOps to SMT not yet supported");
  };
};

struct VerifClockedAssumeOpConversion
    : OpConversionPattern<verif::ClockedAssumeOp> {
  using OpConversionPattern<verif::ClockedAssumeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(verif::ClockedAssumeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return op.emitError(
        "Conversion of ClockedAssumeOps to SMT not yet supported");
  };
};

/// Lower a verif::LecOp operation to a miter circuit encoded in SMT.
/// More information on miter circuits can be found, e.g., in this paper:
/// Brand, D., 1993, November. Verification of large synthesized designs. In
/// Proceedings of 1993 International Conference on Computer Aided Design
/// (ICCAD) (pp. 534-537). IEEE.
struct LogicEquivalenceCheckingOpConversion
    : OpConversionPattern<verif::LogicEquivalenceCheckingOp> {
  using OpConversionPattern<
      verif::LogicEquivalenceCheckingOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(verif::LogicEquivalenceCheckingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *firstOutputs = adaptor.getFirstCircuit().front().getTerminator();
    auto *secondOutputs = adaptor.getSecondCircuit().front().getTerminator();

    if (firstOutputs->getNumOperands() == 0) {
      // Trivially equivalent
      Value trueVal =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true));
      rewriter.replaceOp(op, trueVal);
      return success();
    }

    smt::SolverOp solver =
        rewriter.create<smt::SolverOp>(loc, rewriter.getI1Type(), ValueRange{});
    rewriter.createBlock(&solver.getBodyRegion());

    // First, convert the block arguments of the miter bodies.
    if (failed(rewriter.convertRegionTypes(&adaptor.getFirstCircuit(),
                                           *typeConverter)))
      return failure();
    if (failed(rewriter.convertRegionTypes(&adaptor.getSecondCircuit(),
                                           *typeConverter)))
      return failure();

    // Second, create the symbolic values we replace the block arguments with
    SmallVector<Value> inputs;
    for (auto arg : adaptor.getFirstCircuit().getArguments())
      inputs.push_back(rewriter.create<smt::DeclareFunOp>(loc, arg.getType()));

    // Third, inline the blocks
    // Note: the argument value replacement does not happen immediately, but
    // only after all the operations are already legalized.
    // Also, it has to be ensured that the original argument type and the type
    // of the value with which is is to be replaced match. The value is looked
    // up (transitively) in the replacement map at the time the replacement
    // pattern is committed.
    rewriter.mergeBlocks(&adaptor.getFirstCircuit().front(), solver.getBody(),
                         inputs);
    rewriter.mergeBlocks(&adaptor.getSecondCircuit().front(), solver.getBody(),
                         inputs);
    rewriter.setInsertionPointToEnd(solver.getBody());

    // Fourth, convert the yielded values back to the source type system (since
    // the operations of the inlined blocks will be converted by other patterns
    // later on and we should make sure the IR is well-typed after each pattern
    // application), and build the 'assert'.
    SmallVector<Value> outputsDifferent;
    for (auto [out1, out2] :
         llvm::zip(firstOutputs->getOperands(), secondOutputs->getOperands())) {
      Value o1 = typeConverter->materializeTargetConversion(
          rewriter, loc, typeConverter->convertType(out1.getType()), out1);
      Value o2 = typeConverter->materializeTargetConversion(
          rewriter, loc, typeConverter->convertType(out1.getType()), out2);
      outputsDifferent.emplace_back(
          rewriter.create<smt::DistinctOp>(loc, o1, o2));
    }

    rewriter.eraseOp(firstOutputs);
    rewriter.eraseOp(secondOutputs);

    Value toAssert;
    if (outputsDifferent.size() == 1)
      toAssert = outputsDifferent[0];
    else
      toAssert = rewriter.create<smt::OrOp>(loc, outputsDifferent);

    rewriter.create<smt::AssertOp>(loc, toAssert);

    // Fifth, check for satisfiablility and report the result back.
    Value falseVal =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(false));
    Value trueVal =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true));
    auto checkOp = rewriter.create<smt::CheckOp>(loc, rewriter.getI1Type());
    rewriter.createBlock(&checkOp.getSatRegion());
    rewriter.create<smt::YieldOp>(loc, falseVal);
    rewriter.createBlock(&checkOp.getUnknownRegion());
    rewriter.create<smt::YieldOp>(loc, falseVal);
    rewriter.createBlock(&checkOp.getUnsatRegion());
    rewriter.create<smt::YieldOp>(loc, trueVal);
    rewriter.setInsertionPointAfter(checkOp);
    rewriter.create<smt::YieldOp>(loc, checkOp->getResults());

    rewriter.replaceOp(op, solver->getResults());
    return success();
  }
};

/// Lower a verif::BMCOp operation to an MLIR program that performs the bounded
/// model check
struct VerifBMCOpConversion : OpConversionPattern<verif::BMCOp> {
  using OpConversionPattern<verif::BMCOp>::OpConversionPattern;

  VerifBMCOpConversion(TypeConverter &converter, MLIRContext *context,
                       Namespace &names)
      : OpConversionPattern(converter, context), names(names) {}

  LogicalResult
  matchAndRewrite(verif::BMCOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Type> oldLoopInputTy(op.getLoop().getArgumentTypes());
    SmallVector<Type> oldCircuitInputTy(op.getCircuit().getArgumentTypes());
    SmallVector<Type> loopInputTy, circuitInputTy, initOutputTy, loopOutputTy,
        circuitOutputTy;
    if (failed(typeConverter->convertTypes(oldLoopInputTy, loopInputTy)))
      return failure();
    if (failed(typeConverter->convertTypes(oldCircuitInputTy, circuitInputTy)))
      return failure();
    if (failed(typeConverter->convertTypes(
            op.getInit().front().back().getOperandTypes(), initOutputTy)))
      return failure();
    // loop and init should have same output types
    loopOutputTy = initOutputTy;
    if (failed(typeConverter->convertTypes(
            op.getCircuit().front().back().getOperandTypes(), circuitOutputTy)))
      return failure();
    if (failed(rewriter.convertRegionTypes(&op.getInit(), *typeConverter)))
      return failure();
    if (failed(rewriter.convertRegionTypes(&op.getLoop(), *typeConverter)))
      return failure();
    if (failed(rewriter.convertRegionTypes(&op.getCircuit(), *typeConverter)))
      return failure();

    unsigned numRegs =
        cast<IntegerAttr>(op->getAttr("num_regs")).getValue().getZExtValue();

    auto initFuncTy = rewriter.getFunctionType({}, initOutputTy);
    auto loopFuncTy = rewriter.getFunctionType(loopInputTy, loopOutputTy);
    auto circuitFuncTy =
        rewriter.getFunctionType(circuitInputTy, circuitOutputTy);

    func::FuncOp initFuncOp, loopFuncOp, circuitFuncOp;

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(
          op->getParentOfType<ModuleOp>().getBody());
      initFuncOp = rewriter.create<func::FuncOp>(loc, names.newName("bmc_init"),
                                                 initFuncTy);
      rewriter.inlineRegionBefore(op.getInit(), initFuncOp.getFunctionBody(),
                                  initFuncOp.end());
      loopFuncOp = rewriter.create<func::FuncOp>(loc, names.newName("bmc_loop"),
                                                 loopFuncTy);
      rewriter.inlineRegionBefore(op.getLoop(), loopFuncOp.getFunctionBody(),
                                  loopFuncOp.end());
      circuitFuncOp = rewriter.create<func::FuncOp>(
          loc, names.newName("bmc_circuit"), circuitFuncTy);
      rewriter.inlineRegionBefore(op.getCircuit(),
                                  circuitFuncOp.getFunctionBody(),
                                  circuitFuncOp.end());
      auto funcOps = {&initFuncOp, &loopFuncOp, &circuitFuncOp};
      auto outputTys = {initOutputTy, loopOutputTy, circuitOutputTy};
      for (auto [funcOp, outputTy] : llvm::zip(funcOps, outputTys)) {
        auto operands = funcOp->getBody().front().back().getOperands();
        rewriter.eraseOp(&funcOp->getFunctionBody().front().back());
        rewriter.setInsertionPointToEnd(&funcOp->getBody().front());
        SmallVector<Value> toReturn;
        for (unsigned i = 0; i < outputTy.size(); ++i)
          toReturn.push_back(typeConverter->materializeTargetConversion(
              rewriter, loc, outputTy[i], operands[i]));
        rewriter.create<func::ReturnOp>(loc, toReturn);
      }
    }

    auto solver =
        rewriter.create<smt::SolverOp>(loc, rewriter.getI1Type(), ValueRange{});
    rewriter.createBlock(&solver.getBodyRegion());

    // Call init func to get initial clock value
    ValueRange initVals =
        rewriter.create<func::CallOp>(loc, initFuncOp)->getResults();

    // InputDecls order should be <circuit arguments> <state arguments>
    // <wasViolated>
    size_t initIndex = 0;
    SmallVector<Value> inputDecls;
    for (auto [oldTy, newTy] : llvm::zip(oldCircuitInputTy, circuitInputTy)) {
      if (isa<seq::ClockType>(oldTy))
        inputDecls.push_back(initVals[initIndex++]);
      else
        inputDecls.push_back(rewriter.create<smt::DeclareFunOp>(loc, newTy));
    }

    // Add the rest of the init vals (state args)
    for (; initIndex < initVals.size(); ++initIndex)
      inputDecls.push_back(initVals[initIndex]);

    Value lowerBound =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
    Value step =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(1));
    Value upperBound =
        rewriter.create<arith::ConstantOp>(loc, adaptor.getBoundAttr());
    Value constFalse =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(false));
    Value constTrue =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true));
    inputDecls.push_back(constFalse); // wasViolated?

    // TODO: swapping to a whileOp here would allow early exit once the property
    // is violated
    // Perform model check up to the provided bound
    auto forOp = rewriter.create<scf::ForOp>(
        loc, lowerBound, upperBound, step, inputDecls,
        [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
          // Execute the circuit
          ValueRange circuitCallOuts =
              builder
                  .create<func::CallOp>(
                      loc, circuitFuncOp,
                      iterArgs.take_front(circuitFuncOp.getNumArguments()))
                  ->getResults();
          auto checkOp =
              rewriter.create<smt::CheckOp>(loc, builder.getI1Type());
          {
            OpBuilder::InsertionGuard guard(builder);
            builder.createBlock(&checkOp.getSatRegion());
            builder.create<smt::YieldOp>(loc, constTrue);
            builder.createBlock(&checkOp.getUnknownRegion());
            builder.create<smt::YieldOp>(loc, constTrue);
            builder.createBlock(&checkOp.getUnsatRegion());
            builder.create<smt::YieldOp>(loc, constFalse);
          }

          Value violated = builder.create<arith::OrIOp>(
              loc, checkOp.getResult(0), iterArgs.back());

          SmallVector<Value> newDecls;

          // Call loop func to update clock value
          ValueRange loopVals =
              builder
                  .create<func::CallOp>(loc, loopFuncOp, iterArgs.drop_back())
                  ->getResults();

          size_t loopIndex = 0;
          for (auto [oldTy, newTy] :
               llvm::zip(TypeRange(oldCircuitInputTy).drop_back(numRegs),
                         TypeRange(circuitInputTy).drop_back(numRegs))) {
            if (isa<seq::ClockType>(oldTy))
              newDecls.push_back(loopVals[loopIndex++]);
            else
              newDecls.push_back(builder.create<smt::DeclareFunOp>(loc, newTy));
          }
          newDecls.append(
              SmallVector<Value>(circuitCallOuts.take_back(numRegs)));

          // Add the rest of the loop state args
          for (; loopIndex < loopVals.size(); ++loopIndex)
            newDecls.push_back(loopVals[loopIndex]);

          newDecls.push_back(violated);

          builder.create<scf::YieldOp>(loc, newDecls);
        });

    Value res = rewriter.create<arith::XOrIOp>(loc, forOp->getResults().back(),
                                               constTrue);
    rewriter.create<smt::YieldOp>(loc, res);
    rewriter.replaceOp(op, solver.getResults());
    return success();
  }

  Namespace &names;
};

} // namespace

//===----------------------------------------------------------------------===//
// Convert Verif to SMT pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertVerifToSMTPass
    : public circt::impl::ConvertVerifToSMTBase<ConvertVerifToSMTPass> {
  void runOnOperation() override;
};
} // namespace

void circt::populateVerifToSMTConversionPatterns(TypeConverter &converter,
                                                 RewritePatternSet &patterns,
                                                 Namespace &names) {
  patterns.add<VerifAssertOpConversion, VerifAssumeOpConversion,
               VerifCoverOpConversion, VerifClockedAssertOpConversion,
               VerifClockedAssumeOpConversion, VerifClockedCoverOpConversion,
               LogicEquivalenceCheckingOpConversion>(converter,
                                                     patterns.getContext());
  patterns.add<VerifBMCOpConversion>(converter, patterns.getContext(), names);
}

void ConvertVerifToSMTPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addIllegalDialect<verif::VerifDialect>();
  target.addLegalDialect<smt::SMTDialect, arith::ArithDialect, scf::SCFDialect,
                         func::FuncDialect>();
  target.addLegalOp<UnrealizedConversionCastOp>();

  RewritePatternSet patterns(&getContext());
  TypeConverter converter;
  populateHWToSMTTypeConverter(converter);

  SymbolCache symCache;
  symCache.addDefinitions(getOperation());
  Namespace names;
  names.add(symCache);
  populateVerifToSMTConversionPatterns(converter, patterns, names);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}
