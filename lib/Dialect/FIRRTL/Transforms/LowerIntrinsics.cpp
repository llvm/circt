//===- LowerIntrinsics.cpp - Lower Intrinsics -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerIntrinsics pass.  This pass processes FIRRTL
// generic intrinsic operations and rewrites to their implementation.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLIntrinsics.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"

using namespace circt;
using namespace firrtl;

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
    return gi.hasNInputs(0) || gi.sizedOutput<UIntType>(1) || gi.hasNParam(1) ||
           gi.namedParam("FORMAT");
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
           gi.hasOutputElement(1, "result") || gi.hasNParam(1) ||
           gi.namedParam("FORMAT");
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto bty = gi.getOutputBundle().getType();
    auto newop = rewriter.create<PlusArgsValueIntrinsicOp>(
        gi.op.getLoc(), bty.getElementType(size_t{0}),
        bty.getElementType(size_t{1}), gi.getParamValue<StringAttr>("FORMAT"));
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

class CirctClockInverterConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(1) || gi.typedInput<ClockType>(0) ||
           gi.typedOutput<ClockType>() || gi.hasNParam(0);
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    // TODO: As temporary accomodation, consider propagating name to op
    // during intmodule->op conversion, as code previously materialized
    // a wire to hold the name of the instance.
    // In the future, input FIRRTL can just be "node clock_inv = ....".
    rewriter.replaceOpWithNewOp<ClockInverterIntrinsicOp>(
        gi.op, adaptor.getOperands()[0]);

    // auto name = inst.getInstanceName();
    // Value outWire = builder.create<WireOp>(out.getType(), name).getResult();
    // builder.create<StrictConnectOp>(outWire, out);
    // inst.getResult(1).replaceAllUsesWith(outWire);
  }
};

class CirctMux2CellConverter
    : public IntrinsicOpConverter<Mux2CellIntrinsicOp> {
  using IntrinsicOpConverter::IntrinsicOpConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(3) || gi.typedInput<UIntType>(0) || gi.hasNParam(0) ||
           gi.hasOutput();
  }
};

class CirctMux4CellConverter
    : public IntrinsicOpConverter<Mux4CellIntrinsicOp> {
  using IntrinsicOpConverter::IntrinsicOpConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(5) || gi.typedInput<UIntType>(0) || gi.hasNParam(0) ||
           gi.hasOutput();
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
           gi.sizedOutput<UIntType>(1) || gi.hasNParam(1, 2) ||
           gi.namedIntParam("delay") || gi.namedIntParam("length", true);
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

template <class Op>
class CirctVerifConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(1) || gi.sizedInput<UIntType>(0, 1) ||
           gi.hasNParam(0, 1) || gi.namedParam("label", true) ||
           gi.hasNoOutput();
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto label = gi.getParamValue<StringAttr>("label");

    rewriter.replaceOpWithNewOp<Op>(gi.op, adaptor.getOperands()[0], label);
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
class CirctAssertAssumeConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.typedInput<ClockType>(0) || gi.sizedInput<UIntType>(1, 1) ||
           gi.sizedInput<UIntType>(2, 1) || gi.hasNParam(0, 3) ||
           gi.namedParam("format", /*optional=*/true) ||
           gi.namedParam("label", /*optional=*/true) ||
           gi.namedParam("guards", /*optional=*/true) || gi.hasNoOutput();
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
           gi.sizedInput<UIntType>(2, 1) || gi.hasNParam(0, 2) ||
           gi.namedParam("label", /*optional=*/true) ||
           gi.namedParam("guards", /*optional=*/true);
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

} // namespace

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct LowerIntrinsicsPass : public LowerIntrinsicsBase<LowerIntrinsicsPass> {
  LogicalResult initialize(MLIRContext *context) override;
  void runOnOperation() override;

  std::shared_ptr<IntrinsicLowerings> lowering;
};
} // namespace

/// Initialize the conversions for use during execution.
LogicalResult LowerIntrinsicsPass::initialize(MLIRContext *context) {
  IntrinsicLowerings lowering(context);

  lowering.add<CirctSizeofConverter>("circt.sizeof", "circt_sizeof");
  lowering.add<CirctIsXConverter>("circt.isX", "circt_isX");
  lowering.add<CirctPlusArgTestConverter>("circt.plusargs.test",
                                          "circt_plusargs_test");
  lowering.add<CirctPlusArgValueConverter>("circt.plusargs.value",
                                           "circt_plusargs_value");
  lowering.add<CirctClockGateConverter>("circt.clock_gate", "circt_clock_gate");
  lowering.add<CirctClockInverterConverter>("circt.clock_inv",
                                            "circt_clock_inv");
  lowering.add<CirctMux2CellConverter>("circt.mux2cell", "circt_mux2cell");
  lowering.add<CirctMux4CellConverter>("circt.mux4cell", "circt_mux4cell");
  lowering.add<CirctHasBeenResetConverter>("circt.has_been_reset",
                                           "circt_has_been_reset");

  lowering.add<CirctLTLBinaryConverter<LTLAndIntrinsicOp>>("circt.ltl.and",
                                                           "circt_ltl_and");
  lowering.add<CirctLTLBinaryConverter<LTLOrIntrinsicOp>>("circt.ltl.or",
                                                          "circt_ltl_or");
  lowering.add<CirctLTLBinaryConverter<LTLConcatIntrinsicOp>>(
      "circt.ltl.concat", "circt_ltl_concat");
  lowering.add<CirctLTLBinaryConverter<LTLImplicationIntrinsicOp>>(
      "circt.ltl.implication", "circt_ltl_implication");
  lowering.add<CirctLTLBinaryConverter<LTLDisableIntrinsicOp>>(
      "circt.ltl.disable", "circt_ltl_disable");
  lowering.add<CirctLTLUnaryConverter<LTLNotIntrinsicOp>>("circt.ltl.not",
                                                          "circt_ltl_not");
  lowering.add<CirctLTLUnaryConverter<LTLEventuallyIntrinsicOp>>(
      "circt.ltl.eventually", "circt_ltl_eventually");

  lowering.add<CirctLTLDelayConverter>("circt.ltl.delay", "circt_ltl_delay");
  lowering.add<CirctLTLClockConverter>("circt.ltl.clock", "circt_ltl_clock");

  lowering.add<CirctVerifConverter<VerifAssertIntrinsicOp>>(
      "circt.verif.assert", "circt_verif_assert");
  lowering.add<CirctVerifConverter<VerifAssumeIntrinsicOp>>(
      "circt.verif.assume", "circt_verif_assume");
  lowering.add<CirctVerifConverter<VerifCoverIntrinsicOp>>("circt.verif.cover",
                                                           "circt_verif_cover");

  lowering.add<CirctProbeConverter>("circt.fpga_probe", "circt_fpga_probe");

  lowering.add<CirctAssertAssumeConverter<AssertOp>>(
      "circt.chisel_assert_assume", "circt_chisel_assert_assume");
  lowering.add<CirctAssertAssumeConverter<AssertOp, /*ifElseFatal=*/true>>(
      "circt.chisel_ifelsefatal", "circt_chisel_ifelsefatal");
  lowering.add<CirctAssertAssumeConverter<AssumeOp>>("circt.chisel_assume",
                                                     "circt_chisel_assume");
  lowering.add<CirctCoverConverter>("circt.chisel_cover", "circt_chisel_cover");

  this->lowering = std::make_shared<IntrinsicLowerings>(std::move(lowering));
  return success();
}

// This is the main entrypoint for the lowering pass.
void LowerIntrinsicsPass::runOnOperation() {
  if (failed(lowering->lower(getOperation())))
    return signalPassFailure();
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createLowerIntrinsicsPass() {
  return std::make_unique<LowerIntrinsicsPass>();
}
