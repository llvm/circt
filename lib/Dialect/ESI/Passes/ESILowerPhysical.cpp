//===- ESILowerPhysical.cpp - Lower ESI to physical -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower ESI to ESI "physical level" ops conversions and pass.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/LLVM.h"

#include "mlir/Transforms/DialectConversion.h"

namespace circt {
namespace esi {
#define GEN_PASS_DEF_LOWERESITOPHYSICAL
#include "circt/Dialect/ESI/ESIPasses.h.inc"
} // namespace esi
} // namespace circt

using namespace circt;
using namespace circt::esi;
using namespace circt::esi::detail;
using namespace circt::hw;

namespace {
/// Lower `ChannelBufferOp`s, breaking out the various options. For now, just
/// replace with the specified number of pipeline stages (since that's the only
/// option).
struct ChannelBufferLowering : public OpConversionPattern<ChannelBufferOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ChannelBufferOp buffer, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // anonymous namespace

LogicalResult ChannelBufferLowering::matchAndRewrite(
    ChannelBufferOp buffer, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = buffer.getLoc();

  auto type = buffer.getType();

  // Expand 'abstract' buffer into 'physical' stages.
  auto stages = buffer.getStagesAttr();
  uint64_t numStages = 1;
  if (stages) {
    // Guaranteed positive by the parser.
    numStages = stages.getValue().getLimitedValue();
  }
  Value input = buffer.getInput();
  StringAttr bufferName = buffer.getNameAttr();
  for (uint64_t i = 0; i < numStages; ++i) {
    // Create the stages, connecting them up as we build.
    auto stage = rewriter.create<PipelineStageOp>(loc, type, buffer.getClk(),
                                                  buffer.getRst(), input);
    if (bufferName) {
      SmallString<64> stageName(
          {bufferName.getValue(), "_stage", std::to_string(i)});
      stage->setAttr("name", StringAttr::get(rewriter.getContext(), stageName));
    }
    input = stage;
  }

  // Replace the buffer.
  rewriter.replaceOp(buffer, input);
  return success();
}

namespace {
/// Lower `ChannelBufferOp`s, breaking out the various options. For now, just
/// replace with the specified number of pipeline stages (since that's the only
/// option).
struct FIFOLowering : public OpConversionPattern<FIFOOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FIFOOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // anonymous namespace

LogicalResult
FIFOLowering::matchAndRewrite(FIFOOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto outputType = op.getType();
  BackedgeBuilder bb(rewriter, loc);
  auto i1 = rewriter.getI1Type();
  auto c1 = rewriter.create<hw::ConstantOp>(loc, rewriter.getI1Type(),
                                            rewriter.getBoolAttr(true));
  mlir::TypedValue<ChannelType> chanInput = op.getInput();
  if (chanInput.getType().getDataDelay() != 0)
    return op.emitOpError(
        "currently only supports input channels with zero data delay");

  Backedge inputEn = bb.get(i1);
  Value rawData;
  Value dataNotAvailable;
  if (chanInput.getType().getSignaling() == ChannelSignaling::ValidReady) {
    auto unwrapValidReady =
        rewriter.create<UnwrapValidReadyOp>(loc, chanInput, inputEn);
    rawData = unwrapValidReady.getRawOutput();
    dataNotAvailable =
        comb::createOrFoldNot(loc, unwrapValidReady.getValid(), rewriter);
    dataNotAvailable.getDefiningOp()->setAttr(
        "sv.namehint", rewriter.getStringAttr("dataNotAvailable"));
  } else if (chanInput.getType().getSignaling() == ChannelSignaling::FIFO) {
    auto unwrapPull = rewriter.create<UnwrapFIFOOp>(loc, chanInput, inputEn);
    rawData = unwrapPull.getData();
    dataNotAvailable = unwrapPull.getEmpty();
  } else {
    return rewriter.notifyMatchFailure(
        op, "only supports ValidReady and FIFO signaling");
  }

  Backedge outputRdEn = bb.get(i1);
  auto seqFifo = rewriter.create<seq::FIFOOp>(
      loc, outputType.getInner(), i1, i1, Type(), Type(), rawData, outputRdEn,
      inputEn, op.getClk(), op.getRst(), op.getDepthAttr(),
      rewriter.getI64IntegerAttr(outputType.getDataDelay()), IntegerAttr(),
      IntegerAttr());
  auto inputNotEmpty = rewriter.create<comb::XorOp>(loc, dataNotAvailable, c1);
  inputNotEmpty->setAttr("sv.namehint",
                         rewriter.getStringAttr("inputNotEmpty"));
  auto seqFifoNotFull =
      rewriter.create<comb::XorOp>(loc, seqFifo.getFull(), c1);
  seqFifoNotFull->setAttr("sv.namehint",
                          rewriter.getStringAttr("seqFifoNotFull"));
  inputEn.setValue(
      rewriter.create<comb::AndOp>(loc, inputNotEmpty, seqFifoNotFull));
  static_cast<Value>(inputEn).getDefiningOp()->setAttr(
      "sv.namehint", rewriter.getStringAttr("inputEn"));

  Value output;
  if (outputType.getSignaling() == ChannelSignaling::ValidReady) {
    auto wrap = rewriter.create<WrapValidReadyOp>(
        loc, mlir::TypeRange{outputType, i1}, seqFifo.getOutput(),
        comb::createOrFoldNot(loc, seqFifo.getEmpty(), rewriter));
    output = wrap.getChanOutput();
    outputRdEn.setValue(
        rewriter.create<comb::AndOp>(loc, wrap.getValid(), wrap.getReady()));
    static_cast<Value>(outputRdEn)
        .getDefiningOp()
        ->setAttr("sv.namehint", rewriter.getStringAttr("outputRdEn"));
  } else if (outputType.getSignaling() == ChannelSignaling::FIFO) {
    auto wrap =
        rewriter.create<WrapFIFOOp>(loc, mlir::TypeRange{outputType, i1},
                                    seqFifo.getOutput(), seqFifo.getEmpty());
    output = wrap.getChanOutput();
    outputRdEn.setValue(wrap.getRden());
  } else {
    return rewriter.notifyMatchFailure(op, "only supports ValidReady and FIFO");
  }

  rewriter.replaceOp(op, output);
  return success();
}

namespace {
/// Lower pure modules into hw.modules.
struct PureModuleLowering : public OpConversionPattern<ESIPureModuleOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ESIPureModuleOp pureMod, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // anonymous namespace

LogicalResult
PureModuleLowering::matchAndRewrite(ESIPureModuleOp pureMod, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  auto loc = pureMod.getLoc();
  Block *body = &pureMod.getBody().front();

  // Track existing names (so we can de-dup) and get op result when we want to
  // replace it with the block args.
  DenseMap<StringAttr, ESIPureModuleInputOp> inputPortNames;
  // Build the port list for `hw.module` construction.
  SmallVector<hw::PortInfo> ports;
  // List the input and output ops.
  SmallVector<ESIPureModuleInputOp> inputs;
  SmallVector<ESIPureModuleOutputOp> outputs;
  SmallVector<Attribute> params;

  for (Operation &op : llvm::make_early_inc_range(body->getOperations())) {
    if (auto port = dyn_cast<ESIPureModuleInputOp>(op)) {
      // If we already have an input port of the same name, replace the result
      // value with the previous one. Checking that the types match is done in
      // the pure module verifier.
      auto existingPort = inputPortNames.find(port.getNameAttr());
      if (existingPort != inputPortNames.end()) {
        rewriter.replaceAllUsesWith(port.getResult(),
                                    existingPort->getSecond().getResult());
        rewriter.eraseOp(port);
        continue;
      }
      // Normal port construction.
      ports.push_back(
          hw::PortInfo{{port.getNameAttr(), port.getResult().getType(),
                        hw::ModulePort::Direction::Input},
                       inputs.size(),
                       {},
                       port.getLoc()});
      inputs.push_back(port);
    } else if (auto port = dyn_cast<ESIPureModuleOutputOp>(op)) {
      ports.push_back(
          hw::PortInfo{{port.getNameAttr(), port.getValue().getType(),
                        hw::ModulePort::Direction::Output},
                       outputs.size(),
                       {},
                       port.getLoc()});
      outputs.push_back(port);
    } else if (auto param = dyn_cast<ESIPureModuleParamOp>(op)) {
      params.push_back(
          ParamDeclAttr::get(param.getNameAttr(), param.getType()));
      rewriter.eraseOp(param);
    }
  }

  // Create the replacement `hw.module`.
  auto hwMod = rewriter.create<hw::HWModuleOp>(
      loc, pureMod.getNameAttr(), ports, ArrayAttr::get(getContext(), params));
  hwMod->setDialectAttrs(pureMod->getDialectAttrs());
  rewriter.eraseBlock(hwMod.getBodyBlock());
  rewriter.inlineRegionBefore(*body->getParent(), hwMod.getBodyRegion(),
                              hwMod.getBodyRegion().end());
  body = hwMod.getBodyBlock();

  // Re-wire the inputs and erase them.
  for (auto input : inputs) {
    BlockArgument newArg;
    rewriter.modifyOpInPlace(hwMod, [&]() {
      newArg = body->addArgument(input.getResult().getType(), input.getLoc());
    });
    rewriter.replaceAllUsesWith(input.getResult(), newArg);
    rewriter.eraseOp(input);
  }

  // Assemble the output values.
  SmallVector<Value> hwOutputOperands;
  for (auto output : outputs) {
    hwOutputOperands.push_back(output.getValue());
    rewriter.eraseOp(output);
  }
  rewriter.setInsertionPointToEnd(body);
  rewriter.create<hw::OutputOp>(pureMod.getLoc(), hwOutputOperands);

  // Erase the original op.
  rewriter.eraseOp(pureMod);
  return success();
}

namespace {
/// Run all the physical lowerings.
struct ESIToPhysicalPass
    : public circt::esi::impl::LowerESIToPhysicalBase<ESIToPhysicalPass> {
  void runOnOperation() override;
};
} // anonymous namespace

void ESIToPhysicalPass::runOnOperation() {
  // Set up a conversion and give it a set of laws.
  ConversionTarget target(getContext());
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  target.addIllegalOp<ChannelBufferOp, ESIPureModuleOp, FIFOOp>();

  // Add all the conversion patterns.
  RewritePatternSet patterns(&getContext());
  patterns.insert<ChannelBufferLowering, PureModuleLowering, FIFOLowering>(
      &getContext());

  // Run the conversion.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
circt::esi::createESIPhysicalLoweringPass() {
  return std::make_unique<ESIToPhysicalPass>();
}
