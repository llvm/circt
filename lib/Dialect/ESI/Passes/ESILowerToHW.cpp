//===- ESILowerToHW.cpp - Lower ESI to HW -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower to HW/SV conversions and pass.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/SymCache.h"

#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"

namespace circt {
namespace esi {
#define GEN_PASS_DEF_LOWERESITOHW
#include "circt/Dialect/ESI/ESIPasses.h.inc"
} // namespace esi
} // namespace circt

using namespace circt;
using namespace circt::esi;
using namespace circt::esi::detail;
using namespace circt::hw;
using namespace circt::sv;

namespace {
/// Lower PipelineStageOp ops to an HW implementation. Unwrap and re-wrap
/// appropriately. Another conversion will take care merging the resulting
/// adjacent wrap/unwrap ops.
struct PipelineStageLowering : public OpConversionPattern<PipelineStageOp> {
public:
  PipelineStageLowering(ESIHWBuilder &builder, MLIRContext *ctxt)
      : OpConversionPattern(ctxt), builder(builder) {}
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PipelineStageOp stage, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;

private:
  ESIHWBuilder &builder;
};
} // anonymous namespace

LogicalResult PipelineStageLowering::matchAndRewrite(
    PipelineStageOp stage, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = stage.getLoc();
  auto chPort = dyn_cast<ChannelType>(stage.getInput().getType());
  if (!chPort)
    return rewriter.notifyMatchFailure(stage, "stage had wrong type");
  Operation *symTable = stage->getParentWithTrait<OpTrait::SymbolTable>();
  auto stageModule = builder.declareStage(symTable, stage);

  size_t width = circt::hw::getBitWidth(chPort.getInner());

  ArrayAttr stageParams =
      builder.getStageParameterList(rewriter.getUI32IntegerAttr(width));

  // Unwrap the channel. The ready signal is a Value we haven't created yet,
  // so create a temp value and replace it later. Give this constant an
  // odd-looking type to make debugging easier.
  circt::BackedgeBuilder back(rewriter, loc);
  circt::Backedge wrapReady = back.get(rewriter.getI1Type());
  auto unwrap =
      rewriter.create<UnwrapValidReadyOp>(loc, stage.getInput(), wrapReady);

  StringRef pipeStageName = "pipelineStage";
  if (auto name = stage->getAttrOfType<StringAttr>("name"))
    pipeStageName = name.getValue();

  // Instantiate the "ESI_PipelineStage" external module.
  circt::Backedge stageReady = back.get(rewriter.getI1Type());
  llvm::SmallVector<Value> operands = {stage.getClk(), stage.getRst()};
  operands.push_back(unwrap.getRawOutput());
  operands.push_back(unwrap.getValid());
  operands.push_back(stageReady);
  auto stageInst = rewriter.create<hw::InstanceOp>(
      loc, stageModule, pipeStageName, operands, stageParams);
  auto stageInstResults = stageInst.getResults();

  // Set a_ready (from the unwrap) back edge correctly to its output from
  // stage.
  wrapReady.setValue(stageInstResults[0]);
  Value x, xValid;
  x = stageInstResults[1];
  xValid = stageInstResults[2];

  // Wrap up the output of the HW stage module.
  auto wrap = rewriter.create<WrapValidReadyOp>(
      loc, chPort, rewriter.getI1Type(), x, xValid);
  // Set the stages x_ready backedge correctly.
  stageReady.setValue(wrap.getReady());

  rewriter.replaceOp(stage, wrap.getChanOutput());
  return success();
}

namespace {
struct NullSourceOpLowering : public OpConversionPattern<NullSourceOp> {
public:
  NullSourceOpLowering(MLIRContext *ctxt) : OpConversionPattern(ctxt) {}
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NullSourceOp nullop, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // anonymous namespace

LogicalResult NullSourceOpLowering::matchAndRewrite(
    NullSourceOp nullop, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto innerType = cast<ChannelType>(nullop.getOut().getType()).getInner();
  Location loc = nullop.getLoc();
  int64_t width = hw::getBitWidth(innerType);
  if (width == -1)
    return rewriter.notifyMatchFailure(
        nullop, "NullOp lowering only supports hw types");
  auto valid =
      rewriter.create<hw::ConstantOp>(nullop.getLoc(), rewriter.getI1Type(), 0);
  auto zero =
      rewriter.create<hw::ConstantOp>(loc, rewriter.getIntegerType(width), 0);
  auto typedZero = rewriter.create<hw::BitcastOp>(loc, innerType, zero);
  auto wrap = rewriter.create<WrapValidReadyOp>(loc, typedZero, valid);
  wrap->setAttr("name", rewriter.getStringAttr("nullsource"));
  rewriter.replaceOp(nullop, {wrap.getChanOutput()});
  return success();
}

namespace {
/// Eliminate back-to-back wrap-unwraps to reduce the number of ESI channels.
struct RemoveWrapUnwrap : public ConversionPattern {
public:
  RemoveWrapUnwrap(MLIRContext *context)
      : ConversionPattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Value valid, ready, data;
    WrapValidReadyOp wrap = dyn_cast<WrapValidReadyOp>(op);
    UnwrapValidReadyOp unwrap = dyn_cast<UnwrapValidReadyOp>(op);
    if (wrap) {
      // Lower away snoop ops.
      for (auto user : wrap.getChanOutput().getUsers())
        if (auto snoop = dyn_cast<SnoopValidReadyOp>(user))
          rewriter.replaceOp(
              snoop, {wrap.getValid(), wrap.getReady(), wrap.getRawInput()});

      if (ChannelType::hasNoConsumers(wrap.getChanOutput())) {
        auto c1 = rewriter.create<hw::ConstantOp>(wrap.getLoc(),
                                                  rewriter.getI1Type(), 1);
        rewriter.replaceOp(wrap, {nullptr, c1});
        return success();
      }

      if (!ChannelType::hasOneConsumer(wrap.getChanOutput()))
        return rewriter.notifyMatchFailure(
            wrap, "This conversion only supports wrap-unwrap back-to-back. "
                  "Wrap didn't have exactly one use.");
      if (!(unwrap = dyn_cast<UnwrapValidReadyOp>(
                ChannelType::getSingleConsumer(wrap.getChanOutput())
                    ->getOwner())))
        return rewriter.notifyMatchFailure(
            wrap, "This conversion only supports wrap-unwrap back-to-back. "
                  "Could not find 'unwrap'.");

      data = operands[0];
      valid = operands[1];
      ready = unwrap.getReady();
    } else if (unwrap) {
      wrap = dyn_cast<WrapValidReadyOp>(operands[0].getDefiningOp());
      if (!wrap)
        return rewriter.notifyMatchFailure(
            operands[0].getDefiningOp(),
            "This conversion only supports wrap-unwrap back-to-back. "
            "Could not find 'wrap'.");
      valid = wrap.getValid();
      data = wrap.getRawInput();
      ready = operands[1];

      // Lower away snoop ops.
      for (auto user : operands[0].getUsers())
        if (auto snoop = dyn_cast<SnoopValidReadyOp>(user))
          rewriter.replaceOp(snoop, {valid, ready, data});
    } else {
      return failure();
    }

    if (!ChannelType::hasOneConsumer(wrap.getChanOutput()))
      return rewriter.notifyMatchFailure(wrap, [](Diagnostic &d) {
        d << "This conversion only supports wrap-unwrap back-to-back. "
             "Wrap didn't have exactly one use.";
      });
    rewriter.replaceOp(wrap, {nullptr, ready});
    rewriter.replaceOp(unwrap, {data, valid});
    return success();
  }
};
} // anonymous namespace

namespace {
/// Use the op canonicalizer to lower away the op. Assumes the canonicalizer
/// deletes the op.
template <typename Op>
struct CanonicalizerOpLowering : public OpConversionPattern<Op> {
public:
  CanonicalizerOpLowering(MLIRContext *ctxt) : OpConversionPattern<Op>(ctxt) {}

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (failed(Op::canonicalize(op, rewriter)))
      return rewriter.notifyMatchFailure(op->getLoc(), "canonicalizer failed");
    return success();
  }
};
} // anonymous namespace

namespace {
struct ESItoHWPass : public circt::esi::impl::LowerESItoHWBase<ESItoHWPass> {
  void runOnOperation() override;
};
} // anonymous namespace

namespace {
/// Lower a `wrap.iface` to `wrap.vr` by extracting the wires then feeding the
/// new `wrap.vr`.
struct WrapInterfaceLower : public OpConversionPattern<WrapSVInterfaceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WrapSVInterfaceOp wrap, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // anonymous namespace

LogicalResult
WrapInterfaceLower::matchAndRewrite(WrapSVInterfaceOp wrap, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  auto operands = adaptor.getOperands();
  if (operands.size() != 1)
    return rewriter.notifyMatchFailure(wrap, [&operands](Diagnostic &d) {
      d << "wrap.iface has 1 argument. Got " << operands.size() << "operands";
    });
  auto sinkModport = dyn_cast<GetModportOp>(operands[0].getDefiningOp());
  if (!sinkModport)
    return failure();
  auto ifaceInstance =
      dyn_cast<InterfaceInstanceOp>(sinkModport.getIface().getDefiningOp());
  if (!ifaceInstance)
    return failure();

  auto loc = wrap.getLoc();
  auto validSignal = rewriter.create<ReadInterfaceSignalOp>(
      loc, ifaceInstance, ESIHWBuilder::validStr);
  Value dataSignal;
  dataSignal = rewriter.create<ReadInterfaceSignalOp>(loc, ifaceInstance,
                                                      ESIHWBuilder::dataStr);
  auto wrapVR = rewriter.create<WrapValidReadyOp>(loc, dataSignal, validSignal);
  rewriter.create<AssignInterfaceSignalOp>(
      loc, ifaceInstance, ESIHWBuilder::readyStr, wrapVR.getReady());
  rewriter.replaceOp(wrap, {wrapVR.getChanOutput()});
  return success();
}

namespace {
/// Lower an unwrap interface to just extract the wires and feed them into an
/// `unwrap.vr`.
struct UnwrapInterfaceLower : public OpConversionPattern<UnwrapSVInterfaceOp> {
public:
  UnwrapInterfaceLower(MLIRContext *ctxt) : OpConversionPattern(ctxt) {}
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnwrapSVInterfaceOp wrap, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // anonymous namespace

LogicalResult UnwrapInterfaceLower::matchAndRewrite(
    UnwrapSVInterfaceOp unwrap, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto operands = adaptor.getOperands();
  if (operands.size() != 2)
    return rewriter.notifyMatchFailure(unwrap, [&operands](Diagnostic &d) {
      d << "Unwrap.iface has 2 arguments. Got " << operands.size()
        << "operands";
    });

  auto sourceModport = dyn_cast<GetModportOp>(operands[1].getDefiningOp());
  if (!sourceModport)
    return failure();
  auto ifaceInstance =
      dyn_cast<InterfaceInstanceOp>(sourceModport.getIface().getDefiningOp());
  if (!ifaceInstance)
    return failure();

  auto loc = unwrap.getLoc();
  auto readySignal = rewriter.create<ReadInterfaceSignalOp>(
      loc, ifaceInstance, ESIHWBuilder::readyStr);
  auto unwrapVR =
      rewriter.create<UnwrapValidReadyOp>(loc, operands[0], readySignal);
  rewriter.create<AssignInterfaceSignalOp>(
      loc, ifaceInstance, ESIHWBuilder::validStr, unwrapVR.getValid());

  rewriter.create<AssignInterfaceSignalOp>(
      loc, ifaceInstance, ESIHWBuilder::dataStr, unwrapVR.getRawOutput());
  rewriter.eraseOp(unwrap);
  return success();
}

namespace {
/// Lower `CosimEndpointOp` ops to a SystemVerilog extern module and a Capnp
/// gasket op.
struct CosimToHostLowering : public OpConversionPattern<CosimToHostEndpointOp> {
public:
  CosimToHostLowering(ESIHWBuilder &b)
      : OpConversionPattern(b.getContext(), 1), builder(b) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CosimToHostEndpointOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;

private:
  ESIHWBuilder &builder;
};
} // anonymous namespace

LogicalResult CosimToHostLowering::matchAndRewrite(
    CosimToHostEndpointOp ep, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = ep.getLoc();
  auto *ctxt = rewriter.getContext();
  circt::BackedgeBuilder bb(rewriter, loc);

  Value toHost = adaptor.getToHost();
  Type type = toHost.getType();
  uint64_t width = getWidth(type);

  // Set all the parameters.
  SmallVector<Attribute, 8> params;
  params.push_back(ParamDeclAttr::get("ENDPOINT_ID", ep.getIdAttr()));
  params.push_back(ParamDeclAttr::get("TO_HOST_TYPE_ID", getTypeID(type)));
  params.push_back(ParamDeclAttr::get(
      "TO_HOST_SIZE_BITS", rewriter.getI32IntegerAttr(width > 0 ? width : 1)));

  // Set up the egest route to drive the EP's toHost ports.
  auto sendReady = bb.get(rewriter.getI1Type());
  UnwrapValidReadyOp unwrapSend =
      rewriter.create<UnwrapValidReadyOp>(loc, toHost, sendReady);
  Value castedSendData;
  if (width > 0)
    castedSendData = rewriter.create<hw::BitcastOp>(
        loc, rewriter.getIntegerType(width), unwrapSend.getRawOutput());
  else
    castedSendData = rewriter.create<hw::ConstantOp>(
        loc, rewriter.getIntegerType(1), rewriter.getBoolAttr(false));

  // Build or get the cached Cosim Endpoint module parameterization.
  Operation *symTable = ep->getParentWithTrait<OpTrait::SymbolTable>();
  HWModuleExternOp endpoint =
      builder.declareCosimEndpointToHostModule(symTable);

  // Create replacement Cosim_Endpoint instance.
  Value operands[] = {
      adaptor.getClk(),
      adaptor.getRst(),
      unwrapSend.getValid(),
      castedSendData,
  };
  auto cosimEpModule = rewriter.create<hw::InstanceOp>(
      loc, endpoint, ep.getIdAttr(), operands, ArrayAttr::get(ctxt, params));
  sendReady.setValue(cosimEpModule.getResult(0));

  // Replace the CosimEndpointOp op.
  rewriter.eraseOp(ep);

  return success();
}

namespace {
/// Lower `CosimEndpointOp` ops to a SystemVerilog extern module and a Capnp
/// gasket op.
struct CosimFromHostLowering
    : public OpConversionPattern<CosimFromHostEndpointOp> {
public:
  CosimFromHostLowering(ESIHWBuilder &b)
      : OpConversionPattern(b.getContext(), 1), builder(b) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CosimFromHostEndpointOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;

private:
  ESIHWBuilder &builder;
};
} // anonymous namespace

LogicalResult CosimFromHostLowering::matchAndRewrite(
    CosimFromHostEndpointOp ep, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = ep.getLoc();
  auto *ctxt = rewriter.getContext();
  circt::BackedgeBuilder bb(rewriter, loc);

  ChannelType type = ep.getFromHost().getType();
  uint64_t width = getWidth(type);

  // Set all the parameters.
  SmallVector<Attribute, 8> params;
  params.push_back(ParamDeclAttr::get("ENDPOINT_ID", ep.getIdAttr()));
  params.push_back(ParamDeclAttr::get("FROM_HOST_TYPE_ID", getTypeID(type)));
  params.push_back(
      ParamDeclAttr::get("FROM_HOST_SIZE_BITS",
                         rewriter.getI32IntegerAttr(width > 0 ? width : 1)));

  // Get information necessary for injest path.
  auto recvReady = bb.get(rewriter.getI1Type());

  // Build or get the cached Cosim Endpoint module parameterization.
  Operation *symTable = ep->getParentWithTrait<OpTrait::SymbolTable>();
  HWModuleExternOp endpoint =
      builder.declareCosimEndpointFromHostModule(symTable);

  // Create replacement Cosim_Endpoint instance.
  Value operands[] = {adaptor.getClk(), adaptor.getRst(), recvReady};
  auto cosimEpModule = rewriter.create<hw::InstanceOp>(
      loc, endpoint, ep.getIdAttr(), operands, ArrayAttr::get(ctxt, params));

  // Set up the injest path.
  Value recvDataFromCosim = cosimEpModule.getResult(1);
  Value recvValidFromCosim = cosimEpModule.getResult(0);
  Value castedRecvData;
  if (width > 0)
    castedRecvData =
        rewriter.create<hw::BitcastOp>(loc, type.getInner(), recvDataFromCosim);
  else
    castedRecvData = rewriter.create<hw::ConstantOp>(
        loc, rewriter.getIntegerType(0),
        rewriter.getIntegerAttr(rewriter.getIntegerType(0), 0));
  WrapValidReadyOp wrapRecv = rewriter.create<WrapValidReadyOp>(
      loc, castedRecvData, recvValidFromCosim);
  recvReady.setValue(wrapRecv.getReady());

  // Replace the CosimEndpointOp op.
  rewriter.replaceOp(ep, wrapRecv.getChanOutput());

  return success();
}

namespace {
/// Lower `CompressedManifestOps` ops to a module containing an on-chip ROM.
/// Said module has registered input and outputs, so it has two cycles latency
/// between changing the address and the data being reflected on the output.
struct ManifestRomLowering : public OpConversionPattern<CompressedManifestOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  constexpr static StringRef manifestRomName = "__ESI_Manifest_ROM";

  LogicalResult
  matchAndRewrite(CompressedManifestOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

protected:
  LogicalResult createRomModule(CompressedManifestOp op,
                                ConversionPatternRewriter &rewriter) const;
};
} // anonymous namespace

LogicalResult ManifestRomLowering::createRomModule(
    CompressedManifestOp op, ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  auto mlirModBody = op->getParentOfType<mlir::ModuleOp>();
  rewriter.setInsertionPointToStart(mlirModBody.getBody());

  // Find possible existing module (which may have been created as a dummy
  // module) and erase it.
  if (Operation *existingExtern = mlirModBody.lookupSymbol(manifestRomName)) {
    if (!isa<hw::HWModuleExternOp>(existingExtern))
      return rewriter.notifyMatchFailure(
          op,
          "Found " + manifestRomName + " but it wasn't an HWModuleExternOp");
    rewriter.eraseOp(existingExtern);
  }

  // Create the real module.
  PortInfo ports[] = {
      {{rewriter.getStringAttr("clk"), rewriter.getType<seq::ClockType>(),
        ModulePort::Direction::Input}},
      {{rewriter.getStringAttr("address"), rewriter.getIntegerType(29),
        ModulePort::Direction::Input}},
      {{rewriter.getStringAttr("data"), rewriter.getI64Type(),
        ModulePort::Direction::Output}},
  };
  auto rom = rewriter.create<HWModuleOp>(
      loc, rewriter.getStringAttr(manifestRomName), ports);
  Block *romBody = rom.getBodyBlock();
  rewriter.setInsertionPointToStart(romBody);
  Value clk = romBody->getArgument(0);
  Value inputAddress = romBody->getArgument(1);

  // Manifest the compressed manifest into 64-bit words.
  ArrayRef<uint8_t> maniBytes = op.getCompressedManifest().getData();
  SmallVector<uint64_t> words;
  words.push_back(maniBytes.size());

  for (size_t i = 0; i < maniBytes.size() - 7; i += 8) {
    uint64_t word = 0;
    for (size_t b = 0; b < 8; ++b)
      word |= static_cast<uint64_t>(maniBytes[i + b]) << (8 * b);
    words.push_back(word);
  }
  size_t overHang = maniBytes.size() % 8;
  if (overHang != 0) {
    uint64_t word = 0;
    for (size_t i = 0; i < overHang; ++i)
      word |= static_cast<uint64_t>(maniBytes[maniBytes.size() - overHang + i])
              << (i * 8);
    words.push_back(word);
  }

  // From the words, create an the register which will hold the manifest (and
  // hopefully synthized to a ROM).
  SmallVector<Attribute> wordAttrs;
  for (uint64_t word : words)
    wordAttrs.push_back(rewriter.getI64IntegerAttr(word));
  auto manifestConstant = rewriter.create<hw::AggregateConstantOp>(
      loc, hw::UnpackedArrayType::get(rewriter.getI64Type(), words.size()),
      rewriter.getArrayAttr(wordAttrs));
  auto manifestReg =
      rewriter.create<sv::RegOp>(loc, manifestConstant.getType());
  rewriter.create<sv::AssignOp>(loc, manifestReg, manifestConstant);

  // Slim down the address, register it, do the lookup, and register the output.
  size_t addrBits = llvm::Log2_64_Ceil(words.size());
  auto slimmedIdx =
      rewriter.create<comb::ExtractOp>(loc, inputAddress, 0, addrBits);
  Value inputAddresReg = rewriter.create<seq::CompRegOp>(loc, slimmedIdx, clk);
  auto readIdx =
      rewriter.create<sv::ArrayIndexInOutOp>(loc, manifestReg, inputAddresReg);
  auto readData = rewriter.create<sv::ReadInOutOp>(loc, readIdx);
  Value readDataReg = rewriter.create<seq::CompRegOp>(loc, readData, clk);
  if (auto *term = romBody->getTerminator())
    rewriter.eraseOp(term);
  rewriter.create<hw::OutputOp>(loc, ValueRange{readDataReg});
  return success();
}

LogicalResult ManifestRomLowering::matchAndRewrite(
    CompressedManifestOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  LogicalResult ret = createRomModule(op, rewriter);
  rewriter.eraseOp(op);
  return ret;
}

namespace {
/// Lower `CompressedManifestOps` ops to a SystemVerilog module which sets the
/// Cosim manifest using a DPI support module.
struct CosimManifestLowering : public ManifestRomLowering {
public:
  using ManifestRomLowering::ManifestRomLowering;

  LogicalResult
  matchAndRewrite(CompressedManifestOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // anonymous namespace

LogicalResult CosimManifestLowering::matchAndRewrite(
    CompressedManifestOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  MLIRContext *ctxt = rewriter.getContext();
  Location loc = op.getLoc();

  // Cosim can optionally include a manifest simulation, so produce it in case
  // the Cosim BSP wants it.
  LogicalResult ret = createRomModule(op, rewriter);
  if (failed(ret))
    return ret;

  // Declare external module.
  Attribute params[] = {
      ParamDeclAttr::get("COMPRESSED_MANIFEST_SIZE", rewriter.getI32Type())};
  PortInfo ports[] = {
      {{rewriter.getStringAttr("compressed_manifest"),
        rewriter.getType<hw::ArrayType>(
            rewriter.getI8Type(),
            ParamDeclRefAttr::get(
                rewriter.getStringAttr("COMPRESSED_MANIFEST_SIZE"),
                rewriter.getI32Type())),
        ModulePort::Direction::Input},
       0},
  };
  rewriter.setInsertionPointToEnd(
      op->getParentOfType<mlir::ModuleOp>().getBody());
  auto cosimManifestExternModule = rewriter.create<HWModuleExternOp>(
      loc, rewriter.getStringAttr("Cosim_Manifest"), ports, "Cosim_Manifest",
      ArrayAttr::get(ctxt, params));

  hw::ModulePortInfo portInfo({});
  auto manifestMod = rewriter.create<hw::HWModuleOp>(
      loc, rewriter.getStringAttr("__ESIManifest"), portInfo,
      [&](OpBuilder &rewriter, const hw::HWModulePortAccessor &) {
        // Assemble the manifest data into a constant.
        SmallVector<Attribute> bytes;
        for (uint8_t b : op.getCompressedManifest().getData())
          bytes.push_back(rewriter.getI8IntegerAttr(b));
        auto manifestConstant = rewriter.create<hw::AggregateConstantOp>(
            loc, hw::ArrayType::get(rewriter.getI8Type(), bytes.size()),
            rewriter.getArrayAttr(bytes));
        auto manifestLogic =
            rewriter.create<sv::LogicOp>(loc, manifestConstant.getType());
        rewriter.create<sv::AssignOp>(loc, manifestLogic, manifestConstant);
        auto manifest = rewriter.create<sv::ReadInOutOp>(loc, manifestLogic);

        // Then instantiate the external module.
        rewriter.create<hw::InstanceOp>(
            loc, cosimManifestExternModule, "__manifest",
            ArrayRef<Value>({manifest}),
            rewriter.getArrayAttr({ParamDeclAttr::get(
                "COMPRESSED_MANIFEST_SIZE",
                rewriter.getI32IntegerAttr(bytes.size()))}));
      });

  rewriter.setInsertionPoint(op);
  rewriter.create<hw::InstanceOp>(loc, manifestMod, "__manifest",
                                  ArrayRef<Value>({}));

  rewriter.eraseOp(op);
  return success();
}
void ESItoHWPass::runOnOperation() {
  auto top = getOperation();
  auto *ctxt = &getContext();

  ConversionTarget noBundlesTarget(*ctxt);
  noBundlesTarget.markUnknownOpDynamicallyLegal(
      [](Operation *) { return true; });
  noBundlesTarget.addIllegalOp<PackBundleOp>();
  noBundlesTarget.addIllegalOp<UnpackBundleOp>();
  RewritePatternSet bundlePatterns(&getContext());
  bundlePatterns.add<CanonicalizerOpLowering<PackBundleOp>>(&getContext());
  bundlePatterns.add<CanonicalizerOpLowering<UnpackBundleOp>>(&getContext());
  if (failed(applyPartialConversion(getOperation(), noBundlesTarget,
                                    std::move(bundlePatterns))))
    signalPassFailure();

  // Set up a conversion and give it a set of laws.
  ConversionTarget pass1Target(*ctxt);
  pass1Target.addLegalDialect<comb::CombDialect>();
  pass1Target.addLegalDialect<HWDialect>();
  pass1Target.addLegalDialect<SVDialect>();
  pass1Target.addLegalDialect<seq::SeqDialect>();
  pass1Target.addLegalOp<WrapValidReadyOp, UnwrapValidReadyOp>();

  pass1Target.addIllegalOp<WrapSVInterfaceOp, UnwrapSVInterfaceOp>();
  pass1Target.addIllegalOp<PipelineStageOp>();
  pass1Target.addIllegalOp<CompressedManifestOp>();

  // Add all the conversion patterns.
  ESIHWBuilder esiBuilder(top);
  RewritePatternSet pass1Patterns(ctxt);
  pass1Patterns.insert<PipelineStageLowering>(esiBuilder, ctxt);
  pass1Patterns.insert<WrapInterfaceLower>(ctxt);
  pass1Patterns.insert<UnwrapInterfaceLower>(ctxt);
  pass1Patterns.insert<CosimToHostLowering>(esiBuilder);
  pass1Patterns.insert<CosimFromHostLowering>(esiBuilder);
  pass1Patterns.insert<NullSourceOpLowering>(ctxt);

  if (platform == Platform::cosim)
    pass1Patterns.insert<CosimManifestLowering>(ctxt);
  else if (platform == Platform::fpga)
    pass1Patterns.insert<ManifestRomLowering>(ctxt);
  else
    pass1Patterns.insert<RemoveOpLowering<CompressedManifestOp>>(ctxt);

  // Run the conversion.
  if (failed(
          applyPartialConversion(top, pass1Target, std::move(pass1Patterns))))
    signalPassFailure();

  ConversionTarget pass2Target(*ctxt);
  pass2Target.addLegalDialect<comb::CombDialect>();
  pass2Target.addLegalDialect<HWDialect>();
  pass2Target.addLegalDialect<SVDialect>();
  pass2Target.addIllegalDialect<ESIDialect>();

  RewritePatternSet pass2Patterns(ctxt);
  pass2Patterns.insert<CanonicalizerOpLowering<UnwrapFIFOOp>>(ctxt);
  pass2Patterns.insert<CanonicalizerOpLowering<WrapFIFOOp>>(ctxt);
  pass2Patterns.insert<RemoveWrapUnwrap>(ctxt);
  if (failed(
          applyPartialConversion(top, pass2Target, std::move(pass2Patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> circt::esi::createESItoHWPass() {
  return std::make_unique<ESItoHWPass>();
}
