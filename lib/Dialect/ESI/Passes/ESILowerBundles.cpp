//===- ESILowerBundles.cpp - Lower ESI bundles pass -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/PortConverter.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/SymCache.h"

#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace circt::esi;
using namespace circt::esi::detail;
using namespace circt::hw;

namespace {

/// Implement the Valid/Ready signaling standard.
class BundlePort : public PortConversion {
public:
  BundlePort(PortConverterImpl &converter, hw::PortInfo origPort)
      : PortConversion(converter, origPort) {}

protected:
  void mapInputSignals(OpBuilder &b, Operation *inst, Value instValue,
                       SmallVectorImpl<Value> &newOperands,
                       ArrayRef<Backedge> newResults) override;
  void mapOutputSignals(OpBuilder &b, Operation *inst, Value instValue,
                        SmallVectorImpl<Value> &newOperands,
                        ArrayRef<Backedge> newResults) override;

  void buildInputSignals() override;
  void buildOutputSignals() override;

private:
  SmallVector<hw::PortInfo, 4> newInputChannels;
  SmallVector<hw::PortInfo, 4> newOutputChannels;
};

class ESIBundleConversionBuilder : public PortConversionBuilder {
public:
  using PortConversionBuilder::PortConversionBuilder;
  FailureOr<std::unique_ptr<PortConversion>> build(hw::PortInfo port) override {
    return llvm::TypeSwitch<Type, FailureOr<std::unique_ptr<PortConversion>>>(
               port.type)
        .Case([&](esi::ChannelBundleType)
                  -> FailureOr<std::unique_ptr<PortConversion>> {
          return {std::make_unique<BundlePort>(converter, port)};
        })
        .Default([&](auto) { return PortConversionBuilder::build(port); });
  }
};
} // namespace

void BundlePort::mapInputSignals(OpBuilder &b, Operation *inst, Value instValue,
                                 SmallVectorImpl<Value> &newOperands,
                                 ArrayRef<Backedge> newResults) {

  SmallVector<Value, 4> unpackOperands;
  unpackOperands.push_back(inst->getOperand(origPort.argNum));
  llvm::append_range(unpackOperands,
                     llvm::map_range(newOutputChannels, [&](hw::PortInfo port) {
                       return newResults[port.argNum];
                     }));
  SmallVector<Type, 5> unpackResults(llvm::map_range(
      newInputChannels, [](hw::PortInfo port) { return port.type; }));

  auto unpack = OpBuilder(inst).create<UnpackBundleOp>(
      origPort.loc, unpackResults, unpackOperands);

  for (auto [idx, inPort] : llvm::enumerate(newInputChannels))
    newOperands[inPort.argNum] = unpack.getResult(idx);
}

void BundlePort::mapOutputSignals(OpBuilder &b, Operation *inst,
                                  Value instValue,
                                  SmallVectorImpl<Value> &newOperands,
                                  ArrayRef<Backedge> newResults) {
  auto bundleType = cast<ChannelBundleType>(origPort.type);

  SmallVector<Value, 4> packOperands(
      llvm::map_range(newOutputChannels, [&](hw::PortInfo port) {
        return newResults[port.argNum];
      }));
  SmallVector<Type, 5> packResults;
  packResults.push_back(bundleType);
  llvm::append_range(packResults,
                     llvm::map_range(newInputChannels, [](hw::PortInfo port) {
                       return port.type;
                     }));

  auto pack = OpBuilder(inst).create<PackBundleOp>(origPort.loc, packResults,
                                                   packOperands);

  for (auto [idx, inPort] : llvm::enumerate(newInputChannels))
    newOperands[inPort.argNum] = pack.getFromChannels()[idx];
  inst->getResult(origPort.argNum).replaceAllUsesWith(pack.getBundle());
}

void BundlePort::buildInputSignals() {
  auto bundleType = cast<ChannelBundleType>(origPort.type);
  SmallVector<Value, 4> newInputValues;
  SmallVector<BundledChannel, 4> outputChannels;

  SmallVector<Type, 4> packOpResultTypes;
  packOpResultTypes.push_back(bundleType);
  for (BundledChannel ch : bundleType.getChannels()) {
    // 'to' on an input bundle becomes an input channel.
    if (ch.direction == ChannelDirection::to) {
      hw::PortInfo newPort;
      newInputValues.push_back(converter.createNewInput(
          origPort, "_" + ch.name.getValue(), ch.type, newPort));
      newInputChannels.push_back(newPort);
    } else {
      // 'from' on an input bundle becomes an output channel.
      packOpResultTypes.push_back(ch.type);
      outputChannels.push_back(ch);
    }
  }

  PackBundleOp pack;
  if (body) {
    ImplicitLocOpBuilder b(origPort.loc, body, body->begin());
    pack = b.create<PackBundleOp>(packOpResultTypes, newInputValues);
    body->getArgument(origPort.argNum).replaceAllUsesWith(pack.getBundle());
  }

  newOutputChannels.resize(outputChannels.size());
  for (auto [idx, ch] : llvm::enumerate(outputChannels))
    converter.createNewOutput(origPort, "_" + ch.name.getValue(), ch.type,
                              pack ? pack.getFromChannels()[idx] : nullptr,
                              newOutputChannels[idx]);
}

void BundlePort::buildOutputSignals() {
  auto bundleType = cast<ChannelBundleType>(origPort.type);
  SmallVector<Value, 4> unpackOperands;
  if (body)
    unpackOperands.push_back(
        body->getTerminator()->getOperand(origPort.argNum));
  SmallVector<BundledChannel, 4> outputChannels;

  SmallVector<Type, 4> unpackOpResultTypes;
  for (BundledChannel ch : bundleType.getChannels()) {
    // 'from' on an input bundle becomes an input channel.
    if (ch.direction == ChannelDirection::from) {
      hw::PortInfo newPort;
      unpackOperands.push_back(converter.createNewInput(
          origPort, "_" + ch.name.getValue(), ch.type, newPort));
      newInputChannels.push_back(newPort);
    } else {
      // 'to' on an input bundle becomes an output channel.
      unpackOpResultTypes.push_back(ch.type);
      outputChannels.push_back(ch);
    }
  }

  UnpackBundleOp unpack;
  if (body) {
    ImplicitLocOpBuilder b(origPort.loc, body, body->begin());
    unpack = b.create<UnpackBundleOp>(unpackOpResultTypes, unpackOperands);
  }

  newOutputChannels.resize(outputChannels.size());
  for (auto [idx, ch] : llvm::enumerate(outputChannels))
    converter.createNewOutput(origPort, "_" + ch.name.getValue(), ch.type,
                              unpack ? unpack.getToChannels()[idx] : nullptr,
                              newOutputChannels[idx]);
}

namespace {
/// Convert all the ESI bundle ports on modules to channel ports.
struct ESIBundlesPass : public LowerESIBundlesBase<ESIBundlesPass> {
  void runOnOperation() override;
};
} // anonymous namespace

/// Iterate through the `hw.module[.extern]`s and lower their ports.
void ESIBundlesPass::runOnOperation() {
  MLIRContext &ctxt = getContext();
  ModuleOp top = getOperation();

  // Find all modules and run port conversion on them.
  circt::hw::InstanceGraph &instanceGraph =
      getAnalysis<circt::hw::InstanceGraph>();
  for (auto mod : top.getOps<HWMutableModuleLike>()) {
    if (failed(PortConverter<ESIBundleConversionBuilder>(instanceGraph, mod)
                   .run()))
      return signalPassFailure();
  }

  ConversionTarget tgt(ctxt);
  tgt.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  tgt.addIllegalOp<PackBundleOp, UnpackBundleOp>();
  RewritePatternSet patterns(&ctxt);
  PackBundleOp::getCanonicalizationPatterns(patterns, &ctxt);
  UnpackBundleOp::getCanonicalizationPatterns(patterns, &ctxt);
  if (failed(applyPartialConversion(getOperation(), tgt, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
circt::esi::createESIBundleLoweringPass() {
  return std::make_unique<ESIBundlesPass>();
}
