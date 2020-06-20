//===- HandshakeToFIRRTL.cpp - Translate Handshake into FIRRTL ------------===//
//
// Copyright 2019 The CIRCT Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "circt/Conversion/HandshakeToFIRRTL/HandshakeToFIRRTL.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/FIRRTL/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include <fstream>
#include <iostream>
#include <map>

using namespace mlir;
using namespace mlir::handshake;
using namespace circt;
using namespace circt::firrtl;

// Only support integer type
FIRRTLType getBundleType(mlir::Type type, bool isOutput) {
  auto dataId = Identifier::get("data", type.getContext());
  auto validId = Identifier::get("valid", type.getContext());
  auto readyId = Identifier::get("ready", type.getContext());

  auto validType = UIntType::get(type.getContext(), 1);
  auto readyType = UIntType::get(type.getContext(), 1);
  auto flipValidType = FlipType::get(validType);
  auto flipReadyType = FlipType::get(readyType);

  using BundleElement = std::pair<Identifier, FIRRTLType>;
  llvm::SmallVector<BundleElement, 3> elements;

  if (type.isSignedInteger() || type.isUnsignedInteger()) {

    // Construct the data field of the FIRRTL bundle
    auto width = type.dyn_cast<IntegerType>().getWidth();

    if (type.isSignedInteger()) {
      auto dataType = SIntType::get(type.getContext(), width);
      auto flipDataType = FlipType::get(dataType);

      if (isOutput) {
        BundleElement dataElement = std::make_pair(dataId, flipDataType);
        elements.push_back(dataElement);
      } else {
        BundleElement dataElement = std::make_pair(dataId, dataType);
        elements.push_back(dataElement);
      }
    } else {
      auto dataType = UIntType::get(type.getContext(), width);
      auto flipDataType = FlipType::get(dataType);

      if (isOutput) {
        BundleElement dataElement = std::make_pair(dataId, flipDataType);
        elements.push_back(dataElement);
      } else {
        BundleElement dataElement = std::make_pair(dataId, dataType);
        elements.push_back(dataElement);
      }
    }
  }

  // Construct the valid/ready field of the FIRRTL bundle
  if (isOutput) {
    BundleElement validElement = std::make_pair(validId, flipValidType);
    elements.push_back(validElement);
    BundleElement readyElement = std::make_pair(readyId, readyType);
    elements.push_back(readyElement);
  } else {
    BundleElement validElement = std::make_pair(validId, validType);
    elements.push_back(validElement);
    BundleElement readyElement = std::make_pair(readyId, flipReadyType);
    elements.push_back(readyElement);
  }

  ArrayRef<BundleElement> BundleElements = ArrayRef<BundleElement>(elements);
  return BundleType::get(BundleElements, type.getContext());
}

namespace {
class FIRRTLTypeConverter : public TypeConverter {
public:
  FIRRTLTypeConverter();

  static Optional<Type> convertType(Type type);
};
} // End anonymous namespace

FIRRTLTypeConverter::FIRRTLTypeConverter() { addConversion(convertType); }

Optional<Type> FIRRTLTypeConverter::convertType(Type type) {
  return type;
}

struct FuncOpLowering : public OpConversionPattern<handshake::FuncOp> {
  using OpConversionPattern<handshake::FuncOp>::OpConversionPattern;

  LogicalResult match(Operation *op) const override {
    return success();
  }

  void rewrite(handshake::FuncOp funcOp, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const override {
    using ModulePort = std::pair<StringAttr, FIRRTLType>;
    llvm::SmallVector<ModulePort, 8> modulePorts;

    // Add all the input ports
    int ins_idx = 0;
    for (auto &arg : funcOp.getArguments()) {
      mlir::Type portType = arg.getType();
      StringAttr portName = 
          rewriter.getStringAttr("in" + std::to_string(ins_idx));

      // Convert normal type to FIRRTL bundle type
      FIRRTLType bundlePortType = getBundleType(portType, false);
      ModulePort modulePort = std::make_pair(portName, bundlePortType);
      modulePorts.push_back(modulePort);
      ins_idx += 1;
    }

    // Add all the output ports
    int outs_idx = 0;
    for (auto portType : funcOp.getType().getResults()) {
      StringAttr portName = 
          rewriter.getStringAttr("out" + std::to_string(outs_idx));

      // Convert normal type to FIRRTL bundle type
      FIRRTLType bundlePortType = getBundleType(portType, true);
      ModulePort modulePort = std::make_pair(portName, bundlePortType);
      modulePorts.push_back(modulePort);
      outs_idx += 1;
    }

    auto newFuncOp = rewriter.create<firrtl::FModuleOp>(
        funcOp.getLoc(), rewriter.getStringAttr(funcOp.getName()), 
        ArrayRef<ModulePort>(modulePorts));
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());

    rewriter.eraseOp(funcOp);
  }
};

namespace {
class HandshakeToFIRRTLPass 
    : public mlir::PassWrapper<HandshakeToFIRRTLPass, 
                               OperationPass<ModuleOp>> {
public:
  void runOnOperation() override {
    auto op = getOperation();

    ConversionTarget target(getContext());
    target.addLegalDialect<FIRRTLDialect>();
    target.addIllegalDialect<handshake::HandshakeOpsDialect>();

    OwningRewritePatternList patterns;
    patterns.insert<FuncOpLowering>(op.getContext());

    FIRRTLTypeConverter typeConverter;

    if (failed(applyPartialConversion(op, target, patterns, &typeConverter)))
      signalPassFailure();
  }
};
} // end anonymous namespace

void handshake::registerHandshakeToFIRRTLPasses() {
    PassRegistration<HandshakeToFIRRTLPass>(
      "lower-handshake-to-firrtl",
      "Lowering to FIRRTL Dialect");
}
