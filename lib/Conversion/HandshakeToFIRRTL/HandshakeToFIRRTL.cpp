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
  if (type.isa<BundleType>()) {
    return type.dyn_cast<BundleType>();
  }

  using BundleElement = std::pair<Identifier, FIRRTLType>;
  llvm::SmallVector<BundleElement, 3> elements;

  auto dataId = Identifier::get("data", type.getContext());

  if (type.isSignedInteger() || type.isUnsignedInteger() || 
      type.isSignlessInteger()) {

    // Construct the data field of the FIRRTL bundle
    auto width = type.dyn_cast<IntegerType>().getWidth();

    if (type.isSignedInteger() || type.isSignlessInteger()) {
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

  auto validId = Identifier::get("valid", type.getContext());
  auto validType = UIntType::get(type.getContext(), 1);
  auto flipValidType = FlipType::get(validType);

  auto readyId = Identifier::get("ready", type.getContext());
  auto readyType = UIntType::get(type.getContext(), 1);
  auto flipReadyType = FlipType::get(readyType);

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

void mergeToEntryBlock(FModuleOp &moduleOp,
                       ConversionPatternRewriter &rewriter) {
  
  // Get pointers of the first two blocks
  auto iterator = moduleOp.getBody().begin();
  Block *dstBlock = &*iterator;
  Block *srcBlock = &*(++ iterator);
  auto *termOp = dstBlock->getTerminator();

  // Connect all uses of each argument of the second block to the corresponding
  // argument of the first block
  for (int i = 0, e = srcBlock->getNumArguments(); i < e; i ++) {
    auto srcArgument = srcBlock->getArgument(i);
    auto dstArgument = dstBlock->getArgument(i);
    srcArgument.replaceAllUsesWith(dstArgument);
  }

  // Move all operations of the second block to the first block
  while(!srcBlock->empty()){
    Operation *op = &srcBlock->front();
    op->moveBefore(termOp);
  }
  rewriter.eraseBlock(srcBlock);
  //rewriter.eraseOp(termOp);
}

// Only support single input merge (merge -> connect)
void convertMergeOp(FModuleOp &moduleOp,
                    ConversionPatternRewriter &rewriter) {
  for (Block &block : moduleOp) {
    for (Operation &op : block) {
      if (auto mergeOp = dyn_cast<handshake::MergeOp>(op)) {
        rewriter.setInsertionPointAfter(&op);

        // Create wire for the results of merge operation
        auto result = op.getResult(0);
        auto resultType = result.getType();
        auto resultBundleType = getBundleType(resultType, false);

        auto wireOp = rewriter.create<WireOp>(op.getLoc(), resultBundleType,
                                      rewriter.getStringAttr(""));
        auto newResult = wireOp.getResult();
        result.replaceAllUsesWith(newResult);

        // Create connection between operands and result
        rewriter.setInsertionPointAfter(wireOp);
        if (op.getNumOperands() == 1) {
          auto operand = op.getOperand(0);
          rewriter.create<ConnectOp>(wireOp.getLoc(), newResult, operand);
        }
        rewriter.eraseOp(&op);
      }
    }
  }
}

//ArrayRef<std::pair<StringAttr, FIRRTLType>> getFModulePorts(
//    Operation &op, ConversionPatternRewriter &rewriter) {
//  using ModulePort = std::pair<StringAttr, FIRRTLType>;
//  llvm::SmallVector<ModulePort, 4> modulePorts;
//
//  // Add all the input ports
//  int ins_idx = 0;
//  for (auto portType : op.getOperands().getTypes()) {
//    StringAttr portName = 
//        rewriter.getStringAttr("arg" + std::to_string(ins_idx));
//
//    // Convert normal type to FIRRTL bundle type
//    FIRRTLType bundlePortType = getBundleType(portType, false);
//    ModulePort modulePort = std::make_pair(portName, bundlePortType);
//    modulePorts.push_back(modulePort);
//    ins_idx += 1;
//  }
//
//  // Add all the output ports
//  int outs_idx = ins_idx;
//  for (auto portType : op.getResults().getTypes()) {
//    StringAttr portName = 
//        rewriter.getStringAttr("arg" + std::to_string(outs_idx));
//
//    // Convert normal type to FIRRTL bundle type
//    FIRRTLType bundlePortType = getBundleType(portType, true);
//    ModulePort modulePort = std::make_pair(portName, bundlePortType);
//    modulePorts.push_back(modulePort);
//    outs_idx += 1;
//  }
//
//  return ArrayRef<ModulePort>(modulePorts);
//}

std::map<StringRef, SubfieldOp> createSubfieldOps(BlockArgument port, 
    Location termOpLoc, FIRRTLType dataType, 
    FIRRTLType validType, FIRRTLType readyType, 
    ConversionPatternRewriter &rewriter) {
  std::map<StringRef, SubfieldOp> subfieldOpMap;

  auto dataOp = rewriter.create<SubfieldOp>(
      termOpLoc, dataType, port, rewriter.getStringAttr("data"));
  subfieldOpMap.insert(std::pair<StringRef, SubfieldOp>(
      StringRef("data"), dataOp));

  auto validOp = rewriter.create<SubfieldOp>(
      termOpLoc, validType, port, rewriter.getStringAttr("valid"));
  subfieldOpMap.insert(std::pair<StringRef, SubfieldOp>(
      StringRef("valid"), validOp));

  auto readyOp = rewriter.create<SubfieldOp>(
      termOpLoc, readyType, port, rewriter.getStringAttr("ready"));
  subfieldOpMap.insert(std::pair<StringRef, SubfieldOp>(
      StringRef("ready"), readyOp));
  
  return subfieldOpMap;
}

std::map<StringRef, firrtl::ConstantOp> createLowHighConstantOps(
    BlockArgument port, Location termOpLoc, FIRRTLType dataType, 
    ConversionPatternRewriter &rewriter){
  std::map<StringRef, firrtl::ConstantOp> LowHighConstantOpMap;

  auto signalType = UIntType::get(rewriter.getContext(), 1);
  auto signalStdType = IntegerType::get(1, IntegerType::Unsigned, 
                                        rewriter.getContext());

  // Create zero data constant op
  auto dataStdType = IntegerType::get(dataType.getBitWidthOrSentinel(), 
      IntegerType::Signed, rewriter.getContext());
  auto zeroDataAttr = rewriter.getIntegerAttr(dataStdType, 0);

  auto zeroDataOp = rewriter.create<firrtl::ConstantOp>(
      termOpLoc, dataType, zeroDataAttr);
  LowHighConstantOpMap.insert(std::pair<StringRef, firrtl::ConstantOp>(
      StringRef("zero"), zeroDataOp));

  // Create low signal constant op
  auto lowSignalAttr = rewriter.getIntegerAttr(signalStdType, 0);
  auto lowSignalOp = rewriter.create<firrtl::ConstantOp>(
      termOpLoc, signalType, lowSignalAttr);
  LowHighConstantOpMap.insert(std::pair<StringRef, firrtl::ConstantOp>(
      StringRef("low"), lowSignalOp));

  // Create low signal constant op
  auto highSignalAttr = rewriter.getIntegerAttr(signalStdType, 1);
  auto highSignalOp = rewriter.create<firrtl::ConstantOp>(
      termOpLoc, signalType, highSignalAttr);
  LowHighConstantOpMap.insert(std::pair<StringRef, firrtl::ConstantOp>(
      StringRef("high"), highSignalOp));
  
  return LowHighConstantOpMap;
}

template <typename FIRRTLOpType>
FModuleOp createBinaryOpModule(FModuleOp moduleOp, Operation &binaryOp, 
                               ConversionPatternRewriter &rewriter) {
  // Create new FIRRTL module for binary operation
  rewriter.setInsertionPoint(moduleOp);
  using ModulePort = std::pair<StringAttr, FIRRTLType>;
  llvm::SmallVector<ModulePort, 4> modulePorts;

  // Add all the input ports
  int ins_idx = 0;
  for (auto portType : binaryOp.getOperands().getTypes()) {
    StringAttr portName = 
        rewriter.getStringAttr("arg" + std::to_string(ins_idx));

    // Convert normal type to FIRRTL bundle type
    FIRRTLType bundlePortType = getBundleType(portType, false);
    ModulePort modulePort = std::make_pair(portName, bundlePortType);
    modulePorts.push_back(modulePort);
    ins_idx += 1;
  }

  // Add all the output ports
  int outs_idx = ins_idx;
  for (auto portType : binaryOp.getResults().getTypes()) {
    StringAttr portName = 
        rewriter.getStringAttr("arg" + std::to_string(outs_idx));

    // Convert normal type to FIRRTL bundle type
    FIRRTLType bundlePortType = getBundleType(portType, true);
    ModulePort modulePort = std::make_pair(portName, bundlePortType);
    modulePorts.push_back(modulePort);
    outs_idx += 1;
  }

  auto binaryOpModule = rewriter.create<FModuleOp>(moduleOp.getLoc(), 
      rewriter.getStringAttr(binaryOp.getName().getStringRef()), 
      ArrayRef<ModulePort>(modulePorts));
  
  auto &entryBlock = binaryOpModule.getBody().front();
  auto arg0Port = entryBlock.getArgument(0);
  auto arg1Port = entryBlock.getArgument(1);
  auto resultPort = entryBlock.getArgument(2);
  auto *termOp = entryBlock.getTerminator();

  // Get all useful types
  StringRef dataString = StringRef("data");
  StringRef validString = StringRef("valid");
  StringRef readyString = StringRef("ready");

  auto argsType = arg0Port.getType().dyn_cast<BundleType>();
  auto dataType = argsType.getElementType(dataString);
  auto validType = argsType.getElementType(validString);
  auto flipReadyType = argsType.getElementType(readyString);

  auto resultType = resultPort.getType().dyn_cast<BundleType>();
  auto flipDataType = resultType.getElementType(dataString);
  auto flipValidType = resultType.getElementType(validString);
  auto readyType = resultType.getElementType(readyString);

  // Construct arg0, arg1, result, constant signals
  rewriter.setInsertionPoint(termOp);
  auto arg0Map = createSubfieldOps(arg0Port, termOp->getLoc(), 
      dataType, validType, flipReadyType, rewriter);
  auto arg1Map = createSubfieldOps(arg1Port, termOp->getLoc(), 
      dataType, validType, flipReadyType, rewriter);
  auto resultMap = createSubfieldOps(resultPort, termOp->getLoc(), 
      flipDataType, flipValidType, readyType, rewriter);
  auto constantMap = createLowHighConstantOps(arg0Port, termOp->getLoc(), 
                                              dataType, rewriter);

  // Construct combinational logics
  auto arg0Valid = arg0Map[validString].getResult();
  auto arg1Valid = arg1Map[validString].getResult();
  auto argsValid = rewriter.create<AndPrimOp>(
      termOp->getLoc(), validType, arg0Valid, arg1Valid);

  auto resultReady = resultMap[readyString].getResult();
  auto condition = rewriter.create<AndPrimOp>(
      termOp->getLoc(), validType, argsValid, resultReady);
    
  auto arg0Data = arg0Map[dataString].getResult();
  auto arg1Data = arg1Map[dataString].getResult();
  auto result = rewriter.create<FIRRTLOpType>(
      termOp->getLoc(), dataType, arg0Data, arg1Data);
  
  //TODO
  auto whenOp = rewriter.create<WhenOp>(
      termOp->getLoc(), condition, true);

  return binaryOpModule;
}

Type getInstBundleType(FModuleOp &opModule) {
  using BundleElement = std::pair<Identifier, FIRRTLType>;
  llvm::SmallVector<BundleElement, 3> elements;

  int arg_idx = 0;
  for (auto &arg : opModule.getArguments()) {
    std::string argName = "arg" + std::to_string(arg_idx);
    auto argId = Identifier::get(argName, opModule.getContext());
    auto argType = FlipType::get(arg.getType().dyn_cast<BundleType>());
    BundleElement argElement = std::make_pair(argId, argType);
    elements.push_back(argElement);
    arg_idx += 1;
  }

  ArrayRef<BundleElement> BundleElements = ArrayRef<BundleElement>(elements);
  return BundleType::get(BundleElements, opModule.getContext());
}

void createInstOp(FModuleOp &opModule, Operation &op, int inst_idx, 
                  ConversionPatternRewriter &rewriter) {
  auto instType = getInstBundleType(opModule);
  auto moduleName = opModule.getOperationName();
  auto instName = moduleName + std::to_string(inst_idx);
  auto instNameAttr = rewriter.getStringAttr(instName.getSingleStringRef());
  rewriter.create<firrtl::InstanceOp>(op.getLoc(), instType, 
                                      moduleName, instNameAttr);
  
  //TODO
}

void convertStandardOp(FModuleOp moduleOp,
                       ConversionPatternRewriter &rewriter) {
  for (Block &block : moduleOp) {
    int inst_idx = 0;
    for (Operation &op : block) {
      if (dyn_cast<AddIOp>(op)) {
        FModuleOp binaryOpModule = 
            createBinaryOpModule<firrtl::AddPrimOp>(moduleOp, op, rewriter);
        createInstOp(binaryOpModule, op, inst_idx, rewriter);

        // For debug
        Region *currentRegion = moduleOp.getParentRegion();
        for (auto &block : *currentRegion) {
          for (auto &op : block) {
            llvm::outs() << op << "\n";
          }
        }
      }
      inst_idx += 1;
    }
  }
}

// Only support single block function op
void convertReturnOp(FModuleOp &moduleOp,
                     ConversionPatternRewriter &rewriter, uint numInput) {
  for (Block &block : moduleOp) {
    for (Operation &op : block) {
      if (auto returnOp = dyn_cast<handshake::ReturnOp>(op)) {
        rewriter.setInsertionPointAfter(&op);

        for (int i = 0, e = op.getNumOperands(); i < e; i ++) {
          auto operand = op.getOperand(i);
          auto result = block.getArgument(numInput + i);
          rewriter.create<ConnectOp>(op.getLoc(), result, operand);
        }
        rewriter.eraseOp(&op);
      }
    }
  }
}

namespace {
class FIRRTLTypeConverter : public TypeConverter {
public:
  FIRRTLTypeConverter() {
    addConversion(convertType);
  };

  static Optional<Type> convertType(Type type) {
    if (type.isa<FIRRTLType>()) {
      return type;
    } else {
      return getBundleType(type, false);
    }
  };
};
} // End anonymous namespace

struct HandshakeFuncOpLowering : public OpConversionPattern<handshake::FuncOp> {
  using OpConversionPattern<handshake::FuncOp>::OpConversionPattern;

  LogicalResult match(Operation *op) const override {
    return success();
  }

  void rewrite(handshake::FuncOp funcOp, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const override {
    using ModulePort = std::pair<StringAttr, FIRRTLType>;
    llvm::SmallVector<ModulePort, 8> modulePorts;

    // Signature conversion (converts function arguments)
    int arg_count = funcOp.getNumArguments() + funcOp.getNumResults();
    TypeConverter::SignatureConversion newSignature(arg_count);

    // Add all the input ports
    int ins_idx = 0;
    for (auto &arg : funcOp.getArguments()) {
      mlir::Type portType = arg.getType();
      StringAttr portName = 
          rewriter.getStringAttr("arg" + std::to_string(ins_idx));

      // Convert normal type to FIRRTL bundle type
      FIRRTLType bundlePortType = getBundleType(portType, false);
      ModulePort modulePort = std::make_pair(portName, bundlePortType);
      modulePorts.push_back(modulePort);

      // Remap inputs of the old signature
      newSignature.addInputs(ins_idx, bundlePortType);
      ins_idx += 1;
    }

    // Add all the output ports
    int outs_idx = ins_idx;
    for (auto portType : funcOp.getType().getResults()) {
      StringAttr portName = 
          rewriter.getStringAttr("arg" + std::to_string(outs_idx));

      // Convert normal type to FIRRTL bundle type
      FIRRTLType bundlePortType = getBundleType(portType, true);
      ModulePort modulePort = std::make_pair(portName, bundlePortType);
      modulePorts.push_back(modulePort);

      // Add outputs to the new signature
      newSignature.addInputs(bundlePortType);
      outs_idx += 1;
    }

    auto moduleOp = rewriter.create<FModuleOp>(
        funcOp.getLoc(), rewriter.getStringAttr(funcOp.getName()), 
        ArrayRef<ModulePort>(modulePorts));
    rewriter.inlineRegionBefore(funcOp.getBody(), moduleOp.getBody(),
                                moduleOp.end());
    
    mergeToEntryBlock(moduleOp, rewriter);
    convertMergeOp(moduleOp, rewriter);
    convertStandardOp(moduleOp, rewriter);
    convertReturnOp(moduleOp, rewriter, ins_idx);

    //rewriter.applySignatureConversion(&moduleOp.getBody(), newSignature);
    rewriter.eraseOp(funcOp);
  }
};

namespace {
class HandshakeToFIRRTLPass 
    : public mlir::PassWrapper<HandshakeToFIRRTLPass, 
                               OperationPass<handshake::FuncOp>> {
public:
  void runOnOperation() override {
    auto op = getOperation();

    ConversionTarget target(getContext());
    target.addLegalDialect<FIRRTLDialect>();
    target.addIllegalDialect<handshake::HandshakeOpsDialect>();

    OwningRewritePatternList patterns;
    patterns.insert<HandshakeFuncOpLowering>(op.getContext());

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
