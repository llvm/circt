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

// Convert standard type to FIRRTL bundle type
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

// Merge the second block (inlined block from original funcOp) to the first 
// block (the new created empty block of FModuleOp)
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

// Create a new FModuleOp operation as submodule of the top module
FModuleOp createFModuleOp(FModuleOp moduleOp, Operation &originalOp,
                           ConversionPatternRewriter &rewriter) {
  // Create new FIRRTL module for binary operation
  rewriter.setInsertionPoint(moduleOp);
  using ModulePort = std::pair<StringAttr, FIRRTLType>;
  llvm::SmallVector<ModulePort, 4> modulePorts;

  // Add all the input ports
  int ins_idx = 0;
  for (auto portType : originalOp.getOperands().getTypes()) {
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
  for (auto portType : originalOp.getResults().getTypes()) {
    StringAttr portName = 
        rewriter.getStringAttr("arg" + std::to_string(outs_idx));

    // Convert normal type to FIRRTL bundle type
    FIRRTLType bundlePortType = getBundleType(portType, true);
    ModulePort modulePort = std::make_pair(portName, bundlePortType);
    modulePorts.push_back(modulePort);
    outs_idx += 1;
  }

  auto originalOpModule = rewriter.create<FModuleOp>(moduleOp.getLoc(), 
      rewriter.getStringAttr(originalOp.getName().getStringRef()), 
      ArrayRef<ModulePort>(modulePorts));
  return originalOpModule;
}

// Create a series of subfieldOps for extract elements of the bundle type ports
// of the new created FModuleOp
std::map<StringRef, SubfieldOp> createPortSubfieldOps(BlockArgument port, 
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

// Create some auxiliary constantOp: zero value, low, high voltage
std::map<StringRef, firrtl::ConstantOp> createLowHighConstantOps(
    Location termOpLoc, FIRRTLType dataType, 
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

// Support all unary operations
template <typename FIRRTLOpType>
FModuleOp createUnaryOpFModule(FModuleOp moduleOp, Operation &unaryOp, 
                               ConversionPatternRewriter &rewriter) {
  auto unaryOpModule = createFModuleOp(moduleOp, unaryOp, rewriter);
  
  auto &entryBlock = unaryOpModule.getBody().front();
  auto argPort = entryBlock.getArgument(0);
  auto resultPort = entryBlock.getArgument(1);
  auto *termOp = entryBlock.getTerminator();

  // Get all useful types
  StringRef dataString = StringRef("data");
  StringRef validString = StringRef("valid");
  StringRef readyString = StringRef("ready");

  auto argType = argPort.getType().dyn_cast<BundleType>();
  auto dataType = argType.getElementType(dataString);
  auto validType = argType.getElementType(validString);
  auto flipReadyType = argType.getElementType(readyString);

  auto resultType = resultPort.getType().dyn_cast<BundleType>();
  auto flipDataType = resultType.getElementType(dataString);
  auto flipValidType = resultType.getElementType(validString);
  auto readyType = resultType.getElementType(readyString);

  // Construct arg0, arg1, result, constant signal maps
  rewriter.setInsertionPoint(termOp);
  auto argMap = createPortSubfieldOps(argPort, termOp->getLoc(), 
      dataType, validType, flipReadyType, rewriter);
  auto resultMap = createPortSubfieldOps(resultPort, termOp->getLoc(), 
      flipDataType, flipValidType, readyType, rewriter);
  auto constantMap = createLowHighConstantOps(termOp->getLoc(), 
                                              dataType, rewriter);

  // Construct combinational logics
  auto argValid = argMap[validString].getResult();
  auto resultReady = resultMap[readyString].getResult();
  auto condition = rewriter.create<AndPrimOp>(
      termOp->getLoc(), validType, argValid, resultReady);
    
  auto argData = argMap[dataString].getResult();
  auto unaryFIRRTLOp = rewriter.create<FIRRTLOpType>(
      termOp->getLoc(), dataType, argData);

  auto whenOp = rewriter.create<WhenOp>(
      termOp->getLoc(), condition, true);
  auto resultData = resultMap[dataString].getResult();
  auto resultValid = resultMap[validString].getResult();
  auto argReady = argMap[readyString].getResult();

  // Connect then block of the whenOp
  rewriter.setInsertionPoint(whenOp.thenRegion().front().getTerminator());
  
  auto unaryResult = unaryFIRRTLOp.getResult();
  rewriter.create<ConnectOp>(termOp->getLoc(), resultData, unaryResult);
  auto highSignal = constantMap[StringRef("high")].getResult();
  rewriter.create<ConnectOp>(termOp->getLoc(), resultValid, highSignal);
  rewriter.create<ConnectOp>(termOp->getLoc(), argReady, highSignal);

  // Connect else block of the whenOp
  rewriter.setInsertionPoint(whenOp.elseRegion().front().getTerminator());

  auto zeroValue = constantMap[StringRef("zero")].getResult();
  rewriter.create<ConnectOp>(termOp->getLoc(), resultData, zeroValue);
  auto lowSignal = constantMap[StringRef("low")].getResult();
  rewriter.create<ConnectOp>(termOp->getLoc(), resultValid, lowSignal);
  rewriter.create<ConnectOp>(termOp->getLoc(), argReady, lowSignal);

  return unaryOpModule;
}

// Support all binary operations
template <typename FIRRTLOpType>
FModuleOp createBinaryOpFModule(FModuleOp moduleOp, Operation &binaryOp, 
                               ConversionPatternRewriter &rewriter) {
  auto binaryOpModule = createFModuleOp(moduleOp, binaryOp, rewriter);
  
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

  // Construct arg0, arg1, result, constant signal maps
  rewriter.setInsertionPoint(termOp);
  auto arg0Map = createPortSubfieldOps(arg0Port, termOp->getLoc(), 
      dataType, validType, flipReadyType, rewriter);
  auto arg1Map = createPortSubfieldOps(arg1Port, termOp->getLoc(), 
      dataType, validType, flipReadyType, rewriter);
  auto resultMap = createPortSubfieldOps(resultPort, termOp->getLoc(), 
      flipDataType, flipValidType, readyType, rewriter);
  auto constantMap = createLowHighConstantOps(termOp->getLoc(), 
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
  auto binaryFIRRTLOp = rewriter.create<FIRRTLOpType>(
      termOp->getLoc(), dataType, arg0Data, arg1Data);

  auto whenOp = rewriter.create<WhenOp>(
      termOp->getLoc(), condition, true);
  auto resultData = resultMap[dataString].getResult();
  auto resultValid = resultMap[validString].getResult();
  auto arg0Ready = arg0Map[readyString].getResult();
  auto arg1Ready = arg1Map[readyString].getResult();

  // Connect then block of the whenOp
  rewriter.setInsertionPoint(whenOp.thenRegion().front().getTerminator());
  
  auto binaryResult = binaryFIRRTLOp.getResult();
  rewriter.create<ConnectOp>(termOp->getLoc(), resultData, binaryResult);
  auto highSignal = constantMap[StringRef("high")].getResult();
  rewriter.create<ConnectOp>(termOp->getLoc(), resultValid, highSignal);
  rewriter.create<ConnectOp>(termOp->getLoc(), arg0Ready, highSignal);
  rewriter.create<ConnectOp>(termOp->getLoc(), arg1Ready, highSignal);

  // Connect else block of the whenOp
  rewriter.setInsertionPoint(whenOp.elseRegion().front().getTerminator());

  auto zeroValue = constantMap[StringRef("zero")].getResult();
  rewriter.create<ConnectOp>(termOp->getLoc(), resultData, zeroValue);
  auto lowSignal = constantMap[StringRef("low")].getResult();
  rewriter.create<ConnectOp>(termOp->getLoc(), resultValid, lowSignal);
  rewriter.create<ConnectOp>(termOp->getLoc(), arg0Ready, lowSignal);
  rewriter.create<ConnectOp>(termOp->getLoc(), arg1Ready, lowSignal);

  return binaryOpModule;
}

// Convert orignal multiple bundle type port to a flattend bundle type 
// containing all the origianl bundle ports
Type getInstPortBundleType(FModuleOp &subModuleOp) {
  using BundleElement = std::pair<Identifier, FIRRTLType>;
  llvm::SmallVector<BundleElement, 3> elements;

  int arg_idx = 0;
  for (auto &arg : subModuleOp.getArguments()) {
    std::string argName = "arg" + std::to_string(arg_idx);
    auto argId = Identifier::get(argName, subModuleOp.getContext());
    auto argType = FlipType::get(arg.getType().dyn_cast<BundleType>());
    BundleElement argElement = std::make_pair(argId, argType);
    elements.push_back(argElement);
    arg_idx += 1;
  }

  ArrayRef<BundleElement> BundleElements = ArrayRef<BundleElement>(elements);
  return BundleType::get(BundleElements, subModuleOp.getContext());
}

// Create instanceOp in the top FModuleOp region
void createInstOp(FModuleOp &subModuleOp, Operation &originalOp, 
                  ConversionPatternRewriter &rewriter) {
  rewriter.setInsertionPoint(&originalOp);
  auto instType = getInstPortBundleType(subModuleOp);
  auto exprModuleName = subModuleOp.getName();
  auto instNameAttr = rewriter.getStringAttr("");
  auto instOp = rewriter.create<firrtl::InstanceOp>(originalOp.getLoc(), 
      instType, exprModuleName, instNameAttr);
  
  auto instResult = instOp.getResult();
  auto numInputs = originalOp.getNumOperands();

  int port_idx = 0;
  for (auto &element : instType.dyn_cast<BundleType>().getElements()) {
    auto elementName = element.first;
    auto elementType = element.second;
    auto subfieldOp = rewriter.create<SubfieldOp>(
        originalOp.getLoc(), elementType, instResult, 
        rewriter.getStringAttr(elementName.strref()));
    
    // Connect input ports
    if (port_idx < numInputs) {
      auto operand = originalOp.getOperand(port_idx);
      auto result = subfieldOp.getResult();
      rewriter.create<ConnectOp>(originalOp.getLoc(), result, operand);
    }
    
    // Connect output ports
    else {
      auto result = originalOp.getResult(0);
      auto wireOp = rewriter.create<WireOp>(originalOp.getLoc(), 
          elementType, rewriter.getStringAttr(""));
      auto newResult = wireOp.getResult();
      result.replaceAllUsesWith(newResult);

      auto operand = subfieldOp.getResult();
      rewriter.create<ConnectOp>(originalOp.getLoc(), newResult, operand);
    }
    port_idx += 1;
  }
  rewriter.eraseOp(&originalOp);
}

// Convert standardOps to FIRRTL representation
void convertStandardOp(FModuleOp moduleOp,
                       ConversionPatternRewriter &rewriter) {
  for (Block &block : moduleOp) {
    int inst_idx = 0;
    for (Operation &op : block) {
      if (dyn_cast<AddIOp>(op)) {
        auto subModule = createBinaryOpFModule<firrtl::AddPrimOp>(
                         moduleOp, op, rewriter);
        createInstOp(subModule, op, rewriter);
      } else if (dyn_cast<SubIOp>(op)) {
        auto subModule = createBinaryOpFModule<firrtl::SubPrimOp>(
                         moduleOp, op, rewriter);
        createInstOp(subModule, op, rewriter);
      }
      //// For debug
      //Region *currentRegion = moduleOp.getParentRegion();
      //for (auto &block : *currentRegion) {
      //  for (auto &op : block) {
      //    llvm::outs() << op << "\n";
      //  }
      //}
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
