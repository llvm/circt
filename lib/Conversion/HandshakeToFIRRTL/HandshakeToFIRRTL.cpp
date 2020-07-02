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

using ValueVector = std::vector<Value>;
using ValueVectorList = std::vector<ValueVector>;

// Merge the second block (the block inlined from original funcOp) to the first 
// block (the new created empty block of the top module)
void mergeToEntryBlock(FModuleOp &topModuleOp,
                       ConversionPatternRewriter &rewriter) {
  
  // Get pointers of the first two blocks
  auto blockIterator = topModuleOp.getBody().begin();
  Block *entryBlock = &*blockIterator;
  Block *secondBlock = &*(++ blockIterator);
  Operation *termOp = entryBlock->getTerminator();

  // Connect all uses of each argument of the second block to the corresponding
  // argument of the first block
  for (int i = 0, e = secondBlock->getNumArguments(); i < e; i ++) {
    BlockArgument oldArgument = secondBlock->getArgument(i);
    BlockArgument newArgument = entryBlock->getArgument(i);
    oldArgument.replaceAllUsesWith(newArgument);
  }

  // Move all operations of the second block to the first block
  while(!secondBlock->empty()){
    Operation *op = &secondBlock->front();
    op->moveBefore(termOp);
  }
  rewriter.eraseBlock(secondBlock);
}

// TODO: update code structure
template <typename OpType>
FIRRTLType buildBundleType(OpType dataType, bool isOutput, 
                           MLIRContext *context) {
  using BundleElement = std::pair<Identifier, FIRRTLType>;
  llvm::SmallVector<BundleElement, 3> elements;

  // Construct the data field of the FIRRTL bundle if not a NoneType
  if (dataType) {
    Identifier dataId = Identifier::get("data", context);

    if (isOutput) {
      elements.push_back(std::make_pair(dataId, FlipType::get(dataType)));
    } else {
      elements.push_back(std::make_pair(dataId, dataType));
    }
  }

  // Construct the valid/ready field of the FIRRTL bundle
  Identifier validId = Identifier::get("valid", context);
  Identifier readyId = Identifier::get("ready", context);
  UIntType signalType = UIntType::get(context, 1);

  if (isOutput) {
    elements.push_back(std::make_pair(validId, FlipType::get(signalType)));
    elements.push_back(std::make_pair(readyId, signalType));
  } else {
    elements.push_back(std::make_pair(validId, signalType));
    elements.push_back(std::make_pair(readyId, FlipType::get(signalType)));
  }

  return BundleType::get(ArrayRef<BundleElement>(elements), context);
}

// Convert a standard type to corresponding FIRRTL bundle type
FIRRTLType getBundleType(mlir::Type type, bool isOutput) {
  
  // If the targeted type is already converted to a bundle type elsewhere, 
  // itself will be returned after cast.
  if (type.isa<BundleType>()) {
    return type.cast<BundleType>();
  }

  if (type.isSignedInteger()) {
    unsigned width = type.cast<IntegerType>().getWidth();
    SIntType dataType = SIntType::get(type.getContext(), width);
    return buildBundleType<SIntType>(dataType, isOutput, type.getContext());
  }
  else if (type.isUnsignedInteger()) {
    unsigned width = type.cast<IntegerType>().getWidth();
    UIntType dataType = UIntType::get(type.getContext(), width);
    return buildBundleType<UIntType>(dataType, isOutput, type.getContext());
  }
  // ISSUE: How to handle signless integers? Should we use the AsSIntPrimOp or 
  // AsUIntPrimOp to convert? Now we simply consider them as signed integers.
  else if (type.isSignlessInteger()) {
    unsigned width = type.cast<IntegerType>().getWidth();
    SIntType dataType = SIntType::get(type.getContext(), width);
    return buildBundleType<SIntType>(dataType, isOutput, type.getContext());
  }
  // Now we convert index type as signed integers.
  else if (type.isIndex()) {
    unsigned width = type.cast<IndexType>().kInternalStorageBitWidth;
    SIntType dataType = SIntType::get(type.getContext(), width);
    return buildBundleType<SIntType>(dataType, isOutput, type.getContext());
  }
  else {
    SIntType dataType = nullptr;
    return buildBundleType<SIntType>(dataType, isOutput, type.getContext());
  }
}

// Check whether the same submodule has been created
FModuleOp checkSubModuleOp(FModuleOp topModuleOp, Operation &oldOp) {
  Region *currentRegion = topModuleOp.getParentRegion();
  for (auto &block : *currentRegion) {
    for (auto &op : block) {
      auto oldOpName = oldOp.getName().getStringRef();
      if (auto topModuleOp = dyn_cast<FModuleOp>(op)) {
        
        // Check whether the name of the current operation is as same as the 
        // targeted operation, if so, directly return the current operation 
        // rather than create a new FModule operation
        auto currentOpName = topModuleOp.getName();
        if (oldOpName == currentOpName) {
          return topModuleOp;
        }
      }
    }
  }
  return FModuleOp(nullptr);
}

// Create a new FModuleOp operation as submodule of the top module
FModuleOp insertSubModuleOp(FModuleOp topModuleOp, Operation &oldOp,
                            ConversionPatternRewriter &rewriter) {
  // Create new FIRRTL module for binary operation
  rewriter.setInsertionPoint(topModuleOp);
  using ModulePort = std::pair<StringAttr, FIRRTLType>;
  llvm::SmallVector<ModulePort, 4> modulePorts;

  // Add all the input ports
  int ins_idx = 0;
  for (auto portType : oldOp.getOperands().getTypes()) {
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
  for (auto portType : oldOp.getResults().getTypes()) {
    StringAttr portName = 
        rewriter.getStringAttr("arg" + std::to_string(outs_idx));

    // Convert normal type to FIRRTL bundle type
    FIRRTLType bundlePortType = getBundleType(portType, true);
    ModulePort modulePort = std::make_pair(portName, bundlePortType);
    modulePorts.push_back(modulePort);
    outs_idx += 1;
  }

  auto subModule = rewriter.create<FModuleOp>(topModuleOp.getLoc(), 
      rewriter.getStringAttr(oldOp.getName().getStringRef()), 
      ArrayRef<ModulePort>(modulePorts));
  return subModule;
}

// Convert orignal multiple bundle type port to a flattend bundle type 
// containing all the origianl bundle ports
Type getInstOpType(FModuleOp subModuleOp) {
  using BundleElement = std::pair<Identifier, FIRRTLType>;
  llvm::SmallVector<BundleElement, 4> elements;

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
void insertInstOp(mlir::Type instType, Operation &oldOp, 
                  ConversionPatternRewriter &rewriter) {
  rewriter.setInsertionPoint(&oldOp);
  auto subModuleName = oldOp.getName().getStringRef();
  auto instNameAttr = rewriter.getStringAttr("");
  auto instOp = rewriter.create<firrtl::InstanceOp>(oldOp.getLoc(), 
      instType, subModuleName, instNameAttr);
  
  auto instResult = instOp.getResult();
  auto numInputs = oldOp.getNumOperands();

  int port_idx = 0;
  for (auto &element : instType.dyn_cast<BundleType>().getElements()) {
    auto elementName = element.first;
    auto elementType = element.second;
    auto subfieldOp = rewriter.create<SubfieldOp>(
        oldOp.getLoc(), elementType, instResult, 
        rewriter.getStringAttr(elementName.strref()));
    
    // Connect input ports
    if (port_idx < numInputs) {
      auto operand = oldOp.getOperand(port_idx);
      auto result = subfieldOp.getResult();
      rewriter.create<ConnectOp>(oldOp.getLoc(), result, operand);
    }
    
    // Connect output ports
    else {
      auto result = oldOp.getResult(0);
      auto wireOp = rewriter.create<WireOp>(oldOp.getLoc(), 
          elementType, rewriter.getStringAttr(""));
      auto newResult = wireOp.getResult();
      result.replaceAllUsesWith(newResult);

      auto operand = subfieldOp.getResult();
      rewriter.create<ConnectOp>(oldOp.getLoc(), newResult, operand);
    }
    port_idx += 1;
  }
  rewriter.eraseOp(&oldOp);
}

// Extract values of all subfields of all ports of the submodule
ValueVectorList extractSubfields(FModuleOp subModuleOp, 
                                 ConversionPatternRewriter &rewriter) {
  ValueVectorList valueVectorList;
  auto &entryBlock = subModuleOp.getBody().front();
  auto *termOp = entryBlock.getTerminator();
  rewriter.setInsertionPoint(termOp);

  for (auto &arg : entryBlock.getArguments()) {
    ValueVector valueVector;
    auto argType = arg.getType().cast<BundleType>();
    for (auto &element : argType.getElements()) {
      auto elementName = element.first.strref();
      auto elementType = element.second;
      auto subfieldOp = rewriter.create<SubfieldOp>(termOp->getLoc(), 
          elementType, arg, rewriter.getStringAttr(elementName));
      auto value = subfieldOp.getResult();
      valueVector.push_back(value);
    }
    valueVectorList.push_back(valueVector);
  }

  return valueVectorList;
}

// Create a low voltage constant operation and return its result value
Value insertNegOp(Location insertLoc, ConversionPatternRewriter &rewriter) {
  auto signalType = UIntType::get(rewriter.getContext(), 1);
  auto StandardUIntType = IntegerType::get(1, IntegerType::Unsigned, 
                                           rewriter.getContext());
  auto lowSignalAttr = rewriter.getIntegerAttr(StandardUIntType, 0);
  auto lowSignalOp = rewriter.create<firrtl::ConstantOp>(
      insertLoc, signalType, lowSignalAttr);
  return lowSignalOp.getResult();
}

// Create a high voltage constant operation and return its result value
Value insertPosOp(Location insertLoc, ConversionPatternRewriter &rewriter) {
  auto signalType = UIntType::get(rewriter.getContext(), 1);
  auto StandardUIntType = IntegerType::get(1, IntegerType::Unsigned, 
                                           rewriter.getContext());
  auto highSignalAttr = rewriter.getIntegerAttr(StandardUIntType, 1);
  auto highSignalOp = rewriter.create<firrtl::ConstantOp>(
      insertLoc, signalType, highSignalAttr);
  return highSignalOp.getResult();
}

// TODO
//void buildMergeOp(FModuleOp subModuleOp, ValueMapArray subfieldArray, 
//                  ConversionPatternRewriter &rewriter) {
//  auto &entryBlock = subModuleOp.getBody().front();
//  auto *termOp = entryBlock.getTerminator();
//  rewriter.setInsertionPoint(termOp);
//
//
//}

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

// Build binary operations for the new sub-module
template <typename OpType>
void buildBinaryOp(FModuleOp subModuleOp, ValueVectorList subfieldList, 
                   ConversionPatternRewriter &rewriter) {
  auto &entryBlock = subModuleOp.getBody().front();
  auto *termOp = entryBlock.getTerminator();
  rewriter.setInsertionPoint(termOp);

  // Get subfields values
  auto arg0Subfield = subfieldList[0];
  auto arg1Subfiled = subfieldList[1];
  auto resultSubfiled = subfieldList[2];

  auto arg0Data = arg0Subfield[0];
  auto arg0Valid = arg0Subfield[1];
  auto arg0Ready = arg0Subfield[2];

  auto arg1Data = arg1Subfiled[0];
  auto arg1Valid = arg1Subfiled[1];
  auto arg1Ready = arg1Subfiled[2];

  auto resultData = resultSubfiled[0];
  auto resultValid = resultSubfiled[1];
  auto resultReady = resultSubfiled[2];

  // Connect data signals
  auto CombDataOp = rewriter.create<OpType>(termOp->getLoc(), 
      arg0Data.getType(), arg0Data, arg1Data);
  auto combData = CombDataOp.getResult();
  rewriter.create<ConnectOp>(termOp->getLoc(), resultData, combData);

  // Connect valid signals
  auto combValidOp = rewriter.create<AndPrimOp>(termOp->getLoc(), 
      arg0Valid.getType(), arg0Valid, arg1Valid);
  auto combValid = combValidOp.getResult();
  rewriter.create<ConnectOp>(termOp->getLoc(), resultValid, combValid);

  // Connect ready signals
  auto combReadyOp = rewriter.create<AndPrimOp>(termOp->getLoc(), 
      combValid.getType(), combValid, resultReady);
  auto combReady = combReadyOp.getResult();
  rewriter.create<ConnectOp>(termOp->getLoc(), arg0Ready, combReady);
  rewriter.create<ConnectOp>(termOp->getLoc(), arg1Ready, combReady);
}

// Convert operations which require to create new sub-module to FIRRTL 
// representation. (Create sub-module => instantiate => connect)
void convertSubModuleOp(FModuleOp topModuleOp,
                        ConversionPatternRewriter &rewriter) {
  for (Block &block : topModuleOp) {
    for (Operation &op : block) {

      if (isa<AddIOp>(op)) {
        // Check whether Sub-module already exists, if not, we will create and 
        // insert a new empty sub-module
        auto subModuleOp = checkSubModuleOp(topModuleOp, op);
        if (!subModuleOp) {
          subModuleOp = insertSubModuleOp(topModuleOp, op, rewriter);

          // Build the combinational logic in the new created submodule
          auto subfieldList = extractSubfields(subModuleOp, rewriter);
          buildBinaryOp<firrtl::AddPrimOp>(subModuleOp, subfieldList, rewriter);
        }

        // Create and insert instance operation into top-module, and connect with
        // other operations
        auto instOpType = getInstOpType(subModuleOp);
        insertInstOp(instOpType, op, rewriter);
      }
      
      // For debug
      //Region *currentRegion = topModuleOp.getParentRegion();
      //for (auto &block : *currentRegion) {
      //  for (auto &op : block) {
      //    llvm::outs() << op << "\n";
      //  }
      //}
    }
  }
}

// Only support single block function op
void convertReturnOp(FModuleOp &topModuleOp, unsigned numInput,
                     ConversionPatternRewriter &rewriter) {
  for (Block &block : topModuleOp) {
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

    auto topModuleOp = rewriter.create<FModuleOp>(
        funcOp.getLoc(), rewriter.getStringAttr(funcOp.getName()), 
        ArrayRef<ModulePort>(modulePorts));
    rewriter.inlineRegionBefore(funcOp.getBody(), topModuleOp.getBody(),
                                topModuleOp.end());
    
    mergeToEntryBlock(topModuleOp, rewriter);
    convertMergeOp(topModuleOp, rewriter);
    convertSubModuleOp(topModuleOp, rewriter);
    convertReturnOp(topModuleOp, ins_idx, rewriter);

    //rewriter.applySignatureConversion(&topModuleOp.getBody(), newSignature);
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
