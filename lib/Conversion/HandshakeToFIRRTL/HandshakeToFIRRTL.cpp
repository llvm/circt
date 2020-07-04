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

//===----------------------------------------------------------------------===//
// Auxiliary Functions
//===----------------------------------------------------------------------===//

FIRRTLType buildBundleType(FIRRTLType dataType, bool isFlip, 
                           MLIRContext *context) {
  using BundleElement = std::pair<Identifier, FIRRTLType>;
  llvm::SmallVector<BundleElement, 3> elements;

  // Construct the data field of the FIRRTL bundle if not a NoneType
  if (dataType) {
    Identifier dataId = Identifier::get("data", context);

    if (isFlip) {
      elements.push_back(std::make_pair(dataId, FlipType::get(dataType)));
    } else {
      elements.push_back(std::make_pair(dataId, dataType));
    }
  }

  // Construct the valid/ready field of the FIRRTL bundle
  Identifier validId = Identifier::get("valid", context);
  Identifier readyId = Identifier::get("ready", context);
  UIntType signalType = UIntType::get(context, 1);

  if (isFlip) {
    elements.push_back(std::make_pair(validId, FlipType::get(signalType)));
    elements.push_back(std::make_pair(readyId, signalType));
  } else {
    elements.push_back(std::make_pair(validId, signalType));
    elements.push_back(std::make_pair(readyId, FlipType::get(signalType)));
  }

  return BundleType::get(ArrayRef<BundleElement>(elements), context);
}

// Convert a standard type to corresponding FIRRTL bundle type
FIRRTLType getBundleType(Type type, bool isFlip) {

  // If the targeted type is already converted to a bundle type elsewhere, 
  // itself will be returned after cast.
  if (type.isa<BundleType>()) {
    return type.cast<BundleType>();
  }

  // Convert old type to a bundle type, currently only support integer or index
  // or none type.
  MLIRContext *context = type.getContext();

  switch (type.getKind()) {
  case StandardTypes::Integer: {
    IntegerType integerType = type.cast<IntegerType>();
    unsigned width = integerType.getWidth();
    
    switch (integerType.getSignedness()) {
      case IntegerType::Signed:
        return buildBundleType(SIntType::get(context, width), isFlip, context);
      case IntegerType::Unsigned:
        return buildBundleType(UIntType::get(context, width), isFlip, context);
      // ISSUE: How to handle signless integers? Should we use the AsSIntPrimOp
      // or AsUIntPrimOp to convert?
      case IntegerType::Signless:
        return buildBundleType(SIntType::get(context, width), isFlip, context);
    }
  }
  // ISSUE: How to handle index integers?
  case StandardTypes::Index: {
    unsigned width = type.cast<IndexType>().kInternalStorageBitWidth;
    return buildBundleType(SIntType::get(context, width), isFlip, context);
  }
  case StandardTypes::None:
    return buildBundleType(nullptr, isFlip, context);
  default:
    return FIRRTLType(nullptr);
  }
}

// Create a low voltage constant operation and return its result value
Value createNegOp(Location insertLoc, ConversionPatternRewriter &rewriter) {
  auto signalType = UIntType::get(rewriter.getContext(), 1);
  auto StandardUIntType = IntegerType::get(1, IntegerType::Unsigned, 
                                           rewriter.getContext());
  auto lowSignalAttr = rewriter.getIntegerAttr(StandardUIntType, 0);
  auto lowSignalOp = rewriter.create<firrtl::ConstantOp>(
      insertLoc, signalType, lowSignalAttr);
  return lowSignalOp.getResult();
}

// Create a high voltage constant operation and return its result value
Value createPosOp(Location insertLoc, ConversionPatternRewriter &rewriter) {
  auto signalType = UIntType::get(rewriter.getContext(), 1);
  auto StandardUIntType = IntegerType::get(1, IntegerType::Unsigned, 
                                           rewriter.getContext());
  auto highSignalAttr = rewriter.getIntegerAttr(StandardUIntType, 1);
  auto highSignalOp = rewriter.create<firrtl::ConstantOp>(
      insertLoc, signalType, highSignalAttr);
  return highSignalOp.getResult();
}

//===----------------------------------------------------------------------===//
// Create Top FIRRTL Module Functions
//===----------------------------------------------------------------------===//

FModuleOp createTopModuleOp(handshake::FuncOp funcOp, 
                            ConversionPatternRewriter &rewriter) {
  using ModulePort = std::pair<StringAttr, FIRRTLType>;
  llvm::SmallVector<ModulePort, 8> modulePorts;

  // Add all the input ports
  unsigned args_idx = 0;
  for (auto &arg : funcOp.getArguments()) {
    Type portType = arg.getType();
    StringAttr portName = 
        rewriter.getStringAttr("arg" + std::to_string(args_idx));

    FIRRTLType bundlePortType = getBundleType(portType, false);
    modulePorts.push_back(std::make_pair(portName, bundlePortType));
    args_idx += 1;
  }

  // Add all the output ports
  for (auto portType : funcOp.getType().getResults()) {
    StringAttr portName = 
        rewriter.getStringAttr("arg" + std::to_string(args_idx));

    FIRRTLType bundlePortType = getBundleType(portType, true);
    modulePorts.push_back(std::make_pair(portName, bundlePortType));
    args_idx += 1;
  }

  // Create top FIRRTL module for subtituting the funcOp
  auto topModuleOp = rewriter.create<FModuleOp>(
      funcOp.getLoc(), rewriter.getStringAttr(funcOp.getName()), 
      ArrayRef<ModulePort>(modulePorts));
  rewriter.inlineRegionBefore(funcOp.getBody(), topModuleOp.getBody(),
                              topModuleOp.end());

  return topModuleOp;
}

// Merge the second block (the block inlined from original funcOp) to the first 
// block (the new created empty block of the top module)
void mergeToEntryBlock(FModuleOp topModuleOp,
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

//===----------------------------------------------------------------------===//
// Create Sub FIRRTL Module Functions
//===----------------------------------------------------------------------===//

// Check whether the same submodule has been created
FModuleOp checkSubModuleOp(FModuleOp topModuleOp, Operation &oldOp) {
  for (auto &op : topModuleOp.getParentRegion()->front()) {
    if (auto subModuleOp = dyn_cast<FModuleOp>(op)) {
      
      // Check whether the name of the current operation is as same as the 
      // targeted operation, if so, return the current operation rather than
      // create a new FModule operation
      if (oldOp.getName().getStringRef() == subModuleOp.getName()) {
        return subModuleOp;
      }
    }
  }
  return FModuleOp(nullptr);
}

// Create a new FModuleOp operation as submodule of the top module
FModuleOp createSubModuleOp(FModuleOp topModuleOp, Operation &oldOp,
                            ConversionPatternRewriter &rewriter) {
  // Create new FIRRTL module
  rewriter.setInsertionPoint(topModuleOp);
  using ModulePort = std::pair<StringAttr, FIRRTLType>;
  llvm::SmallVector<ModulePort, 4> modulePorts;

  // Add all the input ports
  int ins_idx = 0;
  for (auto portType : oldOp.getOperands().getTypes()) {
    StringAttr portName = 
        rewriter.getStringAttr("arg" + std::to_string(ins_idx));

    FIRRTLType bundlePortType = getBundleType(portType, false);
    modulePorts.push_back(std::make_pair(portName, bundlePortType));
    ins_idx += 1;
  }

  // Add all the output ports
  int outs_idx = ins_idx;
  for (auto portType : oldOp.getResults().getTypes()) {
    StringAttr portName = 
        rewriter.getStringAttr("arg" + std::to_string(outs_idx));

    FIRRTLType bundlePortType = getBundleType(portType, true);
    modulePorts.push_back(std::make_pair(portName, bundlePortType));
    outs_idx += 1;
  }

  auto subModule = rewriter.create<FModuleOp>(topModuleOp.getLoc(), 
      rewriter.getStringAttr(oldOp.getName().getStringRef()), 
      ArrayRef<ModulePort>(modulePorts));
  return subModule;
}

//===----------------------------------------------------------------------===//
// Construct Sub-Module Combinational Logic Functions
//===----------------------------------------------------------------------===//

// Extract values of all subfields of all ports of the submodule
ValueVectorList extractSubfields(Block &entryBlock, Location insertLoc,
                                 ConversionPatternRewriter &rewriter) {
  ValueVectorList valueVectorList;

  for (auto &arg : entryBlock.getArguments()) {
    ValueVector valueVector;
    auto argType = arg.getType().cast<BundleType>();
    for (auto &element : argType.getElements()) {
      auto elementName = element.first.strref();
      auto elementType = element.second;
      auto subfieldOp = rewriter.create<SubfieldOp>(insertLoc, 
          elementType, arg, rewriter.getStringAttr(elementName));
      auto value = subfieldOp.getResult();
      valueVector.push_back(value);
    }
    valueVectorList.push_back(valueVector);
  }

  return valueVectorList;
}

// Multiple input merge operation. Now we presume only one input is active, an
// simple arbitration algorithm is used here: the 1th input always has the
// highest priority.
void buildMergeLogic(ValueVectorList subfieldList, Location insertLoc, 
                     ConversionPatternRewriter &rewriter) {

  // Get result subfield values
  auto resultSubfield = subfieldList.back();
  auto resultData = resultSubfield[0];
  auto resultValid = resultSubfield[1];
  auto resultReady = resultSubfield[2];

  // Connect ready signal for all inputs
  for (int i = 0, e = subfieldList.size() - 1; i < e; i ++) {
    auto argReady = subfieldList[i][2];
    rewriter.create<ConnectOp>(insertLoc, argReady, resultReady);
  }

  // Walk through all inputs to create a chain of when operation
  for (int i = 0, e = subfieldList.size() - 1; i < e; i ++) {
    
    // Get current input subfield values
    auto argSubfield = subfieldList[i];
    auto argData = argSubfield[0];
    auto argValid = argSubfield[1];

    // If current input is not the last input, a new when operation will be
    // created, and connections will be created in the thenRegion of the new
    // when operation.
    if (i != e - 1) {
      auto whenOp = rewriter.create<WhenOp>(insertLoc, argValid, true);
      rewriter.setInsertionPointToStart(&whenOp.thenRegion().front());

      rewriter.create<ConnectOp>(insertLoc, resultData, argData);
      rewriter.create<ConnectOp>(insertLoc, resultValid, argValid);
      rewriter.setInsertionPointToStart(&whenOp.elseRegion().front());
    } else {
      rewriter.create<ConnectOp>(insertLoc, resultData, argData);
      rewriter.create<ConnectOp>(insertLoc, resultValid, argValid);
    }
  }
}

void buildControlMergeLogic(ValueVectorList subfieldList, Location insertLoc, 
                            ConversionPatternRewriter &rewriter) {

  // Get result subfield values
  auto numPort = subfieldList.size();
  auto resultSubfield = subfieldList[numPort - 2];
  auto resultValid = resultSubfield[0];
  auto resultReady = resultSubfield[1];

  // The last result of control_merge indicates which input is active.
  auto controlSubfield = subfieldList[numPort - 1];
  auto controlData = controlSubfield[0];
  auto controlValid = controlSubfield[1];
  auto controlReady = controlSubfield[2];

  // Connect ready signal for all inputs
  auto argReadyOp = rewriter.create<AndPrimOp>(
      insertLoc, resultReady.getType(), resultReady, controlReady);
  for (int i = 0, e = numPort - 2; i < e; i ++) {
    auto argReady = subfieldList[i][1];
    rewriter.create<ConnectOp>(insertLoc, argReady, argReadyOp.getResult());
  }

  // Walk through all inputs to create a chain of when operation
  for (int i = 0, e = numPort - 2; i < e; i ++) {

    // Get current input subfield values
    auto argSubfield = subfieldList[i];
    auto argValid = argSubfield[0];

    auto controlType = FlipType::get(controlData.getType().cast<FIRRTLType>());
    auto controlAttr = rewriter.getIntegerAttr(
        IntegerType::get(32, rewriter.getContext()), i);

    // If current input is not the last input, a new when operation will be
    // created, and connections will be created in the thenRegion of the new
    // when operation.
    if (i != e - 1) {
      auto whenOp = rewriter.create<WhenOp>(insertLoc, argValid, true);
      rewriter.setInsertionPointToStart(&whenOp.thenRegion().front());

      auto controlOp = rewriter.create<firrtl::ConstantOp>(
          insertLoc, controlType, controlAttr);
      rewriter.create<ConnectOp>(insertLoc, controlData, controlOp.getResult());
      rewriter.create<ConnectOp>(insertLoc, controlValid, argValid); // Mark
      rewriter.create<ConnectOp>(insertLoc, resultValid, argValid);

      rewriter.setInsertionPointToStart(&whenOp.elseRegion().front());
    } else {
      auto controlOp = rewriter.create<firrtl::ConstantOp>(
          insertLoc, controlType, controlAttr);
      rewriter.create<ConnectOp>(insertLoc, controlData, controlOp.getResult());
      rewriter.create<ConnectOp>(insertLoc, controlValid, argValid); // Mark
      rewriter.create<ConnectOp>(insertLoc, resultValid, argValid);
    }
  }
}

void buildBranchLogic(ValueVectorList subfieldList, Location insertLoc, 
                      ConversionPatternRewriter &rewriter) {
  auto argSubfield = subfieldList[0];
  auto resultSubfield = subfieldList[1];

  bool isControl = (argSubfield.size() == 2 && resultSubfield.size() == 2);
  if (isControl) {
    auto argValid = argSubfield[0];
    auto argReady = argSubfield[1];

    auto resultValid = resultSubfield[0];
    auto resultReady = resultSubfield[1];

    rewriter.create<ConnectOp>(insertLoc, resultValid, argValid);
    rewriter.create<ConnectOp>(insertLoc, argReady, resultReady);
  } else {
    auto argData = argSubfield[0];
    auto argValid = argSubfield[1];
    auto argReady = argSubfield[2];

    auto resultData = resultSubfield[0];
    auto resultValid = resultSubfield[1];
    auto resultReady = resultSubfield[2];

    rewriter.create<ConnectOp>(insertLoc, resultData, argData);
    rewriter.create<ConnectOp>(insertLoc, resultValid, argValid);
    rewriter.create<ConnectOp>(insertLoc, argReady, resultReady);
  }
}

void buildConditionalBranchLogic(
    ValueVectorList subfieldList, Location insertLoc, 
    ConversionPatternRewriter &rewriter) {

}

void buildMuxLogic(ValueVectorList subfieldList, Location insertLoc, 
                   ConversionPatternRewriter &rewriter) {

}

void buildForkLogic(ValueVectorList subfieldList, Location insertLoc, 
                    ConversionPatternRewriter &rewriter) {

}

void buildSinkLogic(ValueVectorList subfieldList, Location insertLoc, 
                    ConversionPatternRewriter &rewriter) {

}

void buildConstantLogic(ValueVectorList subfieldList, Location insertLoc, 
                        ConversionPatternRewriter &rewriter) {

}

void buildJoinLogic(ValueVectorList subfieldList, Location insertLoc, 
                    ConversionPatternRewriter &rewriter) {

}

// Build binary logic for the new sub-module
template <typename OpType>
void buildBinaryLogic(ValueVectorList subfieldList, Location insertLoc, 
                      ConversionPatternRewriter &rewriter) {
  // Get subfields values
  auto arg0Subfield = subfieldList[0];
  auto arg1Subfield = subfieldList[1];
  auto resultSubfield = subfieldList[2];

  auto arg0Data = arg0Subfield[0];
  auto arg0Valid = arg0Subfield[1];
  auto arg0Ready = arg0Subfield[2];

  auto arg1Data = arg1Subfield[0];
  auto arg1Valid = arg1Subfield[1];
  auto arg1Ready = arg1Subfield[2];

  auto resultData = resultSubfield[0];
  auto resultValid = resultSubfield[1];
  auto resultReady = resultSubfield[2];

  // Connect data signals
  auto CombDataOp = rewriter.create<OpType>(
      insertLoc, arg0Data.getType(), arg0Data, arg1Data);
  rewriter.create<ConnectOp>(insertLoc, resultData, CombDataOp.getResult());

  // Connect valid signals
  auto combValidOp = rewriter.create<AndPrimOp>(
      insertLoc, arg0Valid.getType(), arg0Valid, arg1Valid);
  rewriter.create<ConnectOp>(insertLoc, resultValid, combValidOp.getResult());

  // Connect ready signals
  auto combReadyOp = rewriter.create<AndPrimOp>(
      insertLoc, resultReady.getType(), resultReady, combValidOp.getResult());
  rewriter.create<ConnectOp>(insertLoc, arg0Ready, combReadyOp.getResult());
  rewriter.create<ConnectOp>(insertLoc, arg1Ready, combReadyOp.getResult());
}

//===----------------------------------------------------------------------===//
// Convert Old Operations Functions
//===----------------------------------------------------------------------===//

// Create instanceOp in the top FModuleOp region
void createInstOp(Operation &oldOp, FModuleOp subModuleOp, 
                  ConversionPatternRewriter &rewriter) {
  // Convert orignal multiple bundle type port to a flattend bundle type 
  // containing all the origianl bundle ports
  using BundleElement = std::pair<Identifier, FIRRTLType>;
  llvm::SmallVector<BundleElement, 4> elements;
  MLIRContext *context = subModuleOp.getContext();

  int arg_idx = 0;
  for (auto &arg : subModuleOp.getArguments()) {
    std::string argName = "arg" + std::to_string(arg_idx);
    auto argId = Identifier::get(argName, context);

    // All ports of the instance operation are flipped
    auto argType = FlipType::get(arg.getType().dyn_cast<BundleType>());
    elements.push_back(std::make_pair(argId, argType));
    arg_idx += 1;
  }
  auto instType = BundleType::get(ArrayRef<BundleElement>(elements), context);

  // Insert instanceOp
  rewriter.setInsertionPointAfter(&oldOp);
  auto instOp = rewriter.create<firrtl::InstanceOp>(oldOp.getLoc(), 
      instType, oldOp.getName().getStringRef(), rewriter.getStringAttr(""));

  // Connect instanceOp with other operations in the top module
  int port_idx = 0;
  for (auto &element : instType.cast<BundleType>().getElements()) {
    auto elementName = element.first;
    auto elementType = element.second;
    auto subfieldOp = rewriter.create<SubfieldOp>(
        oldOp.getLoc(), elementType, instOp.getResult(), 
        rewriter.getStringAttr(elementName.strref()));
    
    // Connect input ports
    if (port_idx < oldOp.getNumOperands()) {
      auto operand = oldOp.getOperand(port_idx);
      auto result = subfieldOp.getResult();
      rewriter.create<ConnectOp>(oldOp.getLoc(), result, operand);
    }
    
    // Connect output ports
    else {
      auto result = oldOp.getResult(0);
      auto newResult = subfieldOp.getResult();
      result.replaceAllUsesWith(newResult);
    }
    port_idx += 1;
  }
  rewriter.eraseOp(&oldOp);
}

// Only support single block function op
void convertReturnOp(Operation &oldOp, Block &entryBlock, unsigned numInput,
                     ConversionPatternRewriter &rewriter) {
  rewriter.setInsertionPointAfter(&oldOp);
  for (int i = 0, e = oldOp.getNumOperands(); i < e; i ++) {
    rewriter.create<ConnectOp>(oldOp.getLoc(), 
        entryBlock.getArgument(numInput + i), oldOp.getOperand(i));
  }
  rewriter.eraseOp(&oldOp);
}

//===----------------------------------------------------------------------===//
// MLIR Pass Entry
//===----------------------------------------------------------------------===//

struct HandshakeFuncOpLowering : public OpConversionPattern<handshake::FuncOp> {
  using OpConversionPattern<handshake::FuncOp>::OpConversionPattern;

  LogicalResult match(Operation *op) const override {
    return success();
  }

  void rewrite(handshake::FuncOp funcOp, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const override {
    auto topModuleOp = createTopModuleOp(funcOp, rewriter);
    mergeToEntryBlock(topModuleOp, rewriter);
    auto &entryBlock = topModuleOp.getBody().front();

    // Traverse and convert all operations in funcOp.
    for (Operation &op : entryBlock) {
      if (isa<handshake::ReturnOp>(op)) {
        convertReturnOp(op, entryBlock, funcOp.getNumArguments(), rewriter);
      } 
      // This branch take cares of all operations that require to create new 
      // submodules, and instantiation.
      else if (op.getDialect()->getNamespace() != StringRef("firrtl")) {
        // Check whether Sub-module already exists, if not, we will create 
        // and insert a new empty sub-module
        auto subModuleOp = checkSubModuleOp(topModuleOp, op);
        if (!subModuleOp) {
          subModuleOp = createSubModuleOp(topModuleOp, op, rewriter);
          
          auto &entryBlock = subModuleOp.getBody().front();
          auto *termOp = entryBlock.getTerminator();
          auto insertLoc = termOp->getLoc();
          rewriter.setInsertionPoint(termOp);

          auto subfieldList = extractSubfields(entryBlock, insertLoc, rewriter);
          if (isa<AddIOp>(op)) {
            buildBinaryLogic<AddPrimOp>(subfieldList, insertLoc, rewriter);
          } else if (isa<handshake::MergeOp>(op)) {
            buildMergeLogic(subfieldList, insertLoc, rewriter);
          }
        }
        createInstOp(op, subModuleOp, rewriter);
      }
    }
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

    if (failed(applyPartialConversion(op, target, patterns)))
      signalPassFailure();
  }
};
} // end anonymous namespace

void handshake::registerHandshakeToFIRRTLPasses() {
    PassRegistration<HandshakeToFIRRTLPass>(
      "lower-handshake-to-firrtl",
      "Lowering to FIRRTL Dialect");
}
