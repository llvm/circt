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
#include "circt/Dialect/FIRRTL/Ops.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
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

  // Construct the data field of the FIRRTL bundle if not a NoneType
  if (dataType) {
    Identifier dataId = Identifier::get("data", context);

    if (isFlip)
      elements.push_back(std::make_pair(dataId, FlipType::get(dataType)));
    else
      elements.push_back(std::make_pair(dataId, dataType));
  }

  return BundleType::get(ArrayRef<BundleElement>(elements), context);
}

// Convert a standard type to corresponding FIRRTL bundle type
FIRRTLType getBundleType(Type type, bool isFlip) {

  // If the targeted type is already converted to a bundle type elsewhere,
  // itself will be returned after cast.
  if (type.isa<BundleType>())
    return type.cast<BundleType>();

  // Convert old type to a bundle type, currently only support integer or index
  // or none type.
  MLIRContext *context = type.getContext();

  switch (type.getKind()) {
  case StandardTypes::Integer: {
    IntegerType integerType = type.cast<IntegerType>();
    int width = integerType.getWidth();

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
    int width = type.cast<IndexType>().kInternalStorageBitWidth;
    return buildBundleType(SIntType::get(context, width), isFlip, context);
  }
  case StandardTypes::None:
    return buildBundleType(nullptr, isFlip, context);
  default:
    assert(false &&
           "Unsupported data type. Supported data types: integer, index, none");
    return FIRRTLType(nullptr);
  }
}

// Create a low voltage constant operation and return its result value
Value createConstantOp(Location insertLoc, FIRRTLType opType, int value,
                       ConversionPatternRewriter &rewriter) {
  IntegerAttr constantOpAttr = rewriter.getIntegerAttr(
      IntegerType::get(opType.getBitWidthOrSentinel(), rewriter.getContext()),
      value);
  firrtl::ConstantOp constantOp =
      rewriter.create<firrtl::ConstantOp>(insertLoc, opType, constantOpAttr);
  return constantOp.getResult();
}

std::string getSubModuleName(Operation &oldOp) {
  std::string subModuleName = oldOp.getName().getStringRef().str() + "_" +
                              std::to_string(oldOp.getNumOperands()) + "ins_" +
                              std::to_string(oldOp.getNumResults()) + "outs";

  // Add compare operation information
  if (auto comOp = dyn_cast<mlir::CmpIOp>(oldOp))
    subModuleName += "_" + stringifyEnum(comOp.getPredicate()).str();

  // Add elastic component control information
  else if (auto ctrlAttr = oldOp.getAttr("control")) {
    if (ctrlAttr.cast<BoolAttr>().getValue())
      subModuleName += "_ctrl";
  }
  return subModuleName;
}

//===----------------------------------------------------------------------===//
// Create Top FIRRTL Module Functions
//===----------------------------------------------------------------------===//

FModuleOp createTopModuleOp(handshake::FuncOp funcOp,
                            ConversionPatternRewriter &rewriter) {
  using ModulePort = std::pair<StringAttr, FIRRTLType>;
  llvm::SmallVector<ModulePort, 8> modulePorts;

  // Add all the input ports
  int args_idx = 0;
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
  FModuleOp topModuleOp = rewriter.create<FModuleOp>(
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
  Block *secondBlock = &*(++blockIterator);
  Operation *termOp = entryBlock->getTerminator();

  // Connect all uses of each argument of the second block to the corresponding
  // argument of the first block
  for (int i = 0, e = secondBlock->getNumArguments(); i < e; i++) {
    BlockArgument oldArgument = secondBlock->getArgument(i);
    BlockArgument newArgument = entryBlock->getArgument(i);
    oldArgument.replaceAllUsesWith(newArgument);
  }

  // Move all operations of the second block to the first block
  while (!secondBlock->empty()) {
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
      if (StringRef(getSubModuleName(oldOp)) == subModuleOp.getName()) {
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

  auto subModule = rewriter.create<FModuleOp>(
      topModuleOp.getLoc(),
      rewriter.getStringAttr(StringRef(getSubModuleName(oldOp))),
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
    BundleType argType = arg.getType().cast<BundleType>();
    for (auto &element : argType.getElements()) {
      StringRef elementName = element.first.strref();
      FIRRTLType elementType = element.second;
      SubfieldOp subfieldOp = rewriter.create<SubfieldOp>(
          insertLoc, elementType, arg, rewriter.getStringAttr(elementName));
      Value value = subfieldOp.getResult();
      valueVector.push_back(value);
    }
    valueVectorList.push_back(valueVector);
  }

  return valueVectorList;
}

// Multiple input merge operation. Now we presume only one input is active, an
// simple arbitration algorithm is used here: the former input always has the
// higher priority.
// We also presume merge is a non-block element.
void buildMergeLogic(ValueVectorList subfieldList, Location insertLoc,
                     ConversionPatternRewriter &rewriter) {

  // Get result subfield values
  ValueVector resultSubfield = subfieldList.back();
  Value resultValid = resultSubfield[0];
  Value resultReady = resultSubfield[1];
  Value resultData = resultSubfield[2];

  // Connect ready signal for all inputs
  for (int i = 0, e = subfieldList.size() - 1; i < e; i++) {
    Value argReady = subfieldList[i][1];
    rewriter.create<ConnectOp>(insertLoc, argReady, resultReady);
  }

  // Walk through all inputs to create a chain of when operation
  for (int i = 0, e = subfieldList.size() - 1; i < e; i++) {

    // Get current input subfield values
    ValueVector argSubfield = subfieldList[i];
    Value argValid = argSubfield[0];
    Value argData = argSubfield[2];

    // If current input is not the last input, a new when operation will be
    // created, and connections will be created in the thenRegion of the new
    // when operation.
    if (i != e - 1) {
      WhenOp whenOp = rewriter.create<WhenOp>(insertLoc, argValid, true);

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

// Same as buildMergeLogic, except that a control output port is included for
// indicating the current active input port.
void buildControlMergeLogic(ValueVectorList subfieldList, Location insertLoc,
                            ConversionPatternRewriter &rewriter) {

  // Get result subfield values
  auto numPort = subfieldList.size();
  ValueVector resultSubfield = subfieldList[numPort - 2];
  Value resultValid = resultSubfield[0];
  Value resultReady = resultSubfield[1];

  // The last result of control_merge indicates which input is active.
  ValueVector controlSubfield = subfieldList[numPort - 1];
  Value controlValid = controlSubfield[0];
  Value controlReady = controlSubfield[1];
  Value controlData = controlSubfield[2];

  // Connect ready signal for all inputs
  AndPrimOp argReadyOp = rewriter.create<AndPrimOp>(
      insertLoc, resultReady.getType(), resultReady, controlReady);
  for (int i = 0, e = numPort - 2; i < e; i++) {
    Value argReady = subfieldList[i][1];
    rewriter.create<ConnectOp>(insertLoc, argReady, argReadyOp.getResult());
  }

  // Walk through all inputs to create a chain of when operation
  for (int i = 0, e = numPort - 2; i < e; i++) {

    // Get current input subfield values
    ValueVector argSubfield = subfieldList[i];
    Value argValid = argSubfield[0];

    FIRRTLType controlType =
        FlipType::get(controlData.getType().cast<FIRRTLType>());

    // If current input is not the last input, a new when operation will be
    // created, and connections will be created in the thenRegion of the new
    // when operation.
    if (i != e - 1) {
      WhenOp whenOp = rewriter.create<WhenOp>(insertLoc, argValid, true);

      rewriter.setInsertionPointToStart(&whenOp.thenRegion().front());
      Value controlValue =
          createConstantOp(insertLoc, controlType, i, rewriter);
      rewriter.create<ConnectOp>(insertLoc, controlData, controlValue);
      rewriter.create<ConnectOp>(insertLoc, controlValid, argValid); // Mark
      rewriter.create<ConnectOp>(insertLoc, resultValid, argValid);

      rewriter.setInsertionPointToStart(&whenOp.elseRegion().front());
    } else {
      Value controlValue =
          createConstantOp(insertLoc, controlType, i, rewriter);
      rewriter.create<ConnectOp>(insertLoc, controlData, controlValue);
      rewriter.create<ConnectOp>(insertLoc, controlValid, argValid); // Mark
      rewriter.create<ConnectOp>(insertLoc, resultValid, argValid);
    }
  }
}

void buildMuxLogic(ValueVectorList subfieldList, Location insertLoc,
                   ConversionPatternRewriter &rewriter) {
  // Get select subfield values
  ValueVector selectSubfield = subfieldList.front();
  Value selectValid = selectSubfield[0];
  Value selectReady = selectSubfield[1];
  Value selectData = selectSubfield[2];
  FIRRTLType selectType = selectData.getType().cast<FIRRTLType>();

  // Get result subfield values
  ValueVector resultSubfield = subfieldList.back();
  Value resultValid = resultSubfield[0];
  Value resultReady = resultSubfield[1];
  Value resultData = resultSubfield[2];

  WhenOp validWhenOp = rewriter.create<WhenOp>(insertLoc, selectValid, false);
  rewriter.setInsertionPointToStart(&validWhenOp.thenRegion().front());

  for (int i = 1, e = subfieldList.size() - 1; i < e; i++) {
    ValueVector argSubfield = subfieldList[i];
    Value argValid = argSubfield[0];
    Value argReady = argSubfield[1];
    Value argData = argSubfield[2];

    if (i != e - 1) {
      Value constantValue =
          createConstantOp(insertLoc, selectType, i, rewriter);
      EQPrimOp conditionOp = rewriter.create<EQPrimOp>(
          insertLoc, UIntType::get(rewriter.getContext(), 1), selectData,
          constantValue);
      WhenOp branchWhenOp =
          rewriter.create<WhenOp>(insertLoc, conditionOp.getResult(), true);

      rewriter.setInsertionPointToStart(&branchWhenOp.thenRegion().front());
      rewriter.create<ConnectOp>(insertLoc, resultValid, argValid);
      rewriter.create<ConnectOp>(insertLoc, resultData, argData);
      rewriter.create<ConnectOp>(insertLoc, argReady, resultReady);
      AndPrimOp selectReadyOp = rewriter.create<AndPrimOp>(
          insertLoc, argValid.getType(), argValid, resultReady);
      rewriter.create<ConnectOp>(insertLoc, selectReady,
                                 selectReadyOp.getResult());

      rewriter.setInsertionPointToStart(&branchWhenOp.elseRegion().front());
    } else {
      rewriter.create<ConnectOp>(insertLoc, resultValid, argValid);
      rewriter.create<ConnectOp>(insertLoc, resultData, argData);
      rewriter.create<ConnectOp>(insertLoc, argReady, resultReady);
      AndPrimOp selectReadyOp = rewriter.create<AndPrimOp>(
          insertLoc, argValid.getType(), argValid, resultReady);
      rewriter.create<ConnectOp>(insertLoc, selectReady,
                                 selectReadyOp.getResult());
    }
  }
}

// Single input and single output branch operation
void buildBranchLogic(Operation &oldOp, ValueVectorList subfieldList,
                      Location insertLoc, ConversionPatternRewriter &rewriter) {
  ValueVector argSubfield = subfieldList[0];
  ValueVector resultSubfield = subfieldList[1];

  Value argValid = argSubfield[0];
  Value argReady = argSubfield[1];
  Value resultValid = resultSubfield[0];
  Value resultReady = resultSubfield[1];

  rewriter.create<ConnectOp>(insertLoc, resultValid, argValid);
  rewriter.create<ConnectOp>(insertLoc, argReady, resultReady);

  if (!cast<handshake::BranchOp>(oldOp).isControl()) {
    Value argData = argSubfield[2];
    Value resultData = resultSubfield[2];

    rewriter.create<ConnectOp>(insertLoc, resultData, argData);
  }
}

// Conditional branch operation with two output
void buildConditionalBranchLogic(Operation &oldOp, ValueVectorList subfieldList,
                                 Location insertLoc,
                                 ConversionPatternRewriter &rewriter) {

  ValueVector controlSubfield = subfieldList[0];
  ValueVector argSubfield = subfieldList[1];
  ValueVector result0Subfield = subfieldList[2];
  ValueVector result1Subfield = subfieldList[3];

  Value controlValid = controlSubfield[0];
  Value controlReady = controlSubfield[1];
  Value controlData = controlSubfield[2];

  Value argValid = argSubfield[0];
  Value argReady = argSubfield[1];
  Value result0Valid = result0Subfield[0];
  Value result0Ready = result0Subfield[1];
  Value result1Valid = result1Subfield[0];
  Value result1Ready = result1Subfield[1];

  WhenOp validWhenOp = rewriter.create<WhenOp>(insertLoc, controlValid, false);

  rewriter.setInsertionPointToStart(&validWhenOp.thenRegion().front());
  WhenOp branchWhenOp = rewriter.create<WhenOp>(insertLoc, controlData, true);

  // When control signal is true, the first branch is selected
  rewriter.setInsertionPointToStart(&branchWhenOp.thenRegion().front());
  rewriter.create<ConnectOp>(insertLoc, result0Valid, argValid);
  rewriter.create<ConnectOp>(insertLoc, argReady, result0Ready);

  if (!cast<handshake::ConditionalBranchOp>(oldOp).isControl()) {
    Value argData = argSubfield[2];
    Value result0Data = result0Subfield[2];
    rewriter.create<ConnectOp>(insertLoc, result0Data, argData);
  }

  AndPrimOp controlReadyOp0 = rewriter.create<AndPrimOp>(
      insertLoc, argValid.getType(), argValid, result0Ready);
  rewriter.create<ConnectOp>(insertLoc, controlReady,
                             controlReadyOp0.getResult());

  // When control signal is false, the second branch is selected
  rewriter.setInsertionPointToStart(&branchWhenOp.elseRegion().front());
  rewriter.create<ConnectOp>(insertLoc, result1Valid, argValid);
  rewriter.create<ConnectOp>(insertLoc, argReady, result1Ready);

  if (!cast<handshake::ConditionalBranchOp>(oldOp).isControl()) {
    Value argData = argSubfield[2];
    Value result1Data = result1Subfield[2];
    rewriter.create<ConnectOp>(insertLoc, result1Data, argData);
  }

  AndPrimOp controlReadyOp1 = rewriter.create<AndPrimOp>(
      insertLoc, argValid.getType(), argValid, result1Ready);
  rewriter.create<ConnectOp>(insertLoc, controlReady,
                             controlReadyOp1.getResult());
}

// ISSUE: Currently fork is as same as lazy fork
// This should be a timing components?
void buildForkLogic(Operation &oldOp, ValueVectorList subfieldList,
                    Location insertLoc, ConversionPatternRewriter &rewriter) {
  ValueVector argSubfield = subfieldList.front();
  Value argValid = argSubfield[0];
  Value argReady = argSubfield[1];

  Value *tmpReady = &subfieldList[1][1];
  for (int i = 2, e = subfieldList.size(); i < e; i++) {
    Value resultReady = subfieldList[i][1];
    AndPrimOp tmpReadyOp = rewriter.create<AndPrimOp>(
        insertLoc, resultReady.getType(), resultReady, *tmpReady);
    *tmpReady = tmpReadyOp.getResult();
  }
  rewriter.create<ConnectOp>(insertLoc, argReady, *tmpReady);

  AndPrimOp resultValidOp = rewriter.create<AndPrimOp>(
      insertLoc, argValid.getType(), argValid, *tmpReady);
  for (int i = 1, e = subfieldList.size(); i < e; i++) {
    ValueVector resultfield = subfieldList[i];
    Value resultValid = resultfield[0];

    rewriter.create<ConnectOp>(insertLoc, resultValid,
                               resultValidOp.getResult());

    if (!cast<handshake::ForkOp>(oldOp).isControl()) {
      Value argData = argSubfield[2];
      Value resultData = resultfield[2];
      rewriter.create<ConnectOp>(insertLoc, resultData, argData);
    }
  }
}

void buildLazyForkLogic(Operation &oldOp, ValueVectorList subfieldList,
                        Location insertLoc,
                        ConversionPatternRewriter &rewriter) {
  ValueVector argSubfield = subfieldList.front();
  Value argValid = argSubfield[0];
  Value argReady = argSubfield[1];

  Value *tmpReady = &subfieldList[1][1];
  for (int i = 2, e = subfieldList.size(); i < e; i++) {
    Value resultReady = subfieldList[i][1];
    AndPrimOp tmpReadyOp = rewriter.create<AndPrimOp>(
        insertLoc, resultReady.getType(), resultReady, *tmpReady);
    *tmpReady = tmpReadyOp.getResult();
  }
  rewriter.create<ConnectOp>(insertLoc, argReady, *tmpReady);

  AndPrimOp resultValidOp = rewriter.create<AndPrimOp>(
      insertLoc, argValid.getType(), argValid, *tmpReady);
  for (int i = 1, e = subfieldList.size(); i < e; i++) {
    ValueVector resultfield = subfieldList[i];
    Value resultValid = resultfield[0];

    rewriter.create<ConnectOp>(insertLoc, resultValid,
                               resultValidOp.getResult());

    if (!cast<handshake::ForkOp>(oldOp).isControl()) {
      Value argData = argSubfield[2];
      Value resultData = resultfield[2];
      rewriter.create<ConnectOp>(insertLoc, resultData, argData);
    }
  }
}

void buildConstantLogic(Operation &oldOp, ValueVectorList subfieldList,
                        Location insertLoc,
                        ConversionPatternRewriter &rewriter) {

  ValueVector controlSubfield = subfieldList.front();
  Value controlValid = controlSubfield[0];
  Value controlReady = controlSubfield[1];

  ValueVector resultSubfield = subfieldList.back();
  Value resultValid = resultSubfield[0];
  Value resultReady = resultSubfield[1];
  Value resultData = resultSubfield[2];

  FIRRTLType constantType =
      FlipType::get(resultData.getType().cast<FIRRTLType>());
  auto constantAttrValue = oldOp.getAttrOfType<IntegerAttr>("value").getInt();

  rewriter.create<ConnectOp>(insertLoc, resultValid, controlValid);
  rewriter.create<ConnectOp>(insertLoc, controlReady, resultReady);
  Value constantValue =
      createConstantOp(insertLoc, constantType, constantAttrValue, rewriter);
  rewriter.create<ConnectOp>(insertLoc, resultData, constantValue);
}

void buildSinkLogic(ValueVectorList subfieldList, Location insertLoc,
                    ConversionPatternRewriter &rewriter) {
  ValueVector argSubfield = subfieldList.front();
  Value argValid = argSubfield[0];
  Value argReady = argSubfield[1];
  Value argData = argSubfield[2];

  FIRRTLType signalType = argValid.getType().cast<FIRRTLType>();
  Value highSignal = createConstantOp(insertLoc, signalType, 1, rewriter);
  rewriter.create<ConnectOp>(insertLoc, argReady, highSignal);

  rewriter.eraseOp(argValid.getDefiningOp());
  rewriter.eraseOp(argData.getDefiningOp());
}

void buildJoinLogic(ValueVectorList subfieldList, Location insertLoc,
                    ConversionPatternRewriter &rewriter) {
  ValueVector resultSubfield = subfieldList.back();
  Value resultValid = resultSubfield[0];
  Value resultReady = resultSubfield[1];

  Value *tmpValid = &subfieldList[0][0];
  for (int i = 1, e = subfieldList.size() - 1; i < e; i++) {
    Value argValid = subfieldList[i][0];
    AndPrimOp tmpValidOp = rewriter.create<AndPrimOp>(
        insertLoc, argValid.getType(), argValid, *tmpValid);
    *tmpValid = tmpValidOp.getResult();
  }

  rewriter.create<ConnectOp>(insertLoc, resultValid, *tmpValid);
  AndPrimOp argReadyOp = rewriter.create<AndPrimOp>(
      insertLoc, resultReady.getType(), resultReady, *tmpValid);

  for (int i = 0, e = subfieldList.size() - 1; i < e; i++) {
    Value argReady = subfieldList[i][1];
    rewriter.create<ConnectOp>(insertLoc, argReady, argReadyOp);
  }
}

// Build binary logic for the new sub-module
template <typename OpType>
void buildBinaryLogic(ValueVectorList subfieldList, Location insertLoc,
                      ConversionPatternRewriter &rewriter) {
  // Get subfields values
  ValueVector arg0Subfield = subfieldList[0];
  ValueVector arg1Subfield = subfieldList[1];
  ValueVector resultSubfield = subfieldList[2];

  Value arg0Valid = arg0Subfield[0];
  Value arg0Ready = arg0Subfield[1];
  Value arg0Data = arg0Subfield[2];

  Value arg1Valid = arg1Subfield[0];
  Value arg1Ready = arg1Subfield[1];
  Value arg1Data = arg1Subfield[2];

  Value resultValid = resultSubfield[0];
  Value resultReady = resultSubfield[1];
  Value resultData = resultSubfield[2];

  // Connect data signals
  OpType CombDataOp = rewriter.create<OpType>(insertLoc, arg0Data.getType(),
                                              arg0Data, arg1Data);
  rewriter.create<ConnectOp>(insertLoc, resultData, CombDataOp.getResult());

  // Connect valid signals
  AndPrimOp combValidOp = rewriter.create<AndPrimOp>(
      insertLoc, arg0Valid.getType(), arg0Valid, arg1Valid);
  rewriter.create<ConnectOp>(insertLoc, resultValid, combValidOp.getResult());

  // Connect ready signals
  AndPrimOp combReadyOp = rewriter.create<AndPrimOp>(
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
    Identifier argId = Identifier::get(argName, context);

    // All ports of the instance operation are flipped
    FIRRTLType argType = FlipType::get(arg.getType().cast<BundleType>());
    elements.push_back(std::make_pair(argId, argType));
    arg_idx += 1;
  }
  FIRRTLType instType =
      BundleType::get(ArrayRef<BundleElement>(elements), context);

  // Insert instanceOp
  rewriter.setInsertionPointAfter(&oldOp);
  firrtl::InstanceOp instOp = rewriter.create<firrtl::InstanceOp>(
      oldOp.getLoc(), instType, subModuleOp.getName(),
      rewriter.getStringAttr(""));

  // Connect instanceOp with other operations in the top module
  int port_idx = 0;
  for (auto &element : instType.cast<BundleType>().getElements()) {
    Identifier elementName = element.first;
    FIRRTLType elementType = element.second;
    SubfieldOp subfieldOp = rewriter.create<SubfieldOp>(
        oldOp.getLoc(), elementType, instOp.getResult(),
        rewriter.getStringAttr(elementName.strref()));

    // Connect input ports
    if (port_idx < oldOp.getNumOperands()) {
      Value operand = oldOp.getOperand(port_idx);
      Value result = subfieldOp.getResult();
      rewriter.create<ConnectOp>(oldOp.getLoc(), result, operand);
    }

    // Connect output ports
    else {
      Value result = oldOp.getResult(port_idx - oldOp.getNumOperands());
      Value newResult = subfieldOp.getResult();
      result.replaceAllUsesWith(newResult);
    }
    port_idx += 1;
  }
  rewriter.eraseOp(&oldOp);
}

// Only support single block function op
void convertReturnOp(Operation &oldOp, Block &entryBlock,
                     ConversionPatternRewriter &rewriter) {
  rewriter.setInsertionPointAfter(&oldOp);
  int numArg = entryBlock.getNumArguments();
  for (int i = 0, e = oldOp.getNumOperands(); i < e; i++) {
    rewriter.create<ConnectOp>(oldOp.getLoc(),
                               entryBlock.getArgument(numArg - e + i),
                               oldOp.getOperand(i));
  }
  rewriter.eraseOp(&oldOp);
}

// TODO
void insertBuffers() {}

//===----------------------------------------------------------------------===//
// MLIR Pass Entry
//===----------------------------------------------------------------------===//

struct HandshakeFuncOpLowering : public OpConversionPattern<handshake::FuncOp> {
  using OpConversionPattern<handshake::FuncOp>::OpConversionPattern;

  LogicalResult match(Operation *op) const override { return success(); }

  void rewrite(handshake::FuncOp funcOp, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const override {
    CircuitOp circuitOp = rewriter.create<CircuitOp>(
        funcOp.getLoc(), rewriter.getStringAttr(funcOp.getName()));
    rewriter.setInsertionPointToStart(circuitOp.getBody());

    FModuleOp topModuleOp = createTopModuleOp(funcOp, rewriter);
    mergeToEntryBlock(topModuleOp, rewriter);
    Block &entryBlock = topModuleOp.getBody().front();

    // Traverse and convert all operations in funcOp.
    // If you want to extend a non-submodule operation, please follow the
    // pattern of convertReturnOp.
    // Else if you want to extend a submodule operation, please follow the
    // pattern of buildBinaryLogic, and buildMergeLogic.
    for (Operation &op : entryBlock) {
      if (isa<handshake::ReturnOp>(op)) {
        convertReturnOp(op, entryBlock, rewriter);
      }
      // This branch take cares of all non-timing operations that require to
      // create new submodules, and instantiation.
      else if (op.getDialect()->getNamespace() != StringRef("firrtl")) {
        // Check whether Sub-module already exists, if not, we will create
        // and insert a new empty sub-module
        FModuleOp subModuleOp = checkSubModuleOp(topModuleOp, op);
        if (!subModuleOp) {
          subModuleOp = createSubModuleOp(topModuleOp, op, rewriter);

          Block &entryBlock = subModuleOp.getBody().front();
          Operation *termOp = entryBlock.getTerminator();
          Location insertLoc = termOp->getLoc();
          rewriter.setInsertionPoint(termOp);

          ValueVectorList subfieldList =
              extractSubfields(entryBlock, insertLoc, rewriter);

          // Build standard expressions logic
          if (isa<mlir::AddIOp>(op))
            buildBinaryLogic<AddPrimOp>(subfieldList, insertLoc, rewriter);
          else if (isa<mlir::SubIOp>(op))
            buildBinaryLogic<SubPrimOp>(subfieldList, insertLoc, rewriter);
          else if (isa<mlir::MulIOp>(op))
            buildBinaryLogic<MulPrimOp>(subfieldList, insertLoc, rewriter);

          else if (isa<mlir::AndOp>(op))
            buildBinaryLogic<AndPrimOp>(subfieldList, insertLoc, rewriter);
          else if (isa<mlir::OrOp>(op))
            buildBinaryLogic<OrPrimOp>(subfieldList, insertLoc, rewriter);
          else if (isa<mlir::XOrOp>(op))
            buildBinaryLogic<XorPrimOp>(subfieldList, insertLoc, rewriter);

          else if (auto cmpOp = dyn_cast<mlir::CmpIOp>(op)) {
            auto cmpOpAttr = cmpOp.getPredicate();
            switch (cmpOpAttr) {
            case CmpIPredicate::eq:
              buildBinaryLogic<EQPrimOp>(subfieldList, insertLoc, rewriter);
              break;
            case CmpIPredicate::ne:
              buildBinaryLogic<NEQPrimOp>(subfieldList, insertLoc, rewriter);
              break;
            case CmpIPredicate::slt:
              buildBinaryLogic<LTPrimOp>(subfieldList, insertLoc, rewriter);
              break;
            case CmpIPredicate::sle:
              buildBinaryLogic<LEQPrimOp>(subfieldList, insertLoc, rewriter);
              break;
            case CmpIPredicate::sgt:
              buildBinaryLogic<GTPrimOp>(subfieldList, insertLoc, rewriter);
              break;
            case CmpIPredicate::sge:
              buildBinaryLogic<GEQPrimOp>(subfieldList, insertLoc, rewriter);
              break;
            }
          } else if (isa<mlir::ShiftLeftOp>(op))
            buildBinaryLogic<DShlPrimOp>(subfieldList, insertLoc, rewriter);
          else if (isa<mlir::SignedShiftRightOp>(op))
            buildBinaryLogic<DShrPrimOp>(subfieldList, insertLoc, rewriter);

          // Build elastic components logic
          else if (isa<handshake::MergeOp>(op))
            buildMergeLogic(subfieldList, insertLoc, rewriter);
          else if (isa<handshake::ControlMergeOp>(op))
            buildControlMergeLogic(subfieldList, insertLoc, rewriter);

          else if (isa<handshake::MuxOp>(op))
            buildMuxLogic(subfieldList, insertLoc, rewriter);

          else if (isa<handshake::BranchOp>(op))
            buildBranchLogic(op, subfieldList, insertLoc, rewriter);
          else if (isa<handshake::ConditionalBranchOp>(op))
            buildConditionalBranchLogic(op, subfieldList, insertLoc, rewriter);

          else if (isa<handshake::ForkOp>(op))
            buildForkLogic(op, subfieldList, insertLoc, rewriter);
          else if (isa<handshake::LazyForkOp>(op))
            buildLazyForkLogic(op, subfieldList, insertLoc, rewriter);

          else if (isa<handshake::ConstantOp>(op))
            buildConstantLogic(op, subfieldList, insertLoc, rewriter);
          else if (isa<handshake::SinkOp>(op))
            buildSinkLogic(subfieldList, insertLoc, rewriter);
        }

        // Insert instance operation into the top module, for instantiating the
        // new sub-module
        createInstOp(op, subModuleOp, rewriter);
      }
    }
    rewriter.eraseOp(funcOp);

    //// Code for debug
    // for (auto &block : *topModuleOp.getParentRegion()) {
    //  for (auto &op : block) {
    //    llvm::outs() << op << "\n";
    //  }
    //}
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
  PassRegistration<HandshakeToFIRRTLPass>("lower-handshake-to-firrtl",
                                          "Lowering to FIRRTL Dialect");
}
