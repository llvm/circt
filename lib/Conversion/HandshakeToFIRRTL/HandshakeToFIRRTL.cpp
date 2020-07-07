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
//===----------------------------------------------------------------------===//

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

using namespace mlir;
using namespace mlir::handshake;
using namespace circt;
using namespace circt::firrtl;

using ValueVector = std::vector<Value>;
using ValueVectorList = std::vector<ValueVector>;

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

/// Build a FIRRTL bundle type (with data, valid, and ready subfields) given the
/// type of the data subfield.
FIRRTLType buildBundleType(FIRRTLType dataType, bool isFlip,
                           MLIRContext *context) {
  using BundleElement = std::pair<Identifier, FIRRTLType>;
  llvm::SmallVector<BundleElement, 3> elements;

  // Add valid and ready subfield to the bundle.
  Identifier validId = Identifier::get("valid", context);
  Identifier readyId = Identifier::get("ready", context);
  UIntType signalType = UIntType::get(context, 1);
  if (isFlip) {
    elements.push_back(BundleElement(validId, FlipType::get(signalType)));
    elements.push_back(BundleElement(readyId, signalType));
  } else {
    elements.push_back(BundleElement(validId, signalType));
    elements.push_back(BundleElement(readyId, FlipType::get(signalType)));
  }

  // Add data subfield to the bundle if dataType is not a null.
  if (dataType) {
    Identifier dataId = Identifier::get("data", context);
    if (isFlip)
      elements.push_back(BundleElement(dataId, FlipType::get(dataType)));
    else
      elements.push_back(BundleElement(dataId, dataType));
  }

  return BundleType::get(ArrayRef<BundleElement>(elements), context);
}

/// Return a FIRRTL bundle type (with data, valid, and ready subfields) given a
/// standard data type. Current supported data types are integer (signed,
/// unsigned, and signless), index, and none.
FIRRTLType getBundleType(Type type, bool isFlip) {
  // If the input is already converted to a bundle type elsewhere, itself will
  // be returned after cast.
  if (type.isa<BundleType>())
    return type.cast<BundleType>();

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
  // Currently we consider index type as 64-bits signed integer.
  case StandardTypes::Index: {
    int width = type.cast<IndexType>().kInternalStorageBitWidth;
    return buildBundleType(SIntType::get(context, width), isFlip, context);
  }
  case StandardTypes::None:
    return buildBundleType(/*dataType=*/nullptr, isFlip, context);
  default:
    assert(false && "Unsupported data type. Supported data types: integer "
                    "(signed, unsigned, signless), index, none");
    return FIRRTLType(nullptr);
  }
}

Value createConstantOp(Location insertLoc, FIRRTLType opType, int value,
                       ConversionPatternRewriter &rewriter) {
  IntegerAttr constantOpAttr = rewriter.getIntegerAttr(
      IntegerType::get(opType.getBitWidthOrSentinel(), rewriter.getContext()),
      value);
  return rewriter.create<firrtl::ConstantOp>(insertLoc, opType, constantOpAttr)
      .getResult();
}

/// Construct a name for creating FIRRTL sub-module. The returned string
/// contains the following information: 1) standard or handshake operation
/// name; 2) number of inputs; 3) number of outputs; 4) comparison operation
/// type (if applied); 5) whether the elastic component is for the control path
/// (if applied).
std::string getSubModuleName(Operation &oldOp) {
  std::string subModuleName = oldOp.getName().getStringRef().str() + "_" +
                              std::to_string(oldOp.getNumOperands()) + "ins_" +
                              std::to_string(oldOp.getNumResults()) + "outs";

  if (auto comOp = dyn_cast<mlir::CmpIOp>(oldOp))
    subModuleName += "_" + stringifyEnum(comOp.getPredicate()).str();

  if (auto ctrlAttr = oldOp.getAttr("control")) {
    if (ctrlAttr.cast<BoolAttr>().getValue())
      subModuleName += "_ctrl";
  }

  return subModuleName;
}

//===----------------------------------------------------------------------===//
// FIRRTL Top-module Related Functions
//===----------------------------------------------------------------------===//

/// Currently we are not considering clock and registers, thus the generated
/// circuit is pure combinational logic. If graph cycle exists, at least one
/// buffer is required to be inserted for breaking the cycle, which will be
/// supported in the next patch.
FModuleOp createTopModuleOp(handshake::FuncOp funcOp,
                            ConversionPatternRewriter &rewriter) {
  using ModulePort = std::pair<StringAttr, FIRRTLType>;
  llvm::SmallVector<ModulePort, 8> ports;

  // Add all inputs of funcOp.
  int args_idx = 0;
  for (auto &arg : funcOp.getArguments()) {
    StringAttr portName =
        rewriter.getStringAttr("arg" + std::to_string(args_idx));
    FIRRTLType bundlePortType = getBundleType(arg.getType(), /*isFlip=*/false);
    ports.push_back(ModulePort(portName, bundlePortType));
    args_idx += 1;
  }

  // Add all outputs of funcOp.
  for (auto portType : funcOp.getType().getResults()) {
    StringAttr portName =
        rewriter.getStringAttr("arg" + std::to_string(args_idx));
    FIRRTLType bundlePortType = getBundleType(portType, /*isFlip=*/true);
    ports.push_back(ModulePort(portName, bundlePortType));
    args_idx += 1;
  }

  // Create a FIRRTL module, and inline the funcOp into it.
  FModuleOp topModuleOp = rewriter.create<FModuleOp>(
      funcOp.getLoc(), rewriter.getStringAttr(funcOp.getName()),
      ArrayRef<ModulePort>(ports));
  rewriter.inlineRegionBefore(funcOp.getBody(), topModuleOp.getBody(),
                              topModuleOp.end());

  // Merge the second block (inlined from funcOp) of the top-module into the
  // entry block.
  auto blockIterator = topModuleOp.getBody().begin();
  Block *entryBlock = &*blockIterator;
  Block *secondBlock = &*(++blockIterator);

  // Replace uses of each argument of the second block with the corresponding
  // argument of the entry block.
  for (int i = 0, e = secondBlock->getNumArguments(); i < e; ++i) {
    BlockArgument oldArgument = secondBlock->getArgument(i);
    BlockArgument newArgument = entryBlock->getArgument(i);
    oldArgument.replaceAllUsesWith(newArgument);
  }

  // Move all operations of the second block to the entry block.
  while (!secondBlock->empty()) {
    Operation &op = secondBlock->front();
    op.moveBefore(entryBlock->getTerminator());
  }
  rewriter.eraseBlock(secondBlock);

  return topModuleOp;
}

//===----------------------------------------------------------------------===//
// FIRRTL Sub-module Related Functions
//===----------------------------------------------------------------------===//

/// Check whether a submodule with the same name has been created elsewhere.
/// Return the matched submodule if true, otherwise return nullptr.
FModuleOp checkSubModuleOp(FModuleOp topModuleOp, Operation &oldOp) {
  for (auto &op : topModuleOp.getParentRegion()->front()) {
    if (auto subModuleOp = dyn_cast<FModuleOp>(op)) {
      if (StringRef(getSubModuleName(oldOp)) == subModuleOp.getName()) {
        return subModuleOp;
      }
    }
  }
  return FModuleOp(nullptr);
}

/// All standard expressions and handshake elastic components will be converted
/// to a FIRRTL sub-module and be instantiated in the top-module.
FModuleOp createSubModuleOp(FModuleOp topModuleOp, Operation &oldOp,
                            ConversionPatternRewriter &rewriter) {
  rewriter.setInsertionPoint(topModuleOp);
  using ModulePort = std::pair<StringAttr, FIRRTLType>;
  llvm::SmallVector<ModulePort, 4> ports;

  // Add all inputs of oldOp.
  int args_idx = 0;
  for (auto portType : oldOp.getOperands().getTypes()) {
    StringAttr portName =
        rewriter.getStringAttr("arg" + std::to_string(args_idx));
    FIRRTLType bundlePortType = getBundleType(portType, /*isFlip=*/false);
    ports.push_back(ModulePort(portName, bundlePortType));
    args_idx += 1;
  }

  // Add all outputs of oldOp.
  for (auto portType : oldOp.getResults().getTypes()) {
    StringAttr portName =
        rewriter.getStringAttr("arg" + std::to_string(args_idx));
    FIRRTLType bundlePortType = getBundleType(portType, /*isFlip=*/true);
    ports.push_back(ModulePort(portName, bundlePortType));
    args_idx += 1;
  }

  return rewriter.create<FModuleOp>(
      topModuleOp.getLoc(),
      rewriter.getStringAttr(StringRef(getSubModuleName(oldOp))),
      ArrayRef<ModulePort>(ports));
}

//===----------------------------------------------------------------------===//
// Combinational Logic Builders
//===----------------------------------------------------------------------===//

/// Extract all subfields of all ports of the sub-module.
ValueVectorList extractSubfields(FModuleOp subModuleOp, Location insertLoc,
                                 ConversionPatternRewriter &rewriter) {
  ValueVectorList portList;
  for (auto &arg : subModuleOp.getArguments()) {
    ValueVector subfields;
    BundleType argType = arg.getType().cast<BundleType>();
    for (auto &element : argType.getElements()) {
      StringRef elementName = element.first.strref();
      FIRRTLType elementType = element.second;
      SubfieldOp subfieldOp = rewriter.create<SubfieldOp>(
          insertLoc, elementType, arg, rewriter.getStringAttr(elementName));
      subfields.push_back(subfieldOp.getResult());
    }
    portList.push_back(subfields);
  }

  return portList;
}

/// Assume only one input is active. When multiple inputs are active, inputs in
/// the front have higher priority.
/// Please refer to test_merge.mlir test case.
void buildMergeLogic(ValueVectorList portList, Location insertLoc,
                     ConversionPatternRewriter &rewriter) {
  ValueVector resultSubfields = portList.back();
  Value resultValid = resultSubfields[0];
  Value resultReady = resultSubfields[1];
  Value resultData = resultSubfields[2];

  // Walk through each input to create a chain of when operation.
  for (int i = 0, e = portList.size() - 1; i < e; ++i) {
    ValueVector argSubfields = portList[i];
    Value argValid = argSubfields[0];
    Value argReady = argSubfields[1];
    Value argData = argSubfields[2];

    // If the current input is not the last one, the new created when operation
    // will have an else region.
    WhenOp whenOp = rewriter.create<WhenOp>(insertLoc, argValid,
                                            /*withElseRegion=*/(i != e - 1));
    rewriter.setInsertionPointToStart(&whenOp.thenRegion().front());
    rewriter.create<ConnectOp>(insertLoc, resultData, argData);
    rewriter.create<ConnectOp>(insertLoc, resultValid, argValid);
    rewriter.create<ConnectOp>(insertLoc, argReady, resultReady);

    if (i != e - 1)
      rewriter.setInsertionPointToStart(&whenOp.elseRegion().front());
  }
}

/// Assume only one input is active.
/// Please refer to test_cmerge.mlir test case.
void buildControlMergeLogic(Operation &oldOp, ValueVectorList portList,
                            Location insertLoc,
                            ConversionPatternRewriter &rewriter) {
  int numPort = portList.size();

  ValueVector resultSubfields = portList[numPort - 2];
  Value resultValid = resultSubfields[0];
  Value resultReady = resultSubfields[1];

  // The last output indicates which input is active now.
  ValueVector controlSubfields = portList[numPort - 1];
  Value controlValid = controlSubfields[0];
  Value controlReady = controlSubfields[1];
  Value controlData = controlSubfields[2];
  FIRRTLType controlType =
      FlipType::get(controlData.getType().cast<FIRRTLType>());

  AndPrimOp argReadyOp = rewriter.create<AndPrimOp>(
      insertLoc, resultReady.getType(), resultReady, controlReady);

  // Walk through each input to create a chain of when operation.
  for (int i = 0, e = numPort - 2; i < e; ++i) {
    ValueVector argSubfields = portList[i];
    Value argValid = argSubfields[0];
    Value argReady = argSubfields[1];

    // If the current input is not the last one, the new created when operation
    // will have an else region.
    WhenOp whenOp = rewriter.create<WhenOp>(insertLoc, argValid,
                                            /*withElseRegion=*/(i != e - 1));
    rewriter.setInsertionPointToStart(&whenOp.thenRegion().front());
    rewriter.create<ConnectOp>(
        insertLoc, controlData,
        createConstantOp(insertLoc, controlType, i, rewriter));
    rewriter.create<ConnectOp>(insertLoc, controlValid, argValid);
    rewriter.create<ConnectOp>(insertLoc, resultValid, argValid);
    rewriter.create<ConnectOp>(insertLoc, argReady, argReadyOp.getResult());

    if (!cast<handshake::ControlMergeOp>(oldOp)
             .getAttrOfType<BoolAttr>("control")
             .getValue()) {
      Value argData = argSubfields[2];
      Value resultData = resultSubfields[2];
      rewriter.create<ConnectOp>(insertLoc, resultData, argData);
    }

    if (i != e - 1)
      rewriter.setInsertionPointToStart(&whenOp.elseRegion().front());
  }
}

/// Please refer to test_mux.mlir test case.
void buildMuxLogic(ValueVectorList portList, Location insertLoc,
                   ConversionPatternRewriter &rewriter) {
  ValueVector selectSubfields = portList.front();
  Value selectValid = selectSubfields[0];
  Value selectReady = selectSubfields[1];
  Value selectData = selectSubfields[2];
  FIRRTLType selectType = selectData.getType().cast<FIRRTLType>();

  ValueVector resultSubfields = portList.back();
  Value resultValid = resultSubfields[0];
  Value resultReady = resultSubfields[1];
  Value resultData = resultSubfields[2];

  // Mux will work only when the select input is active.
  WhenOp validWhenOp = rewriter.create<WhenOp>(insertLoc, selectValid,
                                               /*withElseRegion=*/false);
  rewriter.setInsertionPointToStart(&validWhenOp.thenRegion().front());

  // Walk through each input to create a chain of when operation.
  for (int i = 1, e = portList.size() - 1; i < e; ++i) {
    ValueVector argSubfields = portList[i];
    Value argValid = argSubfields[0];
    Value argReady = argSubfields[1];
    Value argData = argSubfields[2];

    EQPrimOp conditionOp = rewriter.create<EQPrimOp>(
        insertLoc, UIntType::get(rewriter.getContext(), 1), selectData,
        createConstantOp(insertLoc, selectType, i, rewriter));

    // If the current input is not the last one, the new created when
    // operation will have an else region.
    WhenOp branchWhenOp = rewriter.create<WhenOp>(
        insertLoc, conditionOp.getResult(), /*withElseRegion=*/(i != e - 1));

    rewriter.setInsertionPointToStart(&branchWhenOp.thenRegion().front());
    rewriter.create<ConnectOp>(insertLoc, resultValid, argValid);
    rewriter.create<ConnectOp>(insertLoc, resultData, argData);
    rewriter.create<ConnectOp>(insertLoc, argReady, resultReady);

    // Select will be ready to accept new token when data has been passed from
    // input to output.
    AndPrimOp selectReadyOp = rewriter.create<AndPrimOp>(
        insertLoc, argValid.getType(), argValid, resultReady);
    rewriter.create<ConnectOp>(insertLoc, selectReady,
                               selectReadyOp.getResult());
    if (i != e - 1)
      rewriter.setInsertionPointToStart(&branchWhenOp.elseRegion().front());
  }
}

/// Please refer to test_branch.mlir test case.
void buildBranchLogic(Operation &oldOp, ValueVectorList portList,
                      Location insertLoc, ConversionPatternRewriter &rewriter) {
  ValueVector argSubfields = portList[0];
  ValueVector resultSubfields = portList[1];
  Value argValid = argSubfields[0];
  Value argReady = argSubfields[1];
  Value resultValid = resultSubfields[0];
  Value resultReady = resultSubfields[1];

  rewriter.create<ConnectOp>(insertLoc, resultValid, argValid);
  rewriter.create<ConnectOp>(insertLoc, argReady, resultReady);

  if (!cast<handshake::BranchOp>(oldOp).isControl()) {
    Value argData = argSubfields[2];
    Value resultData = resultSubfields[2];
    rewriter.create<ConnectOp>(insertLoc, resultData, argData);
  }
}

/// Two outputs conditional branch operation.
/// Please refer to test_conditional_branch.mlir test case.
void buildConditionalBranchLogic(Operation &oldOp, ValueVectorList portList,
                                 Location insertLoc,
                                 ConversionPatternRewriter &rewriter) {
  ValueVector controlSubfields = portList[0];
  ValueVector argSubfields = portList[1];
  ValueVector result0Subfields = portList[2];
  ValueVector result1Subfields = portList[3];

  Value controlValid = controlSubfields[0];
  Value controlReady = controlSubfields[1];
  Value controlData = controlSubfields[2];
  Value argValid = argSubfields[0];
  Value argReady = argSubfields[1];
  Value result0Valid = result0Subfields[0];
  Value result0Ready = result0Subfields[1];
  Value result1Valid = result1Subfields[0];
  Value result1Ready = result1Subfields[1];

  // ConditionalBranch will work only when the control input is active.
  WhenOp validWhenOp = rewriter.create<WhenOp>(insertLoc, controlValid,
                                               /*withElseRegion=*/false);
  rewriter.setInsertionPointToStart(&validWhenOp.thenRegion().front());
  WhenOp branchWhenOp = rewriter.create<WhenOp>(insertLoc, controlData,
                                                /*withElseRegion=*/true);

  // When control signal is true, the first branch is selected
  rewriter.setInsertionPointToStart(&branchWhenOp.thenRegion().front());
  rewriter.create<ConnectOp>(insertLoc, result0Valid, argValid);
  rewriter.create<ConnectOp>(insertLoc, argReady, result0Ready);

  if (!cast<handshake::ConditionalBranchOp>(oldOp).isControl()) {
    Value argData = argSubfields[2];
    Value result0Data = result0Subfields[2];
    rewriter.create<ConnectOp>(insertLoc, result0Data, argData);
  }

  AndPrimOp control0ReadyOp = rewriter.create<AndPrimOp>(
      insertLoc, argValid.getType(), argValid, result0Ready);
  rewriter.create<ConnectOp>(insertLoc, controlReady,
                             control0ReadyOp.getResult());

  // When control signal is false, the second branch is selected
  rewriter.setInsertionPointToStart(&branchWhenOp.elseRegion().front());
  rewriter.create<ConnectOp>(insertLoc, result1Valid, argValid);
  rewriter.create<ConnectOp>(insertLoc, argReady, result1Ready);

  if (!cast<handshake::ConditionalBranchOp>(oldOp).isControl()) {
    Value argData = argSubfields[2];
    Value result1Data = result1Subfields[2];
    rewriter.create<ConnectOp>(insertLoc, result1Data, argData);
  }

  AndPrimOp control1ReadyOp = rewriter.create<AndPrimOp>(
      insertLoc, argValid.getType(), argValid, result1Ready);
  rewriter.create<ConnectOp>(insertLoc, controlReady,
                             control1ReadyOp.getResult());
}

/// Please refer to test_lazy_fork.mlir test case.
void buildLazyForkLogic(Operation &oldOp, ValueVectorList portList,
                        Location insertLoc,
                        ConversionPatternRewriter &rewriter) {
  ValueVector argSubfields = portList.front();
  Value argValid = argSubfields[0];
  Value argReady = argSubfields[1];

  // The input will be ready to accept new token when all outputs are ready.
  Value *tmpReady = &portList[1][1];
  for (int i = 2, e = portList.size(); i < e; ++i) {
    Value resultReady = portList[i][1];
    AndPrimOp tmpReadyOp = rewriter.create<AndPrimOp>(
        insertLoc, resultReady.getType(), resultReady, *tmpReady);
    *tmpReady = tmpReadyOp.getResult();
  }
  rewriter.create<ConnectOp>(insertLoc, argReady, *tmpReady);

  // All outputs must be ready for the LazyFork to send the token.
  AndPrimOp resultValidOp = rewriter.create<AndPrimOp>(
      insertLoc, argValid.getType(), argValid, *tmpReady);
  for (int i = 1, e = portList.size(); i < e; ++i) {
    ValueVector resultfield = portList[i];
    Value resultValid = resultfield[0];
    rewriter.create<ConnectOp>(insertLoc, resultValid,
                               resultValidOp.getResult());

    if (!cast<handshake::LazyForkOp>(oldOp).isControl()) {
      Value argData = argSubfields[2];
      Value resultData = resultfield[2];
      rewriter.create<ConnectOp>(insertLoc, resultData, argData);
    }
  }
}

/// Currently Fork is implement as a LazyFork. An eager Fork is supposed to be
/// a timing component, and contains a register for recording which outputs
/// have accepted the token. Eager Fork will be supported in the next patch.
/// Please refer to test_lazy_fork.mlir test case.
void buildForkLogic(Operation &oldOp, ValueVectorList portList,
                    Location insertLoc, ConversionPatternRewriter &rewriter) {
  ValueVector argSubfields = portList.front();
  Value argValid = argSubfields[0];
  Value argReady = argSubfields[1];

  // The input will be ready to accept new token when all outputs are ready.
  Value *tmpReady = &portList[1][1];
  for (int i = 2, e = portList.size(); i < e; ++i) {
    Value resultReady = portList[i][1];
    AndPrimOp tmpReadyOp = rewriter.create<AndPrimOp>(
        insertLoc, resultReady.getType(), resultReady, *tmpReady);
    *tmpReady = tmpReadyOp.getResult();
  }
  rewriter.create<ConnectOp>(insertLoc, argReady, *tmpReady);

  // All outputs must be ready for the LazyFork to send the token.
  AndPrimOp resultValidOp = rewriter.create<AndPrimOp>(
      insertLoc, argValid.getType(), argValid, *tmpReady);
  for (int i = 1, e = portList.size(); i < e; ++i) {
    ValueVector resultfield = portList[i];
    Value resultValid = resultfield[0];
    rewriter.create<ConnectOp>(insertLoc, resultValid,
                               resultValidOp.getResult());

    if (!cast<handshake::ForkOp>(oldOp).isControl()) {
      Value argData = argSubfields[2];
      Value resultData = resultfield[2];
      rewriter.create<ConnectOp>(insertLoc, resultData, argData);
    }
  }
}

/// Please refer to test_constant.mlir test case.
void buildConstantLogic(Operation &oldOp, ValueVectorList portList,
                        Location insertLoc,
                        ConversionPatternRewriter &rewriter) {
  // The first input is control signal which will trigger the Constant
  // operation to emit tokens.
  ValueVector controlSubfields = portList.front();
  Value controlValid = controlSubfields[0];
  Value controlReady = controlSubfields[1];

  ValueVector resultSubfields = portList.back();
  Value resultValid = resultSubfields[0];
  Value resultReady = resultSubfields[1];
  Value resultData = resultSubfields[2];

  FIRRTLType constantType =
      FlipType::get(resultData.getType().cast<FIRRTLType>());
  int constantAttrValue = oldOp.getAttrOfType<IntegerAttr>("value").getInt();

  rewriter.create<ConnectOp>(insertLoc, resultValid, controlValid);
  rewriter.create<ConnectOp>(insertLoc, controlReady, resultReady);
  rewriter.create<ConnectOp>(
      insertLoc, resultData,
      createConstantOp(insertLoc, constantType, constantAttrValue, rewriter));
}

/// Please refer to test_sink.mlir test case.
void buildSinkLogic(ValueVectorList portList, Location insertLoc,
                    ConversionPatternRewriter &rewriter) {
  ValueVector argSubfields = portList.front();
  Value argValid = argSubfields[0];
  Value argReady = argSubfields[1];
  Value argData = argSubfields[2];

  // A Sink operation is always ready to accept tokens.
  FIRRTLType signalType = argValid.getType().cast<FIRRTLType>();
  Value highSignal = createConstantOp(insertLoc, signalType, 1, rewriter);
  rewriter.create<ConnectOp>(insertLoc, argReady, highSignal);

  rewriter.eraseOp(argValid.getDefiningOp());
  rewriter.eraseOp(argData.getDefiningOp());
}

/// Currently only support {control = true}.
/// Please refer to test_join.mlir test case.
void buildJoinLogic(ValueVectorList portList, Location insertLoc,
                    ConversionPatternRewriter &rewriter) {
  ValueVector resultSubfields = portList.back();
  Value resultValid = resultSubfields[0];
  Value resultReady = resultSubfields[1];

  // The output is triggered only after all inputs are valid.
  Value *tmpValid = &portList[0][0];
  for (int i = 1, e = portList.size() - 1; i < e; ++i) {
    Value argValid = portList[i][0];
    AndPrimOp tmpValidOp = rewriter.create<AndPrimOp>(
        insertLoc, argValid.getType(), argValid, *tmpValid);
    *tmpValid = tmpValidOp.getResult();
  }
  rewriter.create<ConnectOp>(insertLoc, resultValid, *tmpValid);

  // The input will be ready to accept new token when old token is sent out.
  AndPrimOp argReadyOp = rewriter.create<AndPrimOp>(
      insertLoc, resultReady.getType(), resultReady, *tmpValid);
  for (int i = 0, e = portList.size() - 1; i < e; ++i) {
    Value argReady = portList[i][1];
    rewriter.create<ConnectOp>(insertLoc, argReady, argReadyOp.getResult());
  }
}

/// Please refer to simple_addi.mlir test case.
template <typename OpType>
void buildBinaryLogic(ValueVectorList portList, Location insertLoc,
                      ConversionPatternRewriter &rewriter) {
  ValueVector arg0Subfield = portList[0];
  ValueVector arg1Subfield = portList[1];
  ValueVector resultSubfields = portList[2];

  Value arg0Valid = arg0Subfield[0];
  Value arg0Ready = arg0Subfield[1];
  Value arg0Data = arg0Subfield[2];
  Value arg1Valid = arg1Subfield[0];
  Value arg1Ready = arg1Subfield[1];
  Value arg1Data = arg1Subfield[2];
  Value resultValid = resultSubfields[0];
  Value resultReady = resultSubfields[1];
  Value resultData = resultSubfields[2];

  // Carry out the binary operation.
  OpType resultDataOp = rewriter.create<OpType>(insertLoc, arg0Data.getType(),
                                                arg0Data, arg1Data);
  rewriter.create<ConnectOp>(insertLoc, resultData, resultDataOp.getResult());

  // Generate valid signal.
  AndPrimOp resultValidOp = rewriter.create<AndPrimOp>(
      insertLoc, arg0Valid.getType(), arg0Valid, arg1Valid);
  rewriter.create<ConnectOp>(insertLoc, resultValid, resultValidOp.getResult());

  // Generate ready signals.
  AndPrimOp argReadyOp = rewriter.create<AndPrimOp>(
      insertLoc, resultReady.getType(), resultReady, resultValidOp.getResult());
  rewriter.create<ConnectOp>(insertLoc, arg0Ready, argReadyOp.getResult());
  rewriter.create<ConnectOp>(insertLoc, arg1Ready, argReadyOp.getResult());
}

//===----------------------------------------------------------------------===//
// Old Operation Conversion Functions
//===----------------------------------------------------------------------===//

/// Create InstanceOp in the top-module. This will be called after the
/// corresponding sub-module and combinational logic are created.
void createInstOp(Operation &oldOp, FModuleOp subModuleOp,
                  ConversionPatternRewriter &rewriter) {
  rewriter.setInsertionPointAfter(&oldOp);
  using BundleElement = std::pair<Identifier, FIRRTLType>;
  llvm::SmallVector<BundleElement, 4> elements;
  MLIRContext *context = subModuleOp.getContext();

  // Bundle all ports of the instance into a new flattened bundle type.
  int args_idx = 0;
  for (auto &arg : subModuleOp.getArguments()) {
    std::string argName = "arg" + std::to_string(args_idx);
    Identifier argId = Identifier::get(argName, context);

    // All ports of the instance operation are flipped.
    FIRRTLType argType = FlipType::get(arg.getType().cast<BundleType>());
    elements.push_back(BundleElement(argId, argType));
    args_idx += 1;
  }

  // Create a instance operation.
  FIRRTLType instType =
      BundleType::get(ArrayRef<BundleElement>(elements), context);
  firrtl::InstanceOp instOp = rewriter.create<firrtl::InstanceOp>(
      oldOp.getLoc(), instType, subModuleOp.getName(),
      rewriter.getStringAttr(""));

  // Connect the new created instance with its predecessors and successors in
  // the top-module.
  int ports_idx = 0;
  for (auto &element : instType.cast<BundleType>().getElements()) {
    Identifier elementName = element.first;
    FIRRTLType elementType = element.second;
    SubfieldOp subfieldOp = rewriter.create<SubfieldOp>(
        oldOp.getLoc(), elementType, instOp.getResult(),
        rewriter.getStringAttr(elementName.strref()));

    // Connect input ports.
    if (ports_idx < oldOp.getNumOperands())
      rewriter.create<ConnectOp>(oldOp.getLoc(), subfieldOp.getResult(),
                                 oldOp.getOperand(ports_idx));
    // Connect output ports.
    else {
      Value result = oldOp.getResult(ports_idx - oldOp.getNumOperands());
      result.replaceAllUsesWith(subfieldOp.getResult());
    }
    ports_idx += 1;
  }
  rewriter.eraseOp(&oldOp);
}

void convertReturnOp(Operation &oldOp, FModuleOp topModuleOp,
                     ConversionPatternRewriter &rewriter) {
  rewriter.setInsertionPointAfter(&oldOp);
  int numArg = topModuleOp.getNumArguments();

  // Connect each operand of the old return operation with the corresponding
  // output ports.
  for (int i = 0, e = oldOp.getNumOperands(); i < e; ++i)
    rewriter.create<ConnectOp>(oldOp.getLoc(),
                               topModuleOp.getArgument(numArg - e + i),
                               oldOp.getOperand(i));
  rewriter.eraseOp(&oldOp);
}

//===----------------------------------------------------------------------===//
// HandshakeToFIRRTL lowering Pass
//===----------------------------------------------------------------------===//

/// Process of lowering:
///
/// 0)  Create and go into a new FIRRTL circuit;
/// 1)  Create and go into a new FIRRTL top-module;
/// 2)  Inline Handshake FuncOp region into the FIRRTL top-module;
/// 3)  Traverse and convert each Standard or Handshake operation:
///   i)    Check if an identical sub-module exists. If so, skip to vi);
///   ii)   Create and go into a new FIRRTL sub-module;
///   iii)  Extract data (if applied), valid, and ready subfield from each port
///         of the sub-module;
///   iv)   Build combinational logic;
///   v)    Exit the sub-module and go back to the top-module;
///   vi)   Create an new instance for the sub-module;
///   vii)  Connect the instance with its predecessors and successors;
/// 4)  Erase the Handshake FuncOp.
///
/// createTopModuleOp():  1) and 2)
/// checkSubModuleOp():   3.i)
/// createSubModuleOp():  3.ii)
/// extractSubfields():   3.iii)
/// build*Logic():        3.iv)
/// createInstOp():       3.v), 3.vi), and 3.vii)
///
/// Please refer to test_addi.mlir test case.
struct HandshakeFuncOpLowering : public OpConversionPattern<handshake::FuncOp> {
  using OpConversionPattern<handshake::FuncOp>::OpConversionPattern;

  LogicalResult match(Operation *op) const override { return success(); }

  void rewrite(handshake::FuncOp funcOp, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const override {
    // Create FIRRTL circuit and top-module operation.
    CircuitOp circuitOp = rewriter.create<CircuitOp>(
        funcOp.getLoc(), rewriter.getStringAttr(funcOp.getName()));
    rewriter.setInsertionPointToStart(circuitOp.getBody());
    FModuleOp topModuleOp = createTopModuleOp(funcOp, rewriter);

    // Traverse and convert each operation in funcOp.
    for (Operation &op : topModuleOp.getBody().front()) {
      if (isa<handshake::ReturnOp>(op))
        convertReturnOp(op, topModuleOp, rewriter);

      // This branch takes care of all non-timing operations that require to
      // be instantiated in the top-module.
      else if (op.getDialect()->getNamespace() != StringRef("firrtl")) {
        FModuleOp subModuleOp = checkSubModuleOp(topModuleOp, op);

        // Check if the sub-module already exists.
        if (!subModuleOp) {
          subModuleOp = createSubModuleOp(topModuleOp, op, rewriter);

          Operation *termOp = subModuleOp.getBody().front().getTerminator();
          Location insertLoc = termOp->getLoc();
          rewriter.setInsertionPoint(termOp);

          ValueVectorList portList =
              extractSubfields(subModuleOp, insertLoc, rewriter);

          if (isa<mlir::AddIOp>(op))
            buildBinaryLogic<AddPrimOp>(portList, insertLoc, rewriter);
          else if (isa<mlir::SubIOp>(op))
            buildBinaryLogic<SubPrimOp>(portList, insertLoc, rewriter);
          else if (isa<mlir::MulIOp>(op))
            buildBinaryLogic<MulPrimOp>(portList, insertLoc, rewriter);

          else if (isa<mlir::AndOp>(op))
            buildBinaryLogic<AndPrimOp>(portList, insertLoc, rewriter);
          else if (isa<mlir::OrOp>(op))
            buildBinaryLogic<OrPrimOp>(portList, insertLoc, rewriter);
          else if (isa<mlir::XOrOp>(op))
            buildBinaryLogic<XorPrimOp>(portList, insertLoc, rewriter);

          else if (auto cmpOp = dyn_cast<mlir::CmpIOp>(op)) {
            auto cmpOpAttr = cmpOp.getPredicate();
            switch (cmpOpAttr) {
            case CmpIPredicate::eq:
              buildBinaryLogic<EQPrimOp>(portList, insertLoc, rewriter);
              break;
            case CmpIPredicate::ne:
              buildBinaryLogic<NEQPrimOp>(portList, insertLoc, rewriter);
              break;
            case CmpIPredicate::slt:
              buildBinaryLogic<LTPrimOp>(portList, insertLoc, rewriter);
              break;
            case CmpIPredicate::sle:
              buildBinaryLogic<LEQPrimOp>(portList, insertLoc, rewriter);
              break;
            case CmpIPredicate::sgt:
              buildBinaryLogic<GTPrimOp>(portList, insertLoc, rewriter);
              break;
            case CmpIPredicate::sge:
              buildBinaryLogic<GEQPrimOp>(portList, insertLoc, rewriter);
              break;
            }
          } else if (isa<mlir::ShiftLeftOp>(op))
            buildBinaryLogic<DShlPrimOp>(portList, insertLoc, rewriter);
          else if (isa<mlir::SignedShiftRightOp>(op))
            buildBinaryLogic<DShrPrimOp>(portList, insertLoc, rewriter);

          // Build elastic components logic
          else if (isa<handshake::MergeOp>(op))
            buildMergeLogic(portList, insertLoc, rewriter);
          else if (isa<handshake::ControlMergeOp>(op))
            buildControlMergeLogic(op, portList, insertLoc, rewriter);

          else if (isa<handshake::MuxOp>(op))
            buildMuxLogic(portList, insertLoc, rewriter);

          else if (isa<handshake::BranchOp>(op))
            buildBranchLogic(op, portList, insertLoc, rewriter);
          else if (isa<handshake::ConditionalBranchOp>(op))
            buildConditionalBranchLogic(op, portList, insertLoc, rewriter);

          else if (isa<handshake::ForkOp>(op))
            buildForkLogic(op, portList, insertLoc, rewriter);
          else if (isa<handshake::LazyForkOp>(op))
            buildLazyForkLogic(op, portList, insertLoc, rewriter);

          else if (isa<handshake::ConstantOp>(op))
            buildConstantLogic(op, portList, insertLoc, rewriter);
          else if (isa<handshake::SinkOp>(op))
            buildSinkLogic(portList, insertLoc, rewriter);

          else if (isa<handshake::JoinOp>(op))
            buildJoinLogic(portList, insertLoc, rewriter);
          else
            assert(false && "Usupported operation type.");
        }

        // Instantiate the new created sub-module.
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
  PassRegistration<HandshakeToFIRRTLPass>("lower-handshake-to-firrtl",
                                          "Lowering to FIRRTL Dialect");
}