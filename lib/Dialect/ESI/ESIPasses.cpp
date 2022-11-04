//===- ESIPasses.cpp - ESI to HW/SV conversion passes -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower ESI to HW and SV.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESIPasses.h"
#include "circt/Dialect/ESI/ESIServices.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/SymCache.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"

#include <memory>

#ifdef CAPNP
#include "capnp/ESICapnp.h"
#endif

using namespace circt;
using namespace circt::comb;
using namespace circt::esi;
using namespace circt::hw;
using namespace circt::sv;

//===----------------------------------------------------------------------===//
// ESI custom op builder.
//===----------------------------------------------------------------------===//

namespace {
/// Assist the lowering steps for conversions which need to create auxiliary IR.
class ESIHWBuilder : public circt::ImplicitLocOpBuilder {
public:
  ESIHWBuilder(Operation *top);

  ArrayAttr getStageParameterList(Attribute value);

  HWModuleExternOp declareStage(Operation *symTable, PipelineStageOp);
  // Will be unused when CAPNP is undefined
  HWModuleExternOp declareCosimEndpointOp(Operation *symTable, Type sendType,
                                          Type recvType) LLVM_ATTRIBUTE_UNUSED;

  InterfaceOp getOrConstructInterface(ChannelType);
  InterfaceOp constructInterface(ChannelType);

  // A bunch of constants for use in various places below.
  const StringAttr a, aValid, aReady, x, xValid, xReady;
  const StringAttr dataOutValid, dataOutReady, dataOut, dataInValid,
      dataInReady, dataIn;
  const StringAttr clk, rst;
  const StringAttr width;

  // Various identifier strings. Keep them all here in case we rename them.
  static constexpr char dataStr[] = "data", validStr[] = "valid",
                        readyStr[] = "ready", sourceStr[] = "source",
                        sinkStr[] = "sink";

private:
  /// Construct a type-appropriate name for the interface, making sure it's not
  /// taken in the symbol table.
  StringAttr constructInterfaceName(ChannelType);

  llvm::DenseMap<Type, HWModuleExternOp> declaredStage;
  llvm::DenseMap<std::pair<Type, Type>, HWModuleExternOp>
      declaredCosimEndpointOp;
  llvm::DenseMap<Type, InterfaceOp> portTypeLookup;
};
} // anonymous namespace

// C++ requires this for showing it what object file it should store these
// symbols in. They should be inline but that feature wasn't added until C++17.
constexpr char ESIHWBuilder::dataStr[], ESIHWBuilder::validStr[],
    ESIHWBuilder::readyStr[], ESIHWBuilder::sourceStr[],
    ESIHWBuilder::sinkStr[];

ESIHWBuilder::ESIHWBuilder(Operation *top)
    : ImplicitLocOpBuilder(UnknownLoc::get(top->getContext()), top),
      a(StringAttr::get(getContext(), "a")),
      aValid(StringAttr::get(getContext(), "a_valid")),
      aReady(StringAttr::get(getContext(), "a_ready")),
      x(StringAttr::get(getContext(), "x")),
      xValid(StringAttr::get(getContext(), "x_valid")),
      xReady(StringAttr::get(getContext(), "x_ready")),
      dataOutValid(StringAttr::get(getContext(), "DataOutValid")),
      dataOutReady(StringAttr::get(getContext(), "DataOutReady")),
      dataOut(StringAttr::get(getContext(), "DataOut")),
      dataInValid(StringAttr::get(getContext(), "DataInValid")),
      dataInReady(StringAttr::get(getContext(), "DataInReady")),
      dataIn(StringAttr::get(getContext(), "DataIn")),
      clk(StringAttr::get(getContext(), "clk")),
      rst(StringAttr::get(getContext(), "rst")),
      width(StringAttr::get(getContext(), "WIDTH")) {

  auto regions = top->getRegions();
  if (regions.size() == 0) {
    top->emitError("ESI HW Builder needs a region to insert HW.");
  }
  auto &region = regions.front();
  if (!region.empty())
    setInsertionPoint(&region.front(), region.front().begin());
}

static StringAttr constructUniqueSymbol(Operation *tableOp,
                                        StringRef proposedNameRef) {
  SmallString<64> proposedName = proposedNameRef;

  // Normalize the type name.
  for (char &ch : proposedName) {
    if (isalpha(ch) || isdigit(ch) || ch == '_')
      continue;
    ch = '_';
  }

  // Make sure that this symbol isn't taken. If it is, append a number and try
  // again.
  size_t baseLength = proposedName.size();
  size_t tries = 0;
  while (SymbolTable::lookupSymbolIn(tableOp, proposedName)) {
    proposedName.resize(baseLength);
    proposedName.append(llvm::utostr(++tries));
  }

  return StringAttr::get(tableOp->getContext(), proposedName);
}

StringAttr ESIHWBuilder::constructInterfaceName(ChannelType port) {
  Operation *tableOp =
      getInsertionPoint()->getParentWithTrait<mlir::OpTrait::SymbolTable>();

  // Get a name based on the type.
  std::string portTypeName;
  llvm::raw_string_ostream nameOS(portTypeName);
  TypeSwitch<Type>(port.getInner())
      .Case([&](hw::ArrayType arr) {
        nameOS << "ArrayOf" << arr.getSize() << 'x' << arr.getElementType();
      })
      .Case([&](hw::StructType t) { nameOS << "Struct"; })
      .Default([&](Type t) { nameOS << port.getInner(); });

  // Don't allow the name to end with '_'.
  ssize_t i = portTypeName.size() - 1;
  while (i >= 0 && portTypeName[i] == '_') {
    --i;
  }
  portTypeName = portTypeName.substr(0, i + 1);

  // All stage names start with this.
  SmallString<64> proposedName("IValidReady_");
  proposedName.append(portTypeName);
  return constructUniqueSymbol(tableOp, proposedName);
}

/// Return a parameter list for the stage module with the specified value.
ArrayAttr ESIHWBuilder::getStageParameterList(Attribute value) {
  auto type = IntegerType::get(width.getContext(), 32, IntegerType::Unsigned);
  auto widthParam = ParamDeclAttr::get(width.getContext(), width, type, value);
  return ArrayAttr::get(width.getContext(), widthParam);
}

/// Write an 'ExternModuleOp' to use a hand-coded SystemVerilog module. Said
/// module implements pipeline stage, adding 1 cycle latency. This particular
/// implementation is double-buffered and fully pipelines the reverse-flow ready
/// signal.
HWModuleExternOp ESIHWBuilder::declareStage(Operation *symTable,
                                            PipelineStageOp stage) {
  Type dataType = stage.innerType();
  HWModuleExternOp &stageMod = declaredStage[dataType];
  if (stageMod)
    return stageMod;

  // Since this module has parameterized widths on the a input and x output,
  // give the extern declation a None type since nothing else makes sense.
  // Will be refining this when we decide how to better handle parameterized
  // types and ops.
  size_t argn = 0;
  size_t resn = 0;
  llvm::SmallVector<PortInfo> ports = {
      {clk, PortDirection::INPUT, getI1Type(), argn++},
      {rst, PortDirection::INPUT, getI1Type(), argn++}};

  ports.push_back({a, PortDirection::INPUT, dataType, argn++});
  ports.push_back({aValid, PortDirection::INPUT, getI1Type(), argn++});
  ports.push_back({aReady, PortDirection::OUTPUT, getI1Type(), resn++});
  ports.push_back({x, PortDirection::OUTPUT, dataType, resn++});

  ports.push_back({xValid, PortDirection::OUTPUT, getI1Type(), resn++});
  ports.push_back({xReady, PortDirection::INPUT, getI1Type(), argn++});

  stageMod = create<HWModuleExternOp>(
      constructUniqueSymbol(symTable, "ESI_PipelineStage"), ports,
      "ESI_PipelineStage", getStageParameterList({}));
  return stageMod;
}

/// Write an 'ExternModuleOp' to use a hand-coded SystemVerilog module. Said
/// module contains a bi-directional Cosimulation DPI interface with valid/ready
/// semantics.
HWModuleExternOp ESIHWBuilder::declareCosimEndpointOp(Operation *symTable,
                                                      Type sendType,
                                                      Type recvType) {
  HWModuleExternOp &endpoint =
      declaredCosimEndpointOp[std::make_pair(sendType, recvType)];
  if (endpoint)
    return endpoint;
  // Since this module has parameterized widths on the a input and x output,
  // give the extern declation a None type since nothing else makes sense.
  // Will be refining this when we decide how to better handle parameterized
  // types and ops.
  PortInfo ports[] = {{clk, PortDirection::INPUT, getI1Type(), 0},
                      {rst, PortDirection::INPUT, getI1Type(), 1},
                      {dataOutValid, PortDirection::OUTPUT, getI1Type(), 0},
                      {dataOutReady, PortDirection::INPUT, getI1Type(), 2},
                      {dataOut, PortDirection::OUTPUT, recvType, 1},
                      {dataInValid, PortDirection::INPUT, getI1Type(), 3},
                      {dataInReady, PortDirection::OUTPUT, getI1Type(), 2},
                      {dataIn, PortDirection::INPUT, sendType, 4}};
  SmallVector<Attribute, 8> params;
  params.push_back(ParamDeclAttr::get("ENDPOINT_ID_EXT", getStringAttr("")));
  params.push_back(
      ParamDeclAttr::get("SEND_TYPE_ID", getIntegerType(64, false)));
  params.push_back(ParamDeclAttr::get("SEND_TYPE_SIZE_BITS", getI32Type()));
  params.push_back(
      ParamDeclAttr::get("RECV_TYPE_ID", getIntegerType(64, false)));
  params.push_back(ParamDeclAttr::get("RECV_TYPE_SIZE_BITS", getI32Type()));
  endpoint = create<HWModuleExternOp>(
      constructUniqueSymbol(symTable, "Cosim_Endpoint"), ports,
      "Cosim_Endpoint", ArrayAttr::get(getContext(), params));
  return endpoint;
}

/// Return the InterfaceType which corresponds to an ESI port type. If it
/// doesn't exist in the cache, build the InterfaceOp and the corresponding
/// type.
InterfaceOp ESIHWBuilder::getOrConstructInterface(ChannelType t) {
  auto ifaceIter = portTypeLookup.find(t);
  if (ifaceIter != portTypeLookup.end())
    return ifaceIter->second;
  auto iface = constructInterface(t);
  portTypeLookup[t] = iface;
  return iface;
}

InterfaceOp ESIHWBuilder::constructInterface(ChannelType chan) {
  return create<InterfaceOp>(constructInterfaceName(chan).getValue(), [&]() {
    create<InterfaceSignalOp>(validStr, getI1Type());
    create<InterfaceSignalOp>(readyStr, getI1Type());
    create<InterfaceSignalOp>(dataStr, chan.getInner());
    llvm::SmallVector<StringRef> validDataStrs;
    validDataStrs.push_back(validStr);
    validDataStrs.push_back(dataStr);
    create<InterfaceModportOp>(sinkStr,
                               /*inputs=*/ArrayRef<StringRef>{readyStr},
                               /*outputs=*/validDataStrs);
    create<InterfaceModportOp>(sourceStr,
                               /*inputs=*/validDataStrs,
                               /*outputs=*/ArrayRef<StringRef>{readyStr});
  });
}

//===----------------------------------------------------------------------===//
// Lower to physical conversions and pass.
//===----------------------------------------------------------------------===//

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
/// Run all the physical lowerings.
struct ESIToPhysicalPass : public LowerESIToPhysicalBase<ESIToPhysicalPass> {
  void runOnOperation() override;
};
} // anonymous namespace

void ESIToPhysicalPass::runOnOperation() {
  // Set up a conversion and give it a set of laws.
  ConversionTarget target(getContext());
  target.addLegalDialect<ESIDialect>();
  target.addIllegalOp<ChannelBufferOp>();

  // Add all the conversion patterns.
  RewritePatternSet patterns(&getContext());
  patterns.insert<ChannelBufferLowering>(&getContext());

  // Run the conversion.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

//===----------------------------------------------------------------------===//
// Lower ESI ports pass.
//===----------------------------------------------------------------------===//

namespace {
/// Convert all the ESI ports on modules to some lower construct. SV interfaces
/// for now on external modules, ready/valid to modules defined internally. In
/// the future, it may be possible to select a different format.
struct ESIPortsPass : public LowerESIPortsBase<ESIPortsPass> {
  void runOnOperation() override;

private:
  bool updateFunc(HWModuleOp mod);
  void updateInstance(HWModuleOp mod, InstanceOp inst);

  bool updateFunc(HWModuleExternOp mod);
  void updateInstance(HWModuleExternOp mod, InstanceOp inst);
  ESIHWBuilder *build;
};
} // anonymous namespace

/// Iterate through the `hw.module[.extern]`s and lower their ports.
void ESIPortsPass::runOnOperation() {
  ModuleOp top = getOperation();
  ESIHWBuilder b(top);
  build = &b;

  // Find all externmodules and try to modify them. Remember the modified ones.
  DenseMap<StringRef, HWModuleExternOp> externModsMutated;
  for (auto mod : top.getOps<HWModuleExternOp>())
    if (updateFunc(mod))
      externModsMutated[mod.getName()] = mod;

  // Find all instances and update them.
  top.walk([&externModsMutated, this](InstanceOp inst) {
    auto mapIter = externModsMutated.find(inst.getModuleName());
    if (mapIter != externModsMutated.end())
      updateInstance(mapIter->second, inst);
  });

  // Find all modules and try to modify them to have wires with valid/ready
  // semantics. Remember the modified ones.
  DenseMap<StringRef, HWModuleOp> modsMutated;
  for (auto mod : top.getOps<HWModuleOp>())
    if (updateFunc(mod))
      modsMutated[mod.getName()] = mod;

  // Find all instances and update them.
  top.walk([&modsMutated, this](InstanceOp inst) {
    auto mapIter = modsMutated.find(inst.getModuleName());
    if (mapIter != modsMutated.end())
      updateInstance(mapIter->second, inst);
  });

  build = nullptr;
}

/// Return a attribute with the specified suffix appended.
static StringAttr appendToRtlName(StringAttr base, StringRef suffix) {
  auto *context = base.getContext();
  return StringAttr::get(context, base.getValue().str() + suffix.str());
}

/// Convert all input and output ChannelTypes into valid/ready wires. Try not to
/// change the order and materialize ops in reasonably intuitive locations.
bool ESIPortsPass::updateFunc(HWModuleOp mod) {
  auto funcType = mod.getFunctionType();
  // Build ops in the module.
  ImplicitLocOpBuilder modBuilder(mod.getLoc(), mod.getBody());
  Type i1 = modBuilder.getI1Type();

  // Get information to be used later on.
  hw::OutputOp outOp = cast<hw::OutputOp>(mod.getBodyBlock()->getTerminator());

  bool updated = false;

  // Reconstruct the list of operand types, changing the type whenever an ESI
  // port is found. Keep the argument attributes, apply the ESI ports attributes
  // to the data port only.
  SmallVector<Type, 16> newArgTypes;

  // 'Ready' signals are outputs. Remember them for later when we deal with the
  // returns.
  SmallVector<std::pair<Value, StringAttr>, 8> newReadySignals;
  SmallVector<Attribute> newArgNames;

  for (size_t argNum = 0, blockArgNum = 0, e = mod.getNumArguments();
       argNum < e; ++argNum, ++blockArgNum) {
    Type argTy = funcType.getInput(argNum);
    auto argNameAttr = getModuleArgumentNameAttr(mod, argNum);

    auto chanTy = argTy.dyn_cast<ChannelType>();
    if (!chanTy) {
      // If not ESI, pass through.
      newArgTypes.push_back(argTy);
      newArgNames.push_back(argNameAttr);
      continue;
    }
    // When we find one, add a data and valid signal to the new args.
    Value data;
    newArgTypes.push_back(chanTy.getInner());
    newArgNames.push_back(argNameAttr);
    data = mod.front().insertArgument(blockArgNum, chanTy.getInner(),
                                      mod.getArgument(argNum).getLoc());
    ++blockArgNum;

    newArgTypes.push_back(i1);
    newArgNames.push_back(appendToRtlName(argNameAttr, "_valid"));
    Value valid = mod.front().insertArgument(blockArgNum, i1,
                                             mod.getArgument(argNum).getLoc());
    ++blockArgNum;
    // Build the ESI wrap operation to translate the lowered signals to what
    // they were. (A later pass takes care of eliminating the ESI ops.)
    auto wrap = modBuilder.create<WrapValidReadyOp>(data, valid);
    // Replace uses of the old ESI port argument with the new one from the wrap.
    mod.front()
        .getArgument(blockArgNum)
        .replaceAllUsesWith(wrap.getChanOutput());
    // Delete the ESI port block argument.
    mod.front().eraseArgument(blockArgNum);
    --blockArgNum;
    newReadySignals.push_back(std::make_pair(
        wrap.getReady(), appendToRtlName(argNameAttr, "_ready")));
    updated = true;
  }

  // Iterate through the outputs, appending to all of the next three lists.
  // Lower the ESI ports.
  SmallVector<Type, 8> newResultTypes;
  SmallVector<Value, 8> newOutputOperands;
  SmallVector<Attribute> newResultNames;

  modBuilder.setInsertionPointToEnd(mod.getBodyBlock());
  for (size_t resNum = 0, numRes = funcType.getNumResults(); resNum < numRes;
       ++resNum) {
    Type resTy = funcType.getResult(resNum);
    auto chanTy = resTy.dyn_cast<ChannelType>();
    Value oldOutputValue = outOp.getOperand(resNum);
    auto oldResultName = getModuleResultNameAttr(mod, resNum);
    if (!chanTy) {
      // If not ESI, pass through.
      newResultTypes.push_back(resTy);
      newResultNames.push_back(oldResultName);
      newOutputOperands.push_back(oldOutputValue);
      continue;
    }

    // Lower the output, adding ready signals directly to the arg list.
    Value ready = mod.front().addArgument(
        i1, modBuilder.getUnknownLoc()); // Ready block arg.
    auto unwrap = modBuilder.create<UnwrapValidReadyOp>(oldOutputValue, ready);
    newOutputOperands.push_back(unwrap.getRawOutput());
    newResultTypes.push_back(chanTy.getInner()); // Raw data.
    newResultNames.push_back(oldResultName);

    newResultTypes.push_back(i1); // Valid.
    newOutputOperands.push_back(unwrap.getValid());
    newResultNames.push_back(appendToRtlName(oldResultName, "_valid"));

    newArgTypes.push_back(i1); // Ready func arg.
    newArgNames.push_back(appendToRtlName(oldResultName, "_ready"));
    updated = true;
  }

  // Append the ready list signals we remembered above.
  for (const auto &readySig : newReadySignals) {
    newResultTypes.push_back(i1);
    newResultNames.push_back(readySig.second);
    newOutputOperands.push_back(readySig.first);
  }

  if (!updated)
    return false;

  // A new output op is necessary.
  outOp.erase();
  modBuilder.create<hw::OutputOp>(newOutputOperands);

  // Set the new types.
  auto newFuncType = modBuilder.getFunctionType(newArgTypes, newResultTypes);
  mod.setType(newFuncType);
  setModuleArgumentNames(mod, newArgNames);
  setModuleResultNames(mod, newResultNames);
  return true;
}

/// Update an instance of an updated module by adding `esi.[un]wrap.vr`
/// ops around the instance. Lowering or folding away `[un]wrap` ops is another
/// pass.
void ESIPortsPass::updateInstance(HWModuleOp mod, InstanceOp inst) {
  ImplicitLocOpBuilder b(inst.getLoc(), inst);
  BackedgeBuilder beb(b, inst.getLoc());
  Type i1 = b.getI1Type();

  // -----
  // Lower the operands.

  SmallVector<Value, 16> newOperands;
  // Store the 'ready' operands from the unwrap as a list of backedges. This
  // doubles as a count of `i1`s to append to the existing results.
  SmallVector<Backedge, 8> inputReadysToConnect;
  for (auto operand : inst.getOperands()) {
    if (!operand.getType().isa<ChannelType>()) {
      newOperands.push_back(operand);
      continue;
    }

    auto ready = beb.get(i1);
    inputReadysToConnect.push_back(ready);
    auto unwrap = b.create<UnwrapValidReadyOp>(operand, ready);
    newOperands.push_back(unwrap.getRawOutput());
    newOperands.push_back(unwrap.getValid());
  }

  // -----
  // Lower the result types.

  SmallVector<Type, 16> resTypes;
  // Backedges which we add as operands to be used later when we build the
  // 'wrap' ops.
  SmallVector<Backedge, 8> outputReadysToConnect;
  for (auto resTy : inst.getResultTypes()) {
    auto cpTy = resTy.dyn_cast<ChannelType>();
    if (!cpTy) {
      resTypes.push_back(resTy);
      continue;
    }
    resTypes.push_back(cpTy.getInner());
    resTypes.push_back(i1);
    Backedge ready = beb.get(i1);
    newOperands.push_back(ready);
    outputReadysToConnect.push_back(ready);
  }
  resTypes.append(inputReadysToConnect.size(), i1);

  // -----
  // Clone the instance.
  b.setInsertionPointAfter(inst);
  auto newInst =
      b.create<InstanceOp>(mod, inst.getInstanceNameAttr(), newOperands,
                           inst.getParameters(), inst.getInnerSymAttr());

  // -----
  // Wrap the results back into ESI channels and connect up all the ready
  // signals.

  size_t newInstResNum = 0;
  size_t readyIdx = 0;
  for (auto res : inst.getResults()) {
    auto cpTy = res.getType().dyn_cast<ChannelType>();
    if (!cpTy) {
      res.replaceAllUsesWith(newInst.getResult(newInstResNum));
      newInstResNum++;
      continue;
    }

    WrapValidReadyOp wrap;
    wrap = b.create<WrapValidReadyOp>(newInst.getResult(newInstResNum),
                                      newInst.getResult(newInstResNum + 1));
    newInstResNum += 2;
    res.replaceAllUsesWith(wrap.getChanOutput());
    outputReadysToConnect[readyIdx].setValue(wrap.getReady());
    readyIdx++;
  }

  for (auto inputReady : inputReadysToConnect) {
    inputReady.setValue(newInst.getResult(newInstResNum));
    newInstResNum++;
  }

  // Erase the old instance.
  inst.erase();
}

/// Convert all input and output ChannelTypes into SV Interfaces. For inputs,
/// just switch the type to `ModportType`. For outputs, append a `ModportType`
/// to the inputs and remove the output channel from the results. Returns true
/// if 'mod' was updated. Delay updating the instances to amortize the IR walk
/// over all the module updates.
bool ESIPortsPass::updateFunc(HWModuleExternOp mod) {
  auto *ctxt = &getContext();

  bool updated = false;

  SmallVector<Attribute> newArgNames, newResultNames;

  // Reconstruct the list of operand types, changing the type whenever an ESI
  // port is found.
  SmallVector<Type, 16> newArgTypes;
  size_t nextArgNo = 0;
  for (auto argTy : mod.getArgumentTypes()) {
    auto chanTy = argTy.dyn_cast<ChannelType>();
    newArgNames.push_back(getModuleArgumentNameAttr(mod, nextArgNo++));

    if (!chanTy) {
      newArgTypes.push_back(argTy);
      continue;
    }

    // When we find one, construct an interface, and add the 'source' modport to
    // the type list.
    auto iface = build->getOrConstructInterface(chanTy);
    newArgTypes.push_back(iface.getModportType(ESIHWBuilder::sourceStr));
    updated = true;
  }

  // Iterate through the results and append to one of the two below lists. The
  // first for non-ESI-ports. The second, ports which have been re-located to an
  // operand.
  SmallVector<Type, 8> newResultTypes;
  SmallVector<DictionaryAttr, 4> newResultAttrs;
  auto funcType = mod.getFunctionType();
  for (size_t resNum = 0, numRes = mod.getNumResults(); resNum < numRes;
       ++resNum) {
    Type resTy = funcType.getResult(resNum);
    auto chanTy = resTy.dyn_cast<ChannelType>();
    auto resNameAttr = getModuleResultNameAttr(mod, resNum);
    if (!chanTy) {
      newResultTypes.push_back(resTy);
      newResultNames.push_back(resNameAttr);
      continue;
    }

    // When we find one, construct an interface, and add the 'sink' modport to
    // the type list.
    InterfaceOp iface = build->getOrConstructInterface(chanTy);
    ModportType sinkPort = iface.getModportType(ESIHWBuilder::sinkStr);
    newArgTypes.push_back(sinkPort);
    newArgNames.push_back(resNameAttr);
    updated = true;
  }

  if (!updated)
    return false;

  // Set the new types.
  auto newFuncType = FunctionType::get(ctxt, newArgTypes, newResultTypes);
  mod.setType(newFuncType);
  setModuleArgumentNames(mod, newArgNames);
  setModuleResultNames(mod, newResultNames);
  return true;
}

static StringRef getOperandName(Value operand) {
  if (BlockArgument arg = operand.dyn_cast<BlockArgument>()) {
    auto op = arg.getParentBlock()->getParentOp();
    if (op && hw::isAnyModule(op))
      return hw::getModuleArgumentName(op, arg.getArgNumber());
  } else {
    auto srcOp = operand.getDefiningOp();
    if (auto instOp = dyn_cast<InstanceOp>(srcOp))
      return instOp.instanceName();

    if (auto srcName = srcOp->getAttrOfType<StringAttr>("name"))
      return srcName.getValue();
  }
  return "";
}

/// Create a reasonable name for a SV interface instance.
static std::string &constructInstanceName(Value operand, InterfaceOp iface,
                                          std::string &name) {
  llvm::raw_string_ostream s(name);
  // Drop the "IValidReady_" part of the interface name.
  s << llvm::toLower(iface.getSymName()[12]) << iface.getSymName().substr(13);

  // Indicate to where the source is connected.
  if (operand.hasOneUse()) {
    Operation *dstOp = *operand.getUsers().begin();
    if (auto instOp = dyn_cast<InstanceOp>(dstOp))
      s << "To" << llvm::toUpper(instOp.instanceName()[0])
        << instOp.instanceName().substr(1);
    else if (auto dstName = dstOp->getAttrOfType<StringAttr>("name"))
      s << "To" << dstName.getValue();
  }

  // Indicate to where the sink is connected.
  StringRef operName = getOperandName(operand);
  if (!operName.empty())
    s << "From" << llvm::toUpper(operName[0]) << operName.substr(1);
  return s.str();
}

/// Update an instance of an updated module by adding `esi.(un)wrap.iface`
/// around the instance. Create a new instance at the end from the lists built
/// up before.
void ESIPortsPass::updateInstance(HWModuleExternOp mod, InstanceOp inst) {
  using namespace circt::sv;
  circt::ImplicitLocOpBuilder instBuilder(inst.getLoc(), inst);
  FunctionType funcTy = mod.getFunctionType();

  // op counter for error reporting purposes.
  size_t opNum = 0;
  // List of new operands.
  SmallVector<Value, 16> newOperands;

  // Fill the new operand list with old plain operands and mutated ones.
  std::string nameStringBuffer; // raw_string_ostream uses std::string.
  for (auto op : inst.getOperands()) {
    auto instChanTy = op.getType().dyn_cast<ChannelType>();
    if (!instChanTy) {
      newOperands.push_back(op);
      ++opNum;
      continue;
    }

    // Get the interface from the cache, and make sure it's the same one as
    // being used in the module.
    auto iface = build->getOrConstructInterface(instChanTy);
    if (iface.getModportType(ESIHWBuilder::sourceStr) !=
        funcTy.getInput(opNum)) {
      inst.emitOpError("ESI ChannelType (operand #")
          << opNum << ") doesn't match module!";
      ++opNum;
      newOperands.push_back(op);
      continue;
    }
    ++opNum;

    // Build a gasket by instantiating an interface, connecting one end to an
    // `esi.unwrap.iface` and the other end to the instance.
    auto ifaceInst =
        instBuilder.create<InterfaceInstanceOp>(iface.getInterfaceType());
    nameStringBuffer.clear();
    ifaceInst->setAttr(
        "name",
        StringAttr::get(mod.getContext(),
                        constructInstanceName(op, iface, nameStringBuffer)));
    GetModportOp sinkModport =
        instBuilder.create<GetModportOp>(ifaceInst, ESIHWBuilder::sinkStr);
    instBuilder.create<UnwrapSVInterfaceOp>(op, sinkModport);
    GetModportOp sourceModport =
        instBuilder.create<GetModportOp>(ifaceInst, ESIHWBuilder::sourceStr);
    // Finally, add the correct modport to the list of operands.
    newOperands.push_back(sourceModport);
  }

  // Go through the results and get both a list of the plain old values being
  // produced and their types.
  SmallVector<Value, 8> newResults;
  SmallVector<Type, 8> newResultTypes;
  for (size_t resNum = 0, numRes = inst.getNumResults(); resNum < numRes;
       ++resNum) {
    Value res = inst.getResult(resNum);
    auto instChanTy = res.getType().dyn_cast<ChannelType>();
    if (!instChanTy) {
      newResults.push_back(res);
      newResultTypes.push_back(res.getType());
      continue;
    }

    // Get the interface from the cache, and make sure it's the same one as
    // being used in the module.
    auto iface = build->getOrConstructInterface(instChanTy);
    if (iface.getModportType(ESIHWBuilder::sinkStr) != funcTy.getInput(opNum)) {
      inst.emitOpError("ESI ChannelType (result #")
          << resNum << ", operand #" << opNum << ") doesn't match module!";
      ++opNum;
      newResults.push_back(res);
      newResultTypes.push_back(res.getType());
      continue;
    }
    ++opNum;

    // Build a gasket by instantiating an interface, connecting one end to an
    // `esi.wrap.iface` and the other end to the instance. Append it to the
    // operand list.
    auto ifaceInst =
        instBuilder.create<InterfaceInstanceOp>(iface.getInterfaceType());
    nameStringBuffer.clear();
    ifaceInst->setAttr(
        "name",
        StringAttr::get(mod.getContext(),
                        constructInstanceName(res, iface, nameStringBuffer)));
    GetModportOp sourceModport =
        instBuilder.create<GetModportOp>(ifaceInst, ESIHWBuilder::sourceStr);
    auto newChannel =
        instBuilder.create<WrapSVInterfaceOp>(res.getType(), sourceModport);
    // Connect all the old users of the output channel with the newly
    // wrapped replacement channel.
    res.replaceAllUsesWith(newChannel);
    GetModportOp sinkModport =
        instBuilder.create<GetModportOp>(ifaceInst, ESIHWBuilder::sinkStr);
    // And add the modport on the other side to the new operand list.
    newOperands.push_back(sinkModport);
  }

  // Create the new instance!
  InstanceOp newInst = instBuilder.create<InstanceOp>(
      mod, inst.getInstanceNameAttr(), newOperands, inst.getParameters(),
      inst.getInnerSymAttr());

  // Go through the old list of non-ESI result values, and replace them with the
  // new non-ESI results.
  for (size_t resNum = 0, numRes = newResults.size(); resNum < numRes;
       ++resNum) {
    newResults[resNum].replaceAllUsesWith(newInst.getResult(resNum));
  }
  // Erase the old instance!
  inst.erase();
}

//===----------------------------------------------------------------------===//
// Lower to HW/SV conversions and pass.
//===----------------------------------------------------------------------===//

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
  auto chPort = stage.getInput().getType().dyn_cast<ChannelType>();
  if (!chPort)
    return rewriter.notifyMatchFailure(stage, "stage had wrong type");
  Operation *symTable = stage->getParentWithTrait<OpTrait::SymbolTable>();
  auto stageModule = builder.declareStage(symTable, stage);

  size_t width = circt::hw::getBitWidth(chPort.getInner());

  ArrayAttr stageParams =
      builder.getStageParameterList(rewriter.getUI32IntegerAttr(width));

  // Unwrap the channel. The ready signal is a Value we haven't created yet, so
  // create a temp value and replace it later. Give this constant an odd-looking
  // type to make debugging easier.
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
  auto stageInst = rewriter.create<InstanceOp>(loc, stageModule, pipeStageName,
                                               operands, stageParams);
  auto stageInstResults = stageInst.getResults();

  // Set a_ready (from the unwrap) back edge correctly to its output from stage.
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
  auto innerType = nullop.getOut().getType().cast<ChannelType>().getInner();
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

  virtual LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Value valid, ready, data;
    WrapValidReadyOp wrap = dyn_cast<WrapValidReadyOp>(op);
    UnwrapValidReadyOp unwrap = dyn_cast<UnwrapValidReadyOp>(op);
    if (wrap) {
      if (!wrap.getChanOutput().hasOneUse() ||
          !(unwrap = dyn_cast<UnwrapValidReadyOp>(
                wrap.getChanOutput().use_begin()->getOwner())))
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
    } else {
      return failure();
    }

    if (!wrap.getChanOutput().hasOneUse())
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
struct ESItoHWPass : public LowerESItoHWBase<ESItoHWPass> {
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
struct CosimLowering : public OpConversionPattern<CosimEndpointOp> {
public:
  CosimLowering(ESIHWBuilder &b)
      : OpConversionPattern(b.getContext(), 1), builder(b) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CosimEndpointOp, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const final;

private:
  ESIHWBuilder &builder;
};
} // anonymous namespace

LogicalResult
CosimLowering::matchAndRewrite(CosimEndpointOp ep, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
#ifndef CAPNP
  (void)builder;
  return rewriter.notifyMatchFailure(
      ep, "Cosim lowering requires the ESI capnp plugin, which was disabled.");
#else
  auto loc = ep.getLoc();
  auto *ctxt = rewriter.getContext();
  auto operands = adaptor.getOperands();
  Value clk = operands[0];
  Value rst = operands[1];
  Value send = operands[2];

  circt::BackedgeBuilder bb(rewriter, loc);
  Type ui64Type =
      IntegerType::get(ctxt, 64, IntegerType::SignednessSemantics::Unsigned);
  capnp::TypeSchema sendTypeSchema(send.getType());
  if (!sendTypeSchema.isSupported())
    return rewriter.notifyMatchFailure(ep, "Send type not supported yet");
  capnp::TypeSchema recvTypeSchema(ep.getRecv().getType());
  if (!recvTypeSchema.isSupported())
    return rewriter.notifyMatchFailure(ep, "Recv type not supported yet");

  // Set all the parameters.
  SmallVector<Attribute, 8> params;
  if (auto ext = ep->getAttrOfType<StringAttr>("name_ext"))
    params.push_back(ParamDeclAttr::get("ENDPOINT_ID_EXT", ext));
  else
    params.push_back(
        ParamDeclAttr::get("ENDPOINT_ID_EXT", StringAttr::get(ctxt, "")));
  params.push_back(ParamDeclAttr::get(
      "SEND_TYPE_ID",
      IntegerAttr::get(ui64Type, sendTypeSchema.capnpTypeID())));
  params.push_back(
      ParamDeclAttr::get("SEND_TYPE_SIZE_BITS",
                         rewriter.getI32IntegerAttr(sendTypeSchema.size())));
  params.push_back(ParamDeclAttr::get(
      "RECV_TYPE_ID",
      IntegerAttr::get(ui64Type, recvTypeSchema.capnpTypeID())));
  params.push_back(
      ParamDeclAttr::get("RECV_TYPE_SIZE_BITS",
                         rewriter.getI32IntegerAttr(recvTypeSchema.size())));

  // Set up the egest route to drive the EP's send ports.
  ArrayType egestBitArrayType =
      ArrayType::get(rewriter.getI1Type(), sendTypeSchema.size());
  auto sendReady = bb.get(rewriter.getI1Type());
  UnwrapValidReadyOp unwrapSend =
      rewriter.create<UnwrapValidReadyOp>(loc, send, sendReady);
  auto encodeData = rewriter.create<CapnpEncodeOp>(loc, egestBitArrayType, clk,
                                                   unwrapSend.getValid(),
                                                   unwrapSend.getRawOutput());

  // Get information necessary for injest path.
  auto recvReady = bb.get(rewriter.getI1Type());
  ArrayType ingestBitArrayType =
      ArrayType::get(rewriter.getI1Type(), recvTypeSchema.size());

  // Build or get the cached Cosim Endpoint module parameterization.
  Operation *symTable = ep->getParentWithTrait<OpTrait::SymbolTable>();
  HWModuleExternOp endpoint = builder.declareCosimEndpointOp(
      symTable, egestBitArrayType, ingestBitArrayType);

  // Create replacement Cosim_Endpoint instance.
  StringAttr nameAttr = ep->getAttr("name").dyn_cast_or_null<StringAttr>();
  StringRef name = nameAttr ? nameAttr.getValue() : "CosimEndpointOp";
  Value epInstInputs[] = {
      clk, rst, recvReady, unwrapSend.getValid(), encodeData.getCapnpBits(),
  };

  auto cosimEpModule =
      rewriter.create<InstanceOp>(loc, endpoint, name, epInstInputs,
                                  ArrayAttr::get(ctxt, params), StringAttr());
  sendReady.setValue(cosimEpModule.getResult(2));

  // Set up the injest path.
  Value recvDataFromCosim = cosimEpModule.getResult(1);
  Value recvValidFromCosim = cosimEpModule.getResult(0);
  auto decodeData =
      rewriter.create<CapnpDecodeOp>(loc, recvTypeSchema.getType(), clk,
                                     recvValidFromCosim, recvDataFromCosim);
  WrapValidReadyOp wrapRecv = rewriter.create<WrapValidReadyOp>(
      loc, decodeData.getDecodedData(), recvValidFromCosim);
  recvReady.setValue(wrapRecv.getReady());

  // Replace the CosimEndpointOp op.
  rewriter.replaceOp(ep, wrapRecv.getChanOutput());

  return success();
#endif // CAPNP
}

namespace {
/// Lower the encode gasket to SV/HW.
struct EncoderLowering : public OpConversionPattern<CapnpEncodeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CapnpEncodeOp enc, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
#ifndef CAPNP
    return rewriter.notifyMatchFailure(enc,
                                       "encode.capnp lowering requires the ESI "
                                       "capnp plugin, which was disabled.");
#else
    capnp::TypeSchema encodeType(enc.getDataToEncode().getType());
    if (!encodeType.isSupported())
      return rewriter.notifyMatchFailure(enc, "Type not supported yet");
    auto operands = adaptor.getOperands();
    Value encoderOutput = encodeType.buildEncoder(rewriter, operands[0],
                                                  operands[1], operands[2]);
    assert(encoderOutput && "Error in TypeSchema.buildEncoder()");
    rewriter.replaceOp(enc, encoderOutput);
    return success();
#endif
  }
};
} // anonymous namespace

namespace {
/// Lower the decode gasket to SV/HW.
struct DecoderLowering : public OpConversionPattern<CapnpDecodeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CapnpDecodeOp dec, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
#ifndef CAPNP
    return rewriter.notifyMatchFailure(dec,
                                       "decode.capnp lowering requires the ESI "
                                       "capnp plugin, which was disabled.");
#else
    capnp::TypeSchema decodeType(dec.getDecodedData().getType());
    if (!decodeType.isSupported())
      return rewriter.notifyMatchFailure(dec, "Type not supported yet");
    auto operands = adaptor.getOperands();
    Value decoderOutput = decodeType.buildDecoder(rewriter, operands[0],
                                                  operands[1], operands[2]);
    assert(decoderOutput && "Error in TypeSchema.buildDecoder()");
    rewriter.replaceOp(dec, decoderOutput);
    return success();
#endif
  }
};
} // namespace

void ESItoHWPass::runOnOperation() {
  auto top = getOperation();
  auto ctxt = &getContext();

  // Set up a conversion and give it a set of laws.
  ConversionTarget pass1Target(*ctxt);
  pass1Target.addLegalDialect<CombDialect>();
  pass1Target.addLegalDialect<HWDialect>();
  pass1Target.addLegalDialect<SVDialect>();
  pass1Target.addLegalOp<WrapValidReadyOp, UnwrapValidReadyOp>();
  pass1Target.addLegalOp<CapnpDecodeOp, CapnpEncodeOp>();

  pass1Target.addIllegalOp<WrapSVInterfaceOp, UnwrapSVInterfaceOp>();
  pass1Target.addIllegalOp<PipelineStageOp>();

  // Add all the conversion patterns.
  ESIHWBuilder esiBuilder(top);
  RewritePatternSet pass1Patterns(ctxt);
  pass1Patterns.insert<PipelineStageLowering>(esiBuilder, ctxt);
  pass1Patterns.insert<WrapInterfaceLower>(ctxt);
  pass1Patterns.insert<UnwrapInterfaceLower>(ctxt);
  pass1Patterns.insert<CosimLowering>(esiBuilder);
  pass1Patterns.insert<NullSourceOpLowering>(ctxt);

  // Run the conversion.
  if (failed(
          applyPartialConversion(top, pass1Target, std::move(pass1Patterns))))
    signalPassFailure();

  ConversionTarget pass2Target(*ctxt);
  pass2Target.addLegalDialect<CombDialect>();
  pass2Target.addLegalDialect<HWDialect>();
  pass2Target.addLegalDialect<SVDialect>();
  pass2Target.addIllegalDialect<ESIDialect>();

  RewritePatternSet pass2Patterns(ctxt);
  pass2Patterns.insert<RemoveWrapUnwrap>(ctxt);
  pass2Patterns.insert<EncoderLowering>(ctxt);
  pass2Patterns.insert<DecoderLowering>(ctxt);
  if (failed(
          applyPartialConversion(top, pass2Target, std::move(pass2Patterns))))
    signalPassFailure();
}

//===----------------------------------------------------------------------===//
// Emit ESI collateral pass. Collateral includes the capnp schema and a JSON
// descriptor of the service hierarchy.
//===----------------------------------------------------------------------===//

static llvm::json::Value toJSON(Type type) {
  // TODO: This is far from complete. Build out as necessary.
  using llvm::json::Array;
  using llvm::json::Object;
  using llvm::json::Value;

  StringRef dialect = type.getDialect().getNamespace();
  std::string m;
  Object o = TypeSwitch<Type, Object>(type)
                 .Case([&](ChannelType t) {
                   m = "channel";
                   return Object({{"inner", toJSON(t.getInner())}});
                 })
                 .Case([&](AnyType t) {
                   m = "any";
                   return Object();
                 })
                 .Case([&](StructType t) {
                   m = "struct";
                   Array fields;
                   for (auto field : t.getElements())
                     fields.push_back(Object({{"name", field.name.getValue()},
                                              {"type", toJSON(field.type)}}));
                   return Object({{"fields", Value(std::move(fields))}});
                 })
                 .Default([&](Type t) {
                   llvm::raw_string_ostream(m) << t;
                   return Object();
                 });
  o["dialect"] = dialect;
  if (m.length())
    o["mnemonic"] = m;
  return o;
}

// Serialize an attribute to a JSON value.
static llvm::json::Value toJSON(Attribute attr) {
  // TODO: This is far from complete. Build out as necessary.
  using llvm::json::Value;
  return TypeSwitch<Attribute, Value>(attr)
      .Case([&](StringAttr a) { return a.getValue(); })
      .Case([&](IntegerAttr a) { return a.getValue().getLimitedValue(); })
      .Case([&](TypeAttr a) {
        Type t = a.getValue();
        llvm::json::Object typeMD;
        typeMD["type_desc"] = toJSON(t);

        std::string buf;
        llvm::raw_string_ostream(buf) << t;
        typeMD["mlir_name"] = buf;

        if (auto chanType = t.dyn_cast<ChannelType>()) {
          Type inner = chanType.getInner();
          typeMD["hw_bitwidth"] = hw::getBitWidth(inner);
#ifdef CAPNP
          capnp::TypeSchema schema(inner);
          typeMD["capnp_type_id"] = schema.capnpTypeID();
          typeMD["capnp_name"] = schema.name().str();
#endif
        } else {
          typeMD["hw_bitwidth"] = hw::getBitWidth(t);
        }
        return typeMD;
      })
      .Case([&](ArrayAttr a) {
        return llvm::json::Array(
            llvm::map_range(a, [](Attribute a) { return toJSON(a); }));
      })
      .Case([&](DictionaryAttr a) {
        llvm::json::Object dict;
        for (auto &entry : a.getValue())
          dict[entry.getName().getValue()] = toJSON(entry.getValue());
        return dict;
      })
      .Case([&](InnerRefAttr ref) {
        llvm::json::Object dict;
        dict["outer_sym"] = ref.getModule().getValue();
        dict["inner"] = ref.getName().getValue();
        return dict;
      })
      .Default([&](Attribute a) {
        std::string buff;
        llvm::raw_string_ostream(buff) << a;
        return buff;
      });
}

namespace {
/// Run all the physical lowerings.
struct ESIEmitCollateralPass
    : public ESIEmitCollateralBase<ESIEmitCollateralPass> {
  void runOnOperation() override;

  /// Emit service hierarchy info in JSON format.
  void emitServiceJSON();
};
} // anonymous namespace

void ESIEmitCollateralPass::emitServiceJSON() {
  ModuleOp mod = getOperation();
  auto *ctxt = &getContext();
  SymbolCache topSyms;
  topSyms.addDefinitions(mod);

  std::string jsonStrBuffer;
  llvm::raw_string_ostream os(jsonStrBuffer);
  llvm::json::OStream j(os, 2);

  // Emit the list of ports of a service declaration.
  auto emitPorts = [&](ServiceDeclOpInterface decl) {
    SmallVector<ServicePortInfo> ports;
    decl.getPortList(ports);
    for (ServicePortInfo port : ports) {
      j.object([&] {
        j.attribute("name", port.name.getValue());
        if (port.toClientType)
          j.attribute("to-client-type", toJSON(port.toClientType));
        if (port.toServerType)
          j.attribute("to-server-type", toJSON(port.toServerType));
      });
    }
  };

  j.object([&] {
    // Emit a list of the service declarations in a design.
    j.attributeArray("declarations", [&] {
      for (auto *op : llvm::make_pointer_range(mod.getOps())) {
        if (auto decl = dyn_cast<ServiceDeclOpInterface>(op)) {
          j.object([&] {
            j.attribute("name", SymbolTable::getSymbolName(op).getValue());
            j.attributeArray("ports", [&] { emitPorts(decl); });
          });
        }
      }
    });

    j.attributeArray("top_levels", [&] {
      for (auto topModName : tops) {
        j.object([&] {
          auto sym = FlatSymbolRefAttr::get(ctxt, topModName);
          Operation *hwMod = topSyms.getDefinition(sym);
          j.attribute("module", toJSON(sym));
          j.attributeArray("services", [&] {
            hwMod->walk([&](ServiceHierarchyMetadataOp md) {
              j.object([&] {
                j.attribute("service", md.getServiceSymbol());
                j.attribute("instance_path",
                            toJSON(md.getServerNamePathAttr()));
              });
            });
          });
        });
      }
    });

    // Get a list of metadata ops which originated in modules (path is empty).
    DenseMap<hw::HWModuleLike, SmallVector<ServiceHierarchyMetadataOp, 0>>
        modsWithLocalServices;
    for (auto hwmod : mod.getOps<hw::HWModuleLike>()) {
      SmallVector<ServiceHierarchyMetadataOp, 0> metadataOps;
      hwmod.walk([&metadataOps](ServiceHierarchyMetadataOp md) {
        if (md.getServerNamePath().empty())
          metadataOps.push_back(md);
      });
      if (!metadataOps.empty())
        modsWithLocalServices[hwmod] = metadataOps;
    }

    // Then output metadata for those modules exclusively.
    j.attributeArray("modules", [&] {
      for (auto &modWithSvc : modsWithLocalServices) {
        j.object([&] {
          j.attribute("symbol", modWithSvc.first.moduleName());
          j.attributeArray("services", [&] {
            for (ServiceHierarchyMetadataOp metadata : modWithSvc.getSecond()) {
              j.object([&] {
                j.attribute("service", metadata.getServiceSymbol());
                j.attribute("impl_type", metadata.getImplType());
                if (metadata.getImplDetailsAttr())
                  j.attribute("impl_details",
                              toJSON(metadata.getImplDetailsAttr()));
                j.attributeArray("clients", [&] {
                  for (auto client : metadata.getClients())
                    j.value(toJSON(client));
                });
              });
            }
          });
        });
      }
    });
  });

  j.flush();
  OpBuilder b = OpBuilder::atBlockEnd(mod.getBody());
  auto verbatim = b.create<sv::VerbatimOp>(b.getUnknownLoc(),
                                           StringAttr::get(ctxt, os.str()));
  auto outputFileAttr = OutputFileAttr::getFromFilename(ctxt, "services.json");
  verbatim->setAttr("output_file", outputFileAttr);

  // By now, we should be done with all of the service declarations and metadata
  // ops so we should delete them.
  mod.walk([&](ServiceHierarchyMetadataOp op) { op.erase(); });
  // Track declarations which are still used so that the service impl reqs are
  // still valid.
  DenseSet<StringAttr> stillUsed;
  mod.walk([&](ServiceImplementReqOp req) {
    auto sym = req.getServiceSymbol();
    if (sym.has_value())
      stillUsed.insert(StringAttr::get(req.getContext(), *sym));
  });
  mod.walk([&](ServiceDeclOpInterface decl) {
    if (!stillUsed.contains(SymbolTable::getSymbolName(decl)))
      decl.getOperation()->erase();
  });
}

void ESIEmitCollateralPass::runOnOperation() {
  ModuleOp mod = getOperation();
  auto *ctxt = &getContext();

  emitServiceJSON();

  // Check for cosim endpoints in the design. If the design doesn't have any
  // we don't need a schema.
  WalkResult cosimWalk =
      mod.walk([](CosimEndpointOp _) { return WalkResult::interrupt(); });
  if (!cosimWalk.wasInterrupted())
    return;

  // Generate the schema
  std::string schemaStrBuffer;
  llvm::raw_string_ostream os(schemaStrBuffer);
  if (failed(exportCosimSchema(mod, os))) {
    signalPassFailure();
    return;
  }

  // And stuff if in a verbatim op with a filename, optionally.
  OpBuilder b = OpBuilder::atBlockEnd(mod.getBody());
  auto verbatim = b.create<sv::VerbatimOp>(b.getUnknownLoc(),
                                           StringAttr::get(ctxt, os.str()));
  if (!schemaFile.empty()) {
    auto outputFileAttr = OutputFileAttr::getFromFilename(ctxt, schemaFile);
    verbatim->setAttr("output_file", outputFileAttr);
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
circt::esi::createESIEmitCollateralPass() {
  return std::make_unique<ESIEmitCollateralPass>();
}
std::unique_ptr<OperationPass<ModuleOp>>
circt::esi::createESIPhysicalLoweringPass() {
  return std::make_unique<ESIToPhysicalPass>();
}
std::unique_ptr<OperationPass<ModuleOp>>
circt::esi::createESIPortLoweringPass() {
  return std::make_unique<ESIPortsPass>();
}
std::unique_ptr<OperationPass<ModuleOp>> circt::esi::createESItoHWPass() {
  return std::make_unique<ESItoHWPass>();
}

void circt::esi::registerESIPasses() { registerPasses(); }
