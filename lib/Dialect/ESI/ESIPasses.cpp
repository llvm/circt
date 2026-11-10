//===- ESIPasses.cpp - Common code for ESI passes ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace circt::esi;
using namespace circt::esi::detail;
using namespace circt::hw;

StringAttr circt::esi::detail::getTypeID(Type t) {
  std::string typeID;
  llvm::raw_string_ostream(typeID) << t;
  return StringAttr::get(t.getContext(), typeID);
}

uint64_t circt::esi::detail::getWidth(Type t) {
  if (auto ch = dyn_cast<ChannelType>(t))
    t = ch.getInner();
  if (auto win = dyn_cast<WindowType>(t))
    t = win.getLoweredType();
  return hw::getBitWidth(t);
}

//===----------------------------------------------------------------------===//
// ESI custom op builder.
//===----------------------------------------------------------------------===//

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
      width(StringAttr::get(getContext(), "WIDTH")),
      dataDelay(StringAttr::get(getContext(), "DATA_DELAY")) {

  auto regions = top->getRegions();
  if (regions.empty()) {
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

/// Return a parameter list for the stage module with the specified values.
ArrayAttr ESIHWBuilder::getStageParameterList(Attribute widthValue,
                                              Attribute dataDelayValue) {
  auto type = IntegerType::get(width.getContext(), 32, IntegerType::Unsigned);
  return ArrayAttr::get(
      width.getContext(),
      {ParamDeclAttr::get(width.getContext(), width, type, widthValue),
       ParamDeclAttr::get(width.getContext(), dataDelay, type,
                          dataDelayValue)});
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
      {{clk, getClockType(), ModulePort::Direction::Input}, argn++},
      {{rst, getI1Type(), ModulePort::Direction::Input}, argn++}};

  ports.push_back({{a, dataType, ModulePort::Direction::Input}, argn++});
  ports.push_back(
      {{aValid, getI1Type(), ModulePort::Direction::Input}, argn++});
  ports.push_back(
      {{aReady, getI1Type(), ModulePort::Direction::Output}, resn++});
  ports.push_back({{x, dataType, ModulePort::Direction::Output}, resn++});

  ports.push_back(
      {{xValid, getI1Type(), ModulePort::Direction::Output}, resn++});
  ports.push_back(
      {{xReady, getI1Type(), ModulePort::Direction::Input}, argn++});

  stageMod = HWModuleExternOp::create(
      *this, constructUniqueSymbol(symTable, "ESI_PipelineStage"), ports,
      "ESI_PipelineStage", getStageParameterList({}, {}));
  return stageMod;
}

/// Write an 'ExternModuleOp' to use a hand-coded SystemVerilog module. Said
/// module contains a bi-directional Cosimulation DPI interface with valid/ready
/// semantics.
HWModuleExternOp
ESIHWBuilder::declareCosimEndpointToHostModule(Operation *symTable) {
  if (declaredCosimEndpointToHostModule)
    return *declaredCosimEndpointToHostModule;

  SmallVector<Attribute, 8> params;
  params.push_back(
      ParamDeclAttr::get("ENDPOINT_ID", NoneType::get(getContext())));
  params.push_back(
      ParamDeclAttr::get("TO_HOST_TYPE_ID", NoneType::get(getContext())));
  params.push_back(ParamDeclAttr::get("TO_HOST_SIZE_BITS", getI32Type()));

  auto dataInType = hw::IntType::get(hw::ParamDeclRefAttr::get(
      StringAttr::get(getContext(), "TO_HOST_SIZE_BITS"),
      getIntegerType(32, false)));
  PortInfo ports[] = {
      {{clk, getClockType(), ModulePort::Direction::Input}, 0},
      {{rst, getI1Type(), ModulePort::Direction::Input}, 1},
      {{dataInValid, getI1Type(), ModulePort::Direction::Input}, 2},
      {{dataInReady, getI1Type(), ModulePort::Direction::Output}, 3},
      {{dataIn, dataInType, ModulePort::Direction::Input}, 4}};

  declaredCosimEndpointToHostModule = HWModuleExternOp::create(
      *this, constructUniqueSymbol(symTable, "Cosim_Endpoint_ToHost"), ports,
      "Cosim_Endpoint_ToHost", ArrayAttr::get(getContext(), params));
  return *declaredCosimEndpointToHostModule;
}

HWModuleExternOp
ESIHWBuilder::declareCosimEndpointFromHostModule(Operation *symTable) {
  if (declaredCosimEndpointFromHostModule)
    return *declaredCosimEndpointFromHostModule;

  SmallVector<Attribute, 8> params;
  params.push_back(
      ParamDeclAttr::get("ENDPOINT_ID", NoneType::get(getContext())));
  params.push_back(
      ParamDeclAttr::get("FROM_HOST_TYPE_ID", NoneType::get(getContext())));
  params.push_back(ParamDeclAttr::get("FROM_HOST_SIZE_BITS", getI32Type()));

  auto dataInType = hw::IntType::get(hw::ParamDeclRefAttr::get(
      StringAttr::get(getContext(), "FROM_HOST_SIZE_BITS"),
      getIntegerType(32, false)));
  PortInfo ports[] = {
      {{clk, getClockType(), ModulePort::Direction::Input}, 0},
      {{rst, getI1Type(), ModulePort::Direction::Input}, 1},
      {{dataOutValid, getI1Type(), ModulePort::Direction::Output}, 2},
      {{dataOutReady, getI1Type(), ModulePort::Direction::Input}, 3},
      {{dataOut, dataInType, ModulePort::Direction::Output}, 4}};

  declaredCosimEndpointFromHostModule = HWModuleExternOp::create(
      *this, constructUniqueSymbol(symTable, "Cosim_Endpoint_FromHost"), ports,
      "Cosim_Endpoint_FromHost", ArrayAttr::get(getContext(), params));
  return *declaredCosimEndpointFromHostModule;
}

Type ESIHWBuilder::getClockType() { return seq::ClockType::get(getContext()); }

void circt::esi::registerESIPasses() { registerPasses(); }
