//===- ESIOps.cpp - ESI op code defs ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is where op definitions live.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVTypes.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

using namespace circt;
using namespace circt::esi;

//===----------------------------------------------------------------------===//
// ChannelBufferOp functions.
//===----------------------------------------------------------------------===//

ParseResult ChannelBufferOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();

  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/3,
                              /*delimiter=*/OpAsmParser::Delimiter::None))
    return failure();

  Type innerOutputType;
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(innerOutputType))
    return failure();
  auto outputType =
      ChannelType::get(parser.getBuilder().getContext(), innerOutputType);
  result.addTypes({outputType});

  auto clkTy = seq::ClockType::get(result.getContext());
  auto i1 = IntegerType::get(result.getContext(), 1);
  if (parser.resolveOperands(operands, {clkTy, i1, outputType},
                             inputOperandsLoc, result.operands))
    return failure();
  return success();
}

void ChannelBufferOp::print(OpAsmPrinter &p) {
  p << " " << getClk() << ", " << getRst() << ", " << getInput();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << innerType();
}

circt::esi::ChannelType ChannelBufferOp::channelType() {
  return cast<circt::esi::ChannelType>(getInput().getType());
}

LogicalResult ChannelBufferOp::verify() {
  if (getInput().getType().getSignaling() != ChannelSignaling::ValidReady)
    return emitOpError("currently only supports valid-ready signaling");
  if (getOutput().getType().getDataDelay() != 0)
    return emitOpError("currently only supports channels with zero data delay");
  return success();
}

//===----------------------------------------------------------------------===//
// PipelineStageOp functions.
//===----------------------------------------------------------------------===//

ParseResult PipelineStageOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();

  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  Type innerOutputType;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/3) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(innerOutputType))
    return failure();
  auto type =
      ChannelType::get(parser.getBuilder().getContext(), innerOutputType);
  result.addTypes({type});

  auto clkTy = seq::ClockType::get(result.getContext());
  auto i1 = IntegerType::get(result.getContext(), 1);
  if (parser.resolveOperands(operands, {clkTy, i1, type}, inputOperandsLoc,
                             result.operands))
    return failure();
  return success();
}

void PipelineStageOp::print(OpAsmPrinter &p) {
  p << " " << getClk() << ", " << getRst() << ", " << getInput();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << innerType();
}

circt::esi::ChannelType PipelineStageOp::channelType() {
  return cast<circt::esi::ChannelType>(getInput().getType());
}

//===----------------------------------------------------------------------===//
// Wrap / unwrap.
//===----------------------------------------------------------------------===//

ParseResult WrapValidReadyOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();

  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 2> opList;
  Type innerOutputType;
  if (parser.parseOperandList(opList, 2, OpAsmParser::Delimiter::None) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(innerOutputType))
    return failure();

  auto boolType = parser.getBuilder().getI1Type();
  auto outputType =
      ChannelType::get(parser.getBuilder().getContext(), innerOutputType);
  result.addTypes({outputType, boolType});
  if (parser.resolveOperands(opList, {innerOutputType, boolType},
                             inputOperandsLoc, result.operands))
    return failure();
  return success();
}

void WrapValidReadyOp::print(OpAsmPrinter &p) {
  p << " " << getRawInput() << ", " << getValid();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << innerType();
}

void WrapValidReadyOp::build(OpBuilder &b, OperationState &state, Value data,
                             Value valid) {
  build(b, state, ChannelType::get(state.getContext(), data.getType()),
        b.getI1Type(), data, valid);
}

LogicalResult WrapValidReadyOp::verify() {
  if (getChanOutput().getType().getSignaling() != ChannelSignaling::ValidReady)
    return emitOpError("only supports valid-ready signaling");
  return success();
}

ParseResult UnwrapValidReadyOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();

  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 2> opList;
  Type outputType;
  if (parser.parseOperandList(opList, 2, OpAsmParser::Delimiter::None) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(outputType))
    return failure();

  auto inputType =
      ChannelType::get(parser.getBuilder().getContext(), outputType);

  auto boolType = parser.getBuilder().getI1Type();

  result.addTypes({inputType.getInner(), boolType});
  if (parser.resolveOperands(opList, {inputType, boolType}, inputOperandsLoc,
                             result.operands))
    return failure();
  return success();
}

void UnwrapValidReadyOp::print(OpAsmPrinter &p) {
  p << " " << getChanInput() << ", " << getReady();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << getRawOutput().getType();
}

LogicalResult UnwrapValidReadyOp::verify() {
  if (getChanInput().getType().getSignaling() != ChannelSignaling::ValidReady)
    return emitOpError("only supports valid-ready signaling");
  return success();
}

circt::esi::ChannelType WrapValidReadyOp::channelType() {
  return cast<circt::esi::ChannelType>(getChanOutput().getType());
}

void UnwrapValidReadyOp::build(OpBuilder &b, OperationState &state,
                               Value inChan, Value ready) {
  auto inChanType = cast<ChannelType>(inChan.getType());
  build(b, state, inChanType.getInner(), b.getI1Type(), inChan, ready);
}

circt::esi::ChannelType UnwrapValidReadyOp::channelType() {
  return cast<circt::esi::ChannelType>(getChanInput().getType());
}

circt::esi::ChannelType WrapFIFOOp::channelType() {
  return cast<circt::esi::ChannelType>(getChanOutput().getType());
}

ParseResult parseWrapFIFOType(OpAsmParser &p, Type &dataType,
                              Type &chanInputType) {
  auto loc = p.getCurrentLocation();
  ChannelType chType;
  if (p.parseType(chType))
    return failure();
  if (chType.getSignaling() != ChannelSignaling::FIFO)
    return p.emitError(loc, "can only wrap into FIFO type");
  dataType = chType.getInner();
  chanInputType = chType;
  return success();
}

void printWrapFIFOType(OpAsmPrinter &p, WrapFIFOOp wrap, Type dataType,
                       Type chanType) {
  p << chanType;
}

LogicalResult WrapFIFOOp::verify() {
  if (getChanOutput().getType().getSignaling() != ChannelSignaling::FIFO)
    return emitOpError("only supports FIFO signaling");
  return success();
}

circt::esi::ChannelType UnwrapFIFOOp::channelType() {
  return cast<circt::esi::ChannelType>(getChanInput().getType());
}

LogicalResult UnwrapFIFOOp::verify() {
  if (getChanInput().getType().getSignaling() != ChannelSignaling::FIFO)
    return emitOpError("only supports FIFO signaling");
  return success();
}

LogicalResult
UnwrapFIFOOp::inferReturnTypes(MLIRContext *context, std::optional<Location>,
                               ValueRange operands, DictionaryAttr,
                               mlir::OpaqueProperties, mlir::RegionRange,
                               SmallVectorImpl<Type> &inferredResulTypes) {
  inferredResulTypes.push_back(
      cast<ChannelType>(operands[0].getType()).getInner());
  inferredResulTypes.push_back(
      IntegerType::get(context, 1, IntegerType::Signless));
  return success();
}

/// If 'iface' looks like an ESI interface, return the inner data type.
static Type getEsiDataType(circt::sv::InterfaceOp iface) {
  using namespace circt::sv;
  if (!iface.lookupSymbol<InterfaceSignalOp>("valid"))
    return Type();
  if (!iface.lookupSymbol<InterfaceSignalOp>("ready"))
    return Type();
  auto dataSig = iface.lookupSymbol<InterfaceSignalOp>("data");
  if (!dataSig)
    return Type();
  return dataSig.getType();
}

/// Verify that the modport type of 'modportArg' points to an interface which
/// looks like an ESI interface and the inner data from said interface matches
/// the chan type's inner data type.
static LogicalResult verifySVInterface(Operation *op,
                                       circt::sv::ModportType modportType,
                                       ChannelType chanType) {
  auto modport =
      SymbolTable::lookupNearestSymbolFrom<circt::sv::InterfaceModportOp>(
          op, modportType.getModport());
  if (!modport)
    return op->emitError("Could not find modport ")
           << modportType.getModport() << " in symbol table.";
  auto iface = cast<circt::sv::InterfaceOp>(modport->getParentOp());
  Type esiDataType = getEsiDataType(iface);
  if (!esiDataType)
    return op->emitOpError("Interface is not a valid ESI interface.");
  if (esiDataType != chanType.getInner())
    return op->emitOpError("Operation specifies ")
           << chanType << " but type inside doesn't match interface data type "
           << esiDataType << ".";
  return success();
}

LogicalResult WrapSVInterfaceOp::verify() {
  auto modportType = cast<circt::sv::ModportType>(getInterfaceSink().getType());
  auto chanType = cast<ChannelType>(getOutput().getType());
  return verifySVInterface(*this, modportType, chanType);
}

circt::esi::ChannelType WrapSVInterfaceOp::channelType() {
  return cast<circt::esi::ChannelType>(getOutput().getType());
}

LogicalResult UnwrapSVInterfaceOp::verify() {
  auto modportType =
      cast<circt::sv::ModportType>(getInterfaceSource().getType());
  auto chanType = cast<ChannelType>(getChanInput().getType());
  return verifySVInterface(*this, modportType, chanType);
}

circt::esi::ChannelType UnwrapSVInterfaceOp::channelType() {
  return cast<circt::esi::ChannelType>(getChanInput().getType());
}

LogicalResult WrapWindow::verify() {
  hw::UnionType expectedInput = getWindow().getType().getLoweredType();
  if (expectedInput == getFrame().getType())
    return success();
  return emitOpError("Expected input type is ") << expectedInput;
}

LogicalResult
UnwrapWindow::inferReturnTypes(MLIRContext *, std::optional<Location>,
                               ValueRange operands, DictionaryAttr,
                               mlir::OpaqueProperties, mlir::RegionRange,
                               SmallVectorImpl<Type> &inferredReturnTypes) {
  auto windowType = cast<WindowType>(operands.front().getType());
  inferredReturnTypes.push_back(windowType.getLoweredType());
  return success();
}

/// Determine the input type ('frame') from the return type ('window').
static bool parseInferWindowRet(OpAsmParser &p, Type &frame, Type &windowOut) {
  WindowType window;
  if (p.parseType(window))
    return true;
  windowOut = window;
  frame = window.getLoweredType();
  return false;
}

static void printInferWindowRet(OpAsmPrinter &p, Operation *, Type,
                                Type window) {
  p << window;
}

//===----------------------------------------------------------------------===//
// Services ops.
//===----------------------------------------------------------------------===//

/// Get the port declaration op for the specified service decl, port name.
static FailureOr<ServiceDeclOpInterface>
getServiceDecl(Operation *op, SymbolTableCollection &symbolTable,
               hw::InnerRefAttr servicePort) {
  ModuleOp top = op->getParentOfType<mlir::ModuleOp>();
  SymbolTable &topSyms = symbolTable.getSymbolTable(top);

  StringAttr modName = servicePort.getModule();
  auto serviceDecl = topSyms.lookup<ServiceDeclOpInterface>(modName);
  if (!serviceDecl)
    return op->emitOpError("Could not find service declaration ")
           << servicePort.getModuleRef();
  return serviceDecl;
}

/// Get the port info for the specified service decl and port name.
static FailureOr<ServicePortInfo>
getServicePortInfo(Operation *op, SymbolTableCollection &symbolTable,
                   hw::InnerRefAttr servicePort) {
  auto serviceDecl = getServiceDecl(op, symbolTable, servicePort);
  if (failed(serviceDecl))
    return failure();
  auto portInfo = serviceDecl->getPortInfo(servicePort.getName());
  if (failed(portInfo))
    return op->emitOpError("Could not locate port ") << servicePort.getName();
  return portInfo;
}

namespace circt {
namespace esi {
// Check that the channels on two bundles match allowing for AnyType.
// NOLINTNEXTLINE(misc-no-recursion)
LogicalResult checkInnerTypeMatch(Type expected, Type actual) {
  if (expected == actual)
    return success();

  // Check all of the 'container' or special case types.
  return TypeSwitch<Type, LogicalResult>(expected)
      // For 'any' types, we can match anything.
      .Case<AnyType>([&](Type) { return success(); })
      // If they're both channels, check the inner types.
      .Case<ChannelType>([&](ChannelType expectedChannel) {
        auto actualChannel = dyn_cast<ChannelType>(actual);
        if (!actualChannel)
          return failure();
        return checkInnerTypeMatch(expectedChannel.getInner(),
                                   actualChannel.getInner());
      })
      // For structs, check each field.
      .Case<hw::StructType>([&](hw::StructType expectedStruct) {
        auto actualStruct = dyn_cast<hw::StructType>(actual);
        if (!actualStruct)
          return failure();
        auto expectedFields = expectedStruct.getElements();
        auto actualFields = actualStruct.getElements();
        if (expectedFields.size() != actualFields.size())
          return failure();
        for (auto [efield, afield] : llvm::zip(expectedFields, actualFields)) {
          if (efield.name != afield.name)
            return failure();
          if (failed(checkInnerTypeMatch(efield.type, afield.type)))
            return failure();
        }
        return success();
      })
      // For arrays, check the element type and size.
      .Case<hw::ArrayType>([&](hw::ArrayType expectedArray) {
        auto actualArray = dyn_cast<hw::ArrayType>(actual);
        if (!actualArray)
          return failure();
        if (expectedArray.getNumElements() != actualArray.getNumElements())
          return failure();
        return checkInnerTypeMatch(expectedArray.getElementType(),
                                   actualArray.getElementType());
      })
      // For unions, check the element types and names.
      .Case<hw::UnionType>([&](hw::UnionType expectedUnion) {
        auto actualUnion = dyn_cast<hw::UnionType>(actual);
        if (!actualUnion)
          return failure();
        auto expectedElements = expectedUnion.getElements();
        auto actualElements = actualUnion.getElements();
        if (expectedElements.size() != actualElements.size())
          return failure();
        for (auto [efield, afield] :
             llvm::zip(expectedElements, actualElements)) {
          if (efield.name != afield.name)
            return failure();
          if (efield.offset != afield.offset)
            return failure();
          if (failed(checkInnerTypeMatch(efield.type, afield.type)))
            return failure();
        }
        return success();
      })
      // For ESI lists, check the element type.
      .Case<ListType>([&](ListType expectedList) {
        auto actualList = dyn_cast<ListType>(actual);
        if (!actualList)
          return failure();
        return checkInnerTypeMatch(expectedList.getElementType(),
                                   actualList.getElementType());
      })
      // For ESI windows, unwrap and check the inner type.
      .Case<WindowType>([&](WindowType expectedWindow) {
        auto actualWindow = dyn_cast<WindowType>(actual);
        if (!actualWindow)
          return checkInnerTypeMatch(expectedWindow.getInto(), actual);
        return checkInnerTypeMatch(expectedWindow.getInto(),
                                   actualWindow.getInto());
      })
      // For type aliases, unwrap and check the aliased type.
      .Case<hw::TypeAliasType>([&](hw::TypeAliasType expectedAlias) {
        auto actualAlias = dyn_cast<hw::TypeAliasType>(actual);
        if (!actualAlias)
          return checkInnerTypeMatch(expectedAlias.getCanonicalType(), actual);
        return checkInnerTypeMatch(expectedAlias.getCanonicalType(),
                                   actualAlias.getCanonicalType());
      })
      // TODO: other container types.
      .Default([&](Type) { return failure(); });
}

/// Check that the channels on two bundles match allowing for AnyType in the
/// 'svc' bundle.
LogicalResult checkBundleTypeMatch(Operation *req,
                                   ChannelBundleType svcBundleType,
                                   ChannelBundleType reqBundleType,
                                   bool skipDirectionCheck) {
  if (svcBundleType.getChannels().size() != reqBundleType.getChannels().size())
    return req->emitOpError(
        "Request port bundle channel count does not match service "
        "port bundle channel count");

  // Build fast lookup.
  DenseMap<StringAttr, BundledChannel> declBundleChannels;
  for (BundledChannel bc : svcBundleType.getChannels())
    declBundleChannels[bc.name] = bc;

  // Check all the channels.
  for (BundledChannel bc : reqBundleType.getChannels()) {
    auto f = declBundleChannels.find(bc.name);
    if (f == declBundleChannels.end())
      return req->emitOpError(
          "Request channel name not found in service port bundle");
    if (!skipDirectionCheck && f->second.direction != bc.direction)
      return req->emitOpError(
          "Request channel direction does not match service "
          "port bundle channel direction");

    if (failed(checkInnerTypeMatch(f->second.type, bc.type)))
      return req->emitOpError(
                    "Request channel type does not match service port "
                    "bundle channel type")
                 .attachNote()
             << "Service port '" << bc.name.getValue()
             << "' type: " << f->second.type;
  }
  return success();
}

} // namespace esi
} // namespace circt

LogicalResult
RequestConnectionOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto svcPort = getServicePortInfo(*this, symbolTable, getServicePortAttr());
  if (failed(svcPort))
    return failure();
  return checkBundleTypeMatch(*this, svcPort->type, getToClient().getType(),
                              false);
}

LogicalResult ServiceImplementConnReqOp::verifySymbolUses(
    SymbolTableCollection &symbolTable) {
  auto svcPort = getServicePortInfo(*this, symbolTable, getServicePortAttr());
  if (failed(svcPort))
    return failure();
  return checkBundleTypeMatch(*this, svcPort->type, getToClient().getType(),
                              true);
}

void CustomServiceDeclOp::getPortList(SmallVectorImpl<ServicePortInfo> &ports) {
  for (auto toClient : getOps<ServiceDeclPortOp>())
    ports.push_back(ServicePortInfo{
        hw::InnerRefAttr::get(getSymNameAttr(), toClient.getInnerSymAttr()),
        toClient.getToClientType()});
}

//===----------------------------------------------------------------------===//
// Bundle ops.
//===----------------------------------------------------------------------===//

static ParseResult
parseUnPackBundleType(OpAsmParser &parser,
                      SmallVectorImpl<Type> &toChannelTypes,
                      SmallVectorImpl<Type> &fromChannelTypes, Type &type) {

  ChannelBundleType bundleType;
  if (parser.parseType(bundleType))
    return failure();
  type = bundleType;

  for (BundledChannel ch : bundleType.getChannels())
    if (ch.direction == ChannelDirection::to)
      toChannelTypes.push_back(ch.type);
    else if (ch.direction == ChannelDirection::from)
      fromChannelTypes.push_back(ch.type);
    else
      assert(false && "Channel direction invalid");
  return success();
}
template <typename T3, typename T4>
static void printUnPackBundleType(OpAsmPrinter &p, Operation *, T3, T4,
                                  Type bundleType) {
  p.printType(bundleType);
}
void UnpackBundleOp::build(::mlir::OpBuilder &odsBuilder,
                           ::mlir::OperationState &odsState, Value bundle,
                           mlir::ValueRange fromChannels) {
  for (BundledChannel ch :
       cast<ChannelBundleType>(bundle.getType()).getChannels())
    if (ch.direction == ChannelDirection::to)
      odsState.addTypes(ch.type);
  odsState.addOperands(bundle);
  odsState.addOperands(fromChannels);
}

LogicalResult PackBundleOp::verify() {
  if (!getBundle().hasOneUse())
    return emitOpError("bundles must have exactly one user");
  return success();
}
void PackBundleOp::build(::mlir::OpBuilder &odsBuilder,
                         ::mlir::OperationState &odsState,
                         ChannelBundleType bundleType,
                         mlir::ValueRange toChannels) {
  odsState.addTypes(bundleType);
  for (BundledChannel ch : cast<ChannelBundleType>(bundleType).getChannels())
    if (ch.direction == ChannelDirection::from)
      odsState.addTypes(ch.type);
  odsState.addOperands(toChannels);
}

LogicalResult UnpackBundleOp::verify() {
  if (!getBundle().hasOneUse())
    return emitOpError("bundles must have exactly one user");
  return success();
}

void PackBundleOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  if (getNumResults() == 0)
    return;
  setNameFn(getResult(0), "bundle");
  for (auto [idx, from] : llvm::enumerate(llvm::make_filter_range(
           getBundle().getType().getChannels(), [](BundledChannel ch) {
             return ch.direction == ChannelDirection::from;
           })))
    if (idx + 1 < getNumResults())
      setNameFn(getResult(idx + 1), from.name.getValue());
}

void UnpackBundleOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  for (auto [idx, to] : llvm::enumerate(llvm::make_filter_range(
           getBundle().getType().getChannels(), [](BundledChannel ch) {
             return ch.direction == ChannelDirection::to;
           })))
    if (idx < getNumResults())
      setNameFn(getResult(idx), to.name.getValue());
}
//===----------------------------------------------------------------------===//
// Structural ops.
//===----------------------------------------------------------------------===//

LogicalResult ESIPureModuleOp::verify() {
  ESIDialect *esiDialect = getContext()->getLoadedDialect<ESIDialect>();
  Block &body = getBody().front();
  auto channelOrOutput = [](Value v) {
    if (isa<ChannelType, ChannelBundleType>(v.getType()))
      return true;
    if (v.getUsers().empty())
      return false;
    return llvm::all_of(v.getUsers(), [](Operation *op) {
      return isa<ESIPureModuleOutputOp>(op);
    });
  };

  DenseMap<StringAttr, std::tuple<hw::ModulePort::Direction, Type, Operation *>>
      ports;
  for (Operation &op : body.getOperations()) {
    if (igraph::InstanceOpInterface inst =
            dyn_cast<igraph::InstanceOpInterface>(op)) {
      if (llvm::any_of(op.getOperands(), [](Value v) {
            return !(isa<ChannelType, ChannelBundleType>(v.getType()) ||
                     isa<ESIPureModuleInputOp>(v.getDefiningOp()));
          }))
        return inst.emitOpError(
            "instances in ESI pure modules can only contain channel ports or "
            "ports driven by 'input' ops");
      if (!llvm::all_of(op.getResults(), channelOrOutput))
        return inst.emitOpError(
            "instances in ESI pure modules can only contain channel ports or "
            "drive only 'outputs'");
    } else {
      // Pure modules can only contain instance ops and ESI ops.
      if (op.getDialect() != esiDialect)
        return op.emitOpError("operation not allowed in ESI pure modules");
    }

    // Check for port validity.
    if (auto port = dyn_cast<ESIPureModuleInputOp>(op)) {
      auto existing = ports.find(port.getNameAttr());
      Type portType = port.getResult().getType();
      if (existing != ports.end()) {
        auto [dir, type, op] = existing->getSecond();
        if (dir != hw::ModulePort::Direction::Input || type != portType)
          return (port.emitOpError("port '")
                  << port.getName() << "' previously declared as type " << type)
              .attachNote(op->getLoc());
      }
      ports[port.getNameAttr()] = std::make_tuple(
          hw::ModulePort::Direction::Input, portType, port.getOperation());
    } else if (auto port = dyn_cast<ESIPureModuleOutputOp>(op)) {
      auto existing = ports.find(port.getNameAttr());
      if (existing != ports.end())
        return (port.emitOpError("port '")
                << port.getName() << "' previously declared")
            .attachNote(std::get<2>(existing->getSecond())->getLoc());
      ports[port.getNameAttr()] =
          std::make_tuple(hw::ModulePort::Direction::Input,
                          port.getValue().getType(), port.getOperation());
    }
  }
  return success();
}

hw::ModuleType ESIPureModuleOp::getHWModuleType() {
  return hw::ModuleType::get(getContext(), {});
}

SmallVector<::circt::hw::PortInfo> ESIPureModuleOp::getPortList() { return {}; }
::circt::hw::PortInfo ESIPureModuleOp::getPort(size_t idx) {
  ::llvm::report_fatal_error("not supported");
}

size_t ESIPureModuleOp::getNumPorts() { return 0; }
size_t ESIPureModuleOp::getNumInputPorts() { return 0; }
size_t ESIPureModuleOp::getNumOutputPorts() { return 0; }
size_t ESIPureModuleOp::getPortIdForInputId(size_t) {
  assert(0 && "Out of bounds input port id");
  return ~0ULL;
}
size_t ESIPureModuleOp::getPortIdForOutputId(size_t) {
  assert(0 && "Out of bounds output port id");
  return ~0ULL;
}

SmallVector<Location> ESIPureModuleOp::getAllPortLocs() {
  SmallVector<Location> retval;
  return retval;
}

void ESIPureModuleOp::setAllPortLocsAttrs(ArrayRef<Attribute> locs) {
  emitError("No ports for port locations");
}

void ESIPureModuleOp::setAllPortNames(ArrayRef<Attribute> names) {
  emitError("No ports for port naming");
}

void ESIPureModuleOp::setAllPortAttrs(ArrayRef<Attribute> attrs) {
  emitError("No ports for port attributes");
}

void ESIPureModuleOp::removeAllPortAttrs() {
  emitError("No ports for port attributes)");
}

ArrayRef<Attribute> ESIPureModuleOp::getAllPortAttrs() { return {}; }

void ESIPureModuleOp::setHWModuleType(hw::ModuleType type) {
  emitError("No ports for port types");
}

//===----------------------------------------------------------------------===//
// Manifest ops.
//===----------------------------------------------------------------------===//

StringRef ServiceImplRecordOp::getManifestClass() { return "service"; }

void ServiceImplRecordOp::getDetails(SmallVectorImpl<NamedAttribute> &results) {
  auto *ctxt = getContext();
  // AppID, optionally the service name, implementation name and details.
  results.emplace_back(getAppIDAttrName(), getAppIDAttr());
  if (getService())
    results.emplace_back(getServiceAttrName(), getServiceAttr());
  results.emplace_back(getServiceImplNameAttrName(), getServiceImplNameAttr());
  // Don't add another level for the implementation details.
  for (auto implDetail : getImplDetailsAttr().getValue())
    results.push_back(implDetail);

  // All of the manifest data contained by this op.
  SmallVector<Attribute, 8> reqDetails;
  for (auto reqDetail : getReqDetails().front().getOps<IsManifestData>())
    reqDetails.push_back(reqDetail.getDetailsAsDict());
  results.emplace_back(StringAttr::get(ctxt, "clientDetails"),
                       ArrayAttr::get(ctxt, reqDetails));
}

bool parseServiceImplRecordReqDetails(OpAsmParser &parser,
                                      Region &reqDetailsRegion) {
  parser.parseOptionalRegion(reqDetailsRegion);
  if (reqDetailsRegion.empty())
    reqDetailsRegion.emplaceBlock();
  return false;
}

void printServiceImplRecordReqDetails(OpAsmPrinter &p, ServiceImplRecordOp,
                                      Region &reqDetailsRegion) {
  if (!reqDetailsRegion.empty() && !reqDetailsRegion.front().empty())
    p.printRegion(reqDetailsRegion, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/false);
}

StringRef ServiceImplClientRecordOp::getManifestClass() {
  return "serviceClient";
}
void ServiceImplClientRecordOp::getDetails(
    SmallVectorImpl<NamedAttribute> &results) {
  // Relative AppID path, service port, and implementation details. Don't add
  // the bundle type since it is meaningless to the host and just clutters the
  // output.
  results.emplace_back(getRelAppIDPathAttrName(), getRelAppIDPathAttr());
  auto servicePort = getServicePortAttr();
  results.emplace_back(
      getServicePortAttrName(),
      DictionaryAttr::get(
          getContext(),
          {
              NamedAttribute(StringAttr::get(getContext(), "serviceName"),
                             FlatSymbolRefAttr::get(servicePort.getModule())),
              NamedAttribute(StringAttr::get(getContext(), "port"),
                             servicePort.getName()),
          }));
  if (const auto &channelAssignments = getChannelAssignments())
    results.push_back(
        NamedAttribute(getChannelAssignmentsAttrName(), *channelAssignments));
  // Don't add another level for the implementation details.
  if (const auto &implDetails = getImplDetails())
    for (const auto &implDetail : *implDetails)
      results.push_back(implDetail);
}

StringRef ServiceRequestRecordOp::getManifestClass() { return "clientPort"; }

void ServiceRequestRecordOp::getDetails(
    SmallVectorImpl<NamedAttribute> &results) {
  auto *ctxt = getContext();
  results.emplace_back(StringAttr::get(ctxt, "appID"), getRequestorAttr());
  results.emplace_back(getTypeIDAttrName(), getTypeIDAttr());
  auto servicePort = getServicePortAttr();
  results.emplace_back(
      getServicePortAttrName(),
      DictionaryAttr::get(
          getContext(),
          {
              NamedAttribute(StringAttr::get(getContext(), "serviceName"),
                             FlatSymbolRefAttr::get(servicePort.getModule())),
              NamedAttribute(StringAttr::get(getContext(), "port"),
                             servicePort.getName()),
          }));
}

StringRef SymbolMetadataOp::getManifestClass() { return "symInfo"; }

StringRef SymbolConstantsOp::getManifestClass() { return "symConsts"; }
void SymbolConstantsOp::getDetails(SmallVectorImpl<NamedAttribute> &results) {
  for (auto &attr : getConstantsAttr())
    results.push_back(attr);
}

#define GET_OP_CLASSES
#include "circt/Dialect/ESI/ESI.cpp.inc"

#include "circt/Dialect/ESI/ESIInterfaces.cpp.inc"
