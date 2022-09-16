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
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
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

  auto i1 = IntegerType::get(result.getContext(), 1);
  if (parser.resolveOperands(operands, {i1, i1, outputType}, inputOperandsLoc,
                             result.operands))
    return failure();
  return success();
}

void ChannelBufferOp::print(OpAsmPrinter &p) {
  p << " " << getClk() << ", " << getRst() << ", " << getInput() << " ";
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << innerType();
}

circt::esi::ChannelType ChannelBufferOp::channelType() {
  return getInput().getType().cast<circt::esi::ChannelType>();
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

  auto i1 = IntegerType::get(result.getContext(), 1);
  if (parser.resolveOperands(operands, {i1, i1, type}, inputOperandsLoc,
                             result.operands))
    return failure();
  return success();
}

void PipelineStageOp::print(OpAsmPrinter &p) {
  p << " " << getClk() << ", " << getRst() << ", " << getInput() << " ";
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << innerType();
}

circt::esi::ChannelType PipelineStageOp::channelType() {
  return getInput().getType().cast<circt::esi::ChannelType>();
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

circt::esi::ChannelType WrapValidReadyOp::channelType() {
  return getChanOutput().getType().cast<circt::esi::ChannelType>();
}

void UnwrapValidReadyOp::build(OpBuilder &b, OperationState &state,
                               Value inChan, Value ready) {
  auto inChanType = inChan.getType().cast<ChannelType>();
  build(b, state, inChanType.getInner(), b.getI1Type(), inChan, ready);
}

circt::esi::ChannelType UnwrapValidReadyOp::channelType() {
  return getChanInput().getType().cast<circt::esi::ChannelType>();
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
  auto modportType =
      getInterfaceSink().getType().cast<circt::sv::ModportType>();
  auto chanType = getOutput().getType().cast<ChannelType>();
  return verifySVInterface(*this, modportType, chanType);
}

circt::esi::ChannelType WrapSVInterfaceOp::channelType() {
  return getOutput().getType().cast<circt::esi::ChannelType>();
}

LogicalResult UnwrapSVInterfaceOp::verify() {
  auto modportType =
      getInterfaceSource().getType().cast<circt::sv::ModportType>();
  auto chanType = getChanInput().getType().cast<ChannelType>();
  return verifySVInterface(*this, modportType, chanType);
}

circt::esi::ChannelType UnwrapSVInterfaceOp::channelType() {
  return getChanInput().getType().cast<circt::esi::ChannelType>();
}

/// Get the port declaration op for the specified service decl, port name.
template <class OpType>
static OpType getServicePortDecl(Operation *op,
                                 SymbolTableCollection &symbolTable,
                                 hw::InnerRefAttr servicePort) {
  ModuleOp top = op->getParentOfType<mlir::ModuleOp>();
  SymbolTable topSyms = symbolTable.getSymbolTable(top);

  StringAttr modName = servicePort.getModule();
  ServiceDeclOp serviceDeclOp = topSyms.lookup<ServiceDeclOp>(modName);
  if (!serviceDeclOp) {
    return {};
  }

  StringAttr innerSym = servicePort.getName();
  for (auto portDecl : serviceDeclOp.getOps<OpType>())
    if (portDecl.getInnerSymAttr() == innerSym)
      return portDecl;
  return {};
}

/// Check that the type of a given service request matches the services port
/// type.
template <class PortTypeOp, class OpType>
static LogicalResult
reqPortMatches(OpType op, SymbolTableCollection &symbolTable, Type t) {
  hw::InnerRefAttr port = op.getServicePort();
  auto portDecl = getServicePortDecl<PortTypeOp>(op, symbolTable, port);
  auto inoutPortDecl =
      getServicePortDecl<ServiceDeclInOutOp>(op, symbolTable, port);
  if (!portDecl && !inoutPortDecl)
    return op.emitOpError("Could not find service port declaration ")
           << port.getModuleRef() << "::" << port.getName().getValue();

  auto *ctxt = op.getContext();
  auto anyChannelType = ChannelType::get(ctxt, AnyType::get(ctxt));
  if (portDecl) {
    if (portDecl.getType() != t && portDecl.getType() != anyChannelType)
      return op.emitOpError("Request type does not match port type ")
             << portDecl.getType();
    return success();
  }

  assert(inoutPortDecl);
  if (isa<ToClientOp>(op)) {
    if (inoutPortDecl.getInType() != t &&
        inoutPortDecl.getInType() != anyChannelType)
      return op.emitOpError("Request type does not match port type ")
             << inoutPortDecl.getInType();
  } else if (isa<ToServerOp>(op)) {
    if (inoutPortDecl.getOutType() != t &&
        inoutPortDecl.getOutType() != anyChannelType)
      return op.emitOpError("Request type does not match port type ")
             << inoutPortDecl.getOutType();
  }
  return success();
}

LogicalResult RequestToClientConnectionOp::verifySymbolUses(
    SymbolTableCollection &symbolTable) {
  return reqPortMatches<ToClientOp>(*this, symbolTable,
                                    getReceiving().getType());
}

LogicalResult RequestToServerConnectionOp::verifySymbolUses(
    SymbolTableCollection &symbolTable) {
  return reqPortMatches<ToServerOp>(*this, symbolTable, getSending().getType());
}

LogicalResult
RequestInOutChannelOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto portDecl = getServicePortDecl<ServiceDeclInOutOp>(
      this->getOperation(), symbolTable, getServicePort());
  if (!portDecl)
    return emitOpError("Could not find inout service port declaration.");

  auto *ctxt = getContext();
  auto anyChannelType = ChannelType::get(ctxt, AnyType::get(ctxt));

  // Check the input port type.
  if (portDecl.getInType() != getSending().getType() &&
      portDecl.getInType() != anyChannelType)
    return emitOpError("Request to_server type does not match port type ")
           << portDecl.getInType();

  // Check the output port type.
  if (portDecl.getOutType() != getReceiving().getType() &&
      portDecl.getOutType() != anyChannelType)
    return emitOpError("Request to_client type does not match port type ")
           << portDecl.getOutType();

  return success();
}

LogicalResult ServiceHierarchyMetadataOp::verifySymbolUses(
    SymbolTableCollection &symbolTable) {
  ModuleOp top = getOperation()->getParentOfType<mlir::ModuleOp>();
  SymbolTable topSyms = symbolTable.getSymbolTable(top);
  ServiceDeclOp serviceDeclOp =
      topSyms.lookup<ServiceDeclOp>(getServiceSymbol());
  if (!serviceDeclOp)
    return emitOpError("Could not find service declaration ")
           << getServiceSymbol();
  return success();
}

void ServiceImplementReqOp::gatherPairedReqs(
    llvm::SmallVectorImpl<std::pair<RequestToServerConnectionOp,
                                    RequestToClientConnectionOp>> &reqPairs) {

  // Build a mapping of client path names to requests.
  DenseMap<ArrayAttr, SmallVector<RequestToServerConnectionOp, 0>>
      clientNameToServer;
  DenseMap<ArrayAttr, SmallVector<RequestToClientConnectionOp, 0>>
      clientNameToClient;
  for (auto &op : getOps())
    if (auto req = dyn_cast<RequestToClientConnectionOp>(op))
      clientNameToClient[req.getClientNamePathAttr()].push_back(req);
    else if (auto req = dyn_cast<RequestToServerConnectionOp>(op))
      clientNameToServer[req.getClientNamePathAttr()].push_back(req);

  // Find all of the pairs and emit them.
  DenseSet<Operation *> emittedOps;
  for (auto op : getOps<RequestToServerConnectionOp>()) {
    ArrayAttr clientName = op.getClientNamePathAttr();
    const SmallVector<RequestToServerConnectionOp, 0> &ops =
        clientNameToServer[clientName];

    // Only emit a pair if there's one toServer and one toClient request for a
    // given client name path.
    if (ops.size() == 1) {
      auto toClientF = clientNameToClient.find(clientName);
      if (toClientF != clientNameToClient.end() &&
          toClientF->second.size() == 1) {
        reqPairs.push_back(
            std::make_pair(ops.front(), toClientF->second.front()));
        emittedOps.insert(ops.front());
        emittedOps.insert(toClientF->second.front());
        continue;
      }
    }
  }

  // Emit partial pairs for all the remaining requests.
  for (auto &op : getOps()) {
    if (emittedOps.contains(&op))
      continue;
    if (auto req = dyn_cast<RequestToClientConnectionOp>(op))
      reqPairs.push_back(std::make_pair(nullptr, req));
    else if (auto req = dyn_cast<RequestToServerConnectionOp>(op))
      reqPairs.push_back(std::make_pair(req, nullptr));
  }
}

#define GET_OP_CLASSES
#include "circt/Dialect/ESI/ESI.cpp.inc"
