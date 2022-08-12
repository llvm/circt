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
  p << " " << clk() << ", " << rst() << ", " << input() << " ";
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << innerType();
}

circt::esi::ChannelType ChannelBufferOp::channelType() {
  return input().getType().cast<circt::esi::ChannelType>();
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
  p << " " << clk() << ", " << rst() << ", " << input() << " ";
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << innerType();
}

circt::esi::ChannelType PipelineStageOp::channelType() {
  return input().getType().cast<circt::esi::ChannelType>();
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
  p << " " << rawInput() << ", " << valid();
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
  p << " " << chanInput() << ", " << ready();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << rawOutput().getType();
}

circt::esi::ChannelType WrapValidReadyOp::channelType() {
  return chanOutput().getType().cast<circt::esi::ChannelType>();
}

void UnwrapValidReadyOp::build(OpBuilder &b, OperationState &state,
                               Value inChan, Value ready) {
  auto inChanType = inChan.getType().cast<ChannelType>();
  build(b, state, inChanType.getInner(), b.getI1Type(), inChan, ready);
}

circt::esi::ChannelType UnwrapValidReadyOp::channelType() {
  return chanInput().getType().cast<circt::esi::ChannelType>();
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
  auto modportType = interfaceSink().getType().cast<circt::sv::ModportType>();
  auto chanType = output().getType().cast<ChannelType>();
  return verifySVInterface(*this, modportType, chanType);
}

circt::esi::ChannelType WrapSVInterfaceOp::channelType() {
  return output().getType().cast<circt::esi::ChannelType>();
}

LogicalResult UnwrapSVInterfaceOp::verify() {
  auto modportType = interfaceSource().getType().cast<circt::sv::ModportType>();
  auto chanType = chanInput().getType().cast<ChannelType>();
  return verifySVInterface(*this, modportType, chanType);
}

circt::esi::ChannelType UnwrapSVInterfaceOp::channelType() {
  return chanInput().getType().cast<circt::esi::ChannelType>();
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
    if (portDecl.inner_symAttr() == innerSym)
      return portDecl;
  return {};
}

/// Check that the type of a given service request matches the services port
/// type.
template <class PortTypeOp, class OpType>
static LogicalResult
reqPortMatches(OpType op, SymbolTableCollection &symbolTable, Type t) {
  hw::InnerRefAttr port = op.servicePort();
  auto portDecl = getServicePortDecl<PortTypeOp>(op, symbolTable, port);
  auto inoutPortDecl =
      getServicePortDecl<ServiceDeclInOutOp>(op, symbolTable, port);
  if (!portDecl && !inoutPortDecl)
    return op.emitOpError("Could not find service port declaration ")
           << port.getModuleRef() << "::" << port.getName().getValue();

  auto *ctxt = op.getContext();
  auto anyChannelType = ChannelType::get(ctxt, AnyType::get(ctxt));
  if (portDecl) {
    if (portDecl.type() != t && portDecl.type() != anyChannelType)
      return op.emitOpError("Request type does not match port type ")
             << portDecl.type();
    return success();
  }

  assert(inoutPortDecl);
  if (isa<ToClientOp>(op)) {
    if (inoutPortDecl.inType() != t && inoutPortDecl.inType() != anyChannelType)
      return op.emitOpError("Request type does not match port type ")
             << inoutPortDecl.inType();
  } else if (isa<ToServerOp>(op)) {
    if (inoutPortDecl.outType() != t &&
        inoutPortDecl.outType() != anyChannelType)
      return op.emitOpError("Request type does not match port type ")
             << inoutPortDecl.outType();
  }
  return success();
}

LogicalResult RequestToClientConnectionOp::verifySymbolUses(
    SymbolTableCollection &symbolTable) {
  return reqPortMatches<ToClientOp>(*this, symbolTable, receiving().getType());
}

LogicalResult RequestToServerConnectionOp::verifySymbolUses(
    SymbolTableCollection &symbolTable) {
  return reqPortMatches<ToServerOp>(*this, symbolTable, sending().getType());
}

LogicalResult
RequestInOutChannelOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto portDecl = getServicePortDecl<ServiceDeclInOutOp>(
      this->getOperation(), symbolTable, servicePort());
  if (!portDecl)
    return emitOpError("Could not find inout service port declaration.");

  auto *ctxt = getContext();
  auto anyChannelType = ChannelType::get(ctxt, AnyType::get(ctxt));

  // Check the input port type.
  if (portDecl.inType() != sending().getType() &&
      portDecl.inType() != anyChannelType)
    return emitOpError("Request to_server type does not match port type ")
           << portDecl.inType();

  // Check the output port type.
  if (portDecl.outType() != receiving().getType() &&
      portDecl.outType() != anyChannelType)
    return emitOpError("Request to_client type does not match port type ")
           << portDecl.outType();

  return success();
}

#define GET_OP_CLASSES
#include "circt/Dialect/ESI/ESI.cpp.inc"
