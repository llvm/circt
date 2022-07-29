//===- SystemCOps.cpp - Implement the SystemC operations ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SystemC ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SystemC/SystemCOps.h"
#include "mlir/IR/Builders.h"

using namespace circt;
using namespace circt::systemc;

//===----------------------------------------------------------------------===//
// ImplicitSSAName Custom Directive
//===----------------------------------------------------------------------===//

static ParseResult parseImplicitSSAName(OpAsmParser &parser,
                                        StringAttr &nameAttr) {
  nameAttr = parser.getBuilder().getStringAttr(parser.getResultName(0).first);
  return success();
}

static void printImplicitSSAName(OpAsmPrinter &p, Operation *op,
                                 StringAttr nameAttr) {}

//===----------------------------------------------------------------------===//
// SCModuleOp
//===----------------------------------------------------------------------===//

void SCModuleOp::getPortsOfDirection(PortDirection direction,
                                     SmallVector<Value> &outputs) {
  for (int i = 0, e = getNumArguments(); i < e; ++i) {
    if (getPortDirections().getDirection(i) == direction)
      outputs.push_back(getArgument(i));
  }
}

mlir::Region *SCModuleOp::getCallableRegion() { return &getBody(); }

ArrayRef<mlir::Type> SCModuleOp::getCallableResults() {
  return getResultTypes();
}

StringRef SCModuleOp::getModuleName() {
  return (*this)
      ->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
      .getValue();
}

/// Parse an argument list of a systemc.module operation.
static ParseResult parseArgumentList(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::Argument> &args,
    SmallVectorImpl<Type> &argTypes, SmallVectorImpl<Attribute> &argNames,
    SmallVectorImpl<PortDirection> &argDirection) {
  auto parseElt = [&]() -> ParseResult {
    // Parse port direction.
    PortDirection dir = PortDirection::Input;
    if (succeeded(parser.parseOptionalKeyword(
            stringifyPortDirection(PortDirection::InOut))))
      dir = PortDirection::InOut;
    else if (succeeded(parser.parseOptionalKeyword(
                 stringifyPortDirection(PortDirection::Output))))
      dir = PortDirection::Output;
    else if (failed(parser.parseKeyword(
                 stringifyPortDirection(PortDirection::Input),
                 ", 'sc_out', or 'sc_inout'")))
      return failure();

    OpAsmParser::Argument argument;
    auto optArg = parser.parseOptionalArgument(argument);
    if (optArg.hasValue()) {
      if (succeeded(optArg.getValue())) {
        Type argType;
        if (!argument.ssaName.name.empty() &&
            succeeded(parser.parseColonType(argType))) {
          args.push_back(argument);
          argTypes.push_back(argType);
          args.back().type = argType;
          argDirection.push_back(dir);
          argNames.push_back(StringAttr::get(
              parser.getContext(), argument.ssaName.name.drop_front()));
        }
      }
    }
    return success();
  };

  return parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                        parseElt);
}

ParseResult SCModuleOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr entityName;
  SmallVector<OpAsmParser::Argument, 4> args;
  SmallVector<Type, 4> argTypes;
  SmallVector<Attribute> argNames;
  SmallVector<PortDirection> argDirection;

  if (parser.parseSymbolName(entityName, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  if (failed(parseArgumentList(parser, args, argTypes, argNames, argDirection)))
    return failure();

  result.addAttribute("portDirections", PortDirectionsAttr::get(
                                            parser.getContext(), argDirection));
  result.addAttribute("portNames",
                      ArrayAttr::get(parser.getContext(), argNames));

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  auto type = parser.getBuilder().getFunctionType(argTypes, llvm::None);
  result.addAttribute(SCModuleOp::getTypeAttrName(), TypeAttr::get(type));

  auto &body = *result.addRegion();
  if (parser.parseRegion(body, args))
    return failure();
  if (body.empty())
    body.push_back(std::make_unique<Block>().release());

  return success();
}

static void printArgumentList(OpAsmPrinter &printer,
                              ArrayRef<BlockArgument> args,
                              ArrayRef<PortDirection> directions) {
  printer << "(";
  for (size_t i = 0; i < args.size(); ++i) {
    if (i > 0)
      printer << ", ";
    printer << stringifyPortDirection(directions[i]) << " " << args[i] << ": "
            << args[i].getType();
  }
  printer << ")";
}

void SCModuleOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer.printSymbolName(getModuleName());
  printer << " ";
  SmallVector<PortDirection> directions;
  getPortDirections().getPortDirections(directions);
  printArgumentList(printer, getArguments(), directions);
  printer.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      /*elidedAttrs =*/{SymbolTable::getSymbolAttrName(),
                        SCModuleOp::getTypeAttrName(), "portNames",
                        "portDirections"});
  printer << " ";
  printer.printRegion(getBody(), false, false);
}

void SCModuleOp::getAsmBlockArgumentNames(mlir::Region &region,
                                          mlir::OpAsmSetValueNameFn setNameFn) {
  if (region.empty())
    return;

  ArrayAttr portNames = getPortNames();
  for (size_t i = 0, e = getNumArguments(); i != e; ++i) {
    auto str = portNames[i].cast<StringAttr>().getValue();
    setNameFn(getArgument(i), str);
  }
}

LogicalResult SCModuleOp::verify() {
  if (getFunctionType().getNumResults() != 0)
    return emitOpError(
        "incorrect number of function results (always has to be 0)");
  if (getPortNames().size() != getFunctionType().getNumInputs())
    return emitOpError("incorrect number of port names");
  if (getPortDirections().getNumPorts() != getFunctionType().getNumInputs())
    return emitOpError("incorrect number of port directions");

  ArrayAttr portNames = getPortNames();
  for (auto *iter = portNames.begin(); iter != portNames.end(); ++iter) {
    if (iter->cast<StringAttr>().getValue().empty())
      return emitOpError("port name must not be empty");
  }

  return success();
}

LogicalResult SCModuleOp::verifyRegions() {
  DenseMap<StringRef, BlockArgument> portNames;
  DenseMap<StringRef, Operation *> memberNames;
  DenseMap<StringRef, Operation *> localNames;

  bool portsVerified = true;

  for (auto arg : llvm::zip(getPortNames(), getArguments())) {
    StringRef argName = std::get<0>(arg).cast<StringAttr>().getValue();
    BlockArgument argValue = std::get<1>(arg);

    if (portNames.count(argName)) {
      auto diag = mlir::emitError(argValue.getLoc(), "redefines port name '")
                  << argName << "'";
      diag.attachNote(portNames[argName].getLoc())
          << "'" << argName << "' first defined here";
      diag.attachNote(getLoc()) << "in module '@" << getModuleName() << "'";
      portsVerified = false;
      continue;
    }

    portNames.insert({argName, argValue});
  }

  WalkResult result = walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<SCModuleOp>(op->getParentOp()))
      localNames.clear();

    if (auto nameDeclOp = dyn_cast<SystemCNameDeclOpInterface>(op)) {
      StringRef name = nameDeclOp.getName();

      auto reportNameRedefinition = [&](Location firstLoc) -> WalkResult {
        auto diag = mlir::emitError(op->getLoc(), "redefines name '")
                    << name << "'";
        diag.attachNote(firstLoc) << "'" << name << "' first defined here";
        diag.attachNote(getLoc()) << "in module '@" << getModuleName() << "'";
        return WalkResult::interrupt();
      };

      if (portNames.count(name))
        return reportNameRedefinition(portNames[name].getLoc());
      if (memberNames.count(name))
        return reportNameRedefinition(memberNames[name]->getLoc());
      if (localNames.count(name))
        return reportNameRedefinition(localNames[name]->getLoc());

      if (isa<SCModuleOp>(op->getParentOp()))
        memberNames.insert({name, op});
      else
        localNames.insert({name, op});
    }

    return WalkResult::advance();
  });

  if (result.wasInterrupted() || !portsVerified)
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// SignalOp
//===----------------------------------------------------------------------===//

void SignalOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getSignal(), getName());
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/SystemC/SystemC.cpp.inc"
