//===- IbisOps.cpp - Implementation of Ibis dialect ops -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Ibis/IbisOps.h"
#include "circt/Support/ParsingUtils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace circt;
using namespace ibis;

template <typename TSymAttr>
ParseResult parseClassRefFromName(OpAsmParser &parser, Type &classRefType,
                                  TSymAttr sym) {
  // Nothing to parse, since this is already encoded in the child symbol.
  classRefType = ClassRefType::get(parser.getContext(), sym);
  return success();
}

template <typename TSymAttr>
void printClassRefFromName(OpAsmPrinter &p, Operation *op, Type type,
                           TSymAttr sym) {
  // Nothing to print since this information is already encoded in the child
  // symbol.
}

//===----------------------------------------------------------------------===//
// ClassOp
//===----------------------------------------------------------------------===//

ParseResult ClassOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse signature
  StringAttr className;
  if (parser.parseSymbolName(className, mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // parse "this"
  OpAsmParser::Argument thisArg;
  if (parser.parseLParen() ||
      parser.parseArgument(thisArg, /*allowType=*/false) ||
      parser.parseRParen())
    return failure();

  thisArg.type = parser.getBuilder().getType<ClassRefType>(className);

  // Parse the attribute dict.
  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();

  // Parse the class body.
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, {thisArg}))
    return failure();

  return success();
}

void ClassOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printSymbolName(getSymName());
  p << '(';
  p.printRegionArgument(getThis(), {}, /*omitType*/ true);
  p << ')';
  llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("sym_name");
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

void ClassOp::getAsmBlockArgumentNames(mlir::Region &region,
                                       OpAsmSetValueNameFn setNameFn) {
  setNameFn(getThis(), "this");
}

//===----------------------------------------------------------------------===//
// MethodOp
//===----------------------------------------------------------------------===//

ParseResult MethodOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the function signature.
  SmallVector<OpAsmParser::Argument, 4> args;
  SmallVector<Attribute> argNames;
  SmallVector<Type> resultTypes;
  TypeAttr functionType;

  using namespace mlir::function_interface_impl;
  auto *context = parser.getContext();

  // Parse the argument list.
  if (parser.parseArgumentList(args, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true, /*allowAttrs=*/false))
    return failure();

  // Parse the result type.
  if (succeeded(parser.parseOptionalArrow())) {
    Type resultType;
    if (parser.parseType(resultType))
      return failure();
    resultTypes.push_back(resultType);
  }

  // Process the ssa args for the information we're looking for.
  SmallVector<Type> argTypes;
  for (auto &arg : args) {
    argNames.push_back(parsing_util::getNameFromSSA(context, arg.ssaName.name));
    argTypes.push_back(arg.type);
    if (!arg.sourceLoc)
      arg.sourceLoc = parser.getEncodedSourceLoc(arg.ssaName.location);
  }

  functionType =
      TypeAttr::get(FunctionType::get(context, argTypes, resultTypes));

  // Parse the attribute dict.
  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();

  result.addAttribute("argNames", ArrayAttr::get(context, argNames));
  result.addAttribute(MethodOp::getFunctionTypeAttrName(result.name),
                      functionType);

  // Parse the function body.
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, args))
    return failure();

  ensureTerminator(*body, parser.getBuilder(), result.location);
  return success();
}

void MethodOp::print(OpAsmPrinter &p) {
  FunctionType funcTy = getFunctionType();
  p << ' ';
  p.printSymbolName(getSymName());
  function_interface_impl::printFunctionSignature(
      p, *this, funcTy.getInputs(), /*isVariadic=*/false, funcTy.getResults());
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                     getAttributeNames());
  Region &body = getBody();
  if (!body.empty()) {
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

void MethodOp::getAsmBlockArgumentNames(mlir::Region &region,
                                        OpAsmSetValueNameFn setNameFn) {
  if (region.empty())
    return;

  auto func = cast<MethodOp>(region.getParentOp());
  auto argNames = func.getArgNames().getAsRange<StringAttr>();
  auto *block = &region.front();

  for (auto [idx, argName] : llvm::enumerate(argNames))
    if (!argName.getValue().empty())
      setNameFn(block->getArgument(idx), argName);
}

LogicalResult MethodOp::verify() {
  // Check that we have only one return value.
  if (getFunctionType().getNumResults() > 1)
    return failure();
  return success();
}

void ReturnOp::build(OpBuilder &odsBuilder, OperationState &odsState) {}

LogicalResult ReturnOp::verify() {
  // Check that the return operand type matches the function return type.
  auto func = cast<MethodOp>((*this)->getParentOp());
  ArrayRef<Type> resTypes = func.getResultTypes();
  assert(resTypes.size() <= 1);
  assert(getNumOperands() <= 1);

  if (resTypes.empty()) {
    if (getNumOperands() != 0)
      return emitOpError(
          "cannot return a value from a function with no result type");
    return success();
  }

  Value retValue = getRetValue();
  if (!retValue)
    return emitOpError("must return a value");

  Type retType = retValue.getType();
  if (retType != resTypes.front())
    return emitOpError("return type (")
           << retType << ") must match function return type ("
           << resTypes.front() << ")";

  return success();
}

//===----------------------------------------------------------------------===//
// InstanceOp
//===----------------------------------------------------------------------===//

LogicalResult InstanceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto targetClass = getClass(&symbolTable);
  if (!targetClass)
    return emitOpError() << "'" << getClassName() << "' does not exist";

  return success();
}

//===----------------------------------------------------------------------===//
// InstanceOp
//===----------------------------------------------------------------------===//

LogicalResult GetPortOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Lookup the target module type of the instance class reference.
  ModuleOp mod = getOperation()->getParentOfType<ModuleOp>();
  ClassRefType crt = getInstance().getType().cast<ClassRefType>();
  // @teqdruid TODO: make this more efficient using
  // innersymtablecollection when that's available to non-firrtl dialects.
  auto targetClass =
      symbolTable.lookupSymbolIn<ClassOp>(mod, crt.getClassRef());
  assert(targetClass && "should have been verified by the type system");
  // @teqdruid TODO: make this more efficient using
  // innersymtablecollection when that's available to non-firrtl dialects.
  Operation *targetOp =
      symbolTable.lookupSymbolIn(targetClass, getPortSymbolAttr());

  if (!targetOp)
    return emitOpError() << "port '" << getPortSymbolAttr()
                         << "' does not exist in " << targetClass.getName();

  auto portOp = dyn_cast<PortOpInterface>(targetOp);
  if (!portOp)
    return emitOpError() << "symbol '" << getPortSymbolAttr()
                         << "' does not refer to a port";

  Type targetPortType = portOp.getPortType();
  Type thisPortType = getType().getPortType();
  if (targetPortType != thisPortType)
    return emitOpError() << "symbol '" << getPortSymbolAttr()
                         << "' refers to a port of type " << targetPortType
                         << ", but this op has type " << thisPortType;

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Ibis/IbisInterfaces.cpp.inc"

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/Ibis/Ibis.cpp.inc"
