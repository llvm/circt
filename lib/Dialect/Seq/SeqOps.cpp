//===- SeqOps.cpp - Implement the Seq operations ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements sequential ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/SmallString.h"

using namespace mlir;
using namespace circt;
using namespace seq;

// If there was no name specified, check to see if there was a useful name
// specified in the asm file.
static void setNameFromResult(OpAsmParser &parser, OperationState &result) {
  if (result.attributes.getNamed("name"))
    return;
  // If there is no explicit name attribute, get it from the SSA result name.
  // If numeric, just use an empty name.
  StringRef resultName = parser.getResultName(0).first;
  if (!resultName.empty() && isdigit(resultName[0]))
    resultName = "";
  result.addAttribute("name", parser.getBuilder().getStringAttr(resultName));
}

static bool canElideName(OpAsmPrinter &p, Operation *op) {
  if (!op->hasAttr("name"))
    return true;

  auto name = op->getAttrOfType<StringAttr>("name").getValue();
  if (name.empty())
    return true;

  SmallString<32> resultNameStr;
  llvm::raw_svector_ostream tmpStream(resultNameStr);
  p.printOperand(op->getResult(0), tmpStream);
  auto actualName = tmpStream.str().drop_front();
  return actualName == name;
}

//===----------------------------------------------------------------------===//
// CompRegOp

ParseResult CompRegOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc loc = parser.getCurrentLocation();

  if (succeeded(parser.parseOptionalKeyword("sym"))) {
    StringAttr symName;
    if (parser.parseSymbolName(symName, "sym_name", result.attributes))
      return failure();
  }

  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  if (parser.parseOperandList(operands))
    return failure();
  switch (operands.size()) {
  case 0:
    return parser.emitError(loc, "expected operands");
  case 1:
    return parser.emitError(loc, "expected clock operand");
  case 2:
    // No reset.
    break;
  case 3:
    return parser.emitError(loc, "expected resetValue operand");
  case 4:
    // reset and reset value included.
    break;
  default:
    return parser.emitError(loc, "too many operands");
  }

  Type ty;
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(ty))
    return failure();
  Type i1 = IntegerType::get(result.getContext(), 1);

  setNameFromResult(parser, result);

  result.addTypes({ty});
  if (operands.size() == 2)
    return parser.resolveOperands(operands, {ty, i1}, loc, result.operands);
  else
    return parser.resolveOperands(operands, {ty, i1, i1, ty}, loc,
                                  result.operands);
}

void CompRegOp::print(::mlir::OpAsmPrinter &p) {
  SmallVector<StringRef> elidedAttrs;
  if (auto sym = getSymName()) {
    elidedAttrs.push_back("sym_name");
    p << ' ' << "sym ";
    p.printSymbolName(*sym);
  }

  p << ' ' << getInput() << ", " << getClk();
  if (getReset())
    p << ", " << getReset() << ", " << getResetValue() << ' ';

  // Determine if 'name' can be elided.
  if (canElideName(p, *this))
    elidedAttrs.push_back("name");

  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  p << " : " << getInput().getType();
}

/// Suggest a name for each result value based on the saved result names
/// attribute.
void CompRegOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // If the wire has an optional 'name' attribute, use it.
  if (!getName().empty())
    setNameFn(getResult(), getName());
}

//===----------------------------------------------------------------------===//
// FirRegOp

void FirRegOp::build(OpBuilder &builder, OperationState &result, Value input,
                     Value clk, StringAttr name, StringAttr innerSym) {

  OpBuilder::InsertionGuard guard(builder);

  result.addOperands(input);
  result.addOperands(clk);

  result.addAttribute(getNameAttrName(result.name), name);

  if (innerSym)
    result.addAttribute(getInnerSymAttrName(result.name), innerSym);

  result.addTypes(input.getType());
}

void FirRegOp::build(OpBuilder &builder, OperationState &result, Value input,
                     Value clk, StringAttr name, Value reset, Value resetValue,
                     StringAttr innerSym, bool isAsync) {

  OpBuilder::InsertionGuard guard(builder);

  result.addOperands(input);
  result.addOperands(clk);
  result.addOperands(reset);
  result.addOperands(resetValue);

  result.addAttribute(getNameAttrName(result.name), name);
  if (isAsync)
    result.addAttribute(getIsAsyncAttrName(result.name), builder.getUnitAttr());

  if (innerSym)
    result.addAttribute(getInnerSymAttrName(result.name), innerSym);

  result.addTypes(input.getType());
}

ParseResult FirRegOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  llvm::SMLoc loc = parser.getCurrentLocation();

  using Op = OpAsmParser::UnresolvedOperand;

  Op next, clk;
  if (parser.parseOperand(next) || parser.parseKeyword("clock") ||
      parser.parseOperand(clk))
    return failure();

  if (succeeded(parser.parseOptionalKeyword("sym"))) {
    StringAttr symName;
    if (parser.parseSymbolName(symName, "inner_sym", result.attributes))
      return failure();
  }

  // Parse reset [sync|async] %reset, %value
  Optional<std::pair<Op, Op>> resetAndValue;
  if (succeeded(parser.parseOptionalKeyword("reset"))) {
    bool isAsync;
    if (succeeded(parser.parseOptionalKeyword("async")))
      isAsync = true;
    else if (succeeded(parser.parseOptionalKeyword("sync")))
      isAsync = false;
    else
      return parser.emitError(loc, "invalid reset, expected 'sync' or 'async'");
    if (isAsync)
      result.attributes.append("isAsync", builder.getUnitAttr());

    resetAndValue = {{}, {}};
    if (parser.parseOperand(resetAndValue->first) || parser.parseComma() ||
        parser.parseOperand(resetAndValue->second))
      return failure();
  }

  Type ty;
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(ty))
    return failure();
  result.addTypes({ty});

  setNameFromResult(parser, result);

  Type i1 = IntegerType::get(result.getContext(), 1);
  if (parser.resolveOperand(next, ty, result.operands) ||
      parser.resolveOperand(clk, i1, result.operands))
    return failure();

  if (resetAndValue) {
    if (parser.resolveOperand(resetAndValue->first, i1, result.operands) ||
        parser.resolveOperand(resetAndValue->second, ty, result.operands))
      return failure();
  }

  return success();
}

void FirRegOp::print(::mlir::OpAsmPrinter &p) {
  SmallVector<StringRef> elidedAttrs = {getInnerSymAttrName(),
                                        getIsAsyncAttrName()};

  p << ' ' << getNext() << " clock " << getClk();

  if (auto sym = getInnerSym()) {
    p << " sym ";
    p.printSymbolName(*sym);
  }

  if (hasReset()) {
    p << " reset " << (getIsAsync() ? "async" : "sync") << ' ';
    p << getReset() << ", " << getResetValue();
  }

  if (canElideName(p, *this))
    elidedAttrs.push_back("name");

  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  p << " : " << getNext().getType();
}

/// Verifier for the FIR register op.
LogicalResult FirRegOp::verify() {
  if (getReset() || getResetValue() || getIsAsync()) {
    if (!getReset() || !getResetValue())
      return emitOpError("must specify reset and reset value");
  } else {
    if (getIsAsync())
      return emitOpError("register with no reset cannot be async");
  }
  return success();
}

/// Suggest a name for each result value based on the saved result names
/// attribute.
void FirRegOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // If the register has an optional 'name' attribute, use it.
  if (!getName().empty())
    setNameFn(getResult(), getName());
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/Seq/Seq.cpp.inc"
