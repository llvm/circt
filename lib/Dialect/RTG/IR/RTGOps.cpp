//===- RTGOps.cpp - Implement the RTG operations --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the RTG ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace circt;
using namespace rtg;

//===----------------------------------------------------------------------===//
// SequenceClosureOp
//===----------------------------------------------------------------------===//

LogicalResult
SequenceClosureOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  SequenceOp seq =
      symbolTable.lookupNearestSymbolFrom<SequenceOp>(*this, getSequenceAttr());
  if (!seq)
    return emitOpError()
           << "'" << getSequence()
           << "' does not reference a valid 'rtg.sequence' operation";

  if (seq.getBodyRegion().getArgumentTypes() != getArgs().getTypes())
    return emitOpError("referenced 'rtg.sequence' op's argument types must "
                       "match 'args' types");

  return success();
}

//===----------------------------------------------------------------------===//
// SetCreateOp
//===----------------------------------------------------------------------===//

ParseResult SetCreateOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 16> operands;
  Type elemType;

  if (parser.parseOperandList(operands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(elemType))
    return failure();

  result.addTypes({SetType::get(result.getContext(), elemType)});

  for (auto operand : operands)
    if (parser.resolveOperand(operand, elemType, result.operands))
      return failure();

  return success();
}

void SetCreateOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printOperands(getElements());
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << getSet().getType().getElementType();
}

LogicalResult SetCreateOp::verify() {
  if (getElements().size() > 0) {
    // We only need to check the first element because of the `SameTypeOperands`
    // trait.
    if (getElements()[0].getType() != getSet().getType().getElementType())
      return emitOpError() << "operand types must match set element type";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// BagCreateOp
//===----------------------------------------------------------------------===//

ParseResult BagCreateOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 16> elementOperands,
      multipleOperands;
  Type elemType;

  if (!parser.parseOptionalLParen()) {
    while (true) {
      OpAsmParser::UnresolvedOperand elementOperand, multipleOperand;
      if (parser.parseOperand(multipleOperand) || parser.parseKeyword("x") ||
          parser.parseOperand(elementOperand))
        return failure();

      elementOperands.push_back(elementOperand);
      multipleOperands.push_back(multipleOperand);

      if (parser.parseOptionalComma()) {
        if (parser.parseRParen())
          return failure();
        break;
      }
    }
  }

  if (parser.parseColon() || parser.parseType(elemType) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  result.addTypes({BagType::get(result.getContext(), elemType)});

  for (auto operand : elementOperands)
    if (parser.resolveOperand(operand, elemType, result.operands))
      return failure();

  for (auto operand : multipleOperands)
    if (parser.resolveOperand(operand, IndexType::get(result.getContext()),
                              result.operands))
      return failure();

  return success();
}

void BagCreateOp::print(OpAsmPrinter &p) {
  p << " ";
  if (!getElements().empty())
    p << "(";
  llvm::interleaveComma(llvm::zip(getElements(), getMultiples()), p,
                        [&](auto elAndMultiple) {
                          auto [el, multiple] = elAndMultiple;
                          p << multiple << " x " << el;
                        });
  if (!getElements().empty())
    p << ")";

  p << " : " << getBag().getType().getElementType();
  p.printOptionalAttrDict((*this)->getAttrs());
}

LogicalResult BagCreateOp::verify() {
  if (!llvm::all_equal(getElements().getTypes()))
    return emitOpError() << "types of all elements must match";

  if (getElements().size() > 0)
    if (getElements()[0].getType() != getBag().getType().getElementType())
      return emitOpError() << "operand types must match bag element type";

  return success();
}

//===----------------------------------------------------------------------===//
// TestOp
//===----------------------------------------------------------------------===//

LogicalResult TestOp::verifyRegions() {
  if (!getTarget().entryTypesMatch(getBody()->getArgumentTypes()))
    return emitOpError("argument types must match dict entry types");

  return success();
}

//===----------------------------------------------------------------------===//
// TargetOp
//===----------------------------------------------------------------------===//

LogicalResult TargetOp::verifyRegions() {
  if (!getTarget().entryTypesMatch(
          getBody()->getTerminator()->getOperandTypes()))
    return emitOpError("terminator operand types must match dict entry types");

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "circt/Dialect/RTG/IR/RTG.cpp.inc"
