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
#include <numeric>

using namespace mlir;
using namespace circt;
using namespace rtg;

//===----------------------------------------------------------------------===//
// LabelDeclOp
//===----------------------------------------------------------------------===//

APInt LabelDeclOp::getBinary() {
  assert(false && "must not be used for label resources");
  return APInt();
}

void LabelDeclOp::printAssembly(llvm::raw_ostream &stream) {
  // TODO: perform substitutions
  stream << getFormatString();
}

//===----------------------------------------------------------------------===//
// SequenceRefOp
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
// SelectRandomOp
//===----------------------------------------------------------------------===//

LogicalResult SelectRandomOp::verify() {
  if (getSequences().size() != getSequenceArgs().size())
    return emitOpError("number of sequences and sequence arg lists must match");

  if (getSequences().size() != getRatios().size())
    return emitOpError("number of sequences and ratios must match");

  for (auto [seq, args] : llvm::zip(getSequences(), getSequenceArgs()))
    if (TypeRange(cast<SequenceType>(seq.getType()).getArgTypes()) !=
        args.getTypes())
      return emitOpError(
          "sequence argument types do not match sequence requirements");

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
  if (getElements().size() > 0)
    if (getElements()[0].getType() != getSet().getType().getElementType())
      return emitOpError() << "operand types must match set element type";

  return success();
}

//===----------------------------------------------------------------------===//
// BagCreateOp
//===----------------------------------------------------------------------===//

ParseResult BagCreateOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 16> elementOperands,
      weightOperands;
  Type elemType;

  if (!parser.parseOptionalLParen()) {
    while (true) {
      OpAsmParser::UnresolvedOperand elementOperand, weightOperand;
      if (parser.parseOperand(elementOperand) || parser.parseColon() ||
          parser.parseOperand(weightOperand))
        return failure();
      elementOperands.push_back(elementOperand);
      weightOperands.push_back(weightOperand);
      if (parser.parseOptionalComma()) {
        if (parser.parseRParen())
          return failure();
        break;
      }
    }
  }

  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(elemType))
    return failure();

  result.addTypes({BagType::get(result.getContext(), elemType)});

  for (auto operand : elementOperands)
    if (parser.resolveOperand(operand, elemType, result.operands))
      return failure();

  for (auto operand : weightOperands)
    if (parser.resolveOperand(operand, IndexType::get(result.getContext()),
                              result.operands))
      return failure();

  SmallVector<int32_t> segmentSizes(2, elementOperands.size());
  result.addAttribute(
      getOperandSegmentSizesAttrName(result.name),
      DenseI32ArrayAttr::get(result.getContext(), segmentSizes));

  return success();
}

void BagCreateOp::print(OpAsmPrinter &p) {
  p << " ";
  if (!getElements().empty())
    p << "(";
  llvm::interleaveComma(llvm::zip(getElements(), getWeights()), p,
                        [&](auto elAndWeight) {
                          auto [el, weight] = elAndWeight;
                          p << el << " : " << weight;
                        });
  if (!getElements().empty())
    p << ")";

  p.printOptionalAttrDict((*this)->getAttrs(),
                          {getOperandSegmentSizesAttrName()});
  p << " : " << getBag().getType().getElementType();
}

LogicalResult BagCreateOp::verify() {
  if (getElements().size() != getWeights().size())
    return emitOpError() << "number of elements and weights must match";

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
  if (getBody()->getArgumentTypes() != getTargetType().getEntryTypes())
    return emitOpError("argument types must match target entry types");

  return success();
}

//===----------------------------------------------------------------------===//
// TargetOp
//===----------------------------------------------------------------------===//

LogicalResult TargetOp::verifyRegions() {
  if (getBody()->getTerminator()->getOperandTypes() !=
      getTargetType().getEntryTypes())
    return emitOpError(
        "terminator operand types must match target entry types");

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#include "circt/Dialect/RTG/IR/RTGInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/RTG/IR/RTG.cpp.inc"
