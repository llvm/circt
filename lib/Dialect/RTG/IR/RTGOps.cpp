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
// TableGen generated logic.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "circt/Dialect/RTG/IR/RTG.cpp.inc"
