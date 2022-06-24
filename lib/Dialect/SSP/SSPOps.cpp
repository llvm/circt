//===- SSPOps.cpp - SSP operation implementation --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SSP (static scheduling problem) dialect operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SSP/SSPOps.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/Builders.h"

using namespace circt;
using namespace circt::ssp;

//===----------------------------------------------------------------------===//
// OperationOp
//===----------------------------------------------------------------------===//

ParseResult OperationOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  // (Scheduling) operation's name
  StringAttr opName;
  if (parser.parseSymbolName(opName, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Dependences
  SmallVector<OpAsmParser::UnresolvedOperand> unresolvedOperands;
  SmallVector<Attribute> dependences;
  auto parseDependenceSourceWithAttrDict = [&]() -> ParseResult {
    llvm::SMLoc loc = parser.getCurrentLocation();
    IntegerAttr operandIdx;
    FlatSymbolRefAttr sourceOp;
    DictionaryAttr properties;

    // Try to parse a symbol reference first...
    if (!parser.parseOptionalAttribute(sourceOp).hasValue()) {
      // ...and if that fails, attempt to parse an SSA operand.
      OpAsmParser::UnresolvedOperand operand;
      if (failed(parser.parseOperand(operand)))
        return parser.emitError(loc, "expected SSA value or symbol reference");

      operandIdx = builder.getI64IntegerAttr(unresolvedOperands.size());
      unresolvedOperands.push_back(operand);
    }

    NamedAttrList attrDict;
    if (succeeded(parser.parseOptionalAttrDict(attrDict)))
      properties = builder.getDictionaryAttr(attrDict);

    dependences.push_back(DependenceAttr::get(builder.getContext(), operandIdx,
                                              sourceOp, properties));
    return success();
  };

  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren,
                                     parseDependenceSourceWithAttrDict))
    return failure();

  result.addAttribute(builder.getStringAttr("dependences"),
                      builder.getArrayAttr(dependences));

  // Properties
  (void)parser.parseOptionalAttrDict(result.attributes);

  // Resolve operands
  SmallVector<Value> operands;
  if (parser.resolveOperands(unresolvedOperands, builder.getNoneType(),
                             operands))
    return failure();
  result.addOperands(operands);

  // Mockup results
  SmallVector<Type> types(parser.getNumResults(), builder.getNoneType());
  result.addTypes(types);

  return success();
}

void OperationOp::print(OpAsmPrinter &p) {
  // (Scheduling) operation's name
  p << ' ';
  p.printSymbolName(getSymName());

  // Dependences
  p << '(';
  llvm::interleaveComma(getDependences(), p, [&](Attribute attr) {
    DependenceAttr dep = attr.cast<DependenceAttr>();

    if (auto sourceRef = dep.getSourceRef())
      p << sourceRef;
    else if (auto operandIdx = dep.getOperandIdx())
      p.printOperand(getOperand(operandIdx.getValue().getZExtValue()));

    p.printOptionalAttrDict(dep.getProperties().getValue());
  });
  p << ')';

  // Properties
  SmallVector<StringRef> elidedAttrs = {
      SymbolTable::getSymbolAttrName(),
      OperationOp::getDependencesAttrName().getValue()};
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

LogicalResult OperationOp::verify() {
  // TODO: better check that all SSA operands have an associated DependenceAttr.
  if (getDependences().size() < getNumOperands())
    return emitOpError("has malformed `dependences` attribute");
  return success();
}

LogicalResult
OperationOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  for (auto dep : getDependences().getAsRange<DependenceAttr>()) {
    if (auto sourceRef = dep.getSourceRef()) {
      Operation *sourceOp =
          symbolTable.lookupNearestSymbolFrom(*this, sourceRef);
      if (!sourceOp || !isa<OperationOp>(sourceOp))
        return emitOpError("references invalid source operation: ")
               << sourceRef;
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'ed code
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "circt/Dialect/SSP/SSP.cpp.inc"
