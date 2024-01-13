//===- SMTOps.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SMT/SMTOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace circt;
using namespace smt;
using namespace mlir;

//===----------------------------------------------------------------------===//
// BVConstantOp
//===----------------------------------------------------------------------===//

LogicalResult BVConstantOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(
      properties.as<Properties *>()->getValue().getType());
  return success();
}

void BVConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  SmallVector<char, 128> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << "c" << getValue().getValue() << "_bv"
              << getValue().getValue().getBitWidth();
  setNameFn(getResult(), specialName.str());
}

OpFoldResult BVConstantOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "constant has no operands");
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// DeclareConstOp
//===----------------------------------------------------------------------===//

void DeclareConstOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), getNamePrefix().has_value() ? *getNamePrefix() : "");
}

//===----------------------------------------------------------------------===//
// SolverOp
//===----------------------------------------------------------------------===//

LogicalResult SolverOp::verifyRegions() {
  if (getBody()->getTerminator()->getOperands().getTypes() != getResultTypes())
    return emitOpError() << "types of yielded values must match return values";
  if (getBody()->getArgumentTypes() != getInputs().getTypes())
    return emitOpError()
           << "block argument types must match the types of the 'inputs'";
  return success();
}

//===----------------------------------------------------------------------===//
// CheckOp
//===----------------------------------------------------------------------===//

LogicalResult CheckOp::verifyRegions() {
  if (getSatRegion().front().getTerminator()->getOperands().getTypes() !=
      getResultTypes())
    return emitOpError() << "types of yielded values in 'sat' region must "
                            "match return values";
  if (getUnknownRegion().front().getTerminator()->getOperands().getTypes() !=
      getResultTypes())
    return emitOpError() << "types of yielded values in 'unknown' region must "
                            "match return values";
  if (getUnsatRegion().front().getTerminator()->getOperands().getTypes() !=
      getResultTypes())
    return emitOpError() << "types of yielded values in 'unsat' region must "
                            "match return values";

  return success();
}

//===----------------------------------------------------------------------===//
// EqOp
//===----------------------------------------------------------------------===//

static LogicalResult
parseSameOperandTypeVariadicToBoolOp(OpAsmParser &parser,
                                     OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inputs;
  SMLoc loc = parser.getCurrentLocation();
  Type type;

  if (parser.parseOperandList(inputs) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(type))
    return failure();

  result.addTypes(BoolType::get(parser.getContext()));
  if (parser.resolveOperands(inputs, SmallVector<Type>(inputs.size(), type),
                             loc, result.operands))
    return failure();

  return success();
}

ParseResult EqOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseSameOperandTypeVariadicToBoolOp(parser, result);
}

void EqOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getInputs();
  printer.printOptionalAttrDict(getOperation()->getAttrs());
  printer << " : " << getInputs().front().getType();
}

LogicalResult EqOp::verify() {
  if (getInputs().size() < 2)
    return emitOpError() << "'inputs' must have at least size 2, but got "
                         << getInputs().size();

  return success();
}

//===----------------------------------------------------------------------===//
// DistinctOp
//===----------------------------------------------------------------------===//

ParseResult DistinctOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseSameOperandTypeVariadicToBoolOp(parser, result);
}

void DistinctOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getInputs();
  printer.printOptionalAttrDict(getOperation()->getAttrs());
  printer << " : " << getInputs().front().getType();
}

LogicalResult DistinctOp::verify() {
  if (getInputs().size() < 2)
    return emitOpError() << "'inputs' must have at least size 2, but got "
                         << getInputs().size();

  return success();
}

#define GET_OP_CLASSES
#include "circt/Dialect/SMT/SMT.cpp.inc"
