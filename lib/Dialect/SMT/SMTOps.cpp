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
#include "llvm/ADT/APSInt.h"

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

//===----------------------------------------------------------------------===//
// ExtractOp
//===----------------------------------------------------------------------===//

LogicalResult ExtractOp::verify() {
  unsigned rangeWidth = getType().getWidth();
  unsigned inputWidth = cast<BitVectorType>(getInput().getType()).getWidth();
  if (getLowBit() + rangeWidth > inputWidth)
    return emitOpError("range to be extracted is too big, expected range "
                       "starting at index ")
           << getLowBit() << " of length " << rangeWidth
           << " requires input width of at least " << (getLowBit() + rangeWidth)
           << ", but the input width is only " << inputWidth;
  return success();
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

LogicalResult ConcatOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(BitVectorType::get(
      context, cast<BitVectorType>(operands[0].getType()).getWidth() +
                   cast<BitVectorType>(operands[1].getType()).getWidth()));
  return success();
}

//===----------------------------------------------------------------------===//
// RepeatOp
//===----------------------------------------------------------------------===//

LogicalResult RepeatOp::verify() {
  unsigned inputWidth = cast<BitVectorType>(getInput().getType()).getWidth();
  unsigned resultWidth = getType().getWidth();
  if (resultWidth % inputWidth != 0)
    return emitOpError() << "result bit-vector width must be a multiple of the "
                            "input bit-vector width";

  return success();
}

unsigned RepeatOp::getCount() {
  unsigned inputWidth = cast<BitVectorType>(getInput().getType()).getWidth();
  unsigned resultWidth = getType().getWidth();
  return resultWidth / inputWidth;
}

void RepeatOp::build(OpBuilder &builder, OperationState &state, unsigned count,
                     Value input) {
  unsigned inputWidth = cast<BitVectorType>(input.getType()).getWidth();
  Type resultTy = BitVectorType::get(builder.getContext(), inputWidth * count);
  build(builder, state, resultTy, input);
}

ParseResult RepeatOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  Type inputType;
  llvm::SMLoc countLoc = parser.getCurrentLocation();

  APInt count;
  if (parser.parseInteger(count) || parser.parseKeyword("times"))
    return failure();

  if (count.isNonPositive())
    return parser.emitError(countLoc) << "integer must be positive";

  llvm::SMLoc inputLoc = parser.getCurrentLocation();
  if (parser.parseOperand(input) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(inputType))
    return failure();

  if (parser.resolveOperand(input, inputType, result.operands))
    return failure();

  auto bvInputTy = dyn_cast<BitVectorType>(inputType);
  if (!bvInputTy)
    return parser.emitError(inputLoc) << "input must have bit-vector type";

  // Make sure no assertions can trigger and no silent overflows can happen
  // Bit-width is stored as 'uint64_t' parameter in 'BitVectorType'
  const unsigned maxBw = 64;
  if (count.getActiveBits() > maxBw)
    return parser.emitError(countLoc)
           << "integer must fit into " << maxBw << " bits";

  // Store multiplication in an APInt twice the size to not have any overflow
  // and check if it can be truncated to 'maxBw' bits without cutting of
  // important bits.
  APInt resultBw = bvInputTy.getWidth() * count.zext(2 * maxBw);
  if (resultBw.getActiveBits() > maxBw)
    return parser.emitError(countLoc)
           << "result bit-width (provided integer times bit-width of the input "
              "type) must fit into "
           << maxBw << " bits";

  Type resultTy =
      BitVectorType::get(parser.getContext(), resultBw.getZExtValue());
  result.addTypes(resultTy);
  return success();
}

void RepeatOp::print(OpAsmPrinter &printer) {
  printer << " " << getCount() << " times " << getInput();
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getInput().getType();
}

//===----------------------------------------------------------------------===//
// BoolConstantOp
//===----------------------------------------------------------------------===//

void BoolConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), getValue() ? "true" : "false");
}

OpFoldResult BoolConstantOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "constant has no operands");
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// IntConstantOp
//===----------------------------------------------------------------------===//

void IntConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  SmallVector<char, 32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << "c" << getValue();
  setNameFn(getResult(), specialName.str());
}

OpFoldResult IntConstantOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "constant has no operands");
  return getValueAttr();
}

void IntConstantOp::print(OpAsmPrinter &p) {
  p << " " << getValue();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
}

ParseResult IntConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  APInt value;
  if (parser.parseInteger(value))
    return failure();

  result.getOrAddProperties<Properties>().setValue(
      IntegerAttr::get(parser.getContext(), APSInt(value)));

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  result.addTypes(smt::IntType::get(parser.getContext()));
  return success();
}

#define GET_OP_CLASSES
#include "circt/Dialect/SMT/SMT.cpp.inc"
