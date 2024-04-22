//===- MooreOps.cpp - Implement the Moore operations ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Moore dialect operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Support/CustomDirectiveImpl.h"
#include "mlir/IR/Builders.h"

using namespace circt;
using namespace circt::moore;

//===----------------------------------------------------------------------===//
// InstanceOp
//===----------------------------------------------------------------------===//

LogicalResult InstanceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto *module =
      symbolTable.lookupNearestSymbolFrom(*this, getModuleNameAttr());
  if (module == nullptr)
    return emitError("unknown symbol name '") << getModuleName() << "'";

  // It must be some sort of module.
  if (!isa<SVModuleOp>(module))
    return emitError("symbol '")
           << getModuleName()
           << "' must reference a 'moore.module', but got a '"
           << module->getName() << "' instead";

  return success();
}

//===----------------------------------------------------------------------===//
// VariableOp
//===----------------------------------------------------------------------===//

void VariableOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), getName());
}

//===----------------------------------------------------------------------===//
// NetOp
//===----------------------------------------------------------------------===//

void NetOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), getName());
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

void ConstantOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttributeWithoutType(getValueAttr());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  p << " : ";
  p.printType(getType());
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the constant value without bit width.
  APInt value;
  auto valueLoc = parser.getCurrentLocation();

  if (parser.parseInteger(value) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon())
    return failure();

  // Parse the result type..
  UnpackedType type;
  auto typeLoc = parser.getCurrentLocation();
  if (parser.parseType(type))
    return failure();

  // Ensure that the result type is a simple bit vector type.
  auto sbvt = type.getSimpleBitVectorOrNull();
  if (!sbvt)
    return parser.emitError(typeLoc, "expected simple bit vector type");

  // Extend or truncate the constant value to match the size of the type.
  if (sbvt.size > value.getBitWidth()) {
    // sext is always safe here, even for unsigned values, because the
    // parseOptionalInteger method will return something with a zero in the
    // top bits if it is a positive number.
    value = value.sext(sbvt.size);
  } else if (sbvt.size < value.getBitWidth()) {
    // The parser can return an unnecessarily wide result with leading
    // zeros. This isn't a problem, but truncating off bits is bad.
    unsigned neededBits =
        value.isNegative() ? value.getSignificantBits() : value.getActiveBits();
    if (sbvt.size < neededBits)
      return parser.emitError(valueLoc,
                              "constant out of range for result type ")
             << type;
    value = value.trunc(sbvt.size);
  }

  // Build the attribute and op.
  auto attrType = IntegerType::get(parser.getContext(), sbvt.size);
  auto attrValue = IntegerAttr::get(attrType, value);

  result.addAttribute("value", attrValue);
  result.addTypes(type);
  return success();
}

LogicalResult ConstantOp::verify() {
  auto sbvt = getType().getSimpleBitVector();
  auto width = getValue().getBitWidth();
  if (width != sbvt.size)
    return emitError("attribute width ")
           << width << " does not match return type's width " << sbvt.size;
  return success();
}

void ConstantOp::build(OpBuilder &builder, OperationState &result, Type type,
                       const APInt &value) {
  auto sbvt = type.cast<UnpackedType>().getSimpleBitVector();
  assert(sbvt.size == value.getBitWidth() &&
         "APInt width must match simple bit vector's bit width");
  build(builder, result, type,
        builder.getIntegerAttr(builder.getIntegerType(sbvt.size), value));
}

/// This builder allows construction of small signed integers like 0, 1, -1
/// matching a specified MLIR type. This shouldn't be used for general constant
/// folding because it only works with values that can be expressed in an
/// `int64_t`.
void ConstantOp::build(OpBuilder &builder, OperationState &result, Type type,
                       int64_t value) {
  auto sbvt = type.cast<UnpackedType>().getSimpleBitVector();
  build(builder, result, type,
        APInt(sbvt.size, (uint64_t)value, /*isSigned=*/true));
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

LogicalResult ConcatOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  Domain domain = Domain::TwoValued;
  unsigned size = 0;
  for (auto operand : operands) {
    auto type = operand.getType().cast<UnpackedType>().getSimpleBitVector();
    if (type.domain == Domain::FourValued)
      domain = Domain::FourValued;
    size += type.size;
  }
  results.push_back(
      SimpleBitVectorType(domain, Sign::Unsigned, size).getType(context));
  return success();
}

//===----------------------------------------------------------------------===//
// ConversionOp
//===----------------------------------------------------------------------===//

OpFoldResult ConversionOp::fold(FoldAdaptor adaptor) {
  // Fold away no-op casts.
  if (getInput().getType() == getResult().getType())
    return getInput();
  return {};
}

//===----------------------------------------------------------------------===//
// BoolCastOp
//===----------------------------------------------------------------------===//

OpFoldResult BoolCastOp::fold(FoldAdaptor adaptor) {
  // Fold away no-op casts.
  if (getInput().getType() == getResult().getType())
    return getInput();
  return {};
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/Moore/Moore.cpp.inc"
#include "circt/Dialect/Moore/MooreEnums.cpp.inc"
