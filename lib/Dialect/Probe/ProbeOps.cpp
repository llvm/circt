//===- ProbeOps.cpp - Probe dialect operations ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Probe/ProbeOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Probe/ProbeDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace probe;
using namespace mlir;

static Type getElementType(Type type) {
  return cast<RefType>(type).getElementType();
}

LogicalResult SendOp::inferReturnTypes(MLIRContext *context,
                                       std::optional<Location> location,
                                       ValueRange operands,
                                       DictionaryAttr attributes,
                                       mlir::PropertyRef properties,
                                       mlir::RegionRange regions,
                                       SmallVectorImpl<Type> &inferredReturnTypes) {
  if (operands.empty())
    return failure();
  inferredReturnTypes.push_back(RefType::get(operands[0].getType()));
  return success();
}

LogicalResult SendOp::verify() {
  if (getElementType(getResult().getType()) != getInput().getType())
    return emitOpError("result probe element type must match input type");
  return success();
}

LogicalResult ReadOp::inferReturnTypes(MLIRContext *context,
                                       std::optional<Location> location,
                                       ValueRange operands,
                                       DictionaryAttr attributes,
                                       mlir::PropertyRef properties,
                                       mlir::RegionRange regions,
                                       SmallVectorImpl<Type> &inferredReturnTypes) {
  if (operands.empty())
    return failure();
  auto inputType = dyn_cast<RefType>(operands[0].getType());
  if (!inputType) {
    if (location)
      return mlir::emitError(*location, "input must be a probe reference");
    return failure();
  }

  inferredReturnTypes.push_back(inputType.getElementType());
  return success();
}

LogicalResult ReadOp::verify() {
  if (getElementType(getInput().getType()) != getResult().getType())
    return emitOpError("result type must match input probe element type");
  return success();
}

LogicalResult SubfieldOp::inferReturnTypes(MLIRContext *context,
                                           std::optional<Location> location,
                                           ValueRange operands,
                                           DictionaryAttr attributes,
                                           mlir::PropertyRef properties,
                                           mlir::RegionRange regions,
                                           SmallVectorImpl<Type> &inferredReturnTypes) {
  Adaptor adaptor(operands, attributes, properties, regions);
  auto field = adaptor.getFieldAttr();
  if (!field)
    return failure();

  auto inputRefType = dyn_cast<RefType>(adaptor.getInput().getType());
  if (!inputRefType) {
    if (location)
      return mlir::emitError(*location, "input must be a probe reference");
    return failure();
  }

  auto inputType =
      hw::type_dyn_cast<hw::StructType>(inputRefType.getElementType());
  if (!inputType) {
    if (location)
      return mlir::emitError(*location,
                             "input probe element type must be an hw.struct");
    return failure();
  }

  auto fieldType = inputType.getFieldType(field);
  if (!fieldType) {
    if (location)
      return mlir::emitError(*location, "field '")
             << field.getValue() << "' not found in "
             << inputRefType.getElementType();
    return failure();
  }

  inferredReturnTypes.push_back(RefType::get(fieldType));
  return success();
}

LogicalResult SubfieldOp::verify() {
  auto inputType = hw::type_dyn_cast<hw::StructType>(
      getElementType(getInput().getType()));
  if (!inputType)
    return emitOpError("input probe element type must be an hw.struct");

  auto fieldType = inputType.getFieldType(getField());
  if (!fieldType)
    return emitOpError("field '") << getField() << "' not found in "
                                  << getElementType(getInput().getType());

  if (getElementType(getResult().getType()) != fieldType)
    return emitOpError("result probe element type must match selected field "
                       "type ")
           << fieldType;
  return success();
}

LogicalResult SubindexOp::inferReturnTypes(MLIRContext *context,
                                           std::optional<Location> location,
                                           ValueRange operands,
                                           DictionaryAttr attributes,
                                           mlir::PropertyRef properties,
                                           mlir::RegionRange regions,
                                           SmallVectorImpl<Type> &inferredReturnTypes) {
  Adaptor adaptor(operands, attributes, properties, regions);
  auto indexAttr = adaptor.getIndexAttr();
  if (!indexAttr)
    return failure();

  auto inputRefType = dyn_cast<RefType>(adaptor.getInput().getType());
  if (!inputRefType) {
    if (location)
      return mlir::emitError(*location, "input must be a probe reference");
    return failure();
  }

  auto inputType =
      hw::type_dyn_cast<hw::ArrayType>(inputRefType.getElementType());
  if (!inputType) {
    if (location)
      return mlir::emitError(*location,
                             "input probe element type must be an hw.array");
    return failure();
  }

  auto index = indexAttr.getInt();
  if (static_cast<uint64_t>(index) >= inputType.getNumElements()) {
    if (location)
      return mlir::emitError(*location, "index ")
             << index << " out of bounds for " << inputRefType.getElementType();
    return failure();
  }

  inferredReturnTypes.push_back(RefType::get(inputType.getElementType()));
  return success();
}

LogicalResult SubindexOp::verify() {
  auto inputType =
      hw::type_dyn_cast<hw::ArrayType>(getElementType(getInput().getType()));
  if (!inputType)
    return emitOpError("input probe element type must be an hw.array");

  auto index = getIndex();
  if (static_cast<uint64_t>(index) >= inputType.getNumElements())
    return emitOpError("index ") << index << " out of bounds for "
                                  << getElementType(getInput().getType());

  if (getElementType(getResult().getType()) != inputType.getElementType())
    return emitOpError("result probe element type must match array element "
                       "type ")
           << inputType.getElementType();
  return success();
}

LogicalResult CastOp::verify() {
  auto inputType = getElementType(getInput().getType());
  auto resultType = getElementType(getResult().getType());
  if (hw::getCanonicalType(inputType) != hw::getCanonicalType(resultType))
    return emitOpError("input and result probe element types must have the "
                       "same canonical HW type");
  return success();
}

void ProbeDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Probe/Probe.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "circt/Dialect/Probe/Probe.cpp.inc"
