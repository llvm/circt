//===- LTLOps.cpp ==-------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/LTL/LTLTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace circt;
using namespace ltl;
using namespace mlir;

#define GET_OP_CLASSES
#include "circt/Dialect/LTL/LTL.cpp.inc"

//===----------------------------------------------------------------------===//
// ClockOp / DisableOp
//===----------------------------------------------------------------------===//

LogicalResult
ClockOp::inferReturnTypes(MLIRContext *context, std::optional<Location> loc,
                          ValueRange operands, DictionaryAttr attributes,
                          OpaqueProperties properties, RegionRange regions,
                          SmallVectorImpl<Type> &inferredReturnTypes) {
  auto type = operands[0].getType();

  if (isa<IntegerType>(type)) {
    inferredReturnTypes.push_back(ClockedSequenceType::get(context));
  } else if (isa<PropertyType>(type)) {
    inferredReturnTypes.push_back(ClockedPropertyType::get(context));
  } else if (isa<DisabledPropertyType>(type)) {
    inferredReturnTypes.push_back(ClockedDisabledPropertyType::get(context));
  } else {
    inferredReturnTypes.push_back(ClockedSequenceType::get(context));
  }
  return success();
}

LogicalResult
DisableOp::inferReturnTypes(MLIRContext *context, std::optional<Location> loc,
                            ValueRange operands, DictionaryAttr attributes,
                            OpaqueProperties properties, RegionRange regions,
                            SmallVectorImpl<Type> &inferredReturnTypes) {
  auto type = operands[0].getType();
  if (isa<IntegerType>(type) || isa<PropertyType>(type)) {
    inferredReturnTypes.push_back(DisabledPropertyType::get(context));
  } else if (isa<ClockedPropertyType>(type)) {
    inferredReturnTypes.push_back(ClockedDisabledPropertyType::get(context));
  } else if (isa<ClockedSequenceType>(type)) {
    inferredReturnTypes.push_back(ClockedDisabledPropertyType::get(context));
  } else {
    inferredReturnTypes.push_back(DisabledPropertyType::get(context));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Property and Sequence logical ops
//===----------------------------------------------------------------------===//

static LogicalResult
inferPropertyLikeReturnTypes(MLIRContext *context, ValueRange operands,
                             SmallVectorImpl<Type> &results) {
  bool clocked = llvm::any_of(
      operands, [](auto operand) { return isClocked(operand.getType()); });
  bool disabled = llvm::any_of(
      operands, [](auto operand) { return isDisabled(operand.getType()); });

  if (clocked) {
    if (disabled) {
      results.push_back(ClockedDisabledPropertyType::get(context));
    } else {
      results.push_back(ClockedPropertyType::get(context));
    }

  } else {
    if (disabled) {
      results.push_back(DisabledPropertyType::get(context));
    } else {
      results.push_back(PropertyType::get(context));
    }
  }
  return success();
}

static LogicalResult
inferSequenceLikeReturnTypes(MLIRContext *context, ValueRange operands,
                             SmallVectorImpl<Type> &results) {
  bool clocked = llvm::any_of(
      operands, [](auto operand) { return isClocked(operand.getType()); });
  bool disabled = llvm::any_of(
      operands, [](auto operand) { return isDisabled(operand.getType()); });
  bool prop = llvm::any_of(
      operands, [](auto operand) { return isProperty(operand.getType()); });

  if (clocked) {
    if (disabled) {
      results.push_back(ClockedDisabledPropertyType::get(context));
    } else {
      if (prop) {
        results.push_back(ClockedPropertyType::get(context));
      } else {
        results.push_back(ClockedSequenceType::get(context));
      }
    }

  } else {
    if (disabled) {
      results.push_back(DisabledPropertyType::get(context));
    } else {
      if (prop) {
        results.push_back(PropertyType::get(context));
      } else {
        results.push_back(SequenceType::get(context));
      }
    }
  }
  return success();
}

LogicalResult
DelayOp::inferReturnTypes(MLIRContext *context, std::optional<Location> loc,
                          ValueRange operands, DictionaryAttr attributes,
                          OpaqueProperties properties, RegionRange regions,
                          SmallVectorImpl<Type> &inferredReturnTypes) {
  return inferSequenceLikeReturnTypes(context, operands, inferredReturnTypes);
}

LogicalResult
ConcatOp::inferReturnTypes(MLIRContext *context, std::optional<Location> loc,
                           ValueRange operands, DictionaryAttr attributes,
                           OpaqueProperties properties, RegionRange regions,
                           SmallVectorImpl<Type> &inferredReturnTypes) {
  return inferSequenceLikeReturnTypes(context, operands, inferredReturnTypes);
}

LogicalResult
NotOp::inferReturnTypes(MLIRContext *context, std::optional<Location> loc,
                        ValueRange operands, DictionaryAttr attributes,
                        OpaqueProperties properties, RegionRange regions,
                        SmallVectorImpl<Type> &inferredReturnTypes) {
  return inferPropertyLikeReturnTypes(context, operands, inferredReturnTypes);
}

LogicalResult
AndOp::inferReturnTypes(MLIRContext *context, std::optional<Location> loc,
                        ValueRange operands, DictionaryAttr attributes,
                        OpaqueProperties properties, RegionRange regions,
                        SmallVectorImpl<Type> &inferredReturnTypes) {
  return inferPropertyLikeReturnTypes(context, operands, inferredReturnTypes);
}

LogicalResult
OrOp::inferReturnTypes(MLIRContext *context, std::optional<Location> loc,
                       ValueRange operands, DictionaryAttr attributes,
                       OpaqueProperties properties, RegionRange regions,
                       SmallVectorImpl<Type> &inferredReturnTypes) {
  return inferPropertyLikeReturnTypes(context, operands, inferredReturnTypes);
}

LogicalResult ImplicationOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  return inferPropertyLikeReturnTypes(context, operands, inferredReturnTypes);
}

LogicalResult EventuallyOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  return inferPropertyLikeReturnTypes(context, operands, inferredReturnTypes);
}
