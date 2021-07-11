//===- CombOps.cpp - Implement the Comb operations ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements combinational ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

using namespace circt;
using namespace comb;

/// Return true if the specified type is a signless non-zero width integer type,
/// the only type which the comb ops operate.
static bool isCombIntegerType(mlir::Type type) {
  Type canonicalType;
  if (auto typeAlias = type.dyn_cast<hw::TypeAliasType>())
    canonicalType = typeAlias.getCanonicalType();
  else
    canonicalType = type;

  auto intType = canonicalType.dyn_cast<IntegerType>();
  if (!intType || !intType.isSignless())
    return false;

  return intType.getWidth() != 0;
}

//===----------------------------------------------------------------------===//
// ICmpOp
//===----------------------------------------------------------------------===//

ICmpPredicate ICmpOp::getFlippedPredicate(ICmpPredicate predicate) {
  switch (predicate) {
  case ICmpPredicate::eq:
    return ICmpPredicate::eq;
  case ICmpPredicate::ne:
    return ICmpPredicate::ne;
  case ICmpPredicate::slt:
    return ICmpPredicate::sgt;
  case ICmpPredicate::sle:
    return ICmpPredicate::sge;
  case ICmpPredicate::sgt:
    return ICmpPredicate::slt;
  case ICmpPredicate::sge:
    return ICmpPredicate::sle;
  case ICmpPredicate::ult:
    return ICmpPredicate::ugt;
  case ICmpPredicate::ule:
    return ICmpPredicate::uge;
  case ICmpPredicate::ugt:
    return ICmpPredicate::ult;
  case ICmpPredicate::uge:
    return ICmpPredicate::ule;
  }
  llvm_unreachable("unknown comparison predicate");
}

/// Return true if this is an equality test with -1, which is a "reduction
/// and" operation in Verilog.
bool ICmpOp::isEqualAllOnes() {
  if (predicate() != ICmpPredicate::eq)
    return false;

  if (auto op1 =
          dyn_cast_or_null<hw::ConstantOp>(getOperand(1).getDefiningOp()))
    return op1.getValue().isAllOnesValue();
  return false;
}

/// Return true if this is a not equal test with 0, which is a "reduction
/// or" operation in Verilog.
bool ICmpOp::isNotEqualZero() {
  if (predicate() != ICmpPredicate::ne)
    return false;

  if (auto op1 =
          dyn_cast_or_null<hw::ConstantOp>(getOperand(1).getDefiningOp()))
    return op1.getValue().isNullValue();
  return false;
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

static LogicalResult verifySExtOp(SExtOp op) {
  // The source must be equal or smaller than the dest type.  Both are already
  // known to be signless integers.
  auto srcType = op.getOperand().getType().cast<IntegerType>();
  if (srcType.getWidth() > op.getType().getWidth()) {
    op.emitOpError("extension must increase bitwidth of operand");
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Variadic operations
//===----------------------------------------------------------------------===//

static LogicalResult verifyUTVariadicOp(Operation *op) {
  if (op->getOperands().empty())
    return op->emitOpError("requires 1 or more args");

  return success();
}

/// Return true if this is a two operand xor with an all ones constant as its
/// RHS operand.
bool XorOp::isBinaryNot() {
  if (getNumOperands() != 2)
    return false;
  if (auto cst =
          dyn_cast_or_null<hw::ConstantOp>(getOperand(1).getDefiningOp()))
    if (cst.getValue().isAllOnesValue())
      return true;
  return false;
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

static unsigned getTotalWidth(ValueRange inputs)
{
  unsigned resultWidth = 0;
  for (auto input : inputs) {
    resultWidth += input.getType().cast<IntegerType>().getWidth();
  }
  return resultWidth;
}

void ConcatOp::build(OpBuilder &builder, OperationState &result,
                     ValueRange inputs) {
  build(builder, result, builder.getIntegerType(getTotalWidth(inputs)), inputs);
}

void ConcatOp::build(OpBuilder&builder, OperationState &result, Value hd, ValueRange tl)
{
  result.addOperands(ValueRange { hd });
  result.addOperands(tl);
  unsigned hdWidth = hd.getType().cast<IntegerType>().getWidth();
  result.addTypes(builder.getIntegerType(getTotalWidth(tl) + hdWidth));
}

//===----------------------------------------------------------------------===//
// Other Operations
//===----------------------------------------------------------------------===//

static LogicalResult verifyExtractOp(ExtractOp op) {
  unsigned srcWidth = op.input().getType().cast<IntegerType>().getWidth();
  unsigned dstWidth = op.getType().getWidth();
  if (op.lowBit() >= srcWidth || srcWidth - op.lowBit() < dstWidth)
    return op.emitOpError("from bit too large for input"), failure();

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/Comb/Comb.cpp.inc"
