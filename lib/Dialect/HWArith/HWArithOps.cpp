//===- HWArithOps.cpp - Implement the HW arithmetic operations ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the HW arithmetic ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HWArith/HWArithOps.h"
#include "circt/Dialect/HWArith/HWArithTypes.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/APSInt.h"

using namespace circt;
using namespace hwarith;

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

APSInt ConstantOp::getConstantValue() { return rawValueAttr().getAPSInt(); }

OpFoldResult ConstantOp::fold(ArrayRef<Attribute> constants) {
  assert(constants.empty() && "constant has no operands");
  return rawValueAttr();
}

void ConstantOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttribute(rawValueAttr());
  p.printOptionalAttrDict(getOperation()->getAttrs(),
                          /*elidedAttrs=*/{rawValueAttrName()});
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  IntegerAttr valueAttr;

  if (parser.parseAttribute(valueAttr, rawValueAttrName(result.name),
                            result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  result.addTypes(valueAttr.getType());
  return success();
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

static unsigned inferAddResultType(IntegerType::SignednessSemantics &signedness,
                                   IntegerType lhs, IntegerType rhs) {
  // the result width is never less than max(w1, w2) + 1
  unsigned resultWidth = std::max(lhs.getWidth(), rhs.getWidth()) + 1;

  if (lhs.getSignedness() == rhs.getSignedness()) {
    // max(w1, w2) + 1 in case both operands use the same signedness
    // the signedness is also identical to the operands
    signedness = lhs.getSignedness();
  } else {
    // For mixed signedness the result is always signed
    signedness = IntegerType::Signed;

    // Regarding the result width two case need to be considered:
    if ((lhs.isUnsigned() && lhs.getWidth() >= rhs.getWidth()) ||
        (rhs.isUnsigned() && rhs.getWidth() >= lhs.getWidth())) {
      // 1. the unsigned width is >= the signed width,
      // then the width needs to be increased by 1
      ++resultWidth;
    }
    // 2. the unsigned width is < the signed width,
    // then no further adjustment is needed
  }
  return resultWidth;
}

IntegerType AddOp::inferReturnType(MLIRContext *context, IntegerType lhs,
                                   IntegerType rhs) {
  IntegerType::SignednessSemantics signedness;
  unsigned resultWidth = inferAddResultType(signedness, lhs, rhs);

  return IntegerType::get(context, resultWidth, signedness);
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

IntegerType SubOp::inferReturnType(MLIRContext *context, IntegerType lhs,
                                   IntegerType rhs) {
  // The result type rules are identical to the ones for an addition
  // With one exception: all results are signed!
  IntegerType::SignednessSemantics signedness;
  unsigned resultWidth = inferAddResultType(signedness, lhs, rhs);
  signedness = IntegerType::Signed;

  return IntegerType::get(context, resultWidth, signedness);
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

static IntegerType::SignednessSemantics
getSignedInheritedSignedness(IntegerType lhs, IntegerType rhs) {
  // Signed operands are dominant and enforce a signed result
  if (lhs.getSignedness() == rhs.getSignedness()) {
    // the signedness is also identical to the operands
    return lhs.getSignedness();
  } else {
    // For mixed signedness the result is always signed
    return IntegerType::Signed;
  }
}

IntegerType MulOp::inferReturnType(MLIRContext *context, IntegerType lhs,
                                   IntegerType rhs) {
  // the result width stays the same no matter the signedness
  unsigned resultWidth = lhs.getWidth() + rhs.getWidth();
  IntegerType::SignednessSemantics signedness =
      getSignedInheritedSignedness(lhs, rhs);

  return IntegerType::get(context, resultWidth, signedness);
}

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

IntegerType DivOp::inferReturnType(MLIRContext *context, IntegerType lhs,
                                   IntegerType rhs) {
  // The result width is always at least as large as the bit width of lhs
  unsigned resultWidth = lhs.getWidth();

  // if the divisor is signed, then the result width needs to be extended by 1
  if (rhs.isSigned())
    ++resultWidth;

  IntegerType::SignednessSemantics signedness =
      getSignedInheritedSignedness(lhs, rhs);

  return IntegerType::get(context, resultWidth, signedness);
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/HWArith/HWArith.cpp.inc"
