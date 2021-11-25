//===- APIntOps.cpp - Implement the APInt operations ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the lossless arbitrary precision integer arithmetic ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/APInt/APIntOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/APSInt.h"

using namespace circt;
using namespace apint;

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

LogicalResult AddOp::inferReturnTypes(MLIRContext *context,
                                      Optional<Location> loc,
                                      ValueRange operands, DictionaryAttr attrs,
                                      mlir::RegionRange regions,
                                      SmallVectorImpl<Type> &results) {

  auto lhs = operands[0].getType().cast<IntegerType>();
  auto rhs = operands[1].getType().cast<IntegerType>();

  // Bit width rules are taken from Vivado Design Suite User Guide:
  // High-Level Synthesis (v2020.1) page 241
  // https://www.xilinx.com/content/dam/xilinx/support/documentation/sw_manuals/xilinx2020_1/ug902-vivado-high-level-synthesis.pdf

  // the result width never less than max(w1, w2) + 1
  unsigned resultWidth = std::max(lhs.getWidth(), rhs.getWidth()) + 1;
  IntegerType::SignednessSemantics signedness;

  if (lhs.isSigned() == rhs.isSigned()) {
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

  results.push_back(IntegerType::get(context, resultWidth, signedness));

  return success();
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

LogicalResult MulOp::inferReturnTypes(MLIRContext *context,
                                      Optional<Location> loc,
                                      ValueRange operands, DictionaryAttr attrs,
                                      mlir::RegionRange regions,
                                      SmallVectorImpl<Type> &results) {

  auto lhs = operands[0].getType().cast<IntegerType>();
  auto rhs = operands[1].getType().cast<IntegerType>();

  // Bit width rules are taken from Vivado Design Suite User Guide:
  // High-Level Synthesis (v2020.1) page 242
  // https://www.xilinx.com/content/dam/xilinx/support/documentation/sw_manuals/xilinx2020_1/ug902-vivado-high-level-synthesis.pdf

  // the result width stays the same no matter the signedness
  const unsigned resultWidth = lhs.getWidth() + rhs.getWidth();
  IntegerType::SignednessSemantics signedness;

  if (lhs.isSigned() == rhs.isSigned()) {
    // the signedness is also identical to the operands
    signedness = lhs.getSignedness();
  } else {
    // For mixed signedness the result is always signed
    signedness = IntegerType::Signed;
  }

  results.push_back(IntegerType::get(context, resultWidth, signedness));

  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

APSInt ConstantOp::getValue() {
  return (*this)->getAttrOfType<IntegerAttr>("value").getAPSInt();
}

static void printConstantOp(OpAsmPrinter &p, ConstantOp &op) {
  p << " ";
  p.printAttribute(op.valueAttr());
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"value"});
}

static ParseResult parseConstantOp(OpAsmParser &parser,
                                   OperationState &result) {
  IntegerAttr valueAttr;

  if (parser.parseAttribute(valueAttr, "value", result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  result.addTypes(valueAttr.getType());
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/APInt/APInt.cpp.inc"
