//===- AIGOps.cpp - AIG Dialect Operations ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the AIG ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/Naming.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace circt;
using namespace circt::aig;

#define GET_OP_CLASSES
#include "circt/Dialect/AIG/AIG.cpp.inc"

OpFoldResult AndInverterOp::fold(FoldAdaptor adaptor) {
  if (getNumOperands() == 1 && !isInverted(0))
    return getOperand(0);
  return {};
}

LogicalResult AndInverterOp::canonicalize(AndInverterOp op,
                                          PatternRewriter &rewriter) {
  SmallDenseMap<Value, bool> seen;
  SmallVector<Value> uniqueValues;
  SmallVector<bool> uniqueInverts;

  APInt constValue =
      APInt::getAllOnes(op.getResult().getType().getIntOrFloatBitWidth());

  bool invertedConstFound = false;
  bool flippedFound = false;

  for (auto [value, inverted] : llvm::zip(op.getInputs(), op.getInverted())) {
    bool newInverted = inverted;
    if (auto constOp = value.getDefiningOp<hw::ConstantOp>()) {
      if (inverted) {
        constValue &= ~constOp.getValue();
        invertedConstFound = true;
      } else {
        constValue &= constOp.getValue();
      }
      continue;
    }

    if (auto andInverterOp = value.getDefiningOp<aig::AndInverterOp>()) {
      if (andInverterOp.getInputs().size() == 1 &&
          andInverterOp.isInverted(0)) {
        value = andInverterOp.getOperand(0);
        newInverted = andInverterOp.isInverted(0) ^ inverted;
        flippedFound = true;
      }
    }

    auto it = seen.find(value);
    if (it == seen.end()) {
      seen.insert({value, newInverted});
      uniqueValues.push_back(value);
      uniqueInverts.push_back(newInverted);
    } else if (it->second != newInverted) {
      // replace with const 0
      rewriter.replaceOpWithNewOp<hw::ConstantOp>(
          op, APInt::getZero(value.getType().getIntOrFloatBitWidth()));
      return success();
    }
  }

  // If the constant is zero, we can just replace with zero.
  if (constValue.isZero()) {
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, constValue);
    return success();
  }

  // No change.
  if ((uniqueValues.size() == op.getInputs().size() && !flippedFound) ||
      (!constValue.isAllOnes() && !invertedConstFound &&
       uniqueValues.size() + 1 == op.getInputs().size()))
    return failure();

  if (!constValue.isAllOnes()) {
    auto constOp = hw::ConstantOp::create(rewriter, op.getLoc(), constValue);
    uniqueInverts.push_back(false);
    uniqueValues.push_back(constOp);
  }

  // It means the input is reduced to all ones.
  if (uniqueValues.empty()) {
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, constValue);
    return success();
  }

  // build new op with reduced input values
  replaceOpWithNewOpAndCopyNamehint<aig::AndInverterOp>(
      rewriter, op, uniqueValues, uniqueInverts);
  return success();
}

ParseResult AndInverterOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<bool> inverts;
  auto loc = parser.getCurrentLocation();

  while (true) {
    inverts.push_back(succeeded(parser.parseOptionalKeyword("not")));
    operands.push_back(OpAsmParser::UnresolvedOperand());

    if (parser.parseOperand(operands.back()))
      return failure();
    if (parser.parseOptionalComma())
      break;
  }

  Type type;
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseCustomTypeWithFallback(type))
    return failure();

  result.addTypes({type});
  result.addAttribute("inverted",
                      parser.getBuilder().getDenseBoolArrayAttr(inverts));
  if (parser.resolveOperands(operands, type, loc, result.operands))
    return failure();
  return success();
}

void AndInverterOp::print(OpAsmPrinter &odsPrinter) {
  odsPrinter << ' ';
  llvm::interleaveComma(llvm::zip(getInverted(), getInputs()), odsPrinter,
                        [&](auto &&pair) {
                          auto [invert, input] = pair;
                          if (invert) {
                            odsPrinter << "not ";
                          }
                          odsPrinter << input;
                        });
  odsPrinter.printOptionalAttrDict((*this)->getAttrs(), {"inverted"});
  odsPrinter << " : " << getResult().getType();
}

APInt AndInverterOp::evaluate(ArrayRef<APInt> inputs) {
  assert(inputs.size() == getNumOperands() &&
         "Expected as many inputs as operands");
  assert(!inputs.empty() && "Expected non-empty input list");
  APInt result = APInt::getAllOnes(inputs.front().getBitWidth());
  for (auto [idx, input] : llvm::enumerate(inputs)) {
    if (isInverted(idx))
      result &= ~input;
    else
      result &= input;
  }
  return result;
}
