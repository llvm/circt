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
#include "circt/Dialect//Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
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
  unsigned knownZero = 0;

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

    if (auto concat = value.getDefiningOp<comb::ConcatOp>()) {
      if (auto c = concat.getInputs().front().getDefiningOp<hw::ConstantOp>()) {
        if ((!inverted && c.getValue().isZero()) ||
            (inverted && c.getValue().isAllOnes())) {
          knownZero = std::max(knownZero, c.getValue().getBitWidth());
        }
      }
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

  if (knownZero != 0) {
    auto c = rewriter.create<hw::ConstantOp>(op.getLoc(), APInt(knownZero, 0));
    SmallVector<Value> newOperands;
    for (auto v : uniqueValues) {
      newOperands.push_back(rewriter.create<comb::ExtractOp>(
          op->getLoc(), v, 0, v.getType().getIntOrFloatBitWidth() - knownZero));
    }

    auto inverter = rewriter.create<aig::AndInverterOp>(
        op->getLoc(), newOperands, uniqueInverts);
    rewriter.replaceOpWithNewOp<comb::ConcatOp>(op, ArrayRef<Value>{c, inverter});
    return success();
  }

  // No change.
  if ((uniqueValues.size() == op.getInputs().size() && !flippedFound) ||
      (!constValue.isAllOnes() && !invertedConstFound &&
       uniqueValues.size() + 1 == op.getInputs().size()))
    return failure();

  if (!constValue.isAllOnes()) {
    // Partially bit blast.
    auto width = op.getType().getIntOrFloatBitWidth();
    if (width != 1 && uniqueValues.size() != 0) {
      Value newOp = rewriter.create<aig::AndInverterOp>(
          op->getLoc(), uniqueValues, uniqueInverts);

      SmallVector<Value> concatResult;
      int32_t currentBit = -1, start = 0;
      for (size_t i = 0; i <= width; ++i) {
        if (i != width && currentBit == constValue[i])
          continue;

        size_t len = i - start;
        if (currentBit == 0) {
          if (i != 0) {
            concatResult.push_back(rewriter.create<hw::ConstantOp>(
                op.getLoc(), APInt::getZero(len)));
          }
        } else {
          if (i != 0) {
            concatResult.push_back(rewriter.create<comb::ExtractOp>(
                op.getLoc(), newOp, start, len));
          }
        }
        start = i;
        if (i != width)
          currentBit = constValue[i];
      }
      std::reverse(concatResult.begin(), concatResult.end());
      rewriter.replaceOpWithNewOp<comb::ConcatOp>(op, op.getType(),
                                                  concatResult);
      return success();
    } else {
      auto constOp = rewriter.create<hw::ConstantOp>(op.getLoc(), constValue);
      uniqueInverts.push_back(false);
      uniqueValues.push_back(constOp);
    }
  }

  // It means the input is reduced to all ones.
  if (uniqueValues.empty()) {
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, constValue);
    return success();
  }

  // build new op with reduced input values
  rewriter.replaceOpWithNewOp<aig::AndInverterOp>(op, uniqueValues,
                                                  uniqueInverts);
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

LogicalResult CutOp::verify() {
  auto *block = getBody();
  // NOTE: Currently input and output types of the block must be exactly the
  // same. We might want to relax this in the future as a way to represent
  // "vectorized" cuts. For example in the following cut, the block arguments
  // types are i1, but the cut is batch-applied over 8-bit lanes.
  // %0 = aig.cut %a, %b : (i8, i8) -> (i8) {
  //   ^bb0(%arg0: i1, %arg1: i1):
  //     %c = aig.and_inv %arg0, not %arg1 : i1
  //     aig.output %c : i1
  // }

  if (getInputs().size() != block->getNumArguments())
    return emitOpError("the number of inputs and the number of block arguments "
                       "do not match. Expected ")
           << getInputs().size() << " but got " << block->getNumArguments();

  // Check input types.
  for (auto [input, arg] : llvm::zip(getInputs(), block->getArguments()))
    if (input.getType() != arg.getType())
      return emitOpError("input type ")
             << input.getType() << " does not match " << "block argument type "
             << arg.getType();

  if (getNumResults() != block->getTerminator()->getNumOperands())
    return emitOpError("the number of results and the number of terminator "
                       "operands do not match. Expected ")
           << getNumResults() << " but got "
           << block->getTerminator()->getNumOperands();

  // Check output types.
  for (auto [result, arg] :
       llvm::zip(getResults(), block->getTerminator()->getOperands()))
    if (result.getType() != arg.getType())
      return emitOpError("result type ")
             << result.getType() << " does not match "
             << "terminator operand type " << arg.getType();

  return success();
}
