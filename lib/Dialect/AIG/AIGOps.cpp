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
  size_t numConstInputs = 0;

  for (auto [value, inverted] : llvm::zip(op.getInputs(), op.getInverted())) {
    if (auto constOp = value.getDefiningOp<hw::ConstantOp>()) {
      numConstInputs++;
      if (inverted) {
        constValue &= ~constOp.getValue();
        invertedConstFound = true;
      } else {
        constValue &= constOp.getValue();
      }
      continue;
    }

    auto it = seen.find(value);
    if (it == seen.end()) {
      seen.insert({value, inverted});
      uniqueValues.push_back(value);
      uniqueInverts.push_back(inverted);
    } else if (it->second != inverted) {
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
  if (uniqueValues.size() == op.getInputs().size() ||
      (!constValue.isAllOnes() && !invertedConstFound &&
       uniqueValues.size() + 1 == op.getInputs().size()))
    return failure();

  if (!constValue.isAllOnes()) {
    auto constOp = rewriter.create<hw::ConstantOp>(op.getLoc(), constValue);
    uniqueInverts.push_back(false);
    uniqueValues.push_back(constOp);
  }

  // It means the input is reduced to all ones.
  if (uniqueValues.size() == 0) {
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, constValue);
    return success();
  }

  // build new op with reduced input values
  rewriter.replaceOpWithNewOp<aig::AndInverterOp>(op, uniqueValues,
                                                  uniqueInverts);
  return success();
}

mlir::ParseResult AndInverterOp::parse(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
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

  mlir::Type type;
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

void AndInverterOp::print(mlir::OpAsmPrinter &odsPrinter) {
  odsPrinter << ' ';
  llvm::interleaveComma(llvm::zip(getInverted(), getInputs()), odsPrinter,
                        [&](auto &&pair) {
                          auto [invert, input] = pair;
                          if (invert) {
                            odsPrinter << "not ";
                          }
                          odsPrinter << input;
                        });
  llvm::SmallVector<llvm::StringRef, 2> elidedAttrs{"inverted"};
  odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  odsPrinter << " : " << getResult().getType();
}

APInt AndInverterOp::evaluate(ArrayRef<APInt> inputs) {
  assert(inputs.size() == getNumOperands() &&
         "Expected as many inputs as operands");
  assert(inputs.size() != 0 && "Expected non-empty input list");
  APInt result = APInt::getAllOnes(inputs.front().getBitWidth());
  for (auto [idx, input] : llvm::enumerate(inputs)) {
    if (isInverted(idx))
      result &= ~input;
    else
      result &= input;
  }
  return result;
}

void CutOp::build(OpBuilder &builder, OperationState &result,
                  TypeRange resultTypes, ValueRange inputs,
                  std::function<void(mlir::Block::BlockArgListType)> ctor) {
  OpBuilder::InsertionGuard guard(builder);

  auto *block = builder.createBlock(result.addRegion());
  result.addTypes(resultTypes);
  result.addOperands(inputs);
  for (auto input : inputs)
    block->addArgument(input.getType(), input.getLoc());

  if (ctor)
    ctor(block->getArguments());
}

LogicalResult CutOp::verify() {
  if (getInputs().size() != getBodyBlock()->getNumArguments())
    return emitOpError("the number of inputs and the number of block arguments "
                       "do not match. Expected ")
           << getInputs().size() << " but got "
           << getBodyBlock()->getNumArguments();
  if (getNumResults() != getBodyBlock()->getTerminator()->getNumOperands())
    return emitOpError("the number of results and the number of terminator "
                       "operands do not match. Expected ")
           << getNumResults() << " but got "
           << getBodyBlock()->getTerminator()->getNumOperands();
  return success();
}
