//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/CustomDirectiveImpl.h"
#include "circt/Support/Naming.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace circt;
using namespace circt::synth::mig;
using namespace circt::synth::aig;

#define GET_OP_CLASSES
#include "circt/Dialect/Synth/Synth.cpp.inc"

LogicalResult MajorityInverterOp::verify() {
  if (getNumOperands() % 2 != 1)
    return emitOpError("requires an odd number of operands");

  return success();
}

llvm::APInt MajorityInverterOp::evaluate(ArrayRef<APInt> inputs) {
  assert(inputs.size() == getNumOperands() &&
         "Number of inputs must match number of operands");

  if (inputs.size() == 3) {
    auto a = (isInverted(0) ? ~inputs[0] : inputs[0]);
    auto b = (isInverted(1) ? ~inputs[1] : inputs[1]);
    auto c = (isInverted(2) ? ~inputs[2] : inputs[2]);
    return (a & b) | (a & c) | (b & c);
  }

  // General case for odd number of inputs != 3
  auto width = inputs[0].getBitWidth();
  APInt result(width, 0);

  for (size_t bit = 0; bit < width; ++bit) {
    size_t count = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      // Count the number of 1s, considering inversion.
      if (isInverted(i) ^ inputs[i][bit])
        count++;
    }

    if (count > inputs.size() / 2)
      result.setBit(bit);
  }

  return result;
}

OpFoldResult MajorityInverterOp::fold(FoldAdaptor adaptor) {
  // TODO: Implement maj(x, 1, 1) = 1, maj(x, 0, 0) = 0

  SmallVector<APInt, 3> inputValues;
  SmallVector<size_t, 3> nonConstantValues;
  for (auto [i, input] : llvm::enumerate(adaptor.getInputs())) {
    auto attr = llvm::dyn_cast_or_null<IntegerAttr>(input);
    if (attr)
      inputValues.push_back(attr.getValue());
    else
      nonConstantValues.push_back(i);
  }

  if (nonConstantValues.size() == 0)
    return IntegerAttr::get(getType(), evaluate(inputValues));

  if (getNumOperands() != 3)
    return {};

  // Return if the idx-th operand is a constant (inverted if necessary),
  // otherwise return std::nullopt.
  auto getConstant = [&](unsigned index) -> std::optional<llvm::APInt> {
    APInt value;
    if (mlir::matchPattern(getInputs()[index], mlir::m_ConstantInt(&value)))
      return isInverted(index) ? ~value : value;
    return std::nullopt;
  };
  if (nonConstantValues.size() == 1) {
    auto k = nonConstantValues[0]; // for 3 operands
    auto i = (k + 1) % 3;
    auto j = (k + 2) % 3;
    auto c1 = getConstant(i);
    auto c2 = getConstant(j);
    // x 0 0 -> 0
    // x 1 1 -> 1
    // x 0 ~0 -> x
    // x 1 ~1 -> x
    // x ~1 ~1 -> ~1 -> 0
    // ~x 0 0 -> ~x  no fold
    // ~x 0 ~1 -> 0
    if (c1 == c2) {
      if (isInverted(i) != isInverted(j)) {
        if (isInverted(i))
          return getOperand(j);
        else
          return getOperand(i);
      }
      if (isInverted(i)) {
        // return the inverted value
        auto value = cast<IntegerAttr>(adaptor.getInputs()[i]).getValue();
        value = ~value;
        return IntegerAttr::get(
            IntegerType::get(getContext(), value.getBitWidth()), value);
      } else
        return getOperand(i);
    } else {
      if (isInverted(k))
        return {};
      else
        return getOperand(k);
    }
  }
  // else if (nonConstantValues.size() == 2) {
  //   // x x 1 -> x
  //   // x ~x 1 -> 1
  //   // ~x ~x 1 -> ~x
  //   auto k = 3 - (nonConstantValues[0] + nonConstantValues[1]);
  //   auto i = nonConstantValues[0];
  //   auto j = nonConstantValues[1];
  //   auto c1 = adaptor.getInputs()[i];
  //   auto c2 = adaptor.getInputs()[j];
  //   if (c1 == c2) {
  //     if (isInverted(i) != isInverted(j)) {
  //       if (!isInverted(k))
  //         return getOperand(k);
  //       auto value = cast<IntegerAttr>(adaptor.getInputs()[k]).getValue();
  //       value = ~value;
  //       return IntegerAttr::get(
  //           IntegerType::get(getContext(), value.getBitWidth()), value);
  //     } else {
  //             if(isInverted(i))return {};
  //             else return getOperand(i);
  //     }
  //   }
  // }
  return {};
}

LogicalResult MajorityInverterOp::canonicalize(MajorityInverterOp op,
                                               PatternRewriter &rewriter) {
  if (op.getNumOperands() == 1) {
    if (op.getInverted()[0])
      return failure();
    rewriter.replaceOp(op, op.getOperand(0));
    return success();
  }

  // For now, only support 3 operands.
  if (op.getNumOperands() != 3)
    return failure();

  // Replace the op with the idx-th operand (inverted if necessary).
  auto replaceWithIndex = [&](int index) {
    bool inverted = op.isInverted(index);
    if (inverted)
      rewriter.replaceOpWithNewOp<MajorityInverterOp>(
          op, op.getType(), op.getOperand(index), true);
    else
      rewriter.replaceOp(op, op.getOperand(index));
    return success();
  };

  // Pattern match following cases:
  // maj_inv(x, x, y) -> x
  // maj_inv(x, y, not y) -> x
  for (int i = 0; i < 2; ++i) {
    for (int j = i + 1; j < 3; ++j) {
      int k = 3 - (i + j);
      assert(k >= 0 && k < 3);
      // If we have two identical operands, we can fold.
      if (op.getOperand(i) == op.getOperand(j)) {
        // If they are inverted differently, we can fold to the third.
        if (op.isInverted(i) != op.isInverted(j)) {
          return replaceWithIndex(k);
        }
        rewriter.replaceOp(op, op.getOperand(i));
        return success();
      }
    }
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// AIG Operations
//===----------------------------------------------------------------------===//

OpFoldResult AndInverterOp::fold(FoldAdaptor adaptor) {
  if (getNumOperands() == 1 && !isInverted(0))
    return getOperand(0);

  auto inputs = adaptor.getInputs();
  if (inputs.size() == 2 && inputs[1]) {
    auto value = cast<IntegerAttr>(inputs[1]).getValue();
    if (isInverted(1))
      value = ~value;
    if (value.isZero())
      return IntegerAttr::get(
          IntegerType::get(getContext(), value.getBitWidth()), value);
    if (value.isAllOnes()) {
      if (isInverted(0))
        return {};

      return getOperand(0);
    }
  }
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

    if (auto andInverterOp = value.getDefiningOp<synth::aig::AndInverterOp>()) {
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
  replaceOpWithNewOpAndCopyNamehint<synth::aig::AndInverterOp>(
      rewriter, op, uniqueValues, uniqueInverts);
  return success();
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

static Value lowerVariadicAndInverterOp(AndInverterOp op, OperandRange operands,
                                        ArrayRef<bool> inverts,
                                        PatternRewriter &rewriter) {
  switch (operands.size()) {
  case 0:
    assert(0 && "cannot be called with empty operand range");
    break;
  case 1:
    if (inverts[0])
      return AndInverterOp::create(rewriter, op.getLoc(), operands[0], true);
    else
      return operands[0];
  case 2:
    return AndInverterOp::create(rewriter, op.getLoc(), operands[0],
                                 operands[1], inverts[0], inverts[1]);
  default:
    auto firstHalf = operands.size() / 2;
    auto lhs =
        lowerVariadicAndInverterOp(op, operands.take_front(firstHalf),
                                   inverts.take_front(firstHalf), rewriter);
    auto rhs =
        lowerVariadicAndInverterOp(op, operands.drop_front(firstHalf),
                                   inverts.drop_front(firstHalf), rewriter);
    return AndInverterOp::create(rewriter, op.getLoc(), lhs, rhs);
  }
  return Value();
}

LogicalResult circt::synth::AndInverterVariadicOpConversion::matchAndRewrite(
    AndInverterOp op, PatternRewriter &rewriter) const {
  if (op.getInputs().size() <= 2)
    return failure();
  // TODO: This is a naive implementation that creates a balanced binary tree.
  //       We can improve by analyzing the dataflow and creating a tree that
  //       improves the critical path or area.
  rewriter.replaceOp(op, lowerVariadicAndInverterOp(
                             op, op.getOperands(), op.getInverted(), rewriter));
  return success();
}

LogicalResult circt::synth::topologicallySortGraphRegionBlocks(
    mlir::Operation *op,
    llvm::function_ref<bool(mlir::Value, mlir::Operation *)> isOperandReady) {
  // Sort the operations topologically
  auto walkResult = op->walk([&](Region *region) {
    auto regionKindOp =
        dyn_cast<mlir::RegionKindInterface>(region->getParentOp());
    if (!regionKindOp ||
        regionKindOp.hasSSADominance(region->getRegionNumber()))
      return WalkResult::advance();

    // Graph region.
    for (auto &block : *region) {
      if (!mlir::sortTopologically(&block, isOperandReady))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return success(!walkResult.wasInterrupted());
}
