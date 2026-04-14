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
#include "circt/Support/SATSolver.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace circt;
using namespace circt::synth;
using namespace circt::synth::aig;

#define GET_OP_CLASSES
#include "circt/Dialect/Synth/Synth.cpp.inc"

LogicalResult ChoiceOp::verify() {
  if (getNumOperands() < 1)
    return emitOpError("requires at least one operand");
  return success();
}

OpFoldResult ChoiceOp::fold(FoldAdaptor adaptor) {
  if (adaptor.getInputs().size() == 1)
    return getOperand(0);
  return {};
}

// Canonicalize a network of synth.choice operations by computing their
// transitive closure and flattening them into a single choice operation.
// This merges nested choices and deduplicates shared operands.
// Pattern matched:
//   %0 = synth.choice %x, %y, %z
//   %1 = synth.choice %0, %u
//   %2 = synth.choice %z, %v
//     =>
//   %merged = synth.choice %x, %y, %z, %u, %v
LogicalResult ChoiceOp::canonicalize(ChoiceOp op, PatternRewriter &rewriter) {
  llvm::SetVector<Value> worklist;
  llvm::SmallSetVector<Operation *, 4> visitedChoices;

  auto addToWorklist = [&](ChoiceOp choice) -> bool {
    if (choice->getBlock() == op->getBlock() && visitedChoices.insert(choice)) {
      worklist.insert(choice.getInputs().begin(), choice.getInputs().end());
      return true;
    }
    return false;
  };

  addToWorklist(op);

  bool mergedOtherChoices = false;

  // Look up and down at definitions and users.
  for (unsigned i = 0; i < worklist.size(); ++i) {
    Value val = worklist[i];
    if (auto defOp = val.getDefiningOp<synth::ChoiceOp>()) {

      if (addToWorklist(defOp))
        mergedOtherChoices = true;
    }

    for (Operation *user : val.getUsers()) {
      if (auto userChoice = llvm::dyn_cast<synth::ChoiceOp>(user)) {
        if (addToWorklist(userChoice)) {
          mergedOtherChoices = true;
        }
      }
    }
  }

  llvm::SmallVector<mlir::Value> finalOperands;
  for (Value v : worklist) {
    if (!visitedChoices.contains(v.getDefiningOp())) {
      finalOperands.push_back(v);
    }
  }

  if (!mergedOtherChoices && finalOperands.size() == op.getInputs().size())
    return llvm::failure();

  auto newChoice = synth::ChoiceOp::create(rewriter, op->getLoc(), op.getType(),
                                           finalOperands);
  for (Operation *visited : visitedChoices.takeVector())
    rewriter.replaceOp(visited, newChoice);

  for (auto value : newChoice.getInputs())
    rewriter.replaceAllUsesExcept(value, newChoice.getResult(), newChoice);

  return success();
}

//===----------------------------------------------------------------------===//
// AIG Operations
//===----------------------------------------------------------------------===//

bool AndInverterOp::areInputsPermutationInvariant() { return true; }

OpFoldResult AndInverterOp::fold(FoldAdaptor adaptor) {
  if (getNumOperands() == 1 && !isInverted(0))
    return getOperand(0);

  auto inputs = adaptor.getInputs();
  if (inputs.size() == 2)
    if (auto intAttr = dyn_cast_or_null<IntegerAttr>(inputs[1])) {
      auto value = intAttr.getValue();
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

APInt AndInverterOp::evaluateBooleanLogic(
    llvm::function_ref<const APInt &(unsigned)> getInputValue) {
  assert(getNumOperands() > 0 && "Expected non-empty input list");
  APInt result = APInt::getAllOnes(getInputValue(0).getBitWidth());
  for (auto [idx, inverted] : llvm::enumerate(getInverted())) {
    const APInt &input = getInputValue(idx);
    if (inverted)
      result &= ~input;
    else
      result &= input;
  }
  return result;
}

llvm::KnownBits AndInverterOp::computeKnownBits(
    llvm::function_ref<const llvm::KnownBits &(unsigned)> getInputKnownBits) {
  assert(getNumOperands() > 0 && "Expected non-empty input list");

  auto width = getInputKnownBits(0).getBitWidth();
  llvm::KnownBits result(width);
  result.One = APInt::getAllOnes(width);
  result.Zero = APInt::getZero(width);

  for (auto [i, inverted] : llvm::enumerate(getInverted())) {
    auto operandKnownBits = getInputKnownBits(i);
    if (inverted)
      std::swap(operandKnownBits.Zero, operandKnownBits.One);
    result &= operandKnownBits;
  }

  return result;
}

int64_t AndInverterOp::getLogicDepthCost() {
  return llvm::Log2_64_Ceil(getNumOperands());
}

uint64_t AndInverterOp::getLogicAreaCost() {
  return static_cast<uint64_t>(getNumOperands() - 1) *
         getType().getIntOrFloatBitWidth();
}

void AndInverterOp::emitCNF(
    int outVar, llvm::ArrayRef<int> inputVars,
    llvm::function_ref<void(llvm::ArrayRef<int>)> addClause,
    llvm::function_ref<int()> newVar) {
  (void)newVar;
  assert(inputVars.size() == getInputs().size() &&
         "expected one SAT variable per operand");

  SmallVector<int> inputLits;
  inputLits.reserve(inputVars.size());
  for (auto [inputVar, inverted] : llvm::zip(inputVars, getInverted())) {
    assert(inputVar > 0 && "input SAT variables must be positive");
    inputLits.push_back(inverted ? -inputVar : inputVar);
  }
  circt::addAndClauses(outVar, inputLits, addClause);
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
