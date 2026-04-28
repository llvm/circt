//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
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
using namespace circt::comb;
using namespace matchers;

#define GET_OP_CLASSES
#include "circt/Dialect/Synth/Synth.cpp.inc"

namespace {

inline llvm::KnownBits applyInversion(llvm::KnownBits value, bool inverted) {
  if (inverted)
    std::swap(value.Zero, value.One);
  return value;
}

template <typename SubType>
struct ComplementMatcher {
  SubType lhs;
  ComplementMatcher(SubType lhs) : lhs(std::move(lhs)) {}
  bool match(Operation *op) {
    auto boolOp = dyn_cast<BooleanLogicOpInterface>(op);
    return boolOp && boolOp.getInputs().size() == 1 && boolOp.isInverted(0) &&
           lhs.match(op->getOperand(0));
  }
};

template <typename SubType>
static inline ComplementMatcher<SubType> m_Complement(const SubType &subExpr) {
  return ComplementMatcher<SubType>(subExpr);
}

} // namespace

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
// AndInverterOp
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

APInt AndInverterOp::evaluateBooleanLogicWithoutInversion(
    llvm::ArrayRef<APInt> inputs) {
  assert(!inputs.empty() && "expected non-empty input list");
  APInt result = APInt::getAllOnes(inputs.front().getBitWidth());
  for (const APInt &input : inputs)
    result &= input;
  return result;
}

bool AndInverterOp::supportsNumInputs(unsigned numInputs) {
  return numInputs >= 1;
}

llvm::KnownBits AndInverterOp::computeKnownBits(
    llvm::function_ref<const llvm::KnownBits &(unsigned)> getInputKnownBits) {
  assert(getNumOperands() > 0 && "Expected non-empty input list");

  auto width = getInputKnownBits(0).getBitWidth();
  llvm::KnownBits result(width);
  result.One = APInt::getAllOnes(width);
  result.Zero = APInt::getZero(width);

  for (auto [i, inverted] : llvm::enumerate(getInverted()))
    result &= applyInversion(getInputKnownBits(i), inverted);

  return result;
}

int64_t AndInverterOp::getLogicDepthCost() {
  return llvm::Log2_64_Ceil(getNumOperands());
}

std::optional<uint64_t> AndInverterOp::getLogicAreaCost() {
  int64_t bitWidth = hw::getBitWidth(getType());
  if (bitWidth < 0)
    return std::nullopt;
  return static_cast<uint64_t>(getNumOperands() - 1) * bitWidth;
}

void AndInverterOp::emitCNFWithoutInversion(
    int outVar, llvm::ArrayRef<int> inputVars,
    llvm::function_ref<void(llvm::ArrayRef<int>)> addClause,
    llvm::function_ref<int()> newVar) {
  (void)newVar;
  circt::addAndClauses(outVar, inputVars, addClause);
}

//===----------------------------------------------------------------------===//
// XorInverterOp
//===----------------------------------------------------------------------===//

bool XorInverterOp::areInputsPermutationInvariant() { return true; }

OpFoldResult XorInverterOp::fold(FoldAdaptor adaptor) {
  // xor_inv(a) -> a
  if (getNumOperands() == 1 && !isInverted(0))
    return getOperand(0);

  auto inputs = adaptor.getInputs();
  if (inputs.size() == 2)
    if (auto intAttr = dyn_cast_or_null<IntegerAttr>(inputs[1])) {
      auto value = intAttr.getValue();
      if (isInverted(1))
        value = ~value;
      // xor_inv(a, 0000000) -> a
      if (value.isZero())
        return getOperand(0);
    }
  return {};
}

LogicalResult XorInverterOp::canonicalize(XorInverterOp op,
                                          PatternRewriter &rewriter) {

  // Map to store active (non-canceled) operands and their inversion state
  SmallMapVector<Value, bool, 4> activeOperands;

  // XOR identity is zero; accumulate all constant operands here.
  APInt constValue =
      APInt::getZero(op.getResult().getType().getIntOrFloatBitWidth());

  bool constFound = false;
  bool changed = false;

  for (auto [value, inverted] : llvm::zip(op.getInputs(), op.getInverted())) {
    Value currentValue = value;
    bool newInverted = inverted;

    // xor_inv(a, c0, c1) -> xor_inv(a, c0 ^ c1)
    // xor_inv(a, not c0) -> xor_inv(a, ~c0)
    if (auto constOp = currentValue.getDefiningOp<hw::ConstantOp>()) {
      APInt val = constOp.getValue();
      if (newInverted)
        val = ~val;
      constValue ^= val;
      constFound = true;
      continue;
    }

    // xor_inv(a, not (xor_inv/aig_inv not b)) -> xor_inv(a, b)
    Value matchedVal;
    if (newInverted &&
        matchPattern(currentValue, m_Complement(m_Any(&matchedVal)))) {
      currentValue = matchedVal;
      newInverted = false; // double inversion cancels out
      changed = true;
    }

    // xor_inv (a, a, b) -> b
    // xor_inv (a, not a, b) -> ~b
    if (activeOperands.count(currentValue)) {
      // If we see the value again, they cancel out.
      // If one was inverted and the other wasn't (x ^ ~x), it results in a '1'.
      if (activeOperands[currentValue] != newInverted)
        constValue.flipAllBits();
      activeOperands.erase(currentValue);
      changed = true;
    } else {
      activeOperands[currentValue] = newInverted;
    }
  }

  // No constants were folded and no operands cancelled out. There is nothing to
  // do.
  if (!changed && !constFound && activeOperands.size() == op.getInputs().size())
    return failure();

  // xor_inv(a, 1111111) -> xor_inv(not a)
  // xor_inv(a, c0, c1) -> xor_inv(a, c0^c1)
  if (!constValue.isZero()) {
    if (constValue.isAllOnes() && !activeOperands.empty()) {
      // Propagate ones as an inversion on the last operand.
      activeOperands.back().second = !activeOperands.back().second;
    } else {
      if (op.getInputs().size() == 2 && !op.getInverted()[1] &&
          activeOperands.size() == 1)
        return failure();
      auto constOp = hw::ConstantOp::create(rewriter, op.getLoc(), constValue);
      activeOperands.insert({constOp, false});
    }
  }

  if (activeOperands.empty()) {
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(
        op, APInt::getZero(op.getResult().getType().getIntOrFloatBitWidth()));
    return success();
  }

  replaceOpAndCopyNamehint(rewriter, op,
                           XorInverterOp::create(rewriter, op.getLoc(),
                                                 activeOperands.getArrayRef()));
  return success();
}

APInt XorInverterOp::evaluateBooleanLogicWithoutInversion(
    llvm::ArrayRef<APInt> inputs) {
  assert(!inputs.empty() && "expected non-empty input list");
  APInt result = APInt::getZero(inputs.front().getBitWidth());
  for (const APInt &input : inputs)
    result ^= input;
  return result;
}

bool XorInverterOp::supportsNumInputs(unsigned numInputs) {
  return numInputs >= 1;
}

llvm::KnownBits XorInverterOp::computeKnownBits(
    llvm::function_ref<const llvm::KnownBits &(unsigned)> getInputKnownBits) {
  assert(getNumOperands() > 0 && "Expected non-empty input list");

  llvm::KnownBits result(getInputKnownBits(0).getBitWidth());
  for (auto [i, inverted] : llvm::enumerate(getInverted()))
    result ^= applyInversion(getInputKnownBits(i), inverted);
  return result;
}

int64_t XorInverterOp::getLogicDepthCost() {
  return llvm::Log2_64_Ceil(getNumOperands());
}

std::optional<uint64_t> XorInverterOp::getLogicAreaCost() {
  int64_t bitWidth = hw::getBitWidth(getType());
  if (bitWidth < 0)
    return std::nullopt;
  return static_cast<uint64_t>(getNumOperands() - 1) * bitWidth;
}

void XorInverterOp::emitCNFWithoutInversion(
    int outVar, llvm::ArrayRef<int> inputVars,
    llvm::function_ref<void(llvm::ArrayRef<int>)> addClause,
    llvm::function_ref<int()> newVar) {
  circt::addParityClauses(outVar, inputVars, addClause, newVar);
}

//===----------------------------------------------------------------------===//
// DotOp
//===----------------------------------------------------------------------===//

ParseResult DotOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  Type resultType;
  DenseBoolArrayAttr inverted;
  NamedAttrList attrs;

  if (parseVariadicInvertibleOperands(parser, operands, resultType, inverted,
                                      attrs))
    return failure();
  if (operands.size() != 3)
    return parser.emitError(parser.getCurrentLocation())
           << "expected exactly three operands";
  if (parser.resolveOperands(operands, resultType, result.operands))
    return failure();

  result.addTypes(resultType);
  result.addAttributes(attrs);
  result.addAttribute("inverted", inverted);
  return success();
}

void DotOp::print(OpAsmPrinter &printer) {
  printer << ' ';
  printVariadicInvertibleOperands(printer, getOperation(), getOperands(),
                                  getType(), getInvertedAttr(),
                                  (*this)->getAttrDictionary());
}

LogicalResult DotOp::verify() {
  if (getInverted().size() != 3)
    return emitOpError("requires exactly three inversion flags");
  return success();
}

APInt DotOp::evaluateBooleanLogicWithoutInversion(
    llvm::ArrayRef<APInt> inputs) {
  assert(supportsNumInputs(inputs.size()) &&
         "dot expects exactly three operands");
  return evaluateDotLogic(inputs[0], inputs[1], inputs[2]);
}

bool DotOp::areInputsPermutationInvariant() { return false; }

bool DotOp::supportsNumInputs(unsigned numInputs) { return numInputs == 3; }

llvm::KnownBits DotOp::computeKnownBits(
    llvm::function_ref<const llvm::KnownBits &(unsigned)> getInputKnownBits) {
  auto x = applyInversion(getInputKnownBits(0), isInverted(0));
  auto y = applyInversion(getInputKnownBits(1), isInverted(1));
  auto z = applyInversion(getInputKnownBits(2), isInverted(2));
  return evaluateDotLogic(x, y, z);
}

std::optional<uint64_t> DotOp::getLogicAreaCost() {
  int64_t bitWidth = hw::getBitWidth(getType());
  if (bitWidth < 0)
    return std::nullopt;
  return static_cast<uint64_t>(bitWidth);
}

void DotOp::emitCNFWithoutInversion(
    int outVar, llvm::ArrayRef<int> inputVars,
    llvm::function_ref<void(llvm::ArrayRef<int>)> addClause,
    llvm::function_ref<int()> newVar) {
  assert(inputVars.size() == 3 && "expected one SAT variable per operand");
  int andVar = newVar();
  int orVar = newVar();
  // andVar = x and y
  circt::addAndClauses(andVar, {inputVars[0], inputVars[1]}, addClause);
  // orVar = z or andVar
  circt::addOrClauses(orVar, {inputVars[2], andVar}, addClause);
  // outVar = x xor orVar
  circt::addXorClauses(outVar, inputVars[0], orVar, addClause);
}

static Value lowerVariadicInvertibleOp(
    Location loc, ValueRange operands, ArrayRef<bool> inverts,
    PatternRewriter &rewriter,
    llvm::function_ref<Value(Value, bool)> createUnary,
    llvm::function_ref<Value(Value, Value, bool, bool)> createBinary) {
  switch (operands.size()) {
  case 0:
    assert(0 && "cannot be called with empty operand range");
    break;
  case 1:
    return inverts[0] ? createUnary(operands[0], true) : operands[0];
  case 2:
    return createBinary(operands[0], operands[1], inverts[0], inverts[1]);
  default:
    auto firstHalf = operands.size() / 2;
    auto lhs = lowerVariadicInvertibleOp(loc, operands.take_front(firstHalf),
                                         inverts.take_front(firstHalf),
                                         rewriter, createUnary, createBinary);
    auto rhs = lowerVariadicInvertibleOp(loc, operands.drop_front(firstHalf),
                                         inverts.drop_front(firstHalf),
                                         rewriter, createUnary, createBinary);
    return createBinary(lhs, rhs, false, false);
  }
  return Value();
}

template <typename OpTy>
LogicalResult lowerVariadicAndInverterOpConversion(OpTy op,
                                                   PatternRewriter &rewriter) {
  if (op.getInputs().size() <= 2)
    return failure();
  auto result = lowerVariadicInvertibleOp(
      op.getLoc(), op.getOperands(), op.getInverted(), rewriter,
      [&](Value input, bool invert) {
        return OpTy::create(rewriter, op.getLoc(), input, invert);
      },
      [&](Value lhs, Value rhs, bool invertLhs, bool invertRhs) {
        return OpTy::create(rewriter, op.getLoc(), lhs, rhs, invertLhs,
                            invertRhs);
      });
  replaceOpAndCopyNamehint(rewriter, op, result);
  return success();
}

void circt::synth::populateVariadicAndInverterLoweringPatterns(
    RewritePatternSet &patterns) {
  patterns.add(lowerVariadicAndInverterOpConversion<aig::AndInverterOp>);
}

void circt::synth::populateVariadicXorInverterLoweringPatterns(
    RewritePatternSet &patterns) {
  patterns.add(lowerVariadicAndInverterOpConversion<XorInverterOp>);
}

bool circt::synth::isLogicNetworkOp(Operation *op) {
  return isa<synth::BooleanLogicOpInterface, synth::ChoiceOp, comb::ExtractOp,
             comb::ReplicateOp, comb::ConcatOp>(op);
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
