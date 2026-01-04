//===CoreToFSM.cpp - Convert Core Dialects (HW + Seq + Comb) to FSM Dialect===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CoreToFSM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/LLVM.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include <string>

namespace circt {
#define GEN_PASS_DEF_CONVERTCORETOFSM
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace hw;
using namespace comb;
using namespace fsm;

namespace {

// Forward declaration for our helper function
static void generateConcatenatedValues(
    const llvm::SmallVector<llvm::DenseSet<size_t>> &allOperandValues,
    const llvm::SmallVector<unsigned> &shifts,
    llvm::DenseSet<size_t> &finalPossibleValues);

/// Internal helper with visited set to detect cycles.
static void addPossibleValuesImpl(llvm::DenseSet<size_t> &possibleValues,
                                  Value v, llvm::DenseSet<Value> &visited) {
  // Detect cycles - if we've seen this value before, skip it.
  if (!visited.insert(v).second)
    return;

  if (auto c = dyn_cast_or_null<hw::ConstantOp>(v.getDefiningOp())) {
    possibleValues.insert(c.getValueAttr().getValue().getZExtValue());
    return;
  }
  if (auto m = dyn_cast_or_null<MuxOp>(v.getDefiningOp())) {
    addPossibleValuesImpl(possibleValues, m.getTrueValue(), visited);
    addPossibleValuesImpl(possibleValues, m.getFalseValue(), visited);
    return;
  }

  if (auto concatOp = dyn_cast_or_null<ConcatOp>(v.getDefiningOp())) {
    llvm::SmallVector<llvm::DenseSet<size_t>> allOperandValues;
    llvm::SmallVector<unsigned> operandWidths;

    for (Value operand : concatOp.getOperands()) {
      llvm::DenseSet<size_t> operandPossibleValues;
      addPossibleValuesImpl(operandPossibleValues, operand, visited);

      // It's crucial to handle the case where a sub-computation is too complex.
      // If we can't determine specific values for an operand, we must
      // pessimistically assume it can be any value its bitwidth allows.
      auto opType = dyn_cast<IntegerType>(operand.getType());
      if (!opType) {
        concatOp.emitError(
            "FSM extraction only supports integer-typed operands "
            "in concat operations");
        return;
      }
      unsigned width = opType.getWidth();
      if (operandPossibleValues.empty()) {
        uint64_t numStates = 1ULL << width;
        // Add a threshold to prevent combinatorial explosion on large unknown
        // inputs.
        if (numStates > 256) { // Heuristic threshold
          // If the search space is too large, we abandon the analysis for this
          // path. The outer function will fall back to its own full-range
          // default.
          v.getDefiningOp()->emitWarning()
              << "Search space too large (>" << 256
              << " states) for operand with bitwidth " << width
              << "; abandoning analysis for this path";
          return;
        }
        for (uint64_t i = 0; i < numStates; ++i)
          operandPossibleValues.insert(i);
      }

      allOperandValues.push_back(operandPossibleValues);
      operandWidths.push_back(width);
    }

    // The shift for operand `i` is the sum of the widths of operands `i+1` to
    // `n-1`.
    llvm::SmallVector<unsigned> shifts(concatOp.getNumOperands(), 0);
    for (int i = concatOp.getNumOperands() - 2; i >= 0; --i) {
      shifts[i] = shifts[i + 1] + operandWidths[i + 1];
    }

    generateConcatenatedValues(allOperandValues, shifts, possibleValues);
    return;
  }

  // --- Fallback Case ---
  // If the operation is not recognized, assume all possible values for its
  // bitwidth.

  auto addrType = dyn_cast<IntegerType>(v.getType());
  if (!addrType)
    return; // Not an integer type we can analyze

  unsigned bitWidth = addrType.getWidth();
  // Again, use a threshold to avoid trying to enumerate 2^64 values.
  if (bitWidth > 16) {
    if (v.getDefiningOp())
      v.getDefiningOp()->emitWarning()
          << "Bitwidth " << bitWidth
          << " too large (>16); abandoning analysis for this path";
    return;
  }

  uint64_t numRegStates = 1ULL << bitWidth;
  for (size_t i = 0; i < numRegStates; i++) {
    possibleValues.insert(i);
  }
}

static void addPossibleValues(llvm::DenseSet<size_t> &possibleValues, Value v) {
  llvm::DenseSet<Value> visited;
  addPossibleValuesImpl(possibleValues, v, visited);
}

/// Checks if a value is a constant or a tree of muxes with constant leaves.
/// Uses an iterative approach with a visited set to handle cycles.
static bool isConstantLike(Value value) {
  SmallVector<Value> worklist;
  llvm::DenseSet<Value> visited;

  worklist.push_back(value);
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();

    // Skip if already visited (handles cycles).
    if (!visited.insert(current).second)
      continue;

    Operation *definingOp = current.getDefiningOp();
    if (!definingOp)
      return false;

    if (isa<hw::ConstantOp>(definingOp))
      continue;

    if (auto muxOp = dyn_cast<MuxOp>(definingOp)) {
      worklist.push_back(muxOp.getTrueValue());
      worklist.push_back(muxOp.getFalseValue());
      continue;
    }

    // Not a constant or mux - not constant-like.
    return false;
  }
  return true;
}

/// Pushes an ICmp equality comparison through a mux operation.
/// This transforms `icmp eq (mux cond, x, y), b` into
/// `mux cond, (icmp eq x, b), (icmp eq y, b)`.
/// This simplification helps expose constant comparisons that can be folded
/// during FSM extraction, making transition guards easier to analyze.
LogicalResult pushIcmp(ICmpOp op, PatternRewriter &rewriter) {
  APInt lhs, rhs;
  if (op.getPredicate() == ICmpPredicate::eq &&
      op.getLhs().getDefiningOp<MuxOp>() &&
      (isConstantLike(op.getLhs()) ||
       op.getRhs().getDefiningOp<hw::ConstantOp>())) {
    rewriter.setInsertionPointAfter(op);
    auto mux = op.getLhs().getDefiningOp<MuxOp>();
    Value x = mux.getTrueValue();
    Value y = mux.getFalseValue();
    Value b = op.getRhs();
    Location loc = op.getLoc();
    auto eq1 = rewriter.create<ICmpOp>(loc, ICmpPredicate::eq, x, b);
    auto eq2 = rewriter.create<ICmpOp>(loc, ICmpPredicate::eq, y, b);
    auto newMux = rewriter.create<MuxOp>(loc, mux.getCond(), eq1.getResult(),
                                         eq2.getResult());
    op.replaceAllUsesWith(newMux.getOperation());
    op.erase();
    return llvm::success();
  }
  if (op.getPredicate() == ICmpPredicate::eq &&
      op.getRhs().getDefiningOp<MuxOp>() &&
      (isConstantLike(op.getRhs()) ||
       op.getLhs().getDefiningOp<hw::ConstantOp>())) {
    rewriter.setInsertionPointAfter(op);
    auto mux = op.getRhs().getDefiningOp<MuxOp>();
    Value x = mux.getTrueValue();
    Value y = mux.getFalseValue();
    Value b = op.getLhs();
    Location loc = op.getLoc();
    auto eq1 = rewriter.create<ICmpOp>(loc, ICmpPredicate::eq, x, b);
    auto eq2 = rewriter.create<ICmpOp>(loc, ICmpPredicate::eq, y, b);
    auto newMux = rewriter.create<MuxOp>(loc, mux.getCond(), eq1.getResult(),
                                         eq2.getResult());
    op.replaceAllUsesWith(newMux.getOperation());
    op.erase();
    return llvm::success();
  }
  return llvm::failure();
}

/// Iteratively builds all possible concatenated integer values from the
/// Cartesian product of value sets.
static void generateConcatenatedValues(
    const llvm::SmallVector<llvm::DenseSet<size_t>> &allOperandValues,
    const llvm::SmallVector<unsigned> &shifts,
    llvm::DenseSet<size_t> &finalPossibleValues) {

  if (allOperandValues.empty()) {
    finalPossibleValues.insert(0);
    return;
  }

  // Start with the values of the first operand, shifted appropriately.
  llvm::DenseSet<size_t> currentResults;
  for (size_t val : allOperandValues[0])
    currentResults.insert(val << shifts[0]);

  // For each subsequent operand, combine with all existing partial results.
  for (size_t operandIdx = 1; operandIdx < allOperandValues.size();
       ++operandIdx) {
    llvm::DenseSet<size_t> nextResults;
    unsigned shift = shifts[operandIdx];

    for (size_t partialValue : currentResults) {
      for (size_t val : allOperandValues[operandIdx]) {
        nextResults.insert(partialValue | (val << shift));
      }
    }
    currentResults = std::move(nextResults);
  }

  finalPossibleValues = std::move(currentResults);
}

static llvm::DenseMap<Value, int> intToRegMap(SmallVector<seq::CompRegOp> v,
                                              int i) {
  llvm::DenseMap<Value, int> m;
  for (size_t ci = 0; ci < v.size(); ci++) {
    seq::CompRegOp reg = v[ci];
    int bits = reg.getType().getIntOrFloatBitWidth();
    int v = i & ((1 << bits) - 1);
    m[reg] = v;
    i = i >> bits;
  }
  return m;
}

static int regMapToInt(SmallVector<seq::CompRegOp> v,
                       llvm::DenseMap<Value, int> m) {
  int i = 0;
  int width = 0;
  for (size_t ci = 0; ci < v.size(); ci++) {
    seq::CompRegOp reg = v[ci];
    i += m[reg] * 1ULL << width;
    width += (reg.getType().getIntOrFloatBitWidth());
  }
  return i;
}

/// Computes the Cartesian product of a list of sets.
static std::set<llvm::SmallVector<size_t>> calculateCartesianProduct(
    const llvm::SmallVector<llvm::DenseSet<size_t>> &valueSets) {
  std::set<llvm::SmallVector<size_t>> product;
  if (valueSets.empty()) {
    // The Cartesian product of zero sets is a set containing one element:
    // the empty tuple (represented here by an empty vector).
    product.insert({});
    return product;
  }

  // Initialize the product with the elements of the first set, each in its
  // own vector.
  for (size_t value : valueSets.front()) {
    product.insert({value});
  }

  // Iteratively build the product. For each subsequent set, create a new
  // temporary product by appending each of its elements to every combination
  // already generated.
  for (size_t i = 1; i < valueSets.size(); ++i) {
    const auto &currentSet = valueSets[i];
    if (currentSet.empty()) {
      // The Cartesian product with an empty set results in an empty set.
      return {};
    }

    std::set<llvm::SmallVector<size_t>> newProduct;
    for (const auto &existingVector : product) {
      for (size_t newValue : currentSet) {
        llvm::SmallVector<size_t> newVector = existingVector;
        newVector.push_back(newValue);
        newProduct.insert(std::move(newVector));
      }
    }
    product = std::move(newProduct);
  }

  return product;
}

static FrozenRewritePatternSet loadPatterns(MLIRContext &context) {

  RewritePatternSet patterns(&context);
  for (auto *dialect : context.getLoadedDialects())
    dialect->getCanonicalizationPatterns(patterns);
  ICmpOp::getCanonicalizationPatterns(patterns, &context);
  AndOp::getCanonicalizationPatterns(patterns, &context);
  XorOp::getCanonicalizationPatterns(patterns, &context);
  MuxOp::getCanonicalizationPatterns(patterns, &context);
  ConcatOp::getCanonicalizationPatterns(patterns, &context);
  ExtractOp::getCanonicalizationPatterns(patterns, &context);
  AddOp::getCanonicalizationPatterns(patterns, &context);
  OrOp::getCanonicalizationPatterns(patterns, &context);
  MulOp::getCanonicalizationPatterns(patterns, &context);
  hw::ConstantOp::getCanonicalizationPatterns(patterns, &context);
  TransitionOp::getCanonicalizationPatterns(patterns, &context);
  StateOp::getCanonicalizationPatterns(patterns, &context);
  MachineOp::getCanonicalizationPatterns(patterns, &context);
  patterns.add(pushIcmp);
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  return frozenPatterns;
}

static void getReachableStates(llvm::DenseSet<size_t> &visitableStates,
                               HWModuleOp moduleOp, size_t currentStateIndex,
                               SmallVector<seq::CompRegOp> registers,
                               OpBuilder opBuilder, bool isInitialState) {

  IRMapping mapping;
  auto clonedBody =
      llvm::dyn_cast<HWModuleOp>(opBuilder.clone(*moduleOp, mapping));

  llvm::DenseMap<Value, int> stateMap =
      intToRegMap(registers, currentStateIndex);
  Operation *terminator = clonedBody.getBody().front().getTerminator();
  auto output = dyn_cast<hw::OutputOp>(terminator);
  int i = 0;
  SmallVector<Value> values;
  SmallVector<Type> types;
  llvm::DenseMap<int, Value> regMap;

  for (auto [originalRegValue, constStateValue] : stateMap) {

    Value clonedRegValue = mapping.lookup(originalRegValue);
    Operation *clonedRegOp = clonedRegValue.getDefiningOp();
    auto reg = cast<seq::CompRegOp>(clonedRegOp);
    Type constantType = reg.getType();
    IntegerAttr constantAttr =
        opBuilder.getIntegerAttr(constantType, constStateValue);
    opBuilder.setInsertionPoint(clonedRegOp);
    auto otherStateConstant =
        opBuilder.create<hw::ConstantOp>(reg.getLoc(), constantAttr);
    values.push_back(reg.getInput());
    types.push_back(reg.getType());
    clonedRegValue.replaceAllUsesWith(otherStateConstant.getResult());
    regMap[i] = originalRegValue;
    reg.erase();
    i++;
  }
  opBuilder.setInsertionPointToEnd(clonedBody.front().getBlock());
  auto newOutput = opBuilder.create<hw::OutputOp>(output.getLoc(), values);
  output.erase();
  FrozenRewritePatternSet frozenPatterns = loadPatterns(*moduleOp.getContext());

  SmallVector<Operation *> opsToProcess;
  clonedBody.walk([&](Operation *op) { opsToProcess.push_back(op); });

  bool changed = false;
  GreedyRewriteConfig config;
  LogicalResult converged =
      applyOpPatternsGreedily(opsToProcess, frozenPatterns, config, &changed);

  llvm::SmallVector<llvm::DenseSet<size_t>> pv;
  for (size_t j = 0; j < newOutput.getNumOperands(); j++) {
    llvm::DenseSet<size_t> possibleValues;

    Value v = newOutput.getOperand(j);
    addPossibleValues(possibleValues, v);
    pv.push_back(possibleValues);
  }
  std::set<llvm::SmallVector<size_t>> flipped = calculateCartesianProduct(pv);
  for (llvm::SmallVector<size_t> v : flipped) {
    llvm::DenseMap<Value, int> m;
    for (size_t k = 0; k < v.size(); k++) {
      seq::CompRegOp r = registers[k];
      m[r] = v[k];
    }

    int i = regMapToInt(registers, m);
    visitableStates.insert(i);
  }

  clonedBody.erase();
};

// A converter class to handle the logic of converting a single hw.module.
class HWModuleOpConverter {
public:
  HWModuleOpConverter(OpBuilder &builder, HWModuleOp moduleOp,
                      ArrayRef<std::string> stateRegNames)
      : moduleOp(moduleOp), opBuilder(builder), stateRegNames(stateRegNames) {}
  LogicalResult run() {
    SmallVector<seq::CompRegOp> stateRegs;
    SmallVector<seq::CompRegOp> variableRegs;
    WalkResult walkResult = moduleOp.walk([&](seq::CompRegOp reg) {
      // Check that the register type is an integer.
      if (!isa<IntegerType>(reg.getType())) {
        reg.emitError("FSM extraction only supports integer-typed registers");
        return WalkResult::interrupt();
      }
      if (isStateRegister(reg)) {
        stateRegs.push_back(reg);
      } else {
        variableRegs.push_back(reg);
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return failure();
    if (stateRegs.empty()) {
      emitError(moduleOp.getLoc())
          << "Cannot find state register in this FSM. Use the state-regs "
             "option to specify which registers are state registers.";
      return failure();
    }
    llvm::DenseMap<Value, size_t> regToIndexMap;
    int regIndex = 0;
    SmallVector<seq::CompRegOp> registers;
    for (seq::CompRegOp c : stateRegs) {
      regToIndexMap[c] = regIndex;
      regIndex++;
      registers.push_back(c);
    }

    llvm::DenseMap<size_t, StateOp> stateToStateOp;
    llvm::DenseMap<StateOp, size_t> stateOpToState;
    // Collect async reset arguments to exclude from the FSM's function type.
    // FSMs in CIRCT don't have an async reset concept, so these signals are
    // not passed through. The reset behavior is captured in the initial state.
    llvm::DenseSet<size_t> asyncResetArguments;
    auto regsInGroup = stateRegs;
    Location loc = moduleOp.getLoc();
    SmallVector<Type> inputTypes = moduleOp.getInputTypes();

    // Create a new FSM machine with the current state.
    auto resultTypes = moduleOp.getOutputTypes();
    FunctionType machineType =
        FunctionType::get(opBuilder.getContext(), inputTypes, resultTypes);
    StringRef machineName = moduleOp.getName();

    llvm::DenseMap<Value, int> initialStateMap;
    for (seq::CompRegOp reg : moduleOp.getOps<seq::CompRegOp>()) {
      Value resetValue = reg.getResetValue();
      auto definingConstant = resetValue.getDefiningOp<hw::ConstantOp>();
      if (!definingConstant) {
        reg->emitError(
            "cannot find defining constant for reset value of register");
        return failure();
      }
      int resetValueInt =
          definingConstant.getValueAttr().getValue().getZExtValue();
      initialStateMap[reg] = resetValueInt;
    }
    int initialStateIndex = regMapToInt(registers, initialStateMap);

    std::string initialStateName = "state_" + std::to_string(initialStateIndex);

    // Preserve argument and result names, which are stored as attributes.
    SmallVector<NamedAttribute> machineAttrs;
    if (auto argNames = moduleOp->getAttrOfType<ArrayAttr>("argNames"))
      machineAttrs.emplace_back(opBuilder.getStringAttr("argNames"), argNames);
    if (auto resNames = moduleOp->getAttrOfType<ArrayAttr>("resultNames"))
      machineAttrs.emplace_back(opBuilder.getStringAttr("resNames"), resNames);

    // The builder for fsm.MachineOp will create the body region and block
    // arguments.
    opBuilder.setInsertionPoint(moduleOp);
    auto machine = opBuilder.create<MachineOp>(
        loc, machineName, initialStateName, machineType, machineAttrs);

    OpBuilder::InsertionGuard guard(opBuilder);
    opBuilder.setInsertionPointToStart(&machine.getBody().front());
    llvm::DenseMap<seq::CompRegOp, VariableOp> variableMap;
    for (seq::CompRegOp varReg : variableRegs) {
      TypedValue<Type> initialValue = varReg.getResetValue();
      auto definingConstant = initialValue.getDefiningOp<hw::ConstantOp>();
      if (!definingConstant) {
        varReg->emitError("cannot find defining constant for reset value of "
                          "variable register");
        return failure();
      }
      auto variableOp = opBuilder.create<VariableOp>(
          varReg->getLoc(), varReg.getInput().getType(),
          definingConstant.getValueAttr(), varReg.getName().value_or("var"));
      variableMap[varReg] = variableOp;
    }

    // Load rewrite patterns used for canonicalizing the generated FSM.
    FrozenRewritePatternSet frozenPatterns =
        loadPatterns(*moduleOp.getContext());

    SetVector<int> reachableStates;
    SmallVector<int> worklist;

    worklist.push_back(initialStateIndex);
    reachableStates.insert(initialStateIndex);
    // Process states in BFS order. The worklist grows as new reachable states
    // are discovered, so we use an index-based loop.
    for (unsigned i = 0; i < worklist.size(); ++i) {

      int currentStateIndex = worklist[i];

      llvm::DenseMap<Value, int> stateMap =
          intToRegMap(registers, currentStateIndex);

      opBuilder.setInsertionPointToEnd(&machine.getBody().front());

      StateOp stateOp;

      if (!stateToStateOp.contains(currentStateIndex)) {
        stateOp = opBuilder.create<StateOp>(
            loc, "state_" + std::to_string(currentStateIndex));
        stateToStateOp.insert({currentStateIndex, stateOp});
        stateOpToState.insert({stateOp, currentStateIndex});
      } else {
        stateOp = stateToStateOp.lookup(currentStateIndex);
      }
      Region &outputRegion = stateOp.getOutput();
      Block *outputBlock = &outputRegion.front();
      opBuilder.setInsertionPointToStart(outputBlock);
      IRMapping mapping;
      opBuilder.cloneRegionBefore(moduleOp.getModuleBody(), outputRegion,
                                  outputBlock->getIterator(), mapping);
      outputBlock->erase();

      auto *terminator = outputRegion.front().getTerminator();
      auto hwOutputOp = dyn_cast<hw::OutputOp>(terminator);
      assert(hwOutputOp && "Expected terminator to be hw.output op");

      // Position the builder to insert the new terminator right before the
      // old one.
      OpBuilder::InsertionGuard stateGuard(opBuilder);
      opBuilder.setInsertionPoint(hwOutputOp);

      // Create the new fsm.OutputOp with the same operands.

      opBuilder.create<fsm::OutputOp>(hwOutputOp.getLoc(),
                                      hwOutputOp.getOperands());

      // Erase the old terminator.
      hwOutputOp.erase();

      // Iterate through the state configuration to replace registers
      // with constants.
      for (auto &[originalRegValue, variableOp] : variableMap) {
        Value clonedRegValue = mapping.lookup(originalRegValue);
        Operation *clonedRegOp = clonedRegValue.getDefiningOp();
        auto reg = cast<seq::CompRegOp>(clonedRegOp);
        const auto res = variableOp.getResult();
        clonedRegValue.replaceAllUsesWith(res);
        reg.erase();
      }
      for (auto const &[originalRegValue, constStateValue] : stateMap) {
        //  Find the cloned register's result value using the mapping.
        Value clonedRegValue = mapping.lookup(originalRegValue);
        assert(clonedRegValue &&
               "Original register value not found in mapping");
        Operation *clonedRegOp = clonedRegValue.getDefiningOp();

        assert(clonedRegOp && "Cloned value must have a defining op");
        opBuilder.setInsertionPoint(clonedRegOp);
        auto r = cast<seq::CompRegOp>(clonedRegOp);
        TypedValue<IntegerType> registerReset = r.getReset();
        if (registerReset) {
          if (BlockArgument blockArg = dyn_cast<BlockArgument>(registerReset)) {
            asyncResetArguments.insert(blockArg.getArgNumber());
            auto falseConst = opBuilder.create<hw::ConstantOp>(
                blockArg.getLoc(), clonedRegValue.getType(), 0);
            blockArg.replaceAllUsesWith(falseConst.getResult());
          }
          if (auto xorOp = registerReset.getDefiningOp<XorOp>()) {
            if (xorOp.isBinaryNot()) {
              Value rhs = xorOp.getOperand(0);
              if (BlockArgument blockArg = dyn_cast<BlockArgument>(rhs)) {
                asyncResetArguments.insert(blockArg.getArgNumber());
                auto trueConst = opBuilder.create<hw::ConstantOp>(
                    blockArg.getLoc(), blockArg.getType(), 1);
                blockArg.replaceAllUsesWith(trueConst.getResult());
              }
            }
          }
        }
        auto constantOp = opBuilder.create<hw::ConstantOp>(
            clonedRegValue.getLoc(), clonedRegValue.getType(), constStateValue);
        clonedRegValue.replaceAllUsesWith(constantOp.getResult());
        clonedRegOp->erase();
      }
      GreedyRewriteConfig config;
      SmallVector<Operation *> opsToProcess;
      outputRegion.walk([&](Operation *op) { opsToProcess.push_back(op); });
      // Replace references to arguments in the output block with
      // arguments at the top level.
      for (auto arg : outputRegion.front().getArguments()) {
        int argIndex = arg.getArgNumber();
        BlockArgument topLevelArg = machine.getBody().getArgument(argIndex);
        arg.replaceAllUsesWith(topLevelArg);
      }
      outputRegion.front().eraseArguments(
          [](BlockArgument arg) { return true; });
      FrozenRewritePatternSet patterns(opBuilder.getContext());
      config.setScope(&outputRegion);

      bool changed = false;
      LogicalResult converged =
          applyOpPatternsGreedily(opsToProcess, patterns, config, &changed);
      opBuilder.setInsertionPoint(stateOp);
      if (!sortTopologically(&outputRegion.front())) {
        moduleOp.emitError("could not resolve cycles in module");
        return failure();
      }
      Region &transitionRegion = stateOp.getTransitions();
      llvm::DenseSet<size_t> visitableStates;
      getReachableStates(visitableStates, moduleOp, currentStateIndex,
                         registers, opBuilder,
                         currentStateIndex == initialStateIndex);
      for (size_t j : visitableStates) {
        StateOp toState;
        if (!stateToStateOp.contains(j)) {
          opBuilder.setInsertionPointToEnd(&machine.getBody().front());
          toState =
              opBuilder.create<StateOp>(loc, "state_" + std::to_string(j));
          stateToStateOp.insert({j, toState});
          stateOpToState.insert({toState, j});
        } else {
          toState = stateToStateOp[j];
        }
        opBuilder.setInsertionPointToStart(&transitionRegion.front());
        auto transitionOp =
            opBuilder.create<TransitionOp>(loc, "state_" + std::to_string(j));
        Region &guardRegion = transitionOp.getGuard();
        opBuilder.createBlock(&guardRegion);

        Block &guardBlock = guardRegion.front();

        opBuilder.setInsertionPointToStart(&guardBlock);
        IRMapping mapping;
        opBuilder.cloneRegionBefore(moduleOp.getModuleBody(), guardRegion,
                                    guardBlock.getIterator(), mapping);
        guardBlock.erase();
        Block &newGuardBlock = guardRegion.front();
        Operation *terminator = newGuardBlock.getTerminator();
        auto hwOutputOp = dyn_cast<hw::OutputOp>(terminator);
        assert(hwOutputOp && "Expected terminator to be hw.output op");

        llvm::DenseMap<Value, int> toStateMap = intToRegMap(registers, j);
        SmallVector<Value> equalityChecks;
        for (auto &[originalRegValue, variableOp] : variableMap) {
          opBuilder.setInsertionPointToStart(&newGuardBlock);
          Value clonedRegValue = mapping.lookup(originalRegValue);
          Operation *clonedRegOp = clonedRegValue.getDefiningOp();
          auto reg = cast<seq::CompRegOp>(clonedRegOp);
          const auto res = variableOp.getResult();
          clonedRegValue.replaceAllUsesWith(res);
          reg.erase();
        }
        for (auto const &[originalRegValue, constStateValue] : toStateMap) {

          Value clonedRegValue = mapping.lookup(originalRegValue);
          Operation *clonedRegOp = clonedRegValue.getDefiningOp();
          opBuilder.setInsertionPoint(clonedRegOp);
          auto r = cast<seq::CompRegOp>(clonedRegOp);

          Value registerInput = r.getInput();
          TypedValue<IntegerType> registerReset = r.getReset();
          if (registerReset) {
            if (BlockArgument blockArg =
                    dyn_cast<BlockArgument>(registerReset)) {
              auto falseConst = opBuilder.create<hw::ConstantOp>(
                  blockArg.getLoc(), clonedRegValue.getType(), 0);
              blockArg.replaceAllUsesWith(falseConst.getResult());
            }
            if (auto xorOp = registerReset.getDefiningOp<XorOp>()) {
              if (xorOp.isBinaryNot()) {
                Value rhs = xorOp.getOperand(0);
                if (BlockArgument blockArg = dyn_cast<BlockArgument>(rhs)) {
                  auto trueConst = opBuilder.create<hw::ConstantOp>(
                      blockArg.getLoc(), blockArg.getType(), 1);
                  blockArg.replaceAllUsesWith(trueConst.getResult());
                }
              }
            }
          }
          Type constantType = registerInput.getType();
          IntegerAttr constantAttr =
              opBuilder.getIntegerAttr(constantType, constStateValue);
          auto otherStateConstant = opBuilder.create<hw::ConstantOp>(
              hwOutputOp.getLoc(), constantAttr);

          auto doesEqual = opBuilder.create<ICmpOp>(
              hwOutputOp.getLoc(), ICmpPredicate::eq, registerInput,
              otherStateConstant.getResult());
          equalityChecks.push_back(doesEqual.getResult());
        }
        opBuilder.setInsertionPoint(hwOutputOp);
        auto allEqualCheck =
            opBuilder.create<AndOp>(hwOutputOp.getLoc(), equalityChecks, false);
        opBuilder.create<fsm::ReturnOp>(hwOutputOp.getLoc(),
                                        allEqualCheck.getResult());
        hwOutputOp.erase();
        for (BlockArgument arg : newGuardBlock.getArguments()) {
          int argIndex = arg.getArgNumber();
          BlockArgument topLevelArg = machine.getBody().getArgument(argIndex);
          arg.replaceAllUsesWith(topLevelArg);
        }
        newGuardBlock.eraseArguments([](BlockArgument arg) { return true; });
        llvm::DenseMap<Value, int> fromStateMap =
            intToRegMap(registers, currentStateIndex);
        for (auto const &[originalRegValue, constStateValue] : fromStateMap) {
          Value clonedRegValue = mapping.lookup(originalRegValue);
          assert(clonedRegValue &&
                 "Original register value not found in mapping");
          Operation *clonedRegOp = clonedRegValue.getDefiningOp();
          assert(clonedRegOp && "Cloned value must have a defining op");
          opBuilder.setInsertionPoint(clonedRegOp);
          auto constantOp = opBuilder.create<hw::ConstantOp>(
              clonedRegValue.getLoc(), clonedRegValue.getType(),
              constStateValue);
          clonedRegValue.replaceAllUsesWith(constantOp.getResult());
          clonedRegOp->erase();
        }
        Region &actionRegion = transitionOp.getAction();
        if (!variableRegs.empty()) {
          Block *actionBlock = opBuilder.createBlock(&actionRegion);
          opBuilder.setInsertionPointToStart(actionBlock);
          IRMapping mapping;
          opBuilder.cloneRegionBefore(moduleOp.getModuleBody(), actionRegion,
                                      actionBlock->getIterator(), mapping);
          actionBlock->erase();
          Block &newActionBlock = actionRegion.front();
          for (BlockArgument arg : newActionBlock.getArguments()) {
            int argIndex = arg.getArgNumber();
            BlockArgument topLevelArg = machine.getBody().getArgument(argIndex);
            arg.replaceAllUsesWith(topLevelArg);
          }
          newActionBlock.eraseArguments([](BlockArgument arg) { return true; });
          for (auto &[originalRegValue, variableOp] : variableMap) {
            Value clonedRegValue = mapping.lookup(originalRegValue);
            Operation *clonedRegOp = clonedRegValue.getDefiningOp();
            auto reg = cast<seq::CompRegOp>(clonedRegOp);
            opBuilder.setInsertionPointToStart(&newActionBlock);
            auto updateOp = opBuilder.create<UpdateOp>(reg.getLoc(), variableOp,
                                                       reg.getInput());
            const Value res = variableOp.getResult();
            clonedRegValue.replaceAllUsesWith(res);
            reg.erase();
          }
          Operation *terminator = actionRegion.back().getTerminator();
          auto hwOutputOp = dyn_cast<hw::OutputOp>(terminator);
          assert(hwOutputOp && "Expected terminator to be hw.output op");
          hwOutputOp.erase();

          for (auto const &[originalRegValue, constStateValue] : fromStateMap) {
            Value clonedRegValue = mapping.lookup(originalRegValue);
            Operation *clonedRegOp = clonedRegValue.getDefiningOp();
            opBuilder.setInsertionPoint(clonedRegOp);
            auto constantOp = opBuilder.create<hw::ConstantOp>(
                clonedRegValue.getLoc(), clonedRegValue.getType(),
                constStateValue);
            clonedRegValue.replaceAllUsesWith(constantOp.getResult());
            clonedRegOp->erase();
          }

          FrozenRewritePatternSet patterns(opBuilder.getContext());
          GreedyRewriteConfig config;
          SmallVector<Operation *> opsToProcess;
          actionRegion.walk([&](Operation *op) { opsToProcess.push_back(op); });
          config.setScope(&actionRegion);

          bool changed = false;
          LogicalResult converged =
              applyOpPatternsGreedily(opsToProcess, patterns, config, &changed);

          if (!sortTopologically(&actionRegion.front())) {
            transitionOp.emitError(
                "could not resolve cycles in action block of" +
                std::to_string(currentStateIndex) + " to " + std::to_string(j));
            return failure();
          }
        }

        if (!sortTopologically(&newGuardBlock)) {
          transitionOp.emitError("could not resolve cycles in guard block of" +
                                 std::to_string(currentStateIndex) + " to " +
                                 std::to_string(j));
          return failure();
        }
        SmallVector<Operation *> outputOps;
        stateOp.getOutput().walk(
            [&](Operation *op) { outputOps.push_back(op); });

        bool changed = false;
        GreedyRewriteConfig config;
        config.setScope(&stateOp.getOutput());
        LogicalResult converged = applyOpPatternsGreedily(
            outputOps, frozenPatterns, config, &changed);
        if (failed(converged)) {
          stateOp.emitError("Failed to canonicalize the generated state op");
          return failure();
        }    
        SmallVector<Operation *> transitionOps;
        stateOp.getTransitions().walk(
            [&](Operation *op) { transitionOps.push_back(op); });

        GreedyRewriteConfig config2;
        config2.setScope(&stateOp.getTransitions());
        applyOpPatternsGreedily(transitionOps, frozenPatterns, config2,
                                &changed);

        

        for (TransitionOp transition :
             stateOp.getTransitions().getOps<TransitionOp>()) {
          StateOp nextState = transition.getNextStateOp();
          int nextStateIndex = stateOpToState.lookup(nextState);
          auto guardConst = transition.getGuardReturn()
                                .getOperand()
                                .getDefiningOp<hw::ConstantOp>();
          bool nextStateIsReachable =
              !guardConst || (guardConst.getValueAttr().getInt() != 0);
          // If we find a valid next state and haven't seen it before, add it to
          // the worklist and the set of reachable states.
          if (nextStateIsReachable &&
              !reachableStates.contains(nextStateIndex)) {
            worklist.push_back(nextStateIndex);
            reachableStates.insert(nextStateIndex);
          }
        }
      }
    }

    // Clean up unreachable states. States without an output region are
    // placeholder states that were created during reachability analysis but
    // never populated (i.e., they are unreachable from the initial state).
    SmallVector<StateOp> statesToErase;

    // Collect unreachable states (those without an output op).
    for (StateOp stateOp : machine.getOps<StateOp>()) {
      if (!stateOp.getOutputOp()) {
        statesToErase.push_back(stateOp);
      }
    }

    // Erase states in a separate loop to avoid iterator invalidation. We first
    // collect all states to erase, then iterate over that list. This is
    // necessary because erasing a state while iterating over machine.getOps()
    // would invalidate the iterator.
    for (StateOp stateOp : statesToErase) {
      for (TransitionOp transition : machine.getOps<TransitionOp>()) {
        if (transition.getNextStateOp().getSymName() == stateOp.getSymName()) {
          transition.erase();
        }
      }
      stateOp.erase();
    }

    llvm::DenseSet<BlockArgument> asyncResetBlockArguments;
    for (auto arg : machine.getBody().front().getArguments()) {
      if (asyncResetArguments.contains(arg.getArgNumber())) {
        asyncResetBlockArguments.insert(arg);
      }
    }

    // Emit a warning if async reset signals were detected and removed.
    // The FSM dialect does not support async reset, so the reset behavior
    // is only captured in the initial state. The original async reset
    // triggering mechanism is not preserved.
    if (!asyncResetBlockArguments.empty()) {
      moduleOp.emitWarning()
          << "async reset signals detected and removed from FSM; "
             "reset behavior is captured only in the initial state";
    }

    Block &front = machine.getBody().front();
    front.eraseArguments([&](BlockArgument arg) {
      return asyncResetBlockArguments.contains(arg);
    });
    machine.getBody().front().eraseArguments([&](BlockArgument arg) {
      return arg.getType() == seq::ClockType::get(arg.getContext());
    });
    FunctionType oldFunctionType = machine.getFunctionType();
    SmallVector<Type> inputsWithoutClock;
    for (unsigned int i = 0; i < oldFunctionType.getNumInputs(); i++) {
      Type input = oldFunctionType.getInput(i);
      if (input != seq::ClockType::get(input.getContext()) &&
          !asyncResetArguments.contains(i))
        inputsWithoutClock.push_back(input);
    }

    FunctionType newFunctionType = FunctionType::get(
        opBuilder.getContext(), inputsWithoutClock, resultTypes);

    machine.setFunctionType(newFunctionType);
    moduleOp.erase();
    return success();
  }

private:
  /// Helper function to determine if a register is a state register.
  bool isStateRegister(seq::CompRegOp reg) const {
    auto regName = reg.getName();
    if (!regName)
      return false;

    // If user specified state registers, check if this register's name matches
    // any of them.
    if (!stateRegNames.empty()) {
      return llvm::is_contained(stateRegNames, regName->str());
    }

    // Default behavior: infer state registers by checking if the name contains
    // "state".
    return regName->contains("state");
  }

  HWModuleOp moduleOp;
  OpBuilder &opBuilder;
  ArrayRef<std::string> stateRegNames;
};

} // namespace

namespace {
struct CoreToFSMPass : public circt::impl::ConvertCoreToFSMBase<CoreToFSMPass> {
  using ConvertCoreToFSMBase<CoreToFSMPass>::ConvertCoreToFSMBase;

  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder builder(module);

    SmallVector<HWModuleOp> modules;
    for (auto hwModule : module.getOps<HWModuleOp>()) {
      modules.push_back(hwModule);
    }

    for (auto hwModule : modules) {
      builder.setInsertionPoint(hwModule);
      HWModuleOpConverter converter(builder, hwModule, stateRegs);
      if (failed(converter.run())) {
        signalPassFailure();
        return;
      }
    }
  }
};
} // namespace
