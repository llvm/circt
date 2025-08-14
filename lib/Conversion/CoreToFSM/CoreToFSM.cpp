//===- CoreToFSM.cpp - Convert Core to FSM Dialect ------------------------===//
#include "circt/Conversion/CoreToFSM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "circt/Conversion/CoreToFSM.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include <boost/pending/disjoint_sets.hpp>
#include <memory>
#include <string>
#include <vector>
namespace circt {
#define GEN_PASS_DEF_CONVERTCORETOFSM
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace hw;
using namespace fsm;

namespace {

// Forward declaration for our recursive helper function
static void generateConcatenatedValues(
    const std::vector<llvm::DenseSet<size_t>> &allOperandValues,
    const std::vector<unsigned> &shifts,
    llvm::DenseSet<size_t> &finalPossibleValues, size_t operandIdx,
    size_t currentValue);

static void getPossibleValues(llvm::DenseSet<size_t> &possibleValues, Value v) {
  if (circt::hw::ConstantOp c =
          dyn_cast_or_null<hw::ConstantOp>(v.getDefiningOp())) {
    possibleValues.insert(c.getValueAttr().getValue().getZExtValue());
    return;
  }
  if (circt::comb::MuxOp m = dyn_cast_or_null<comb::MuxOp>(v.getDefiningOp())) {
    getPossibleValues(possibleValues, m.getTrueValue());
    getPossibleValues(possibleValues, m.getFalseValue());
    return;
  }

  if (circt::comb::ConcatOp concatOp =
          dyn_cast_or_null<comb::ConcatOp>(v.getDefiningOp())) {
    std::vector<llvm::DenseSet<size_t>> allOperandValues;
    std::vector<unsigned> operandWidths;

    for (Value operand : concatOp.getOperands()) {
      llvm::DenseSet<size_t> operandPossibleValues;
      getPossibleValues(operandPossibleValues, operand);

      // It's crucial to handle the case where a sub-computation is too complex.
      // If we can't determine specific values for an operand, we must
      // pessimistically assume it can be any value its bitwidth allows.
      IntegerType opType = dyn_cast<IntegerType>(operand.getType());
      unsigned width = opType.getWidth();
      if (operandPossibleValues.empty()) {
        uint64_t numStates = 1ULL << width;
        // Add a threshold to prevent combinatorial explosion on large unknown
        // inputs.
        if (numStates > 256) { // Heuristic threshold
          // If the search space is too large, we abandon the analysis for this
          // path. The outer function will fall back to its own full-range
          // default.
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
    std::vector<unsigned> shifts(concatOp.getNumOperands(), 0);
    for (int i = concatOp.getNumOperands() - 2; i >= 0; --i) {
      shifts[i] = shifts[i + 1] + operandWidths[i + 1];
    }

    generateConcatenatedValues(allOperandValues, shifts, possibleValues, 0, 0);
    return;
  }

  // --- Fallback Case ---
  // If the operation is not recognized, assume all possible values for its
  // bitwidth.
  Operation *o = v.getDefiningOp();

  IntegerType addrType = dyn_cast<IntegerType>(v.getType());
  if (!addrType)
    return; // Not an integer type we can analyze

  unsigned bitWidth = addrType.getWidth();
  // Again, use a threshold to avoid trying to enumerate 2^64 values.
  if (bitWidth > 16) {
    return;
  }

  uint64_t numRegStates = 1ULL << bitWidth;
  for (size_t i = 0; i < numRegStates; i++) {
    possibleValues.insert(i);
  }
  return;
};

/// @brief Recursively builds all possible concatenated integer values.
/// @param allOperandValues Vector of sets, where each set contains the possible
/// values for an operand.
/// @param shifts The pre-calculated left-shift amount for each operand index.
/// @param finalPossibleValues The output set to populate with final combined
/// values.
/// @param operandIdx The current operand index we are processing.
/// @param currentValue The bitwise-OR'd value accumulated so far from previous
/// operands.
static void generateConcatenatedValues(
    const std::vector<llvm::DenseSet<size_t>> &allOperandValues,
    const std::vector<unsigned> &shifts,
    llvm::DenseSet<size_t> &finalPossibleValues, size_t operandIdx,
    size_t currentValue) {

  // Base case: If we've processed all operands, the currentValue is complete.
  if (operandIdx >= allOperandValues.size()) {
    finalPossibleValues.insert(currentValue);
    return;
  }

  // Recursive step: For each possible value of the current operand,
  // combine it with the accumulated value and recurse for the next operand.
  const auto &currentOperandPossibleValues = allOperandValues[operandIdx];
  unsigned shift = shifts[operandIdx];

  for (size_t val : currentOperandPossibleValues) {
    // Combine the current operand's value by shifting it and ORing it.
    size_t nextValue = currentValue | (val << shift);
    generateConcatenatedValues(allOperandValues, shifts, finalPossibleValues,
                               operandIdx + 1, nextValue);
  }
}

static llvm::DenseMap<mlir::Value, int> intToRegMap(std::vector<seq::CompRegOp> v, int i){
    llvm::DenseMap<mlir::Value, int> m;
    // int i = 0;
    // int width = 0;
    for(size_t ci = 0; ci < v.size(); ci++){
        seq::CompRegOp reg = v[ci];
        int bits = reg.getType().getIntOrFloatBitWidth();
        int v = i & ((1 << bits) - 1);
        m[reg] = v;
        i = i >> bits;
        // i += m[reg] * 1ULL << width;
        // width += (bits);
    }
    return m;
    // return i;
}
static int regMapToInt(std::vector<seq::CompRegOp> v, llvm::DenseMap<mlir::Value, int> m){
    int i = 0;
    int width = 0;
    for(size_t ci = 0; ci < v.size(); ci++){
        seq::CompRegOp reg = v[ci];
        i += m[reg] * 1ULL << width;
        width += (reg.getType().getIntOrFloatBitWidth());
    }
    return i;
}
/// @brief Computes the Cartesian product of a list of sets.
/// This function takes a vector of sets, where each set contains the possible
/// values for a particular element (e.g., a register). It returns a set of
/// vectors, where each vector represents one complete and unique combination of
/// values, drawing one value from each of the input sets.
/// @param valueSets A vector of DenseSets, each representing the possible
/// values for one component of a state vector.
/// @return A std::set of std::vectors, representing all possible complete
/// state vectors.
static std::set<std::vector<size_t>>
calculateCartesianProduct(const std::vector<llvm::DenseSet<size_t>> &valueSets) {
  std::set<std::vector<size_t>> product;
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

    std::set<std::vector<size_t>> newProduct;
    for (const auto &existingVector : product) {
      for (size_t newValue : currentSet) {
        std::vector<size_t> newVector = existingVector;
        newVector.push_back(newValue);
        newProduct.insert(std::move(newVector));
      }
    }
    product = std::move(newProduct);
  }

  return product;
}

static FrozenRewritePatternSet loadPatterns(MLIRContext &context){

    RewritePatternSet patterns(&context);
    // Collect canonicalization patterns from the dialects you are using.
    // This is what the canonicalizer pass does internally.
    for (auto *dialect : context.getLoadedDialects())
      dialect->getCanonicalizationPatterns(patterns);
    comb::ICmpOp::getCanonicalizationPatterns(patterns, &context);
    comb::AndOp::getCanonicalizationPatterns(patterns, &context);
    comb::XorOp::getCanonicalizationPatterns(patterns, &context);
    comb::MuxOp::getCanonicalizationPatterns(patterns, &context);
    comb::ConcatOp::getCanonicalizationPatterns(patterns,
                                                &context);
    comb::ExtractOp::getCanonicalizationPatterns(patterns,
                                                 &context);
    comb::AddOp::getCanonicalizationPatterns(patterns, &context);
    comb::OrOp::getCanonicalizationPatterns(patterns, &context);
    comb::MulOp::getCanonicalizationPatterns(patterns, &context);
    hw::ConstantOp::getCanonicalizationPatterns(patterns,
                                                &context);
    fsm::TransitionOp::getCanonicalizationPatterns(patterns,
                                                   &context);
    fsm::StateOp::getCanonicalizationPatterns(patterns, &context);
    fsm::MachineOp::getCanonicalizationPatterns(patterns,
                                                &context);

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    return frozenPatterns;
}

static void
getReachableStates(llvm::DenseSet<size_t> &vistableStates,
                   circt::hw::HWModuleOp moduleOp, size_t currentStateIndex,
                   std::vector<seq::CompRegOp> registers,
                   OpBuilder opBuilder, bool isInitialState) {

  IRMapping mapping;
  HWModuleOp clonedBody = llvm::dyn_cast<circt::hw::HWModuleOp>(
      opBuilder.clone(*moduleOp, mapping));

  llvm::DenseMap<mlir::Value, int> stateMap = intToRegMap(registers,currentStateIndex);
  Operation *terminator = clonedBody.getBody().front().getTerminator();
  circt::hw::OutputOp output = dyn_cast<circt::hw::OutputOp>(terminator);
  int i = 0;
  std::vector<mlir::Value> values;
  std::vector<mlir::Type> types;
  llvm::DenseMap<int, mlir::Value> regMap;

  for (auto [originalRegValue, constStateValue] : stateMap) {

    mlir::Value clonedRegValue = mapping.lookup(originalRegValue);
    Operation *clonedRegOp = clonedRegValue.getDefiningOp();
    circt::seq::CompRegOp reg = dyn_cast<circt::seq::CompRegOp>(clonedRegOp);
    mlir::Type constantType = reg.getType();
    IntegerAttr constantAttr =
        opBuilder.getIntegerAttr(constantType, constStateValue);
    opBuilder.setInsertionPoint(clonedRegOp);
    circt::hw::ConstantOp otherStateConstant =
        opBuilder.create<hw::ConstantOp>(reg.getLoc(), constantAttr);
    values.push_back(reg.getInput());
    types.push_back(reg.getType());
    clonedRegValue.replaceAllUsesWith(otherStateConstant.getResult());
    regMap[i] = originalRegValue;
    reg.erase();
    i++;
  }
  opBuilder.setInsertionPointToEnd(clonedBody.front().getBlock());
  circt::hw::OutputOp newOutput =
      opBuilder.create<circt::hw::OutputOp>(output.getLoc(), values);
  output.erase();
  // Collect canonicalization patterns from the dialects you are using.
  // This is what the canonicalizer pass does internally.
  FrozenRewritePatternSet frozenPatterns = loadPatterns(*moduleOp.getContext());

  SmallVector<Operation *> opsToProcess;
  clonedBody.walk([&](Operation *op) { opsToProcess.push_back(op); });

  bool changed = false;
  mlir::GreedyRewriteConfig config;
  LogicalResult converged = mlir::applyOpPatternsGreedily(
      opsToProcess, frozenPatterns, config, &changed);

    std::vector<llvm::DenseSet<size_t>> pv;
    for (size_t j = 0; j < newOutput.getNumOperands(); j++) {
      llvm::DenseSet<size_t> possibleValues;

      Value v = newOutput.getOperand(j);
      getPossibleValues(possibleValues, v);
      pv.push_back(possibleValues);
    }
    std::set<std::vector<size_t>> flipped =  calculateCartesianProduct(pv);
    for(std::vector<size_t> v : flipped) {
        llvm::DenseMap<mlir::Value, int> m;
        for(int k = 0; k < v.size(); k++){
            seq::CompRegOp r = registers[k];
            m[r] = v[k];
        }

        int i = regMapToInt(registers, m);
        vistableStates.insert(i);
    }

  clonedBody.erase();
};





// A converter class to handle the logic of converting a single hw.module.
class HWModuleOpConverter {
public:
  HWModuleOpConverter(OpBuilder &builder, HWModuleOp moduleOp)
      : moduleOp(moduleOp), opBuilder(builder) {}
  LogicalResult run() {
    llvm::SmallVector<circt::seq::CompRegOp> stateRegs;
    llvm::SmallVector<circt::seq::CompRegOp> variableRegs;
    moduleOp.walk([&](circt::seq::CompRegOp reg) {
      if (reg.getName()->contains("state")) {
        stateRegs.push_back(reg);
      } else {
        variableRegs.push_back(reg);
      }
    });
    if (stateRegs.empty()) {
      llvm::outs()
          << "Cannot find state register in this FSM. You might need to "
             "manually specify which registers are state registers.\n";
      return mlir::success();
    }
    llvm::DenseMap<mlir::Value, size_t> regToIndexMap;
    int regIndex = 0;
    std::vector<seq::CompRegOp> registers;
    for(seq::CompRegOp c : stateRegs){
        regToIndexMap[c] = regIndex;
        regIndex++;
        registers.push_back(c);
    }

    llvm::DenseMap<size_t, circt::fsm::StateOp> stateToStateOp;
    llvm::DenseMap<circt::fsm::StateOp, size_t> stateOpToState;
    // gather async reset arguments to delete them from function type
    llvm::DenseSet<size_t> asyncResetArguments;
    auto regsInGroup = stateRegs;
    mlir::Location loc = moduleOp.getLoc();
    SmallVector<mlir::Type> inputTypes = moduleOp.getInputTypes();

    // Create a new FSM machine with the current state.
    auto resultTypes = moduleOp.getOutputTypes();
    FunctionType machineType =
        FunctionType::get(opBuilder.getContext(), inputTypes, resultTypes);
    StringRef machineName = moduleOp.getName();

    // int initialStateIndex = 0;
    // std::vector<llvm::DenseMap<mlir::Value, int>> states =

    //     enumerateStates(regsInGroup);
    llvm::DenseMap<mlir::Value, int> initialStateMap;
    for(seq::CompRegOp reg : moduleOp.getOps<seq::CompRegOp>()){
        mlir::Value resetValue = reg.getResetValue();
        circt::hw::ConstantOp definingConstant =
            resetValue.getDefiningOp<circt::hw::ConstantOp>();
        if (!definingConstant) {
          reg->emitError(
              "Cannot find defining constant for reset value of register: ");
          return failure();
        }
        int resetValueInt = definingConstant.getValueAttr().getValue().getZExtValue();
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
    fsm::MachineOp machine = opBuilder.create<fsm::MachineOp>(
        loc, machineName, initialStateName, machineType, machineAttrs);

    OpBuilder::InsertionGuard guard(opBuilder);
    opBuilder.setInsertionPointToStart(&machine.getBody().front());
    llvm::DenseMap<circt::seq::CompRegOp, circt::fsm::VariableOp> variableMap;
    for (circt::seq::CompRegOp varReg : variableRegs) {
      ::mlir::TypedValue<::mlir::Type> initialValue = varReg.getResetValue();
      circt::hw::ConstantOp definingConstant =
          initialValue.getDefiningOp<circt::hw::ConstantOp>();
      circt::fsm::VariableOp variableOp =
          opBuilder.create<circt::fsm::VariableOp>(
              varReg->getLoc(), varReg.getInput().getType(),
              definingConstant.getValueAttr(), varReg.getName().value_or("a"));
      variableMap[varReg] = variableOp;
    }

    // A valid machine needs at least its initial state defined.
    FrozenRewritePatternSet frozenPatterns = loadPatterns(*moduleOp.getContext());

    SetVector<int> reachableStates;
    SmallVector<int> worklist;

    worklist.push_back(initialStateIndex);
    reachableStates.insert(initialStateIndex);
    unsigned i = 0;
    while (i < worklist.size()) {

      int currentStateIndex = worklist[i++];

     llvm::DenseMap<mlir::Value, int> stateMap = intToRegMap(registers, currentStateIndex);
      

      opBuilder.setInsertionPointToEnd(&machine.getBody().front());

      circt::fsm::StateOp stateOp;

      if (!stateToStateOp.contains(currentStateIndex)) {
        stateOp = opBuilder.create<fsm::StateOp>(
            loc, "state_" + std::to_string(currentStateIndex));
        stateToStateOp.insert({currentStateIndex, stateOp});
        stateOpToState.insert({stateOp, currentStateIndex});
      } else {
        stateOp = stateToStateOp.lookup(currentStateIndex);
      }
      mlir::Region &outputRegion = stateOp.getOutput();
      mlir::Block *outputBlock = &outputRegion.front();
      opBuilder.setInsertionPointToStart(outputBlock);
      IRMapping mapping;
      opBuilder.cloneRegionBefore(moduleOp.getModuleBody(), outputRegion,
                                  outputBlock->getIterator(), mapping);
      outputBlock->erase();

      auto *terminator = outputRegion.front().getTerminator();
      auto hwOutputOp = dyn_cast<hw::OutputOp>(terminator);
      assert(hwOutputOp && "Expected terminator to be hw.OutputOp");

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
        circt::seq::CompRegOp reg =
            dyn_cast<circt::seq::CompRegOp>(clonedRegOp);
        const auto res = variableOp.getResult();
        clonedRegValue.replaceAllUsesWith(res);
        reg.erase();
      }
      for (auto const &[originalRegValue, constStateValue] : stateMap) {
        //  Find the cloned register's result value using the mapping.
        Value clonedRegValue = mapping.lookup(originalRegValue);
        assert(clonedRegValue && "Original register value not found in "
                                 "mapping; this is an internal error.");
        Operation *clonedRegOp = clonedRegValue.getDefiningOp();
        
        assert(clonedRegOp && "Cloned value must have a defining op.");
        opBuilder.setInsertionPoint(clonedRegOp);
        circt::seq::CompRegOp r = dyn_cast<circt::seq::CompRegOp>(clonedRegOp);
        assert(r && "Must be a register.");
        auto input = r.getInput();
        TypedValue<IntegerType> registerReset = r.getReset();
        if (registerReset) { // Ensure the register has a reset.
          if (BlockArgument blockArg =
                  mlir::dyn_cast<mlir::BlockArgument>(registerReset)) {
            asyncResetArguments.insert(blockArg.getArgNumber());
            // blockArg.dump();
            ConstantOp falseConst = opBuilder.create<hw::ConstantOp>(
                blockArg.getLoc(), clonedRegValue.getType(), 0);
            blockArg.replaceAllUsesWith(falseConst.getResult());
            // Check if this argument belongs to the top-level module block.
          }
          // also check for rst_ni case, because OpenTitan uses ni.
          if (circt::comb::XorOp xorOp =
                  registerReset.getDefiningOp<circt::comb::XorOp>()) {
            if (xorOp.isBinaryNot()) {
              mlir::Value rhs = xorOp.getOperand(0);
              if (BlockArgument blockArg =
                      mlir::dyn_cast<mlir::BlockArgument>(rhs)) {
                asyncResetArguments.insert(blockArg.getArgNumber());
                ConstantOp trueConst = opBuilder.create<hw::ConstantOp>(
                    blockArg.getLoc(), blockArg.getType(), 1);
                blockArg.replaceAllUsesWith(trueConst.getResult());
              }
            }
          }
        }
        // Create the hw.constant operation for the specific state value.
        auto constantOp = opBuilder.create<hw::ConstantOp>(
            clonedRegValue.getLoc(), clonedRegValue.getType(), constStateValue);

        // Replace all uses of the cloned register's result with the new
        // constant.
        clonedRegValue.replaceAllUsesWith(constantOp.getResult());

        // Erase the now-dead cloned register operation.

        clonedRegOp->erase();
      }
      mlir::GreedyRewriteConfig config;
      // You could add more patterns here if needed.
      // This function will apply folding and DCE, which is exactly what
      // you need.
      SmallVector<Operation *> opsToProcess;
      outputRegion.walk([&](Operation *op) { opsToProcess.push_back(op); });
      // replace references to arguments in the output block with
      // arguments at the top level
      // Iterate directly over the block's arguments.
      for (auto arg : outputRegion.front().getArguments()) {
        int argIndex = arg.getArgNumber();
        mlir::BlockArgument topLevelArg =
            machine.getBody().getArgument(argIndex);
        // Replace all uses of the old argument with the new one.
        arg.replaceAllUsesWith(topLevelArg);
      }
      // delete the arguments from the output block
      outputRegion.front().eraseArguments(
          [](BlockArgument arg) { return true; });
      // Create your pattern set. For simple DCE and folding, an empty
      //    native pattern set is often sufficient, as folding is built-in.
      FrozenRewritePatternSet patterns(
          opBuilder.getContext()); // Or add specific patterns

      config.setScope(&outputRegion); // IMPORTANT: Tell the rewriter the
                                      // boundary of its work.

      bool changed = false;
      LogicalResult converged = mlir::applyOpPatternsGreedily(
          opsToProcess, patterns, config, &changed);
      opBuilder.setInsertionPoint(stateOp);
      if (!sortTopologically(&outputRegion.front())) {
        moduleOp.emitError("could not resolve cycles in module");
        return failure();
      }
      mlir::Region &transitionRegion = stateOp.getTransitions();
      llvm::DenseSet<size_t> vistableStates;
      getReachableStates(vistableStates, moduleOp, currentStateIndex, registers,
                         opBuilder, currentStateIndex == initialStateIndex);
      for (size_t j : vistableStates) {
        circt::fsm::StateOp toState;
        if (!stateToStateOp.contains(j)) {
          opBuilder.setInsertionPointToEnd(&machine.getBody().front());
          toState =
              opBuilder.create<fsm::StateOp>(loc, "state_" + std::to_string(j));
          stateToStateOp.insert({j, toState});
          stateOpToState.insert({toState, j});
        } else {
          toState = stateToStateOp[j];
        }
        opBuilder.setInsertionPointToStart(&transitionRegion.front());
        circt::fsm::TransitionOp transitionOp =
            opBuilder.create<circt::fsm::TransitionOp>(
                loc, "state_" + std::to_string(j));
        mlir::Region &guardRegion = transitionOp.getGuard();
        opBuilder.createBlock(&guardRegion);

        mlir::Block &guardBlock = guardRegion.front();

        opBuilder.setInsertionPointToStart(&guardBlock);
        IRMapping mapping;
        opBuilder.cloneRegionBefore(moduleOp.getModuleBody(), guardRegion,
                                    guardBlock.getIterator(), mapping);
        guardBlock.erase();
        mlir::Block &newGuardBlock = guardRegion.front();
        Operation *terminator = newGuardBlock.getTerminator();
        hw::OutputOp hwOutputOp = dyn_cast<hw::OutputOp>(terminator);
        assert(hwOutputOp && "Expected terminator to be hw.OutputOp");

        // Position the builder to insert the new terminator right before
        // the old one.

        llvm::DenseMap<mlir::Value, int> toStateMap = intToRegMap(registers, j);//states[j];
        SmallVector<mlir::Value> equalityChecks;
        // check if the input to each register matches the toState
        for (auto &[originalRegValue, variableOp] : variableMap) {
          opBuilder.setInsertionPointToStart(&newGuardBlock);
          Value clonedRegValue = mapping.lookup(originalRegValue);
          Operation *clonedRegOp = clonedRegValue.getDefiningOp();
          circt::seq::CompRegOp reg =
              dyn_cast<circt::seq::CompRegOp>(clonedRegOp);
          const auto res = variableOp.getResult();
          clonedRegValue.replaceAllUsesWith(res);
          reg.erase();
        }
        for (auto const &[originalRegValue, constStateValue] : toStateMap) {

          Value clonedRegValue = mapping.lookup(originalRegValue);
          Operation *clonedRegOp = clonedRegValue.getDefiningOp();
          opBuilder.setInsertionPoint(clonedRegOp);
          circt::seq::CompRegOp r =
              dyn_cast<circt::seq::CompRegOp>(clonedRegOp);

          mlir::Value registerInput = r.getInput();
          TypedValue<IntegerType> registerReset = r.getReset();
          if (registerReset) { // Ensure the register has a reset.
            if (BlockArgument blockArg =
                    mlir::dyn_cast<mlir::BlockArgument>(registerReset)) {
              // asyncResetArguments.insert(blockArg.getArgNumber());
              // blockArg.dump();
              ConstantOp falseConst = opBuilder.create<hw::ConstantOp>(
                  blockArg.getLoc(), clonedRegValue.getType(), 0);
              blockArg.replaceAllUsesWith(falseConst.getResult());
            }
            // also check for rst_ni case, because OpenTitan uses ni.
            if (circt::comb::XorOp xorOp =
                    registerReset.getDefiningOp<circt::comb::XorOp>()) {
              if (xorOp.isBinaryNot()) {
                mlir::Value rhs = xorOp.getOperand(0);
                if (BlockArgument blockArg =
                        mlir::dyn_cast<mlir::BlockArgument>(rhs)) {
                  ConstantOp trueConst = opBuilder.create<hw::ConstantOp>(
                      blockArg.getLoc(), blockArg.getType(), 1);
                  blockArg.replaceAllUsesWith(trueConst.getResult());
                }
              }
            }
          }
          mlir::Type constantType =
              registerInput.getType(); // Use the type of the value you're
                                       // comparing against.

          IntegerAttr constantAttr =
              opBuilder.getIntegerAttr(constantType, constStateValue);
          circt::hw::ConstantOp otherStateConstant =
              opBuilder.create<hw::ConstantOp>(hwOutputOp.getLoc(),
                                               constantAttr);

          circt::comb::ICmpOp doesEqual = opBuilder.create<circt::comb::ICmpOp>(
              hwOutputOp.getLoc(), comb::ICmpPredicate::eq, registerInput,
              otherStateConstant.getResult());
          equalityChecks.push_back(doesEqual.getResult());
        }
        opBuilder.setInsertionPoint(hwOutputOp);
        circt::comb::AndOp allEqualCheck = opBuilder.create<circt::comb::AndOp>(
            hwOutputOp.getLoc(), equalityChecks, false);
        // return `true` iff all registers match their value in the toState.
        opBuilder.create<fsm::ReturnOp>(hwOutputOp.getLoc(),
                                        allEqualCheck.getResult());

        // Erase the old terminator.
        hwOutputOp.erase();
        for (BlockArgument arg : newGuardBlock.getArguments()) {
          int argIndex = arg.getArgNumber();
          mlir::BlockArgument topLevelArg =
              machine.getBody().getArgument(argIndex);
          // Replace all uses of the old argument with the new one.
          arg.replaceAllUsesWith(topLevelArg);
        }
        // delete the arguments from the output block
        newGuardBlock.eraseArguments([](BlockArgument arg) { return true; });
        llvm::DenseMap<mlir::Value, int> fromStateMap = intToRegMap(registers, currentStateIndex);
           // states[currentStateIndex];
        for (auto const &[originalRegValue, constStateValue] : fromStateMap) {
          //  Find the cloned register's result value using the mapping.
          Value clonedRegValue = mapping.lookup(originalRegValue);
          assert(clonedRegValue && "Original register value not found in "
                                   "mapping; this is an internal error.");
          Operation *clonedRegOp = clonedRegValue.getDefiningOp();
          assert(clonedRegOp && "Cloned value must have a defining op.");
          opBuilder.setInsertionPoint(clonedRegOp);

          // Create the hw.constant operation for the specific state value.
          auto constantOp = opBuilder.create<hw::ConstantOp>(
              clonedRegValue.getLoc(), clonedRegValue.getType(),
              constStateValue);

          // Replace all uses of the cloned register's result with the new
          // constant.
          clonedRegValue.replaceAllUsesWith(constantOp.getResult());

          clonedRegOp->erase();
        }
        mlir::Region &actionRegion = transitionOp.getAction();
        if (!variableRegs.empty()) {
          mlir::Block *actionBlock = opBuilder.createBlock(&actionRegion);
          opBuilder.setInsertionPointToStart(actionBlock);
          IRMapping mapping;
          opBuilder.cloneRegionBefore(moduleOp.getModuleBody(), actionRegion,
                                      actionBlock->getIterator(), mapping);
          actionBlock->erase();
          mlir::Block &newActionBlock = actionRegion.front();
          for (BlockArgument arg : newActionBlock.getArguments()) {
            int argIndex = arg.getArgNumber();
            mlir::BlockArgument topLevelArg =
                machine.getBody().getArgument(argIndex);
            arg.replaceAllUsesWith(topLevelArg);
          }
          newActionBlock.eraseArguments([](BlockArgument arg) { return true; });
          for (auto &[originalRegValue, variableOp] : variableMap) {
            //  Find the cloned register's result value using the mapping.
            Value clonedRegValue = mapping.lookup(originalRegValue);
            Operation *clonedRegOp = clonedRegValue.getDefiningOp();
            seq::CompRegOp reg = dyn_cast<seq::CompRegOp>(clonedRegOp);
            opBuilder.setInsertionPointToStart(&newActionBlock);
            fsm::UpdateOp updateOp = opBuilder.create<fsm::UpdateOp>(
                reg.getLoc(), variableOp, reg.getInput());
            const mlir::Value res = variableOp.getResult();
            clonedRegValue.replaceAllUsesWith(res);
            reg.erase();
          }
          Operation *terminator = actionRegion.back().getTerminator();
          hw::OutputOp hwOutputOp = dyn_cast<hw::OutputOp>(terminator);
          hwOutputOp.erase();

          for (auto const &[originalRegValue, constStateValue] : fromStateMap) {
            Value clonedRegValue = mapping.lookup(originalRegValue);
            Operation *clonedRegOp = clonedRegValue.getDefiningOp();
            opBuilder.setInsertionPoint(clonedRegOp);

            // Create the hw.constant operation for the specific state value.
            auto constantOp = opBuilder.create<hw::ConstantOp>(
                clonedRegValue.getLoc(), clonedRegValue.getType(),
                constStateValue);

            // Replace all uses of the cloned register's result with the new
            // constant.
            clonedRegValue.replaceAllUsesWith(constantOp.getResult());

            clonedRegOp->erase();
          }

          // delete the arguments from the output block
          FrozenRewritePatternSet patterns(
              opBuilder.getContext()); // Or add specific patterns
          mlir::GreedyRewriteConfig config;
          SmallVector<Operation *> opsToProcess;
          actionRegion.walk([&](Operation *op) { opsToProcess.push_back(op); });
          config.setScope(&actionRegion); // IMPORTANT: Tell the rewriter the
                                          // boundary of its work.

          bool changed = false;
          LogicalResult converged = mlir::applyOpPatternsGreedily(
              opsToProcess, patterns, config, &changed);

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
        mlir::GreedyRewriteConfig config;
        config.setScope(&stateOp.getOutput());
        LogicalResult converged = mlir::applyOpPatternsGreedily(
            outputOps, frozenPatterns, config, &changed);

        SmallVector<Operation *> transitionOps;
        stateOp.getTransitions().walk(
            [&](Operation *op) { transitionOps.push_back(op); });

        mlir::GreedyRewriteConfig config2;
        config2.setScope(&stateOp.getTransitions());
        mlir::applyOpPatternsGreedily(transitionOps, frozenPatterns, config2,
                                      &changed);

        if (failed(converged)) {
          stateOp.emitError("Failed to canonicalize the generated state op");
          return failure();
        }

        for (TransitionOp transition :
             stateOp.getTransitions().getOps<TransitionOp>()) {
          StateOp nextState = transition.getNextStateOp();
          int nextStateIndex =  stateOpToState.lookup(nextState);
          hw::ConstantOp guardConst =
              transition.getGuardReturn()
                  .getOperand()
                  .getDefiningOp<circt::hw::ConstantOp>();
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

    SmallVector<fsm::StateOp> statesToErase;

    // First, collect the states that need to be erased.
    for (fsm::StateOp stateOp : machine.getOps<fsm::StateOp>()) {
      if (!stateOp.getOutputOp()) {
        statesToErase.push_back(stateOp);
      }
    }

    // Now, erase them in a separate loop.
    for (fsm::StateOp stateOp : statesToErase) {
      for (fsm::TransitionOp transition : machine.getOps<fsm::TransitionOp>()) {
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
    Block &front = machine.getBody().front();
    front.eraseArguments([&](BlockArgument arg) {
      if (asyncResetBlockArguments.contains(arg)) {
        arg.dump();
      }
      return asyncResetBlockArguments.contains(arg);
    });
    machine.getBody().front().eraseArguments([&](BlockArgument arg) {
      return arg.getType() == seq::ClockType::get(arg.getContext());
    });
    FunctionType oldFunctionType = machine.getFunctionType();
    SmallVector<mlir::Type> inputsWithoutClock;
    for (unsigned int i = 0; i < oldFunctionType.getNumInputs(); i++) {
      mlir::Type input = oldFunctionType.getInput(i);
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
  HWModuleOp moduleOp;
  OpBuilder &opBuilder;
};

} // namespace

namespace {
struct CoreToFSMPass : public circt::impl::ConvertCoreToFSMBase<CoreToFSMPass> {
  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder builder(module);

    SmallVector<HWModuleOp> modules;
    for (auto hwModule : module.getOps<HWModuleOp>()) {
      modules.push_back(hwModule);
    }

    for (auto hwModule : modules) {
      builder.setInsertionPoint(hwModule);
      HWModuleOpConverter converter(builder, hwModule);
      if (failed(converter.run())) {
        signalPassFailure();
        return;
      }
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> circt::createConvertCoreToFSMPass() {
  return std::make_unique<CoreToFSMPass>();
}
