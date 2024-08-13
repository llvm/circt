//===- ConvertToArcs.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ConvertToArcs.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "convert-to-arcs"

using namespace circt;
using namespace arc;
using namespace hw;
using llvm::MapVector;

static bool isArcBreakingOp(Operation *op) {
  return op->hasTrait<OpTrait::ConstantLike>() ||
         isa<hw::InstanceOp, seq::CompRegOp, MemoryOp, ClockedOpInterface,
             seq::ClockGateOp, sim::DPICallOp>(op) ||
         op->getNumResults() > 1;
}

//===----------------------------------------------------------------------===//
// Conversion
//===----------------------------------------------------------------------===//

namespace {
struct Converter {
  LogicalResult run(ModuleOp module);
  LogicalResult runOnModule(HWModuleOp module);
  LogicalResult analyzeFanIn();
  void extractArcs(HWModuleOp module);
  LogicalResult absorbRegs(HWModuleOp module);

  /// The global namespace used to create unique definition names.
  Namespace globalNamespace;

  /// All arc-breaking operations in the current module.
  SmallVector<Operation *> arcBreakers;
  SmallDenseMap<Operation *, unsigned> arcBreakerIndices;

  /// A post-order traversal of the operations in the current module.
  SmallVector<Operation *> postOrder;

  /// The set of arc-breaking ops an operation in the current module
  /// contributes to, represented as a bit mask.
  MapVector<Operation *, APInt> faninMasks;

  /// The sets of operations that contribute to the same arc-breaking ops.
  MapVector<APInt, DenseSet<Operation *>> faninMaskGroups;

  /// The arc uses generated by `extractArcs`.
  SmallVector<mlir::CallOpInterface> arcUses;

  /// Whether registers should be made observable by assigning their arcs a
  /// "name" attribute.
  bool tapRegisters;
};
} // namespace

LogicalResult Converter::run(ModuleOp module) {
  for (auto &op : module.getOps())
    if (auto sym =
            op.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
      globalNamespace.newName(sym.getValue());
  for (auto module : module.getOps<HWModuleOp>())
    if (failed(runOnModule(module)))
      return failure();
  return success();
}

LogicalResult Converter::runOnModule(HWModuleOp module) {
  // Find all arc-breaking operations in this module and assign them an index.
  arcBreakers.clear();
  arcBreakerIndices.clear();
  for (Operation &op : *module.getBodyBlock()) {
    if (op.getNumRegions() > 0)
      return op.emitOpError("has regions; not supported by ConvertToArcs");
    if (!isArcBreakingOp(&op) && !isa<hw::OutputOp>(&op))
      continue;
    arcBreakerIndices[&op] = arcBreakers.size();
    arcBreakers.push_back(&op);
  }
  // Skip modules with only `OutputOp`.
  if (module.getBodyBlock()->without_terminator().empty() &&
      isa<hw::OutputOp>(module.getBodyBlock()->getTerminator()))
    return success();
  LLVM_DEBUG(llvm::dbgs() << "Analyzing " << module.getModuleNameAttr() << " ("
                          << arcBreakers.size() << " breakers)\n");

  // For each operation, figure out the set of breaker ops it contributes to,
  // in the form of a bit mask. Then group operations together that contribute
  // to the same set of breaker ops.
  if (failed(analyzeFanIn()))
    return failure();

  // Extract the fanin mask groups into separate combinational arcs and
  // combine them with the registers in the design.
  extractArcs(module);
  if (failed(absorbRegs(module)))
    return failure();
  return success();
}

LogicalResult Converter::analyzeFanIn() {
  SmallVector<std::tuple<Operation *, unsigned>> worklist;

  // Seed the worklist and fanin masks with the arc breaking operations.
  faninMasks.clear();
  for (auto *op : arcBreakers) {
    unsigned index = arcBreakerIndices.lookup(op);
    auto mask = APInt::getOneBitSet(arcBreakers.size(), index);
    faninMasks[op] = mask;
    worklist.push_back({op, 0});
  }

  // Establish a post-order among the operations.
  DenseSet<Operation *> seen;
  DenseSet<Operation *> finished;
  postOrder.clear();
  while (!worklist.empty()) {
    auto &[op, operandIdx] = worklist.back();
    if (operandIdx == op->getNumOperands()) {
      if (!isArcBreakingOp(op) && !isa<hw::OutputOp>(op))
        postOrder.push_back(op);
      finished.insert(op);
      seen.erase(op);
      worklist.pop_back();
      continue;
    }
    auto operand = op->getOperand(operandIdx++); // advance to next operand
    auto *definingOp = operand.getDefiningOp();
    if (!definingOp || isArcBreakingOp(definingOp) ||
        finished.contains(definingOp))
      continue;
    if (!seen.insert(definingOp).second) {
      definingOp->emitError("combinational loop detected");
      return failure();
    }
    worklist.push_back({definingOp, 0});
  }
  LLVM_DEBUG(llvm::dbgs() << "- Sorted " << postOrder.size() << " ops\n");

  // Compute fanin masks in reverse post-order, which will compute the mask
  // for an operation's uses before it computes it for the operation itself.
  // This allows us to compute the set of arc breakers an operation
  // contributes to in one pass.
  for (auto *op : llvm::reverse(postOrder)) {
    auto mask = APInt::getZero(arcBreakers.size());
    for (auto *user : op->getUsers()) {
      auto it = faninMasks.find(user);
      if (it != faninMasks.end())
        mask |= it->second;
    }

    auto duplicateOp = faninMasks.insert({op, mask});
    (void)duplicateOp;
    assert(duplicateOp.second && "duplicate op in order");
  }

  // Group the operations by their fan-in mask.
  faninMaskGroups.clear();
  for (auto [op, mask] : faninMasks)
    if (!isArcBreakingOp(op) && !isa<hw::OutputOp>(op))
      faninMaskGroups[mask].insert(op);
  LLVM_DEBUG(llvm::dbgs() << "- Found " << faninMaskGroups.size()
                          << " fanin mask groups\n");

  return success();
}

void Converter::extractArcs(HWModuleOp module) {
  DenseMap<Value, Value> valueMapping;
  SmallVector<Value> inputs;
  SmallVector<Value> outputs;
  SmallVector<Type> inputTypes;
  SmallVector<Type> outputTypes;
  SmallVector<std::pair<OpOperand *, unsigned>> externalUses;

  arcUses.clear();
  for (auto &group : faninMaskGroups) {
    auto &opSet = group.second;
    OpBuilder builder(module);

    auto block = std::make_unique<Block>();
    builder.setInsertionPointToStart(block.get());
    valueMapping.clear();
    inputs.clear();
    outputs.clear();
    inputTypes.clear();
    outputTypes.clear();
    externalUses.clear();

    Operation *lastOp = nullptr;
    // TODO: Remove the elements from the post order as we go.
    for (auto *op : postOrder) {
      if (!opSet.contains(op))
        continue;
      lastOp = op;
      op->remove();
      builder.insert(op);
      for (auto &operand : op->getOpOperands()) {
        if (opSet.contains(operand.get().getDefiningOp()))
          continue;
        auto &mapped = valueMapping[operand.get()];
        if (!mapped) {
          mapped = block->addArgument(operand.get().getType(),
                                      operand.get().getLoc());
          inputs.push_back(operand.get());
          inputTypes.push_back(mapped.getType());
        }
        operand.set(mapped);
      }
      for (auto result : op->getResults()) {
        bool anyExternal = false;
        for (auto &use : result.getUses()) {
          if (!opSet.contains(use.getOwner())) {
            anyExternal = true;
            externalUses.push_back({&use, outputs.size()});
          }
        }
        if (anyExternal) {
          outputs.push_back(result);
          outputTypes.push_back(result.getType());
        }
      }
    }
    assert(lastOp);
    builder.create<arc::OutputOp>(lastOp->getLoc(), outputs);

    // Create the arc definition.
    builder.setInsertionPoint(module);
    auto defOp = builder.create<DefineOp>(
        lastOp->getLoc(),
        builder.getStringAttr(
            globalNamespace.newName(module.getModuleName() + "_arc")),
        builder.getFunctionType(inputTypes, outputTypes));
    defOp.getBody().push_back(block.release());

    // Create the call to the arc definition to replace the operations that
    // we have just extracted.
    builder.setInsertionPoint(module.getBodyBlock()->getTerminator());
    auto arcOp = builder.create<CallOp>(lastOp->getLoc(), defOp, inputs);
    arcUses.push_back(arcOp);
    for (auto [use, resultIdx] : externalUses)
      use->set(arcOp.getResult(resultIdx));
  }
}

LogicalResult Converter::absorbRegs(HWModuleOp module) {
  // Handle the trivial cases where all of an arc's results are used by
  // exactly one register each.
  unsigned outIdx = 0;
  unsigned numTrivialRegs = 0;
  for (auto callOp : arcUses) {
    auto stateOp = dyn_cast<StateOp>(callOp.getOperation());
    Value clock = stateOp ? stateOp.getClock() : Value{};
    Value reset;
    SmallVector<Value> initialValues;
    SmallVector<seq::CompRegOp> absorbedRegs;
    SmallVector<Attribute> absorbedNames(callOp->getNumResults(), {});
    if (auto names = callOp->getAttrOfType<ArrayAttr>("names"))
      absorbedNames.assign(names.getValue().begin(), names.getValue().end());

    // Go through all every arc result and collect the single register that uses
    // it. If a result has multiple uses or is used by something other than a
    // register, skip the arc for now and handle it later.
    bool isTrivial = true;
    for (auto result : callOp->getResults()) {
      if (!result.hasOneUse()) {
        isTrivial = false;
        break;
      }
      auto regOp = dyn_cast<seq::CompRegOp>(result.use_begin()->getOwner());
      if (!regOp || regOp.getInput() != result ||
          (clock && clock != regOp.getClk())) {
        isTrivial = false;
        break;
      }

      clock = regOp.getClk();
      reset = regOp.getReset();

      // Check that if the register has a reset, it is to a constant zero
      if (reset) {
        Value resetValue = regOp.getResetValue();
        Operation *op = resetValue.getDefiningOp();
        if (!op)
          return regOp->emitOpError(
              "is reset by an input; not supported by ConvertToArcs");
        if (auto constant = dyn_cast<hw::ConstantOp>(op)) {
          if (constant.getValue() != 0)
            return regOp->emitOpError("is reset to a constant non-zero value; "
                                      "not supported by ConvertToArcs");
        } else {
          return regOp->emitOpError("is reset to a value that is not clearly "
                                    "constant; not supported by ConvertToArcs");
        }
      }

      initialValues.push_back(regOp.getPowerOnValue());

      absorbedRegs.push_back(regOp);
      // If we absorb a register into the arc, the arc effectively produces that
      // register's value. So if the register had a name, ensure that we assign
      // that name to the arc's output.
      absorbedNames[result.getResultNumber()] = regOp.getNameAttr();
    }

    // If this wasn't a trivial case keep the arc around for a second iteration.
    if (!isTrivial) {
      arcUses[outIdx++] = callOp;
      continue;
    }
    ++numTrivialRegs;

    // Set the arc's clock to the clock of the registers we've absorbed, bump
    // the latency up by one to account for the registers, add the reset if
    // present and update the output names. Then replace the registers.

    auto arc = dyn_cast<StateOp>(callOp.getOperation());
    if (arc) {
      arc.getClockMutable().assign(clock);
      arc.setLatency(arc.getLatency() + 1);
    } else {
      mlir::IRRewriter rewriter(module->getContext());
      rewriter.setInsertionPoint(callOp);
      arc = rewriter.replaceOpWithNewOp<StateOp>(
          callOp.getOperation(),
          callOp.getCallableForCallee().get<SymbolRefAttr>(),
          callOp->getResultTypes(), clock, Value{}, 1, callOp.getArgOperands());
    }

    if (reset) {
      if (arc.getReset())
        return arc.emitError(
            "StateOp tried to infer reset from CompReg, but already "
            "had a reset.");
      arc.getResetMutable().assign(reset);
    }

    bool onlyDefaultInitializers =
        llvm::all_of(initialValues, [](auto val) -> bool { return !val; });

    if (!onlyDefaultInitializers) {
      if (!arc.getInitials().empty()) {
        return arc.emitError(
            "StateOp tried to infer initial values from CompReg, but already "
            "had an initial value.");
      }
      // Create 0 constants for default initialization
      for (unsigned i = 0; i < initialValues.size(); ++i) {
        if (!initialValues[i]) {
          OpBuilder zeroBuilder(arc);
          initialValues[i] = zeroBuilder.createOrFold<hw::ConstantOp>(
              arc.getLoc(),
              zeroBuilder.getIntegerAttr(arc.getResult(i).getType(), 0));
        }
      }
      arc.getInitialsMutable().assign(initialValues);
    }

    if (tapRegisters && llvm::any_of(absorbedNames, [](auto name) {
          return !cast<StringAttr>(name).getValue().empty();
        }))
      arc->setAttr("names", ArrayAttr::get(module.getContext(), absorbedNames));
    for (auto [arcResult, reg] : llvm::zip(arc.getResults(), absorbedRegs)) {
      auto it = arcBreakerIndices.find(reg);
      arcBreakers[it->second] = {};
      arcBreakerIndices.erase(it);
      reg.replaceAllUsesWith(arcResult);
      reg.erase();
    }
  }
  if (numTrivialRegs > 0)
    LLVM_DEBUG(llvm::dbgs() << "- Trivially converted " << numTrivialRegs
                            << " regs to arcs\n");
  arcUses.truncate(outIdx);

  // Group the remaining registers by their clock, their reset and the operation
  // they use as input. This will allow us to generally collapse registers
  // derived from the same arc into one shuffling arc.
  MapVector<std::tuple<Value, Value, Operation *>, SmallVector<seq::CompRegOp>>
      regsByInput;
  for (auto *op : arcBreakers)
    if (auto regOp = dyn_cast_or_null<seq::CompRegOp>(op)) {
      regsByInput[{regOp.getClk(), regOp.getReset(),
                   regOp.getInput().getDefiningOp()}]
          .push_back(regOp);
    }

  unsigned numMappedRegs = 0;
  for (auto [clockAndResetAndOp, regOps] : regsByInput) {
    numMappedRegs += regOps.size();
    OpBuilder builder(module);
    auto block = std::make_unique<Block>();
    builder.setInsertionPointToStart(block.get());

    SmallVector<Value> inputs;
    SmallVector<Value> outputs;
    SmallVector<Attribute> names;
    SmallVector<Type> types;
    SmallVector<Value> initialValues;
    SmallDenseMap<Value, unsigned> mapping;
    SmallVector<unsigned> regToOutputMapping;
    for (auto regOp : regOps) {
      auto it = mapping.find(regOp.getInput());
      if (it == mapping.end()) {
        it = mapping.insert({regOp.getInput(), inputs.size()}).first;
        inputs.push_back(regOp.getInput());
        types.push_back(regOp.getType());
        outputs.push_back(block->addArgument(regOp.getType(), regOp.getLoc()));
        names.push_back(regOp->getAttrOfType<StringAttr>("name"));
        initialValues.push_back(regOp.getPowerOnValue());
      }
      regToOutputMapping.push_back(it->second);
    }

    auto loc = regOps.back().getLoc();
    builder.create<arc::OutputOp>(loc, outputs);

    builder.setInsertionPoint(module);
    auto defOp =
        builder.create<DefineOp>(loc,
                                 builder.getStringAttr(globalNamespace.newName(
                                     module.getModuleName() + "_arc")),
                                 builder.getFunctionType(types, types));
    defOp.getBody().push_back(block.release());

    builder.setInsertionPoint(module.getBodyBlock()->getTerminator());

    bool onlyDefaultInitializers =
        llvm::all_of(initialValues, [](auto val) -> bool { return !val; });

    if (onlyDefaultInitializers)
      initialValues.clear();
    else
      for (unsigned i = 0; i < initialValues.size(); ++i) {
        if (!initialValues[i])
          initialValues[i] = builder.createOrFold<hw::ConstantOp>(
              loc, builder.getIntegerAttr(types[i], 0));
      }

    auto arcOp =
        builder.create<StateOp>(loc, defOp, std::get<0>(clockAndResetAndOp),
                                /*enable=*/Value{}, 1, inputs, initialValues);
    auto reset = std::get<1>(clockAndResetAndOp);
    if (reset)
      arcOp.getResetMutable().assign(reset);
    if (tapRegisters && llvm::any_of(names, [](auto name) {
          return !cast<StringAttr>(name).getValue().empty();
        }))
      arcOp->setAttr("names", builder.getArrayAttr(names));
    for (auto [reg, resultIdx] : llvm::zip(regOps, regToOutputMapping)) {
      reg.replaceAllUsesWith(arcOp.getResult(resultIdx));
      reg.erase();
    }
  }

  if (numMappedRegs > 0)
    LLVM_DEBUG(llvm::dbgs() << "- Mapped " << numMappedRegs << " regs to "
                            << regsByInput.size() << " shuffling arcs\n");

  return success();
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace circt {
#define GEN_PASS_DEF_CONVERTTOARCS
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

namespace {
struct ConvertToArcsPass : public impl::ConvertToArcsBase<ConvertToArcsPass> {
  using ConvertToArcsBase::ConvertToArcsBase;

  void runOnOperation() override {
    Converter converter;
    converter.tapRegisters = tapRegisters;
    if (failed(converter.run(getOperation())))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
circt::createConvertToArcsPass(const ConvertToArcsOptions &options) {
  return std::make_unique<ConvertToArcsPass>(options);
}
