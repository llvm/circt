//===- InferClockPorts.cpp - Infer clock module ports ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Retype `i1` module ports that carry clocks to `!seq.clock`.
//
// Outputs are only promoted if already driven by a clock, so that logic on a
// clock path (e.g. a gated clock `clk & en`) stays in the `i1` domain.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "seq-infer-clock-ports"

using namespace circt;
using namespace circt::seq;
using namespace mlir;

namespace circt {
namespace seq {
#define GEN_PASS_DEF_INFERCLOCKPORTS
#include "circt/Dialect/Seq/SeqPasses.h.inc"
} // namespace seq
} // namespace circt

namespace {

class InferClockPortsPass
    : public circt::seq::impl::InferClockPortsBase<InferClockPortsPass> {
public:
  using InferClockPortsBase::InferClockPortsBase;

  void runOnOperation() override;

private:
  hw::HWModuleOp getTarget(hw::InstanceOp instance);
  bool isRewritable(hw::HWModuleOp module);
  bool canPromote(hw::HWModuleOp module, size_t portId);
  bool canDriveClock(hw::HWModuleOp module, unsigned output);
  void markClocks(SmallVectorImpl<Value> &worklist);
  void rewriteModule(hw::HWModuleOp module);

  hw::InstanceGraph *instanceGraph = nullptr;

  /// Indexed by input and output number respectively.
  DenseMap<Operation *, BitVector> clockInputs;
  DenseMap<Operation *, BitVector> clockOutputs;

  /// Memoized so the analysis stays linear in the size of the design.
  DenseMap<Operation *, bool> rewritable;
  DenseMap<std::pair<Operation *, unsigned>, bool> drivesClock;
};

} // namespace

/// Returns false if already marked, so that each port is visited only once.
static bool markPort(DenseMap<Operation *, BitVector> &ports, Operation *op,
                     unsigned idx, unsigned numPorts) {
  auto &bits = ports[op];
  if (bits.empty())
    bits.resize(numPorts);
  if (bits.test(idx))
    return false;
  bits.set(idx);
  return true;
}

static hw::OutputOp getOutputOp(hw::HWModuleOp module) {
  return cast<hw::OutputOp>(module.getBodyBlock()->getTerminator());
}

/// Returns null for external and generated modules, whose ports we can't
/// retype.
hw::HWModuleOp InferClockPortsPass::getTarget(hw::InstanceOp instance) {
  auto *node =
      instanceGraph->lookupOrNull(instance.getReferencedModuleNameAttr());
  if (!node)
    return {};
  return dyn_cast<hw::HWModuleOp>(node->getModule().getOperation());
}

/// A module's signature may only change if we can update every instance of it.
bool InferClockPortsPass::isRewritable(hw::HWModuleOp module) {
  auto [it, inserted] = rewritable.try_emplace(module, false);
  if (!inserted)
    return it->second;
  if (!promotePublicPorts && module.isPublic())
    return false;
  for (auto *use : instanceGraph->lookup(module)->uses()) {
    // Entry node edge of a public module; there is no instance to update.
    if (!use->getInstance())
      continue;
    if (!use->getInstance<hw::InstanceOp>())
      return false;
    if (!isa<hw::HWModuleOp>(use->getParent()->getModule().getOperation()))
      return false;
  }
  return rewritable[module] = true;
}

/// Ports with an inner symbol may be referenced elsewhere, so keep their type.
bool InferClockPortsPass::canPromote(hw::HWModuleOp module, size_t portId) {
  auto port = module.getPort(portId);
  return !port.isInOut() && port.type.isInteger(1) && !port.getSym() &&
         isRewritable(module);
}

/// Only promote an output if its driver already is a clock, so no logic is
/// moved into the clock domain. Instance chains are followed iteratively to
/// avoid deep recursion on deep hierarchies.
bool InferClockPortsPass::canDriveClock(hw::HWModuleOp module,
                                        unsigned output) {
  SmallVector<std::pair<Operation *, unsigned>> chain;
  bool result = false;
  while (true) {
    auto key = std::make_pair(module.getOperation(), output);
    auto [it, inserted] = drivesClock.try_emplace(key, false);
    if (!inserted) {
      result = it->second;
      break;
    }
    chain.push_back(key);
    if (!canPromote(module, module.getPortIdForOutputId(output)))
      break;

    auto driver = getOutputOp(module).getOperand(output);
    if (driver.getDefiningOp<seq::FromClockOp>()) {
      result = true;
      break;
    }
    if (auto arg = dyn_cast<BlockArgument>(driver)) {
      result =
          canPromote(module, module.getPortIdForInputId(arg.getArgNumber()));
      break;
    }
    auto instance = driver.getDefiningOp<hw::InstanceOp>();
    if (!instance)
      break;
    module = getTarget(instance);
    if (!module)
      break;
    output = cast<OpResult>(driver).getResultNumber();
  }

  for (auto key : chain)
    drivesClock[key] = result;
  return result;
}

void InferClockPortsPass::markClocks(SmallVectorImpl<Value> &worklist) {
  while (!worklist.empty()) {
    auto value = worklist.pop_back_val();
    if (!value.getType().isInteger(1))
      continue;

    // A clock input needs a clock from every instantiation.
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      auto module = dyn_cast<hw::HWModuleOp>(arg.getOwner()->getParentOp());
      unsigned input = arg.getArgNumber();
      if (!module || !canPromote(module, module.getPortIdForInputId(input)) ||
          !markPort(clockInputs, module, input, module.getNumInputPorts()))
        continue;
      LLVM_DEBUG(llvm::dbgs() << "Clock input " << module.getModuleName() << "."
                              << module.getInputName(input) << "\n");
      ++numInputPortsPromoted;
      for (auto *use : instanceGraph->lookup(module)->uses())
        if (auto instance = use->getInstance<hw::InstanceOp>())
          worklist.push_back(instance.getOperand(input));
      continue;
    }

    if (auto instance = value.getDefiningOp<hw::InstanceOp>()) {
      auto module = getTarget(instance);
      unsigned output = cast<OpResult>(value).getResultNumber();
      if (!module || !canDriveClock(module, output) ||
          !markPort(clockOutputs, module, output, module.getNumOutputPorts()))
        continue;
      LLVM_DEBUG(llvm::dbgs() << "Clock output " << module.getModuleName()
                              << "." << module.getOutputName(output) << "\n");
      ++numOutputPortsPromoted;
      worklist.push_back(getOutputOp(module).getOperand(output));
      continue;
    }

    // Anything else stays `i1` behind its `seq.to_clock`.
  }
}

void InferClockPortsPass::rewriteModule(hw::HWModuleOp module) {
  auto clockType = seq::ClockType::get(&getContext());
  auto *body = module.getBodyBlock();
  auto inputs = clockInputs.lookup(module);
  auto outputs = clockOutputs.lookup(module);
  SmallVector<std::pair<hw::InstanceOp, hw::HWModuleOp>> instances;
  module.walk([&](hw::InstanceOp instance) {
    if (auto target = getTarget(instance))
      if (clockInputs.count(target) || clockOutputs.count(target))
        instances.push_back({instance, target});
  });
  if (inputs.none() && outputs.none() && instances.empty())
    return;

  // Candidates for folding once all operands are updated.
  SmallSetVector<seq::FromClockOp, 8> casts;

  auto retype = [&](Value value, OpBuilder &builder) {
    value.setType(clockType);
    // Existing users still expect `i1`.
    auto cast = seq::FromClockOp::create(builder, value.getLoc(), value);
    value.replaceAllUsesExcept(cast, cast);
    casts.insert(cast);
  };

  // Look through `seq.from_clock` to avoid a redundant cast pair.
  auto toClock = [&](Value value, OpBuilder &builder) -> Value {
    if (auto cast = value.getDefiningOp<seq::FromClockOp>()) {
      casts.insert(cast);
      return cast.getInput();
    }
    return seq::ToClockOp::create(builder, value.getLoc(), value);
  };

  // Retype values first so `toClock` can look through the new casts.
  auto builder = OpBuilder::atBlockBegin(body);
  for (unsigned input : inputs.set_bits())
    retype(body->getArgument(input), builder);
  for (auto [instance, target] : instances) {
    builder.setInsertionPointAfter(instance);
    for (unsigned output : clockOutputs.lookup(target).set_bits())
      retype(instance.getResult(output), builder);
  }

  auto outputOp = getOutputOp(module);
  builder.setInsertionPoint(outputOp);
  for (unsigned output : outputs.set_bits())
    outputOp.setOperand(output, toClock(outputOp.getOperand(output), builder));
  for (auto [instance, target] : instances) {
    builder.setInsertionPoint(instance);
    for (unsigned input : clockInputs.lookup(target).set_bits())
      instance.setOperand(input, toClock(instance.getOperand(input), builder));
  }

  // Pre-existing `seq.to_clock` users of retyped values are now redundant.
  for (auto cast : casts) {
    for (auto *user : llvm::make_early_inc_range(cast->getUsers())) {
      if (auto toClockOp = dyn_cast<seq::ToClockOp>(user)) {
        toClockOp.replaceAllUsesWith(cast.getInput());
        toClockOp.erase();
        ++numCastsRemoved;
      }
    }
    if (cast.use_empty())
      cast.erase();
  }

  if (inputs.none() && outputs.none())
    return;
  auto type = module.getHWModuleType();
  SmallVector<hw::ModulePort> ports(type.getPorts());
  for (unsigned input : inputs.set_bits())
    ports[type.getPortIdForInputId(input)].type = clockType;
  for (unsigned output : outputs.set_bits())
    ports[type.getPortIdForOutputId(output)].type = clockType;
  module.setHWModuleType(hw::ModuleType::get(&getContext(), ports));
}

void InferClockPortsPass::runOnOperation() {
  instanceGraph = &getAnalysis<hw::InstanceGraph>();
  clockInputs.clear();
  clockOutputs.clear();
  rewritable.clear();
  drivesClock.clear();

  SmallVector<Value> worklist;
  getOperation().walk(
      [&](seq::ToClockOp op) { worklist.push_back(op.getInput()); });
  markClocks(worklist);

  if (clockInputs.empty() && clockOutputs.empty())
    return markAllAnalysesPreserved();

  for (auto module : getOperation().getOps<hw::HWModuleOp>())
    rewriteModule(module);
  // Only port types change, not the hierarchy.
  markAnalysesPreserved<hw::InstanceGraph>();
}
