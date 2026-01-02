//===- HWIMConstProp.cpp - Inter-module constant propagation --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements inter-module constant propagation for HW modules using
// a sparse conditional constant propagation (SCCP) style algorithm.
//
// The algorithm uses a three-state lattice for each SSA value:
//   - Unknown: value not yet analyzed (optimistic assumption)
//   - Constant: value is a known compile-time constant
//   - Overdefined: value is not constant (conservative)
//
// The pass operates in phases:
//   1. Initialize: Mark public inputs and external module outputs as
//      overdefined; seed worklists with constant operations.
//   2. Propagate: Process worklists until fixed point, propagating constants
//      across instance boundaries (inputs to module args, outputs to results).
//   3. Finalize: Mark remaining unknown values as overdefined to handle cycles.
//   4. Fold: Replace constant-valued SSA values with hw.constant ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE                                                             \
  impl::HWIMConstPropBase<HWIMConstPropPass>::getArgumentName().data()

using namespace mlir;
using namespace circt;
using namespace hw;

//===----------------------------------------------------------------------===//
// Constant propagation helper
//===----------------------------------------------------------------------===//

namespace {
/// Helper class that performs inter-module constant propagation using a
/// sparse conditional constant propagation (SCCP) style algorithm.
///
/// The lattice has three states:
/// - Unknown: value not yet visited (represented by `std::nullopt`)
/// - Constant: value is a known constant (represented by `IntegerAttr`)
/// - Overdefined: value is not constant (represented by `IntegerAttr{}`)
class ConstantPropagation {
public:
  ConstantPropagation(hw::InstanceGraph &graph) : graph(graph) {}

  /// Initializes the lattice for a module by marking public inputs as
  /// overdefined and processing constants and instances.
  void initialize(HWModuleOp module);

  /// Runs the propagation algorithm until a fixed point is reached.
  void propagate();

  /// Marks all values that remain unknown as overdefined. This handles
  /// cycles in the dataflow graph that could leave values optimistically
  /// unknown.
  void markUnknownValuesOverdefined(hw::HWModuleOp module);

  /// Replaces values with their constant representations and erases dead ops.
  /// Returns the number of values folded and ops erased.
  std::pair<unsigned, unsigned> fold();

  /// Adds all users of a value to the worklist for processing.
  void addToWorklist(Value value, IntegerAttr attr);

  /// Marks a value with a lattice state. If the value already has a different
  /// constant, it becomes overdefined.
  void setLattice(Value value, IntegerAttr attr);

  /// Marks a value as overdefined (not constant).
  void markOverdefined(Value value) { setLattice(value, IntegerAttr{}); }

  /// Propagates lattice values through an operation.
  void propagate(Operation *op);

  /// Returns the lattice value associated with an SSA value.
  /// `std::nullopt` is unknown, `IntegerAttr{}` is overdefined.
  std::optional<IntegerAttr> getLattice(Value value);

private:
  /// The instance graph for looking up module references.
  hw::InstanceGraph &graph;

  /// Maps SSA values to their lattice state.
  DenseMap<Value, IntegerAttr> constValues;

  /// Set of operations currently in the worklist.
  DenseSet<Operation *> worklist;

  /// Two worklists are used: overdefined values are processed first to avoid
  /// redundant recomputation when operands later become overdefined.
  SmallVector<Operation *> overdefQ;
  SmallVector<Operation *> constQ;
};
} // namespace

void ConstantPropagation::initialize(HWModuleOp module) {
  if (module.isPublic()) {
    // Mark public module inputs as overdefined.
    for (auto arg : module.getBodyBlock()->getArguments())
      markOverdefined(arg);
  }

  module.walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<ConstantOp>([&](auto cst) {
          // Constants are omitted from the mapping, but their
          // users are enqueued for propagation.
          addToWorklist(cst, cst.getValueAttr());
        })
        .Case<HWInstanceLike>([&](auto inst) {
          // Mark external/generated module outputs as overdefined.
          bool hasUnknownTarget = llvm::any_of(
              inst.getReferencedModuleNamesAttr(), [&](Attribute ref) {
                Operation *referencedOp =
                    graph.lookup(cast<StringAttr>(ref))->getModule();
                return !isa_and_nonnull<HWModuleOp>(referencedOp);
              });

          if (hasUnknownTarget) {
            for (auto result : inst->getResults())
              markOverdefined(result);
          }
        })
        .Case<hw::WireOp>([&](auto wire) {
          // Mark wires as overdefined since they can be targeted by force.
          markOverdefined(wire.getResult());
        })
        .Default([&](auto op) {
          if (op->getNumResults() == 0)
            return;
          // Mark all non-comb ops and non-integer types as overdefined.
          bool isFoldable = hw::isCombinational(op);
          for (auto result : op->getResults()) {
            Type ty = result.getType();
            if (!hw::type_isa<IntegerType>(ty) || !isFoldable)
              markOverdefined(result);
          }
        });
  });
}

void ConstantPropagation::setLattice(Value value, IntegerAttr attr) {
  auto it = constValues.try_emplace(value, attr);
  if (!it.second) {
    if (it.first->second == attr)
      return;
    attr = it.first->second = IntegerAttr{};
  }
  // Enqueue users to propagate the updated lattice value.
  addToWorklist(value, attr);
}

void ConstantPropagation::addToWorklist(Value value, IntegerAttr attr) {
  for (Operation *user : value.getUsers()) {
    if (worklist.insert(user).second) {
      if (attr)
        constQ.push_back(user);
      else
        overdefQ.push_back(user);
    }
  }
}

std::optional<IntegerAttr> ConstantPropagation::getLattice(Value value) {
  if (auto constant = value.getDefiningOp<hw::ConstantOp>())
    return constant.getValueAttr();

  auto it = constValues.find(value);
  if (it == constValues.end())
    return std::nullopt;

  return it->second;
}

void ConstantPropagation::propagate() {
  while (!overdefQ.empty() || !constQ.empty()) {
    // Process overdefined first: once overdefined, values stay overdefined,
    // so processing them first avoids recomputing constants that will later
    // become overdefined anyway.
    while (!overdefQ.empty()) {
      auto *op = overdefQ.pop_back_val();
      worklist.erase(op);
      propagate(op);
    }
    while (!constQ.empty()) {
      auto *op = constQ.pop_back_val();
      worklist.erase(op);
      propagate(op);
    }
  }
}

void ConstantPropagation::markUnknownValuesOverdefined(hw::HWModuleOp module) {
  for (auto arg : module.getBodyBlock()->getArguments()) {
    if (!getLattice(arg))
      markOverdefined(arg);
  }
  module.walk([&](Operation *op) {
    for (auto result : op->getResults()) {
      if (!getLattice(result))
        markOverdefined(result);
    }
  });
}

void ConstantPropagation::propagate(Operation *op) {
  // Handle output ops: propagate values from module outputs to instance
  // results.
  if (auto output = dyn_cast<OutputOp>(op)) {
    auto module = output->getParentOfType<HWModuleOp>();
    for (auto *node : graph[module]->uses()) {
      Operation *instLike = node->getInstance();
      if (!instLike)
        continue;

      auto inst = cast<HWInstanceLike>(instLike);
      for (auto [out, res] :
           llvm::zip(output.getOutputs(), inst->getResults())) {
        if (auto attr = getLattice(out))
          setLattice(res, *attr);
      }
    }
    return;
  }

  // Handle instances: propagate values from instance inputs to module
  // arguments.
  if (auto inst = dyn_cast<HWInstanceLike>(op)) {
    for (auto ref : inst.getReferencedModuleNamesAttr()) {
      Operation *referencedOp =
          graph.lookup(cast<StringAttr>(ref))->getModule();
      auto module = dyn_cast_or_null<HWModuleOp>(referencedOp);
      if (!module)
        continue;

      Block *body = module.getBodyBlock();
      for (auto [operand, arg] :
           llvm::zip(inst->getOperands(), body->getArguments())) {
        if (auto attr = getLattice(operand))
          setLattice(arg, *attr);
      }
    }
    return;
  }

  // For other operations, try to fold using constant operands.
  // If any operand is unknown, defer processing.
  SmallVector<Attribute> operands;
  for (auto operand : op->getOperands()) {
    auto attr = getLattice(operand);
    if (!attr)
      return;
    operands.push_back(*attr);
  }

  // Attempt to fold the operation with constant operands.
  SmallVector<OpFoldResult, 1> results;
  if (succeeded(op->fold(operands, results)) && !results.empty()) {
    for (auto [res, value] : llvm::zip(op->getResults(), results)) {
      if (auto attr = dyn_cast<Attribute>(value)) {
        if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
          setLattice(res, intAttr);
          continue;
        }
      }
      // Non-integer fold results become overdefined.
      setLattice(res, {});
    }
  } else {
    // Folding failed, mark results as overdefined.
    for (auto res : op->getResults())
      setLattice(res, {});
  }
}

std::pair<unsigned, unsigned> ConstantPropagation::fold() {
  // Build a cache of existing constants per module to avoid creating
  // duplicate hw.constant ops when folding.
  DenseMap<std::pair<HWModuleOp, IntegerAttr>, Value> constants;
  for (auto *node : graph) {
    Operation *moduleOp = node->getModule();
    auto module = llvm::dyn_cast_or_null<HWModuleOp>(moduleOp);
    if (!module)
      continue;
    for (Operation &op : *module.getBodyBlock()) {
      if (auto cst = dyn_cast<ConstantOp>(&op))
        constants.try_emplace({module, cst.getValueAttr()}, cst);
    }
  }

  // Traverse the mapping from values to lattices and replace with constants.
  DenseSet<Operation *> toDelete;
  unsigned numFolded = 0;
  for (auto [value, attr] : constValues) {
    if (!attr)
      continue;

    ImplicitLocOpBuilder builder(value.getLoc(), value.getContext());

    HWModuleOp parent;
    if (auto arg = dyn_cast<BlockArgument>(value))
      parent = cast<HWModuleOp>(arg.getOwner()->getParentOp());
    else
      parent = value.getDefiningOp()->getParentOfType<HWModuleOp>();

    auto it = constants.try_emplace({parent, attr}, Value{});
    if (it.second) {
      builder.setInsertionPointToStart(parent.getBodyBlock());
      it.first->second = builder.create<ConstantOp>(value.getType(), attr);
    }

    value.replaceAllUsesWith(it.first->second);
    LLVM_DEBUG({
      llvm::dbgs() << "In " << parent.getModuleName() << ": Replace with "
                   << attr << ": " << value << '\n';
    });

    ++numFolded;

    if (auto *defOp = value.getDefiningOp()) {
      if (defOp->use_empty() && mlir::isMemoryEffectFree(defOp))
        toDelete.insert(defOp);
    }
  }

  for (Operation *op : toDelete)
    op->erase();

  return {numFolded, (unsigned)toDelete.size()};
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace circt {
namespace hw {
#define GEN_PASS_DEF_HWIMCONSTPROP
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

namespace {
class HWIMConstPropPass
    : public circt::hw::impl::HWIMConstPropBase<HWIMConstPropPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void HWIMConstPropPass::runOnOperation() {
  // Create the constant propagation helper using the instance graph.
  ConstantPropagation prop(getAnalysis<hw::InstanceGraph>());

  // Phase 1: Initialize lattice values for all modules.
  for (auto module : getOperation().getOps<HWModuleOp>())
    prop.initialize(module);

  // Phase 2: Propagate constants across module boundaries until fixed point.
  prop.propagate();

  // Phase 3: Lattice states may remain overly optimistic due to dependency
  // cycles that can occur in non-Chisel designs. To address this, replace
  // unknown values with overdefined ones.
  for (auto module : getOperation().getOps<HWModuleOp>())
    prop.markUnknownValuesOverdefined(module);

  // Phase 4: Propagate again to fold constants that were overdefined before.
  prop.propagate();

  // Phase 5: Replace values with constants and clean up dead operations.
  auto [numFolded, numErased] = prop.fold();
  numValuesFolded += numFolded;
  numOpsErased += numErased;

  // The instance graph is preserved since we don't add or remove modules.
  markAnalysesPreserved<hw::InstanceGraph>();
}
