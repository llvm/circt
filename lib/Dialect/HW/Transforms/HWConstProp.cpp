//===- HWConstProp.cpp - Inter-module constant propagation ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the `HWConstProp` pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hw-constprop"

using namespace mlir;
using namespace circt;
using namespace hw;

//===----------------------------------------------------------------------===//
// Constant propagation helper
//===----------------------------------------------------------------------===//

namespace {
class ConstantPropagation {
public:
  ConstantPropagation(hw::InstanceGraph &graph) : graph(graph) {}

  void initialize(HWModuleOp module);
  void propagate();
  void markUnknownValuesOverdefined(hw::HWModuleOp module);
  std::pair<unsigned, unsigned> fold();

public:
  void enqueue(Value value, IntegerAttr attr);
  void mark(Value value, IntegerAttr attr);
  void markOverdefined(Value value) { mark(value, IntegerAttr{}); }
  void propagate(Operation *op);

  /**
   * Returns the lattice value associated with an SSA value.
   *
   * `std::nullopt` is unknown, `IntegerAttr{}` is overdefined.
   */
  std::optional<IntegerAttr> map(Value value);

private:
  hw::InstanceGraph &graph;
  DenseMap<Value, IntegerAttr> values;
  DenseSet<Operation *> inQueue;
  SmallVector<Operation *> overdefQueue;
  SmallVector<Operation *> valueQueue;
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
          enqueue(cst, cst.getValueAttr());
        })
        .Case<HWInstanceLike>([&](auto inst) {
          // Mark external/generated module outputs as overdefined.
          bool hasUnknownTarget = llvm::any_of(
              inst.getReferencedModuleNamesAttr(), [&](Attribute ref) {
                Operation *referencedOp =
                    graph.lookup(cast<StringAttr>(ref))->getModule();
                auto module = dyn_cast_or_null<HWModuleOp>(referencedOp);
                return !module;
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

void ConstantPropagation::mark(Value value, IntegerAttr attr) {
  auto it = values.try_emplace(value, attr);
  if (!it.second) {
    if (it.first->second == attr)
      return;
    attr = it.first->second = IntegerAttr{};
  }
  enqueue(value, attr);
}

void ConstantPropagation::enqueue(Value value, IntegerAttr attr) {
  for (Operation *user : value.getUsers()) {
    if (inQueue.insert(user).second) {
      if (attr) {
        valueQueue.push_back(user);
      } else {
        overdefQueue.push_back(user);
      }
    }
  }
}

std::optional<IntegerAttr> ConstantPropagation::map(Value value) {
  if (auto constant = value.getDefiningOp<hw::ConstantOp>())
    return constant.getValueAttr();

  auto it = values.find(value);
  if (it == values.end())
    return std::nullopt;

  return it->second;
}

void ConstantPropagation::propagate() {
  while (!overdefQueue.empty() || !valueQueue.empty()) {
    while (!overdefQueue.empty()) {
      auto *op = overdefQueue.pop_back_val();
      inQueue.erase(op);
      propagate(op);
    }
    while (!valueQueue.empty()) {
      auto *op = valueQueue.pop_back_val();
      inQueue.erase(op);
      propagate(op);
    }
  }
}

void ConstantPropagation::markUnknownValuesOverdefined(hw::HWModuleOp module) {
  for (auto arg : module.getBodyBlock()->getArguments()) {
    if (!map(arg))
      markOverdefined(arg);
  }
  module.walk([&](Operation *op) {
    for (auto result : op->getResults()) {
      if (!map(result))
        markOverdefined(result);
    }
  });
}

void ConstantPropagation::propagate(Operation *op) {
  if (auto output = dyn_cast<OutputOp>(op)) {
    auto module = op->getParentOfType<HWModuleOp>();
    for (auto *node : graph[module]->uses()) {
      Operation *instLike = node->getInstance();
      if (!instLike)
        continue;

      auto inst = cast<HWInstanceLike>(instLike);
      for (auto [op, res] :
           llvm::zip(output.getOutputs(), inst->getResults())) {
        if (auto attr = map(op))
          mark(res, *attr);
      }
    }
    return;
  }

  if (auto inst = dyn_cast<HWInstanceLike>(op)) {
    for (auto ref : inst.getReferencedModuleNamesAttr()) {
      Operation *referencedOp =
          graph.lookup(cast<StringAttr>(ref))->getModule();
      auto module = dyn_cast_or_null<HWModuleOp>(referencedOp);
      if (!module)
        continue;

      Block *body = module.getBodyBlock();
      for (auto [op, arg] :
           llvm::zip(inst->getOperands(), body->getArguments())) {
        if (auto attr = map(op))
          mark(arg, *attr);
      }
    }
    return;
  }

  SmallVector<Attribute> operands;
  for (auto op : op->getOperands()) {
    auto attr = map(op);
    if (!attr)
      return;
    operands.push_back(*attr);
  }

  SmallVector<OpFoldResult, 1> results;
  if (succeeded(op->fold(operands, results)) && !results.empty()) {
    for (auto [res, value] : llvm::zip(op->getResults(), results)) {
      if (auto attr = dyn_cast<Attribute>(value)) {
        if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
          mark(res, intAttr);
          continue;
        }
      }
      mark(res, {});
    }
  } else {
    for (auto res : op->getResults()) {
      mark(res, {});
    }
  }
}

std::pair<unsigned, unsigned> ConstantPropagation::fold() {
  // Cache new constants in each module. Traverse the circuit to
  // populate the mapping with values to re-use.
  DenseMap<std::pair<HWModuleOp, IntegerAttr>, Value> constants;
  for (auto *node : graph) {
    Operation *moduleOp = node->getModule();
    if (!moduleOp)
      continue;
    auto module = dyn_cast<HWModuleOp>(moduleOp);
    if (!module)
      continue;
    for (Operation &op : *module.getBodyBlock()) {
      if (auto cst = dyn_cast<ConstantOp>(&op)) {
        constants.try_emplace({module, cst.getValueAttr()}, cst);
      }
    }
  }

  // Traverse the mapping from values to lattices and replace with constants.
  DenseSet<Operation *> toDelete;
  unsigned numFolded = 0;
  for (auto [value, attr] : values) {
    if (!attr)
      continue;

    ImplicitLocOpBuilder builder(value.getLoc(), value.getContext());

    HWModuleOp parent;
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      parent = cast<HWModuleOp>(arg.getOwner()->getParentOp());
    } else {
      parent = value.getDefiningOp()->getParentOfType<HWModuleOp>();
    }

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

    if (auto *op = value.getDefiningOp()) {
      if (op->use_empty() && mlir::isMemoryEffectFree(op)) {
        toDelete.insert(op);
      }
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
#define GEN_PASS_DEF_HWCONSTPROP
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

namespace {
struct HWConstPropPass
    : public circt::hw::impl::HWConstPropBase<HWConstPropPass> {
  void runOnOperation() override;
};
} // namespace

void HWConstPropPass::runOnOperation() {
  ConstantPropagation prop(getAnalysis<hw::InstanceGraph>());

  for (auto module : getOperation().getOps<HWModuleOp>())
    prop.initialize(module);

  prop.propagate();

  // Lattice states may remain overly optimistic due to dependency cycles
  // that can occur in non-Chisel designs. To address this, replace unknown
  // values with overdefined ones.
  for (auto module : getOperation().getOps<HWModuleOp>())
    prop.markUnknownValuesOverdefined(module);

  // Propagate again to fold constants that were overdefined before.
  prop.propagate();

  auto [numFolded, numErased] = prop.fold();
  numValuesFolded += numFolded;
  numOpsErased += numErased;

  markAnalysesPreserved<hw::InstanceGraph>();
}
