//===- HWDeadCodeElim.cpp - Elimination of dead ports ---------------------===//
//
// Proprietary and Confidential Software of SiFive Inc. All Rights Reserved.
// See the LICENSE file for license information.
// SPDX-License-Identifier: UNLICENSED
//
//===----------------------------------------------------------------------===//
//
// This file implements the `HWDeadCodeElim` pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hw-dead-code-elim"

using namespace mlir;
using namespace circt;

using llvm::SmallBitVector;

//===----------------------------------------------------------------------===//
// Liveness analysis for dead-code elimination
//===----------------------------------------------------------------------===//

namespace {
class Liveness {
public:
  Liveness(ModuleOp circuit, hw::InstanceGraph &instanceGraph)
      : circuit(circuit), instanceGraph(instanceGraph) {}

  LogicalResult run();

  // Return true if a value has users and should be preserved.
  bool isLive(Value value) { return liveValues.count(value); }

private:
  void markLive(Value value);
  void markLive(Operation *op);

  void propagate(Value result, sv::WireOp wire);
  void propagate(Value result, hw::InstanceOp inst);

private:
  ModuleOp circuit;
  hw::InstanceGraph &instanceGraph;

  DenseSet<Value> liveValues;
  SmallVector<Value> q;
};
} // namespace

// Returns true if an op can be deleted.
static bool canBeDeleted(Operation *op) {
  if (isa<sv::AssignOp>(op))
    return true;
  if (op->hasAttr("inner_sym"))
    return false;
  if (op->hasTrait<sv::ProceduralOp>())
    return false;
  if (op->hasTrait<sv::NonProceduralOp>())
    return false;
  return mlir::MemoryEffectOpInterface::hasNoEffect(op);
}

LogicalResult Liveness::run() {
  for (auto module : circuit.getOps<hw::HWModuleOp>()) {
    // Mark all public module input and output ports as live.
    if (module.isPublic()) {
      Block *body = module.getBodyBlock();
      for (auto arg : body->getArguments())
        markLive(arg);
      auto term = cast<hw::OutputOp>(body->getTerminator());
      for (auto output : term.getOutputs())
        markLive(output);
    }

    // Inside private modules, find ops which cannot be eliminated.
    module.walk([&](Operation *op) {
      if (isa<hw::HWModuleOp>(op))
        return;

      if (auto inst = dyn_cast<hw::InstanceOp>(op)) {
        // Mark all inputs to external modules as live.
        if (!isa<hw::HWModuleOp>(instanceGraph.getReferencedModule(inst)))
          for (auto operand : inst.getInputs())
            markLive(operand);
        return;
      }

      if (!canBeDeleted(op)) {
        // Mark all ops with side effects as live.
        markLive(op);
        return;
      }
    });
  }

  while (!q.empty()) {
    Value value = q.pop_back_val();
    if (auto arg = value.dyn_cast<BlockArgument>()) {
      // If a block argument is live, mark all corresponding instance
      // arguments as live at all instance sites.
      auto module = cast<hw::HWModuleOp>(arg.getOwner()->getParentOp());
      for (auto *record : instanceGraph[module]->uses()) {
        Operation *instLike = record->getInstance();
        auto inst = dyn_cast_or_null<hw::InstanceOp>(instLike);
        if (!inst)
          continue;
        markLive(inst.getInputs()[arg.getArgNumber()]);
      }
    } else {
      TypeSwitch<Operation *>(value.getDefiningOp())
          .Case<sv::WireOp, hw::InstanceOp>([&](auto op) {
            // Some ops require special propagation rules.
            propagate(value, op);
          })
          .Default([&](auto op) {
            // Otherwise, mark all operands live if any of the results is live.
            markLive(op);
          });
    }
  }
  return success();
}

void Liveness::markLive(Value value) {
  // Mark a value and queue it for propagation.
  if (liveValues.insert(value).second)
    q.push_back(value);

  // If the op is within a nested region, mark the operands of all parent
  // ops as live to capture implicit control dependencies.
  Operation *op = value.getDefiningOp();
  while (op && !isa<hw::HWModuleOp>(op)) {
    for (auto operand : op->getOperands())
      if (liveValues.insert(operand).second)
        q.push_back(operand);
    op = op->getParentOp();
  }
}

void Liveness::markLive(Operation *op) {
  for (auto result : op->getResults())
    if (liveValues.insert(result).second)
      q.push_back(result);

  while (op && !isa<hw::HWModuleOp>(op)) {
    for (auto operand : op->getOperands())
      if (liveValues.insert(operand).second)
        q.push_back(operand);
    op = op->getParentOp();
  }
}

void Liveness::propagate(Value result, sv::WireOp wire) {
  for (auto *user : wire->getUsers()) {
    if (!isa<sv::AssignOp, sv::PAssignOp, sv::BPAssignOp>(user))
      continue;
    if (user->getOperand(0) != wire)
      continue;
    markLive(user->getOperand(1));
  }
}

void Liveness::propagate(Value result, hw::InstanceOp inst) {
  Operation *targetOp = instanceGraph.getReferencedModule(inst);
  if (auto module = dyn_cast<hw::HWModuleOp>(targetOp)) {
    // If the target is a module, mark the corresponding output.
    Block *body = module.getBodyBlock();
    auto term = cast<hw::OutputOp>(body->getTerminator());
    unsigned portNo = result.cast<OpResult>().getResultNumber();
    markLive(term.getOutputs()[portNo]);
  } else {
    // Otherwise, over-approximate and mark all operands.
    for (auto operand : inst->getOperands())
      markLive(operand);
  }
}

//===----------------------------------------------------------------------===//
// Helper to rewrite module ports and bodies
//===----------------------------------------------------------------------===//

namespace {
class ModuleRewriter {
public:
  ModuleRewriter(Liveness &liveness, hw::InstanceGraph &instanceGraph)
      : liveness(liveness), instanceGraph(instanceGraph) {}

  LogicalResult run();

private:
  LogicalResult rewrite(hw::HWModuleOp module);

private:
  Liveness &liveness;
  hw::InstanceGraph &instanceGraph;

  /// For each module that was modified, keep a bit map of the live ports.
  DenseMap<hw::HWModuleOp, std::pair<SmallBitVector, SmallBitVector>> livePorts;
};
} // namespace

LogicalResult ModuleRewriter::run() {
  for (auto *instanceNode : llvm::post_order(&instanceGraph)) {
    Operation *op = instanceNode->getModule();
    if (!op)
      continue;
    auto module = dyn_cast<hw::HWModuleOp>(op);
    if (!module)
      continue;
    if (failed(rewrite(module)))
      return failure();
  }
  return success();
}

LogicalResult ModuleRewriter::rewrite(hw::HWModuleOp module) {
  LLVM_DEBUG(llvm::dbgs() << module.getName() << ":\n");
  SmallVector<unsigned> eraseInputs, eraseOutputs;

  // Identify the set of live arguments.
  Block *body = module.getBodyBlock();
  SmallBitVector liveArguments(module.getArgumentTypes().size());
  for (auto [index, value] : llvm::enumerate(body->getArguments())) {
    if (liveness.isLive(value)) {
      liveArguments.set(index);
      continue;
    }
    LLVM_DEBUG({
      auto port = module.getInOrInoutPort(index);
      llvm::dbgs() << " - input " << port.name << "\n";
    });
    eraseInputs.push_back(index);
  }

  // Identify the set of live results across all instance sites.
  SmallBitVector liveResults(module.getResultTypes().size());
  for (auto *node : instanceGraph[module]->uses()) {
    Operation *instOp = node->getInstance();
    auto inst = dyn_cast_or_null<hw::InstanceOp>(instOp);
    if (!inst) {
      liveResults.set();
      break;
    }
    for (auto [index, value] : llvm::enumerate(inst.getResults())) {
      if (liveness.isLive(value)) {
        liveResults.set(index);
        continue;
      }
    }
  }
  for (unsigned i = 0, n = liveResults.size(); i < n; ++i) {
    if (liveResults[i])
      continue;
    LLVM_DEBUG({
      auto port = module.getOutputPort(i);
      llvm::dbgs() << " - output " << port.name << "\n";
    });
    eraseOutputs.push_back(i);
  }

  assert(
      (!module.isPublic() || (eraseOutputs.empty() && eraseInputs.empty())) &&
      "cannot delete ports of a public module");

  // Erase dead operations.
  SmallVector<Operation *> toErase;
  module.walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<sv::AssignOp, sv::PAssignOp, sv::BPAssignOp>([&](auto assign) {
          if (liveness.isLive(assign.getDest()))
            return;
          toErase.push_back(assign);
        })
        .Case<sv::WireOp>([&](auto wire) {
          if (liveness.isLive(wire.getResult()))
            return;
          toErase.push_back(wire);
        })
        .Case<hw::InstanceOp>([&](auto inst) {
          Operation *targetOp = instanceGraph.getReferencedModule(inst);
          auto module = dyn_cast<hw::HWModuleOp>(targetOp);
          if (!module)
            return;

          auto it = livePorts.find(module);
          if (it == livePorts.end())
            return;
          auto &[liveArgs, liveResults] = it->second;

          ImplicitLocOpBuilder builder(inst.getLoc(), inst);

          SmallVector<Attribute> argNames;
          SmallVector<Value> arguments;
          for (auto [i, arg] : llvm::enumerate(inst.getInputs())) {
            if (!liveArgs[i])
              continue;
            arguments.push_back(arg);
            argNames.push_back(inst.getArgNames()[i]);
          }

          SmallVector<Type> types;
          SmallVector<Attribute> resultNames;
          for (auto [i, res] : llvm::enumerate(inst.getResults())) {
            if (!liveResults[i])
              continue;
            types.push_back(res.getType());
            resultNames.push_back(inst.getResultNames()[i]);
          }

          auto newInst = builder.create<hw::InstanceOp>(
              types, inst.instanceName(), inst.getModuleName(), arguments,
              builder.getArrayAttr(argNames), builder.getArrayAttr(resultNames),
              inst.getParameters(), inst.getInnerSymAttr());

          unsigned nextOutput = 0;
          for (auto &[i, v] : llvm::enumerate(inst.getResults()))
            if (liveResults[i])
              v.replaceAllUsesWith(newInst.getResults()[nextOutput++]);

          instanceGraph.replaceInstance(inst, newInst);
          toErase.push_back(inst);
        })
        .Case<hw::OutputOp>([&](auto oldOutput) {
          if (eraseOutputs.empty())
            return;

          // Rewrite the output op if any of the results are dead.
          SmallVector<Value> results;
          for (auto &[i, result] : llvm::enumerate(oldOutput.getOutputs())) {
            if (liveResults[i])
              results.push_back(result);
          }

          assert(results.size() == liveResults.count() && "invalid results");
          oldOutput->setOperands(results);
        })
        .Default([&](auto op) {
          if (!canBeDeleted(op))
            return;
          for (auto result : op->getResults())
            if (liveness.isLive(result))
              return;
          toErase.push_back(op);
        });
  });

  for (auto *op : llvm::reverse(toErase)) {
    LLVM_DEBUG(llvm::dbgs() << " - erase: " << *op << "\n");
    op->dropAllUses();
    op->erase();
  }

  // Adjust signature if anything changed.
  if (!eraseInputs.empty() || !eraseOutputs.empty()) {
    module.modifyPorts({}, {}, eraseInputs, eraseOutputs);
    livePorts.try_emplace(module, std::make_pair(std::move(liveArguments),
                                                 std::move(liveResults)));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct HWDeadCodeElimPass : public sv::HWDeadCodeElimBase<HWDeadCodeElimPass> {
  void runOnOperation() override;
};
} // namespace

void HWDeadCodeElimPass::runOnOperation() {
  auto &instanceGraph = getAnalysis<hw::InstanceGraph>();
  auto circuit = getOperation();

  Liveness liveness(circuit, instanceGraph);
  if (failed(liveness.run()))
    return signalPassFailure();

  ModuleRewriter rewriter(liveness, instanceGraph);
  if (failed(rewriter.run()))
    return signalPassFailure();
}

std::unique_ptr<Pass> circt::sv::createHWDeadCodeElimPass() {
  return std::make_unique<HWDeadCodeElimPass>();
}
