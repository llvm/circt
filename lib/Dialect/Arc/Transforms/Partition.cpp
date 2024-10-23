#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include <random>

#define DEBUG_TYPE "arc-partition"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_PARTITION
#define GEN_PASS_DEF_PARTITIONCLONE
#define GEN_PASS_DEF_PARTITIONTASKCANONICALIZE
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace arc;

using llvm::SmallMapVector;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {

struct StateMap {
  DenseMap<Value, size_t> lookup;
  SmallVector<Value> ids;

  size_t numStates;
  size_t numOutputs;
};

// State -> state dependencies
struct StateDep {
  // deps[i] contains the dependency of state i (on other states). Inputs are
  // not included.
  SmallVector<BitVector> deps;

  // Is this state depended by a control flow?
  BitVector ctrlFlow;

  void debug(const StateMap &map) const {
    for (auto it : llvm::enumerate(map.ids)) {
      it.value().dump();
      llvm::dbgs() << "Depends on:\n";
      for (auto dep : deps[it.index()].set_bits())
        map.ids[dep].dump();
      llvm::dbgs() << "=======\n";
    }
  }
};

struct StateWeight {
  SmallVector<size_t> weights;

  void debug(const StateMap &map) const {
    for (auto it : llvm::enumerate(map.ids)) {
      it.value().dump();
      llvm::dbgs() << weights[it.index()] << "\n";
    }
  }
};

struct PartitionPlan {
  // Which chunk is responsible for updating this state?
  SmallVector<size_t> chunk;

  // The shadow writeback should happen synchronously (in the output task)
  BitVector syncWriteback;
};

struct PartitionPass : public arc::impl::PartitionBase<PartitionPass> {
  void runOnOperation() override;
  StateMap statAllStates(arc::ModelOp);
  StateDep statAllStateDeps(arc::ModelOp, const StateMap &);
  StateWeight statAllStateWeights(arc::ModelOp, const StateMap &);

  PartitionPlan planPartition(arc::ModelOp, const StateMap &, const StateDep &,
                              const StateWeight &);

  void splitTasks(arc::ModelOp, const StateMap &, const StateDep &,
                  const PartitionPlan &);

  PartitionPass(const PartitionOptions &opts = {});

  PartitionOptions _opts;
};

struct PartitionClonePass
    : public arc::impl::PartitionCloneBase<PartitionClonePass> {
  void runOnOperation() override;
};
} // namespace

PartitionPass::PartitionPass(const PartitionOptions &opts) : _opts(opts) {}

void PartitionPass::runOnOperation() {
  auto root = getOperation();
  const auto &states = statAllStates(root);
  const auto &deps = statAllStateDeps(root, states);
  const auto &weights = statAllStateWeights(root, states);

  LLVM_DEBUG(deps.debug(states));
  LLVM_DEBUG(weights.debug(states));

  const auto &partition = planPartition(root, states, deps, weights);
  splitTasks(root, states, deps, partition);
}

StateMap PartitionPass::statAllStates(arc::ModelOp root) {
  SmallVector<Value> collected;
  DenseMap<Value, size_t> lookup;

  // First pass: all IO & persistent state
  root->walk([&](arc::AllocStateOp op) {
    size_t id = collected.size();
    auto result = op.getResult();
    lookup.insert({result, id});
    collected.push_back(result);
  });

  size_t numStates = collected.size();

  root->walk([&](arc::RootOutputOp op) {
    size_t id = collected.size();
    auto result = op.getResult();
    lookup.insert({result, id});
    collected.push_back(result);
  });

  return StateMap{
      .lookup = lookup,
      .ids = collected,
      .numStates = numStates,
      .numOutputs = lookup.size() - numStates,
  };
}

StateDep PartitionPass::statAllStateDeps(arc::ModelOp root,
                                         const StateMap &map) {
  // Operations' dependency on states.
  // For control flow operations, this includes all parent control flow
  // dependencies, and the dependency of the condition For ordinary
  // computational operations, this includes all dependency carried by operands,
  // adn the dependency carried by control flow

  DenseMap<Operation *, BitVector> cache;
  SmallVector<BitVector> result;
  BitVector ctrlFlow(map.numStates);
  result.resize(map.ids.size(), BitVector(map.numStates));

  cache.insert({root, BitVector(map.numStates)});

  root->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op == root)
      return;
    if (isa<arc::AllocStateOp>(op))
      return;

    BitVector implicits = cache.at(op->getBlock()->getParentOp());
    ctrlFlow |= implicits;

    if (auto stateWrite = dyn_cast<arc::StateWriteOp>(op)) {
      auto state = stateWrite.getState();
      auto val = stateWrite.getValue();
      assert(val.getDefiningOp() && cache.contains(val.getDefiningOp()));
      auto deps = cache.at(val.getDefiningOp());

      if (isa<RootOutputOp>(state.getDefiningOp()))
        return;

      auto &slot = result[map.lookup.at(state)];
      slot |= deps;
      slot |= implicits;
      return;
    }

    BitVector ret = implicits;
    if (isa<arc::RootInputOp, arc::RootOutputOp>(op)) {
      // Do nothing
    } else if (isa<arc::StateReadOp>(op)) {
      auto stateVal = op->getOperand(0);
      if (map.lookup.contains(stateVal)) // Otherwise: is output
        ret.set(map.lookup.at(stateVal));
    } else {
      for (const Value &v : op->getOperands()) {
        if (Operation *def = v.getDefiningOp()) {
          assert(cache.contains(def));
          ret |= cache.at(def);
        } else {
          op->emitError("Unexpected block argument during partition");
        }
      }
    }

    cache.insert({op, ret});
  });

  return StateDep{
      .deps = result,
      .ctrlFlow = ctrlFlow,
  };
}

namespace {
size_t dfsWeight(Operation *op, DenseSet<Operation *> &visited) {
  if (!op)
    return 0; // Operand
  if (isa<StateReadOp>(op))
    return 0;
  if (visited.contains(op))
    return 0;

  // Approximate the weight of the op by the largest width of all inputs and
  // outputs
  size_t maxWidth = 0;
  size_t operandWidth = 0;
  for (const auto &operand : op->getOperands()) {
    size_t width = 0;
    if (operand.getType().isInteger()) {
      width = operand.getType().getIntOrFloatBitWidth();
    } else if (isa<seq::ClockType>(operand.getType())) {
      width = 1;
    } else {
      op->emitError("Use of non-integer operand");
      continue;
    }

    if (width > maxWidth)
      maxWidth = width;
    operandWidth += dfsWeight(operand.getDefiningOp(), visited);
  }

  for (const auto &result : op->getResults()) {
    if (!result.getType().isInteger())
      op->emitError("Use of non-integer result");

    auto width = result.getType().getIntOrFloatBitWidth();
    if (width > maxWidth)
      maxWidth = width;
  }

  visited.insert(op);
  return maxWidth + operandWidth;
}
} // namespace

StateWeight PartitionPass::statAllStateWeights(arc::ModelOp op,
                                               const StateMap &map) {
  SmallVector<std::pair<size_t, DenseSet<Operation *>>> ctx;
  ctx.resize(map.ids.size());

  op->walk([&](StateWriteOp op) {
    size_t id = map.lookup.at(op.getState());
    auto &slot = ctx[id];
    slot.first += dfsWeight(op.getValue().getDefiningOp(), slot.second);
  });

  return StateWeight{
      .weights = llvm::map_to_vector(ctx, [](auto e) { return e.first; })};
}

PartitionPlan PartitionPass::planPartition(arc::ModelOp, const StateMap &map,
                                           const StateDep &deps,
                                           const StateWeight &) {
  // FIXME: impl
  std::mt19937 gen(0x19260817);

  std::uniform_int_distribution<size_t> dist(0, _opts.chunks - 1);

  SmallVector<size_t> chunk;
  chunk.reserve(map.ids.size());
  for (size_t i = 0; i < map.ids.size(); ++i) {
    chunk.push_back(dist(gen));
  }

  return PartitionPlan{
      .chunk = chunk,
      .syncWriteback = deps.ctrlFlow,
  };
}

void PartitionPass::splitTasks(arc::ModelOp root, const StateMap &map,
                               const StateDep &deps,
                               const PartitionPlan &plan) {
  OpBuilder builder(root);

  // Create shadow storages for all chunks, after the last alloc
  // FIXME: find last alloc
  SmallVector<TypedValue<StorageType>> shadowStorages;
  shadowStorages.reserve(_opts.chunks);
  for (size_t t = 0; t < _opts.chunks; ++t) {
    builder.setInsertionPointToStart(&root.getBodyBlock());
    shadowStorages.push_back(builder.create<AllocStorageOp>(
        root.getLoc(), StorageType::get(root.getContext(), 0),
        root.getBody().getArgument(0)));
  }

  // States that are read by other chunks
  BitVector globallyVisibleStates(map.numStates);
  for (size_t i = 0; i < map.numStates; ++i) {
    auto localChunk = plan.chunk[i];
    for (auto dep : deps.deps[i].set_bits()) {
      auto remoteChunk = plan.chunk[dep];
      if (remoteChunk != localChunk)
        globallyVisibleStates.set(dep);
    }
  }

  // Allocate shadow states for globally visible states in their respective
  // shadow storages
  DenseMap<TypedValue<StateType>, TypedValue<StateType>> shadows;
  root.walk([&](AllocStateOp alloc) {
    auto state = alloc.getState();
    if (!globallyVisibleStates.test(map.lookup.at(state)))
      return;
    auto chunk = plan.chunk[map.lookup.at(state)];

    auto storage = shadowStorages[chunk];

    builder.setInsertionPointAfter(alloc);
    auto shadowAlloc =
        builder.create<AllocStateOp>(alloc.getLoc(), alloc.getType(), storage);
    shadows.insert({state, shadowAlloc.getState()});
  });

  // Re-map all state writes, create in-place shadow write pairs
  root.walk([&](StateWriteOp write) {
    auto state = write.getState();

    // Is something we just inserted
    if (isa<TaskOp>(write->getBlock()->getParentOp()))
      return;

    AllocStateOp alloc = dyn_cast<AllocStateOp>(state.getDefiningOp());
    // TODO: extmod
    if (!alloc) {
      assert(write->getBlock() == &root.getBodyBlock() &&
             "Writes to output should only happens at model root");

      builder.setInsertionPointAfter(write);
      auto mainTask = builder.create<TaskOp>(write.getLoc(),
                                             builder.getStringAttr("output"));
      mainTask.getBodyRegion().emplaceBlock();
      write->moveBefore(&mainTask.getBody().front(),
                        mainTask.getBody().front().end());

      return;
    }

    // TODO: condition and enable
    builder.setInsertionPointAfter(write);
    auto mainTask = builder.create<TaskOp>(
        write.getLoc(), builder.getStringAttr(
                            std::to_string(plan.chunk[map.lookup.at(state)])));
    mainTask.getBodyRegion().emplaceBlock();
    if (!shadows.contains(
            state)) { // Only locally visible. Safe to just directly write into
      write->moveBefore(&mainTask.getBody().front(),
                        mainTask.getBody().front().end());
    } else { // Is globally visible state
      // In main task, write into shadow
      builder.setInsertionPointToStart(&mainTask.getBody().front());
      builder.create<StateWriteOp>(write.getLoc(), shadows.at(state),
                                   write.getValue(), Value());

      // Create separate sync task, read from shadow, write into main
      builder.setInsertionPointAfter(mainTask);
      auto syncStage =
          plan.syncWriteback.test(map.lookup.at(state))
              ? std::string("output")
              : std::to_string(plan.chunk[map.lookup.at(state)]) + "_sync";
      auto syncTask = builder.create<TaskOp>(write.getLoc(),
                                             builder.getStringAttr(syncStage));
      auto syncBlock = &syncTask.getBodyRegion().emplaceBlock();
      builder.setInsertionPointToStart(syncBlock);
      auto readout =
          builder.create<StateReadOp>(write.getLoc(), shadows.at(state));
      builder.create<StateWriteOp>(write.getLoc(), state, readout.getValue(),
                                   Value());
      write.erase();
    }
  });
}

void PartitionClonePass::runOnOperation() {
  ModelOp root = getOperation();
  auto builder = OpBuilder(root);

  // Collect all named tasks
  DenseSet<StringRef> taskNames;
  root->walk([&](TaskOp task) {
    if (auto name = task.getTaskName())
      taskNames.insert(*name);
  });

  // Rename all anonymous tasks
  uint64_t anonCnt = 0;
  root->walk([&](TaskOp task) {
    if (!task.getTaskName()) {
      while (taskNames.contains("__task" + std::to_string(anonCnt)))
        anonCnt++;
      auto name = "task__" + std::to_string(anonCnt);
      auto nameAttr = builder.getStringAttr(name);
      taskNames.insert(name);
      task.setTaskName(nameAttr);
    }
  });

  builder.setInsertionPointAfter(root);

  for (const auto &taskName : taskNames) {
    LLVM_DEBUG(llvm::dbgs() << "Creating clone: " << taskName << "\n");
    std::string newName = root.getSymName().str();
    newName += "_";
    newName += taskName;

    arc::ModelOp cloned = dyn_cast<arc::ModelOp>(builder.clone(*root));
    cloned.setSymName(newName);
    cloned.walk([&](TaskOp task) {
      if (task.getTaskName() != taskName)
        task.erase();
    });

    // Unwrap all tasks
    cloned.walk([](TaskOp task) {
      task.walk([&](Operation *op) { op->moveBefore(task); });
      task.erase();
    });
  }

  // Remote tasks in original model
  root.walk([](TaskOp task) { task.erase(); });
}