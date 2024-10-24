#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
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

  // (shadow -> persistent) mapping
  DenseMap<Value, size_t> shadows;

  size_t numStates;
  size_t numOutputs;
};

// State -> state dependencies
struct StateDep {
  SmallVector<BitVector> deps;

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
  SmallVector<size_t> chunk;
};

struct PartitionPass : public arc::impl::PartitionBase<PartitionPass> {
  void runOnOperation() override;
  StateMap statAllStates(arc::ModelOp);
  StateDep statAllStateDeps(arc::ModelOp, const StateMap &);
  StateWeight statAllStateWeights(arc::ModelOp, const StateMap &);

  PartitionPlan planPartition(arc::ModelOp, const StateMap &, const StateDep &,
                              const StateWeight &);

  void markPartition(arc::ModelOp, const StateMap &, const PartitionPlan &);
  void makeSync(arc::ModelOp);

  PartitionPass(const PartitionOptions &opts = {});

  PartitionOptions _opts;
};

struct PartitionClonePass
    : public arc::impl::PartitionCloneBase<PartitionClonePass> {
  void runOnOperation() override;
  PartitionClonePass();
};
} // namespace

PartitionPass::PartitionPass(const PartitionOptions &opts) : _opts(opts) {}

void PartitionPass::runOnOperation() {
  auto root = getOperation();
  const auto &states = statAllStates(root);
  const auto &deps = statAllStateDeps(root, states);
  const auto &weights = statAllStateWeights(root, states);

  deps.debug(states);
  weights.debug(states);

  const auto &partition = planPartition(root, states, deps, weights);
  markPartition(root, states, partition);
  root.dump();
  makeSync(root);
}

StateMap PartitionPass::statAllStates(arc::ModelOp root) {
  SmallVector<Value> collected;
  DenseMap<Value, size_t> lookup;
  DenseMap<Value, size_t> shadows;

  // First pass: all IO & persistent state
  root->walk([&](arc::AllocStateOp op) {
    if (cast<StringAttr>(op->getAttr("partition-role")) == "shadow-state" ||
        cast<StringAttr>(op->getAttr("partition-role")) == "old-clock" ||
        cast<StringAttr>(op->getAttr("partition-role")) == "shadow-old-clock")
      return;
    size_t id = collected.size();
    auto result = op.getResult();
    lookup.insert({result, id});
    collected.push_back(result);
  });

  // Second pass, all shadows
  root->walk([&](arc::AllocStateOp op) {
    if (cast<StringAttr>(op->getAttr("partition-role")) != "shadow-state")
      return;
    auto shadow = op.getResult();
    OpOperand *written = llvm::find_singleton<OpOperand>(
        shadow.getUses(),
        [](OpOperand &operand, bool _allow_repeats) -> OpOperand * {
          if (isa<arc::StateWriteOp>(operand.getOwner()) &&
              operand.getOperandNumber() == 0)
            return &operand;
          else
            return nullptr;
        });
    assert(written && "Shadow states should be written exactly once");
    auto stateVal = written->getOwner()->getOperand(1);
    assert(stateVal.getDefiningOp() &&
           isa<StateReadOp>(stateVal.getDefiningOp()) &&
           lookup.contains(stateVal.getDefiningOp()->getOperand(0)) &&
           "Shadow states should be written by values that are read from a "
           "persistent state");
    auto state = stateVal.getDefiningOp()->getOperand(0);
    shadows.insert({shadow, lookup.at(state)});
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
      .shadows = shadows,
      .numStates = numStates,
      .numOutputs = lookup.size() - numStates,
  };
}

StateDep PartitionPass::statAllStateDeps(arc::ModelOp root,
                                         const StateMap &map) {
  DenseMap<Operation *, BitVector> cache;
  SmallVector<BitVector> result;
  DenseMap<Value, size_t> shadows;
  result.resize(map.ids.size(), BitVector(map.numStates));

  root->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<arc::AllocStateOp>(op))
      return;

    if (auto stateWrite = dyn_cast<arc::StateWriteOp>(op)) {
      auto state = stateWrite.getState();
      auto val = stateWrite.getValue();
      assert(val.getDefiningOp() && cache.contains(val.getDefiningOp()));
      auto deps = cache.at(val.getDefiningOp());
      state.dump();

      if (isa<RootOutputOp>(state.getDefiningOp()))
        return;
      if (auto role = cast<StringAttr>(
              state.getDefiningOp()->getAttr("partition-role"));
          role == "shadow-state" || role == "shadow-old-clock" ||
          role == "old-clock")
        return;

      auto &slot = result[map.lookup.at(state)];
      slot |= deps;
      return;
    }

    BitVector ret(map.numStates);
    if (isa<arc::RootInputOp, arc::RootOutputOp>(op)) {
      // Do nothing
    } else if (isa<arc::StateReadOp>(op)) {
      auto stateVal = op->getOperand(0);
      std::optional<size_t> stateId = {};
      if (map.shadows.contains(stateVal))
        stateId = map.shadows.at(stateVal);
      else if (map.lookup.contains(stateVal))
        stateId = map.lookup.at(stateVal);
      if (stateId)
        ret.set(*stateId);
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
  };
}

namespace {
size_t dfsWeight(Operation *op, DenseSet<Operation *> &visited) {
  if (isa<StateReadOp>(op))
    return 0;
  if (visited.contains(op))
    return 0;

  // Approximate the weight of the op by the largest width of all inputs and
  // outputs
  size_t maxWidth = 0;
  size_t operandWidth = 0;
  for (const auto &operand : op->getOperands()) {
    if (!operand.getType().isInteger())
      op->emitError("Use of non-integer operand");

    auto width = operand.getType().getIntOrFloatBitWidth();
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
    if (isa<AllocStateOp>(op.getState().getDefiningOp()))
      if (auto role = cast<StringAttr>(
              op.getState().getDefiningOp()->getAttr("partition-role"));
          role == "shadow-state" || role == "old-clock" ||
          role == "shadow-old-clock")
        return;
    size_t id = map.lookup.at(op.getState());
    auto &slot = ctx[id];
    slot.first += dfsWeight(op.getValue().getDefiningOp(), slot.second);
  });

  return StateWeight{
      .weights = llvm::map_to_vector(ctx, [](auto e) { return e.first; })};
}

PartitionPlan PartitionPass::planPartition(arc::ModelOp, const StateMap &map,
                                           const StateDep &,
                                           const StateWeight &) {
  // FIXME: impl
  std::mt19937 gen(0x19260817);

  std::uniform_int_distribution<size_t> dist(0, _opts.chunks - 1);

  SmallVector<size_t> chunk;
  chunk.reserve(map.ids.size());
  for (size_t i = 0; i < map.ids.size(); ++i)
    chunk.push_back(dist(gen));

  return PartitionPlan{.chunk = chunk};
}

void PartitionPass::markPartition(arc::ModelOp op, const StateMap &map,
                                  const PartitionPlan &plan) {
  OpBuilder builder(op);
  op.walk([&](arc::StateWriteOp write) {
    auto state = write.getState();
    bool valid =
        (state.getDefiningOp() &&
         isa<arc::RootOutputOp, arc::AllocStateOp>(state.getDefiningOp()));
    assert(valid && "Partition planning should run before state allocation");

    bool isState = isa<AllocStateOp>(state.getDefiningOp());
    if (isState)
      if (auto role = cast<StringAttr>(
              state.getDefiningOp()->getAttr("partition-role"));
          role == "old-clock" || role == "shadow-old-clock")
        return;

    size_t id;
    if (isState && cast<StringAttr>(state.getDefiningOp()->getAttr(
                       "partition-role")) == "shadow-state")
      id = map.shadows.at(state);
    else
      id = map.lookup.at(state);
    size_t chunk = plan.chunk[id];
    write->setAttr("chunk", builder.getStringAttr(std::to_string(chunk)));
  });

  // Also mark on AllocState & RootOutput
  for (auto it : llvm::enumerate(map.ids))
    it.value().getDefiningOp()->setAttr(
        "updated-by-chunk",
        builder.getStringAttr(std::to_string(plan.chunk[it.index()])));
}

void PartitionPass::makeSync(arc::ModelOp root) {
  DenseMap<std::pair<Block *, StringRef>, SyncOp> syncCache;
  DenseMap<Value, Value> toShadow;
  DenseSet<std::pair<Block *, Value>> shadowed;
  OpBuilder builder(root);

  assert(root.getBodyBlock().getNumArguments() ==
         2); // Persistent and temporary storage
  auto storage = root.getBodyBlock().getArgument(0);
  auto tmpStorage = root.getBodyBlock().getArgument(1);
  IRMapping storageRemap;
  storageRemap.map(storage, tmpStorage);

  // Create shadow arguments
  root.walk([&](AllocStateOp alloc) {
    if (!alloc->hasAttr("updated-by-chunk"))
      return;
    builder.setInsertionPointAfter(alloc);
    auto shadow = cast<AllocStateOp>(builder.clone(*alloc, storageRemap));
    shadow->setAttr("partition-role", builder.getStringAttr("shadow-state"));
    toShadow.insert({alloc.getResult(), shadow.getResult()});
  });

  root.walk([&](StateWriteOp write) {
    auto state = write.getState();

    // Is something we just inserted
    if (isa<SyncOp>(write->getBlock()->getParentOp()))
      return;

    AllocStateOp alloc = dyn_cast<AllocStateOp>(state.getDefiningOp());
    // TODO: extmod
    if (!alloc)
      return; // Is output

    if (cast<StringAttr>(alloc->getAttr("partition-role")) != "state")
      return; // Old clock

    write.dump();
    write->getBlock()->getParentOp()->dump();
    auto chunk = cast<StringAttr>(write->getAttr("chunk"));
    assert(
        chunk &&
        "After partition, state writes should always have a designated chunk");

    auto blk = write->getBlock();

    if (shadowed.contains({blk, state}))
      return;
    shadowed.insert({blk, state});

    SyncOp sync;
    if (syncCache.contains({blk, chunk}))
      sync = syncCache.at({blk, chunk});
    else {
      builder.setInsertionPointToEnd(blk);
      sync = builder.create<SyncOp>(blk->getParentOp()->getLoc());
      sync.getBodyRegion().emplaceBlock();
      sync->setAttr("chunk", chunk);
      syncCache.insert({{blk, chunk}, sync});
    }

    write.setOperand(0, toShadow.at(state));

    builder.setInsertionPointToEnd(&sync.getBody().front());
    auto shadowRead =
        builder.create<StateReadOp>(write.getLoc(), toShadow.at(state));
    // FIXME: condition
    builder.create<StateWriteOp>(write.getLoc(), state, shadowRead.getResult(),
                                 Value());
  });
}

PartitionClonePass::PartitionClonePass() {}

void PartitionClonePass::runOnOperation() {
  Operation *root = getOperation();

  root->walk([&](arc::ModelOp model) {
    DenseSet<StringRef> copies;
    model.walk([&](Operation *write) {
      if (auto attr = write->getAttr("chunk")) {
        assert(isa<StringAttr>(attr) && "Chunk attr should be a string");
        StringAttr sa = dyn_cast<StringAttr>(attr);
        copies.insert(sa.getValue());
      }
    });

    OpBuilder builder(root);
    builder.setInsertionPointAfter(model);

    for (const auto &copy : copies) {
      llvm::dbgs() << "Creating clone: " << copy << "\n";
      std::string newName = model.getSymName().str();
      newName += "_";
      newName += copy;

      arc::ModelOp cloned = dyn_cast<arc::ModelOp>(builder.clone(*model));
      cloned.setSymName(newName);
      cloned.walk([&](Operation *op) {
        if (auto attr = op->getAttr("chunk")) {
          StringRef chunk = dyn_cast<StringAttr>(attr).getValue();
          if (chunk != copy)
            op->erase();
        }
      });

      arc::ModelOp clonedSync = dyn_cast<arc::ModelOp>(builder.clone(*cloned));
      clonedSync.setSymName(newName + "_sync");

      cloned.walk([](SyncOp sync) { sync.erase(); });

      // Remove all state operations besides ones inside sync
      SmallVector<Operation *> toBeErased;
      for (auto &op : clonedSync.getBodyBlock().getOperations()) {
        if (isa<SyncOp>(op))
          continue;
        if (ClockTreeOp top = dyn_cast<ClockTreeOp>(op)) {
          auto walkRet = top.walk([&](SyncOp found) {
            found->remove();
            top.getBody().front().clear();

            OpBuilder builder(top);
            builder.setInsertionPointToEnd(&top.getBody().front());
            builder.insert(found);

            return WalkResult::interrupt();
          });
          if (!walkRet.wasInterrupted())
            toBeErased.emplace_back(&op);
        } else if (isa<StateWriteOp, PassThroughOp>(op)) {
          toBeErased.emplace_back(&op);
        }
      }

      for (auto op : toBeErased)
        op->erase();

      // Unwrap all sync
      clonedSync.walk([](SyncOp sync) {
        sync.walk([&](Operation *op) {
          op->moveBefore(sync->getBlock(), sync->getBlock()->end());
        });
        sync.erase();
      });
    }

    root->remove();
    // FIXME: Calling erase or destroy here would result in a failed constraint
    // root->destroy();
  });
}
