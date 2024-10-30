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
  // Bijection (Writable Values) <-> Integral ID
  // The ID is used to index various bitvectors
  // The IDs are allocated in the following order:
  // States, Memories, Outputs
  DenseMap<Value, size_t> lookup;
  SmallVector<Value> ids;

  size_t numStates;
  size_t numMemories;
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
  // The size of the computational subtree
  SmallVector<size_t> compWeights;

  // The size of the total new data transfered to other chunks in each model
  // invocation For regs, it's the size of the reg For memories, it's the sum of
  // all write sizes
  SmallVector<size_t> transferWeights;

  void debug(const StateMap &map) const {
    for (auto it : llvm::enumerate(map.ids)) {
      it.value().dump();
      llvm::dbgs() << compWeights[it.index()] << "\n";
      llvm::dbgs() << transferWeights[it.index()] << "\n";
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

  void validateTaskOrder(arc::ModelOp &);

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
  validateTaskOrder(root);
}

StateMap PartitionPass::statAllStates(arc::ModelOp root) {
  SmallVector<Value> collected;
  DenseMap<Value, size_t> lookup;

  auto stat = [&](Operation *op) {
    size_t id = collected.size();
    auto result = op->getResult(0);
    lookup.insert({result, id});
    collected.push_back(result);
  };

  root->walk([&](arc::AllocStateOp op) { stat(op); });
  size_t numStates = collected.size();

  root->walk([&](arc::AllocMemoryOp op) { stat(op); });
  size_t numMemories = collected.size() - numStates;

  root->walk([&](arc::RootOutputOp op) { stat(op); });
  size_t numOutputs = collected.size() - numMemories - numStates;

  return StateMap{
      .lookup = lookup,
      .ids = collected,
      .numStates = numStates,
      .numMemories = numMemories,
      .numOutputs = numOutputs,
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
  BitVector ctrlFlow(map.numStates + map.numMemories);
  result.resize(map.ids.size(), BitVector(map.numStates + map.numMemories));

  // The root block depends on nothing
  cache.insert({root, BitVector(map.numStates + map.numMemories)});

  root->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op == root)
      return;
    if (isa<arc::AllocStateOp, arc::AllocMemoryOp>(op))
      return;

    // Implicit dependencies coming from surrounding control flow structure
    BitVector implicits = cache.at(op->getBlock()->getParentOp());
    ctrlFlow |= implicits;

    if (auto stateWrite = dyn_cast<arc::StateWriteOp>(op)) {
      // TODO: condition
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
    } else if (auto memWrite = dyn_cast<arc::MemoryWriteOp>(op)) {
      auto mem = memWrite.getMemory();
      auto addr = memWrite.getAddress();
      auto val = memWrite.getData();
      auto enable = memWrite.getData();

      assert(addr.getDefiningOp() && cache.contains(addr.getDefiningOp()));
      assert(val.getDefiningOp() && cache.contains(val.getDefiningOp()));
      if (enable)
        assert(enable.getDefiningOp() &&
               cache.contains(enable.getDefiningOp()));

      auto &slot = result[map.lookup.at(mem)];
      slot |= cache.at(addr.getDefiningOp());
      slot |= cache.at(val.getDefiningOp());
      slot |= cache.at(enable.getDefiningOp());
      slot |= implicits;
      return;
    }

    BitVector ret = implicits;
    if (isa<arc::RootInputOp, arc::RootOutputOp>(op)) {
      // Do nothing
    } else if (isa<arc::StateReadOp, arc::MemoryReadOp>(op)) {
      auto stateVal = op->getOperand(0);
      // Only care about states and memories, not inputs
      if (map.lookup.contains(stateVal)) {
        ret.set(map.lookup.at(stateVal));

        // Memory reads also depends on address
        if (auto memRead = dyn_cast<arc::MemoryReadOp>(op))
          ret |= cache.at(memRead.getAddress().getDefiningOp());
      } else {
        assert(isa<RootInputOp>(stateVal.getDefiningOp()) &&
               "Only inputs are not tracked during partitioning");
      }
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
std::optional<size_t> getTypeWidth(mlir::Type type) {
  if (type.isInteger()) {
    return type.getIntOrFloatBitWidth();
  } else if (auto arrayType = dyn_cast<hw::ArrayType>(type)) {
    // We may need to recursively compute the width of array
    return arrayType.getElementType().getIntOrFloatBitWidth() *
           arrayType.getNumElements();
  } else if (isa<seq::ClockType>(type)) {
    return 1;
  } else {
    return 0;
  }
}
size_t dfsWeight(Operation *op, DenseSet<Operation *> &visited) {
  if (!op)
    return 0; // Operand
  if (isa<StateReadOp, MemoryReadOp>(op))
    return 0;
  if (visited.contains(op))
    return 0;

  // Approximate the weight of the op by the largest width of all inputs and
  // outputs
  size_t maxWidth = 0;
  size_t operandWidth = 0;

  auto statValue = [&](const Value &v) {
    std::optional<size_t> width = getTypeWidth(v.getType());
    if (!width) {
      op->emitError("Unknown width of type: ") << v;
    } else if (*width > maxWidth)
      maxWidth = *width;
  };

  for (const auto &operand : op->getOperands()) {
    statValue(operand);
    operandWidth += dfsWeight(operand.getDefiningOp(), visited);
  }

  for (const auto &result : op->getResults()) {
    statValue(result);
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
    if (op.getCondition())
      slot.first += dfsWeight(op.getCondition().getDefiningOp(), slot.second);
  });
  op->walk([&](MemoryWriteOp op) {
    size_t id = map.lookup.at(op.getMemory());
    auto &slot = ctx[id];
    slot.first += dfsWeight(op.getAddress().getDefiningOp(), slot.second) +
                  dfsWeight(op.getData().getDefiningOp(), slot.second);
    if (op.getEnable())
      slot.first += dfsWeight(op.getEnable().getDefiningOp(), slot.second);
  });

  SmallVector<size_t> written;
  written.resize(map.ids.size(), 0);
  op->walk([&](AllocStateOp alloc) {
    auto id = map.lookup.at(alloc.getState());
    written[id] = alloc.getType().getByteWidth();
  });
  op->walk([&](MemoryWriteOp write) {
    auto id = map.lookup.at(write.getMemory());
    written[id] += write.getData().getType().getIntOrFloatBitWidth();
  });
  op->walk([&](RootOutputOp alloc) {
    auto id = map.lookup.at(alloc.getState());
    written[id] = alloc.getType().getByteWidth();
  });

  return StateWeight{
      .compWeights = llvm::map_to_vector(ctx, [](auto e) { return e.first; }),
      .transferWeights = written,
  };
}

PartitionPlan PartitionPass::planPartition(arc::ModelOp, const StateMap &map,
                                           const StateDep &deps,
                                           const StateWeight &weights) {
  // FIXME: impl
  SmallVector<size_t> chunk;
  chunk.reserve(map.ids.size());
  size_t totSize = 0;

  for (size_t i = 0; i < map.ids.size(); ++i)
    totSize += weights.transferWeights[i];

  double avgSize = totSize / _opts.chunks;
  size_t allocatedSize = 0;
  size_t allocPtr = 0;
  for (size_t i = 0; i < map.ids.size(); ++i) {
    size_t curSize = weights.transferWeights[i];
    while (allocatedSize + curSize * 0.5 > (allocPtr + 1) * avgSize)
      ++allocPtr;
    assert(allocPtr < _opts.chunks);
    chunk.push_back(allocPtr);
    allocatedSize += curSize;
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
  // Note that ctrlFlow implies globally visibe, because ctrl
  // flow-related states are written back in the output task
  // TODO: add an option to disable this behavior:

  BitVector globallyVisibleStates(map.numStates + map.numMemories);
  globallyVisibleStates = deps.ctrlFlow;
  // globallyVisibleStates.flip();
  for (size_t i = 0; i < map.numStates + map.numMemories; ++i) {
    auto localChunk = plan.chunk[i];
    for (auto dep : deps.deps[i].set_bits()) {
      auto remoteChunk = plan.chunk[dep];
      if (remoteChunk != localChunk)
        globallyVisibleStates.set(dep);
    }
  }

  // Allocate shadow states for globally visible states in their respective
  // shadow storages Creation of shadow memory writes are delayed until
  // remapping
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
  root.walk([&](Operation *write) {
    Value stateOrMem;
    if (auto stateWrite = dyn_cast<StateWriteOp>(write)) {
      stateOrMem = stateWrite.getState();
    } else if (auto memWrite = dyn_cast<MemoryWriteOp>(write)) {
      stateOrMem = memWrite.getMemory();
    } else {
      return;
    }

    // Is something we just inserted
    if (isa<TaskOp>(write->getBlock()->getParentOp()))
      return;

    // TODO: extmod
    // This is a write to output.
    // Outputs use new values, and may read from registers.
    // TODO: Rignt now we just place everything into the output task
    // We may want to do some dependency analysis, and only place those
    // with state dependencies into there. See PartitionPass::validateTaskOrder.
    if (isa<RootOutputOp>(stateOrMem.getDefiningOp())) {
      assert(write->getBlock() == &root.getBodyBlock() &&
             "Writes to output should only happens at model root");

      builder.setInsertionPointAfter(write);
      auto mainTask = builder.create<TaskOp>(write->getLoc(),
                                             builder.getStringAttr("output"));
      mainTask.getBodyRegion().emplaceBlock();
      write->moveBefore(&mainTask.getBody().front(),
                        mainTask.getBody().front().end());

      return;
    }

    // TODO: condition and enable
    builder.setInsertionPointAfter(write);
    auto mainTask = builder.create<TaskOp>(
        write->getLoc(), builder.getStringAttr(std::to_string(
                             plan.chunk[map.lookup.at(stateOrMem)])));
    mainTask.getBodyRegion().emplaceBlock();
    auto stateId = map.lookup.at(stateOrMem);
    if (stateId >= map.numStates + map.numMemories ||
        !globallyVisibleStates.test(stateId)) {
      // Only locally visible, or is write to output. Safe to just directly
      // write into
      write->moveBefore(&mainTask.getBody().front(),
                        mainTask.getBody().front().end());
    } else {
      // Is globally visible state
      // In main task, write into shadow
      // Then create separate sync task, read from shadow, write into main

      builder.setInsertionPointAfter(mainTask);
      auto syncStage = plan.syncWriteback.test(stateId)
                           ? std::string("output")
                           : std::to_string(plan.chunk[stateId]) + "_sync";
      auto syncTask = builder.create<TaskOp>(write->getLoc(),
                                             builder.getStringAttr(syncStage));
      auto syncBlock = &syncTask.getBodyRegion().emplaceBlock();

      if (auto stateWrite = dyn_cast<StateWriteOp>(write)) {
        auto state = stateWrite.getState();
        auto shadow = shadows.at(state);

        builder.setInsertionPointToStart(&mainTask.getBody().front());
        builder.create<StateWriteOp>(write->getLoc(), shadow,
                                     stateWrite.getValue(), Value());

        builder.setInsertionPointToStart(syncBlock);
        auto readout =
            builder.create<StateReadOp>(write->getLoc(), shadows.at(state))
                .getValue();
        builder.create<StateWriteOp>(write->getLoc(), state, readout, Value());
      } else if (auto memWrite = dyn_cast<MemoryWriteOp>(write)) {
        // Create shadow right now
        builder.setInsertionPoint(memWrite.getMemory().getDefiningOp());
        auto shadowStorage = shadowStorages[plan.chunk[stateId]];
        auto shadowAddr =
            builder
                .create<AllocStateOp>(
                    write->getLoc(),
                    StateType::get(memWrite.getAddress().getType()),
                    shadowStorage)
                .getResult();
        auto shadowData =
            builder
                .create<AllocStateOp>(
                    write->getLoc(),
                    StateType::get(memWrite.getData().getType()), shadowStorage)
                .getResult();
        auto shadowEnable =
            memWrite.getEnable()
                ? builder
                      .create<AllocStateOp>(
                          write->getLoc(),
                          StateType::get(memWrite.getEnable().getType()),
                          shadowStorage)
                      .getResult()
                : TypedValue<StateType>();

        builder.setInsertionPointToStart(&mainTask.getBody().front());
        builder.create<StateWriteOp>(write->getLoc(), shadowAddr,
                                     memWrite.getAddress(), Value());
        builder.create<StateWriteOp>(write->getLoc(), shadowData,
                                     memWrite.getData(), Value());
        if (shadowEnable)
          builder.create<StateWriteOp>(write->getLoc(), shadowEnable,
                                       memWrite.getEnable(), Value());

        builder.setInsertionPointToStart(syncBlock);
        auto readoutAddr =
            builder.create<StateReadOp>(write->getLoc(), shadowAddr).getValue();
        auto readoutData =
            builder.create<StateReadOp>(write->getLoc(), shadowData).getValue();
        auto readoutEnable =
            shadowEnable
                ? builder.create<StateReadOp>(write->getLoc(), shadowEnable)
                      .getValue()
                : Value();
        builder.create<MemoryWriteOp>(write->getLoc(), memWrite.getMemory(),
                                      readoutAddr, readoutEnable, readoutData);
      }

      write->erase();
    }
  });
}

size_t taskNameToIndex(StringRef name, size_t totChunks) {
  if (name == "output")
    return totChunks * 2;
  if (name.ends_with("sync")) {
    auto us = name.find("_");
    assert(us != std::string::npos);
    auto prefix = name.substr(0, us);
    auto idx = std::stoi(prefix.str());
    return totChunks + idx;
  }
  return std::stoi(name.str());
}

void PartitionPass::validateTaskOrder(arc::ModelOp &root) {
  // First, build a op -> program order mapping, because I didn't find any
  // better way to compare operations in program order
  DenseMap<Operation *, size_t> order;
  size_t ticket = 0;
  std::function<void(Block *)> visit;
  visit = [&](Block *blk) {
    for (auto &op : blk->getOperations()) {
      order.insert({&op, ticket++});
      for (auto &reg : op.getRegions())
        for (auto &blk : reg.getBlocks())
          visit(&blk);
    }
  };
  visit(&root.getBodyBlock());

  // We also stat the surrounding task's id
  DenseMap<Operation *, size_t> surr;
  root.walk([&](TaskOp task) {
    auto idx = taskNameToIndex(*task.getTaskName(), _opts.chunks);
    task.walk([&](Operation *op) { surr.insert({op, idx}); });
  });

  // Iterate through all state reads
  root.walk([&](Operation *read) {
    Value stateOrMem;
    if (auto stateRead = dyn_cast<StateReadOp>(read)) {
      stateOrMem = stateRead.getState();
    } else if (auto memRead = dyn_cast<MemoryReadOp>(read)) {
      stateOrMem = memRead.getMemory();
    } else {
      return;
    }

    auto selfOrder = order.at(read);
    for (const auto &use : stateOrMem.getUses()) {
      if (use == read)
        continue;

      auto user = use.getOwner();
      if (!isa<StateWriteOp, MemoryWriteOp>(user))
        continue;

      auto remoteOrder = order.at(user);

      bool remoteBefore = remoteOrder < selfOrder;

      if (!surr.contains(user))
        continue;
      auto writeTask = surr.at(user);

      for (const auto &selfUse : read->getResult(0).getUses()) {
        auto selfUser = selfUse.getOwner();
        if (!surr.contains(selfUser))
          continue;

        auto useTask = surr.at(selfUser);
        if (useTask == writeTask)
          continue;
        bool violated =
            remoteBefore ? (useTask / _opts.chunks <= writeTask / _opts.chunks)
                         : (useTask / _opts.chunks >= writeTask / _opts.chunks);
        if (violated) {
          // Violation
          read->emitError("Ordering violation: Read ")
              << (remoteBefore ? "after" : "before") << " write " << user
              << " in task idx " << writeTask << ", but is used at " << selfUser
              << " in task idx " << useTask;
        }

        if (selfUser->getNumRegions() > 0) { // With block, check child tasks
          selfUser->walk([&](TaskOp embeddedTask) {
            auto embeddedIdx =
                taskNameToIndex(*embeddedTask.getTaskName(), _opts.chunks);
            if (embeddedIdx == writeTask)
              return;
            bool embeddedViolated =
                remoteBefore
                    ? (embeddedIdx / _opts.chunks <= writeTask / _opts.chunks)
                    : (embeddedIdx / _opts.chunks >= writeTask / _opts.chunks);
            if (embeddedViolated) {
              read->emitError("Ordering violation: Read ")
                  << (remoteBefore ? "after" : "before") << " write " << user
                  << " in task idx " << writeTask << ", but is used at "
                  << selfUser << ", which contains a task at idx "
                  << embeddedIdx << " as children";
            }
          });
        }
      }
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
  // root.walk([](TaskOp task) { task.erase(); });

  // Unwrap tasks in original model
  root.walk([](TaskOp task) {
    task.walk([&](Operation *op) { op->moveBefore(task); });
    task.erase();
  });
}