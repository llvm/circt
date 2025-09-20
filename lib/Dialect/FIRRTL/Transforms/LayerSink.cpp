//===- LayerSink.cpp - Sink ops into layer blocks -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass sinks operations into layer blocks.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/Debug.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Threading.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/ControlFlowSinkUtils.h"
#include "llvm/ADT/PostOrderIterator.h"

#define DEBUG_TYPE "firrtl-layer-sink"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_LAYERSINK
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// NOLINTBEGIN(misc-no-recursion)
namespace {
/// Walk the ops in `block` bottom-up, back-to-front order. The block walk API
/// in upstream MLIR, although it takes an Iterator parameter, will always walk
/// the top block front-to-back. This walk function will actually walk the ops
/// under `block` back-to-front.
template <typename T>
void walkBwd(Block *block, T &&f) {
  for (auto &op :
       llvm::make_early_inc_range(llvm::reverse(block->getOperations()))) {
    for (auto &region : op.getRegions())
      for (auto &block : region.getBlocks())
        walkBwd(&block, f);
    f(&op);
  }
}
} // namespace
// NOLINTEND(misc-no-recursion)

static bool isAncestor(Block *block, Block *other) {
  if (block == other)
    return true;
  return block->getParent()->isProperAncestor(other->getParent());
}

//===----------------------------------------------------------------------===//
// Clone-ability.
//===----------------------------------------------------------------------===//

static bool cloneable(Operation *op) {
  // Clone any constants.
  if (op->hasTrait<OpTrait::ConstantLike>())
    return true;

  // Clone probe ops, to minimize residue within the design.
  if (isa<RefSendOp, RefResolveOp, RefCastOp, RefSubOp>(op))
    return true;

  return false;
}

//===----------------------------------------------------------------------===//
// Effectfulness Analysis.
//===----------------------------------------------------------------------===//

/// A table that can determine whether an operation is effectful.
namespace {
class EffectInfo {
public:
  EffectInfo(CircuitOp circuit, InstanceGraph &instanceGraph) {
    instanceGraph.walkPostOrder(
        [&](auto &node) { update(node.getModule().getOperation()); });
  }

  /// True if the given operation is NOT moveable due to some effect.
  bool effectful(Operation *op) const {
    if (!AnnotationSet(op).empty() || hasDontTouch(op))
      return true;
    if (auto name = dyn_cast<FNamableOp>(op))
      if (!name.hasDroppableName())
        return true;
    if (op->getNumRegions() != 0)
      return true;
    if (auto instance = dyn_cast<InstanceOp>(op))
      return effectfulModules.contains(instance.getModuleNameAttr().getAttr());
    if (isa<FConnectLike, WireOp, RegResetOp, RegOp, MemOp, NodeOp>(op))
      return false;
    return !(mlir::isMemoryEffectFree(op) ||
             mlir::hasSingleEffect<mlir::MemoryEffects::Allocate>(op) ||
             mlir::hasSingleEffect<mlir::MemoryEffects::Read>(op));
  }

private:
  /// Record whether the module contains any effectful ops.
  void update(FModuleOp moduleOp) {
    moduleOp.getBodyBlock()->walk([&](Operation *op) {
      if (effectful(op)) {
        markEffectful(moduleOp);
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }

  /// Record whether the module-like op contains any effectful op.
  void update(FModuleLike moduleOp) {
    // If the module op has any annotations, then we pretend the module contains
    // some kind of important effect, so that we cannot sink its instances.
    if (auto annos = getAnnotationsIfPresent(moduleOp))
      if (!annos.empty())
        return markEffectful(moduleOp);

    for (auto annos : moduleOp.getPortAnnotations())
      if (!cast<ArrayAttr>(annos).empty())
        return markEffectful(moduleOp);

    auto *op = moduleOp.getOperation();
    // Regular modules may be pure.
    if (auto m = dyn_cast<FModuleOp>(op))
      return update(m);
    // Memory modules are pure.
    if (auto m = dyn_cast<FMemModuleOp>(op))
      return;
    // All other kinds of modules are effectful.
    // intmodules, extmodules, classes.
    return markEffectful(moduleOp);
  }

  void update(Operation *op) {
    if (auto moduleOp = dyn_cast<FModuleLike>(op))
      update(moduleOp);
  }

  /// Record that the given module contains an effectful operation.
  void markEffectful(FModuleLike moduleOp) {
    effectfulModules.insert(moduleOp.getModuleNameAttr());
  }

  DenseSet<StringAttr> effectfulModules;
};
} // namespace

//===----------------------------------------------------------------------===//
// Demands Analysis.
//===----------------------------------------------------------------------===//

namespace {
/// The LCA of the blocks in which a value is used/demanded. Lattice value.
struct Demand {
  constexpr Demand() : Demand(nullptr) {}
  constexpr Demand(Block *block) : block(block) {}

  constexpr Demand merge(Demand other) const {
    if (block == other.block)
      return Demand(block);
    if (other.block == nullptr)
      return Demand(block);
    if (block == nullptr)
      return Demand(other.block);

    auto *b = block;
    while (b && !isAncestor(b, other.block))
      b = b->getParentOp()->getBlock();

    return Demand(b);
  }

  bool mergeIn(Demand other) {
    auto prev = *this;
    auto next = merge(other);
    *this = next;
    return prev != next;
  }

  constexpr bool operator==(Demand rhs) const { return block == rhs.block; }
  constexpr bool operator!=(Demand rhs) const { return block != rhs.block; }
  constexpr operator bool() const { return block; }

  Block *block;
};
} // namespace

/// True if this operation is a good site to sink operations.
static bool isValidDest(Operation *op) {
  return op && (isa<LayerBlockOp>(op) || isa<FModuleOp>(op));
}

/// True if we are prevented from sinking operations into the regions of the op.
static bool isBarrier(Operation *op) { return !isValidDest(op); }

/// Adjust the demand based on the location of the op being demanded. Ideally,
/// we can sink an op directly to its site of use. However, there are two
/// issues.
///
/// 1) Typically, an op will dominate every demand, but for hardware
/// declarations such as wires, the declaration will demand any connections
/// driving it. In this case, the relationship is reversed: the demander
/// dominates the demandee. This can cause us to pull connect-like ops up and
/// and out of their enclosing block (ref-defines can be buried under
/// layerblocks). To avoid this, we set an upper bound on the demand: the
/// enclosing block of the demandee.
///
/// 2) not all regions are valid sink targets. If there is a sink-barrier
/// between the operation and its demand, we adjust the demand upwards so that
/// there is no sink barrier between the demandee and the demand site.
static Demand clamp(Operation *op, Demand demand) {
  if (!demand)
    return nullptr;

  auto *upper = op->getBlock();
  assert(upper && "this should not be called on a top-level operation.");

  if (!isAncestor(upper, demand.block))
    demand = upper;

  for (auto *i = demand.block; i != upper; i = i->getParentOp()->getBlock())
    if (isBarrier(i->getParentOp()))
      demand = i->getParentOp()->getBlock();

  return demand;
}

namespace {
struct DemandInfo {
  using WorkStack = std::vector<Operation *>;

  DemandInfo(const EffectInfo &, FModuleOp);

  Demand getDemandFor(Operation *op) const { return table.lookup(op); }

  /// Starting at effectful ops and output ports, propagate the demand of values
  /// through the design, running until fixpoint. At the end, we have an
  /// accurate picture of where every operation can be sunk, while preserving
  /// effects in the program.
  void run(const EffectInfo &, FModuleOp, WorkStack &);

  /// Update the demand for the given op. If the demand changes, place the op
  /// onto the worklist.
  void update(WorkStack &work, Operation *op, Demand demand) {
    if (table[op].mergeIn(clamp(op, demand)))
      work.push_back(op);
  }

  void update(WorkStack &work, Value value, Demand demand) {
    if (auto result = dyn_cast<OpResult>(value))
      update(work, cast<OpResult>(value).getOwner(), demand);
  }

  void updateConnects(WorkStack &, Value, Demand);
  void updateConnects(WorkStack &, Operation *, Demand);

  llvm::DenseMap<Operation *, Demand> table;
};
} // namespace

DemandInfo::DemandInfo(const EffectInfo &effectInfo, FModuleOp moduleOp) {
  WorkStack work;
  Block *body = moduleOp.getBodyBlock();
  ArrayRef<bool> dirs = moduleOp.getPortDirections();
  for (unsigned i = 0, e = moduleOp.getNumPorts(); i < e; ++i)
    if (direction::get(dirs[i]) == Direction::Out)
      updateConnects(work, body->getArgument(i), moduleOp.getBodyBlock());
  moduleOp.getBodyBlock()->walk([&](Operation *op) {
    if (effectInfo.effectful(op))
      update(work, op, op->getBlock());
  });
  run(effectInfo, moduleOp, work);
}

void DemandInfo::run(const EffectInfo &effectInfo, FModuleOp, WorkStack &work) {
  while (!work.empty()) {
    auto *op = work.back();
    work.pop_back();
    auto demand = getDemandFor(op);
    for (auto operand : op->getOperands())
      update(work, operand, demand);
    updateConnects(work, op, demand);
  }
}

// The value represents a hardware declaration, such as a wire or port. Search
// backwards through uses, looking through aliasing operations such as
// subfields, to find connects that drive the given value. All driving
// connects of a value are demanded by the same region as the value. If the
// demand of the connect op is updated, then the demand will propagate
// forwards to its operands through the normal forward-propagation of demand.
void DemandInfo::updateConnects(WorkStack &work, Value value, Demand demand) {
  struct StackElement {
    Value value;
    Value::user_iterator it;
  };

  SmallVector<StackElement> stack;
  stack.push_back({value, value.user_begin()});
  while (!stack.empty()) {
    auto &top = stack.back();
    auto end = top.value.user_end();
    while (true) {
      if (top.it == end) {
        stack.pop_back();
        break;
      }
      auto *user = *(top.it++);
      if (auto connect = dyn_cast<FConnectLike>(user)) {
        if (connect.getDest() == top.value) {
          update(work, connect, demand);
        }
        continue;
      }
      if (isa<SubfieldOp, SubindexOp, SubaccessOp, ObjectSubfieldOp>(user)) {
        for (auto result : user->getResults())
          stack.push_back({result, result.user_begin()});
        break;
      }
    }
  }
}

void DemandInfo::updateConnects(WorkStack &work, Operation *op, Demand demand) {
  if (isa<WireOp, RegResetOp, RegOp, MemOp, ObjectOp>(op)) {
    for (auto result : op->getResults())
      updateConnects(work, result, demand);
  } else if (auto inst = dyn_cast<InstanceOp>(op)) {
    auto dirs = inst.getPortDirections();
    for (unsigned i = 0, e = inst->getNumResults(); i < e; ++i) {
      if (direction::get(dirs[i]) == Direction::In)
        updateConnects(work, inst.getResult(i), demand);
    }
  }
}

//===----------------------------------------------------------------------===//
// ModuleLayerSink
//===----------------------------------------------------------------------===//

namespace {
class ModuleLayerSink {
public:
  static bool run(FModuleOp moduleOp, const EffectInfo &effectInfo) {
    return ModuleLayerSink(moduleOp, effectInfo)();
  }

private:
  ModuleLayerSink(FModuleOp moduleOp, const EffectInfo &effectInfo)
      : moduleOp(moduleOp), effectInfo(effectInfo) {}

  bool operator()();
  void moveLayersToBack(Operation *op);
  void moveLayersToBack(Block *block);

  void erase(Operation *op);
  void sinkOpByCloning(Operation *op);
  void sinkOpByMoving(Operation *op, Block *dest);

  FModuleOp moduleOp;
  const EffectInfo &effectInfo;
  bool changed = false;
};
} // namespace

// NOLINTNEXTLINE(misc-no-recursion)
void ModuleLayerSink::moveLayersToBack(Operation *op) {
  for (auto &r : op->getRegions())
    for (auto &b : r.getBlocks())
      moveLayersToBack(&b);
}

// NOLINTNEXTLINE(misc-no-recursion)
void ModuleLayerSink::moveLayersToBack(Block *block) {
  auto i = block->rbegin();
  auto e = block->rend();

  // Leave layerblocks that are already "at the back" where they are.
  while (i != e) {
    auto *op = &*i;
    moveLayersToBack(op);
    if (!isa<LayerBlockOp>(op))
      break;
    ++i;
  }

  if (i == e)
    return;

  // We have found a non-layerblock op at position `i`.
  // Any block in front of `i`, will be moved to after `i`.
  auto *ip = &*i;
  while (i != e) {
    auto *op = &*i;
    moveLayersToBack(op);
    if (isa<LayerBlockOp>(op)) {
      ++i;
      op->moveAfter(ip);
      changed = true;
    } else {
      ++i;
    }
  }
}

void ModuleLayerSink::erase(Operation *op) {
  op->erase();
  changed = true;
}

void ModuleLayerSink::sinkOpByCloning(Operation *op) {
  // Copy the op down to each block where there is a user.
  // Use a cache to ensure we don't create more than one copy per block.
  auto *src = op->getBlock();
  DenseMap<Block *, Operation *> cache;
  for (unsigned i = 0, e = op->getNumResults(); i < e; ++i) {
    for (auto &use : llvm::make_early_inc_range(op->getResult(i).getUses())) {
      auto *dst = use.getOwner()->getBlock();
      if (dst != src) {
        auto &clone = cache[dst];
        if (!clone)
          clone = OpBuilder::atBlockBegin(dst).clone(*op);
        use.set(clone->getResult(i));
        changed = true;
      }
    }
  }

  // If there are no remaining uses of the original op, erase it.
  if (isOpTriviallyDead(op))
    erase(op);
}

void ModuleLayerSink::sinkOpByMoving(Operation *op, Block *dst) {
  if (dst != op->getBlock()) {
    op->moveBefore(dst, dst->begin());
    changed = true;
  }
}

bool ModuleLayerSink::operator()() {
  moveLayersToBack(moduleOp.getBodyBlock());
  DemandInfo demandInfo(effectInfo, moduleOp);
  walkBwd(moduleOp.getBodyBlock(), [&](Operation *op) {
    auto demand = demandInfo.getDemandFor(op);
    if (!demand)
      return erase(op);
    if (cloneable(op))
      return sinkOpByCloning(op);
    sinkOpByMoving(op, demand.block);
  });

  return changed;
}

//===----------------------------------------------------------------------===//
// LayerSinkPass
//===----------------------------------------------------------------------===//

namespace {
/// A control-flow sink pass.
struct LayerSinkPass final
    : public circt::firrtl::impl::LayerSinkBase<LayerSinkPass> {
  void runOnOperation() override;
};
} // namespace

void LayerSinkPass::runOnOperation() {
  auto circuit = getOperation();
  LLVM_DEBUG(debugPassHeader(this)
                 << "\n"
                 << "Circuit: '" << circuit.getName() << "'\n";);
  auto &instanceGraph = getAnalysis<InstanceGraph>();
  EffectInfo effectInfo(circuit, instanceGraph);

  std::atomic<bool> changed(false);
  parallelForEach(&getContext(), circuit.getOps<FModuleOp>(),
                  [&](FModuleOp moduleOp) {
                    if (ModuleLayerSink::run(moduleOp, effectInfo))
                      changed = true;
                  });

  if (!changed)
    markAllAnalysesPreserved();
  else
    markAnalysesPreserved<InstanceGraph>();
}
