//===- IMDeadCodeElim.cpp - Intermodule Dead Code Elimination ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-imdeadcodeelim"

using namespace circt;
using namespace firrtl;

/// Return true if this is a wire or a register or a node.
static bool isDeclaration(Operation *op) {
  return isa<WireOp, RegResetOp, RegOp, NodeOp, MemOp>(op);
}

/// Return true if this is a wire or register we're allowed to delete.
static bool isDeletableDeclaration(Operation *op) {
  if (auto name = dyn_cast<FNamableOp>(op))
    if (!name.hasDroppableName())
      return false;
  return !hasDontTouch(op);
}

namespace {
struct IMDeadCodeElimPass : public IMDeadCodeElimBase<IMDeadCodeElimPass> {
  void runOnOperation() override;

  void rewriteModuleSignature(FModuleOp module);
  void rewriteModuleBody(FModuleOp module);
  void eraseEmptyModule(FModuleOp module);
  void forwardConstantOutputPort(FModuleOp module);

  void markAlive(Value value) {
    //  If the value is already in `liveSet`, skip it.
    if (liveSet.insert(value).second)
      worklist.push_back(value);
  }

  /// Return true if the value is known alive.
  bool isKnownAlive(Value value) const {
    assert(value && "null should not be used");
    return liveSet.count(value);
  }

  /// Return true if the value is assumed dead.
  bool isAssumedDead(Value value) const { return !isKnownAlive(value); }
  bool isAssumedDead(Operation *op) const {
    return llvm::none_of(op->getResults(),
                         [&](Value value) { return isKnownAlive(value); });
  }

  /// Return true if the block is alive.
  bool isBlockExecutable(Block *block) const {
    return executableBlocks.count(block);
  }

  void visitUser(Operation *op);
  void visitValue(Value value);
  void visitConnect(FConnectLike connect);
  void visitSubelement(Operation *op);
  void markBlockExecutable(Block *block);
  void markDeclaration(Operation *op);
  void markInstanceOp(InstanceOp instanceOp);
  void markUnknownSideEffectOp(Operation *op);

private:
  /// The set of blocks that are known to execute, or are intrinsically alive.
  DenseSet<Block *> executableBlocks;

  /// This keeps track of users the instance results that correspond to output
  /// ports.
  DenseMap<BlockArgument, llvm::TinyPtrVector<Value>>
      resultPortToInstanceResultMapping;
  InstanceGraph *instanceGraph;

  /// A worklist of values whose liveness recently changed, indicating the
  /// users need to be reprocessed.
  SmallVector<Value, 64> worklist;
  llvm::DenseSet<Value> liveSet;
};
} // namespace

void IMDeadCodeElimPass::markDeclaration(Operation *op) {
  assert(isDeclaration(op) && "only a declaration is expected");
  if (!isDeletableDeclaration(op))
    for (auto result : op->getResults())
      markAlive(result);
}

void IMDeadCodeElimPass::markUnknownSideEffectOp(Operation *op) {
  // For operations with side effects, pessimistically mark results and
  // operands as alive.
  for (auto result : op->getResults())
    markAlive(result);
  for (auto operand : op->getOperands())
    markAlive(operand);
}

void IMDeadCodeElimPass::visitUser(Operation *op) {
  LLVM_DEBUG(llvm::dbgs() << "Visit: " << *op << "\n");
  if (auto connectOp = dyn_cast<FConnectLike>(op))
    return visitConnect(connectOp);
  if (isa<SubfieldOp, SubindexOp, SubaccessOp>(op))
    return visitSubelement(op);
}

void IMDeadCodeElimPass::markInstanceOp(InstanceOp instance) {
  // Get the module being referenced.
  Operation *op = instanceGraph->getReferencedModule(instance);

  // If this is an extmodule, just remember that any inputs and inouts are
  // alive.
  if (!isa<FModuleOp>(op)) {
    auto module = dyn_cast<FModuleLike>(op);
    for (auto resultNo : llvm::seq(0u, instance.getNumResults())) {
      auto portVal = instance.getResult(resultNo);
      // If this is an output to the extmodule, we can ignore it.
      if (module.getPortDirection(resultNo) == Direction::Out)
        continue;

      // Otherwise this is an inuput from it or an inout, mark it as alive.
      markAlive(portVal);
    }
    return;
  }

  // Otherwise this is a defined module.
  auto fModule = cast<FModuleOp>(op);
  markBlockExecutable(fModule.getBodyBlock());

  // Ok, it is a normal internal module reference so populate
  // resultPortToInstanceResultMapping.
  for (auto resultNo : llvm::seq(0u, instance.getNumResults())) {
    auto instancePortVal = instance.getResult(resultNo);

    // Otherwise we have a result from the instance.  We need to forward results
    // from the body to this instance result's SSA value, so remember it.
    BlockArgument modulePortVal = fModule.getArgument(resultNo);

    resultPortToInstanceResultMapping[modulePortVal].push_back(instancePortVal);
  }
}

void IMDeadCodeElimPass::markBlockExecutable(Block *block) {
  if (!executableBlocks.insert(block).second)
    return; // Already executable.

  // Mark ports with don't touch as alive.
  for (auto blockArg : block->getArguments())
    if (hasDontTouch(blockArg))
      markAlive(blockArg);

  for (auto &op : *block) {
    if (isDeclaration(&op))
      markDeclaration(&op);
    else if (auto instance = dyn_cast<InstanceOp>(op))
      markInstanceOp(instance);
    else if (isa<FConnectLike>(op))
      // Skip connect op.
      continue;
    else if (!mlir::MemoryEffectOpInterface::hasNoEffect(&op))
      markUnknownSideEffectOp(&op);

    // TODO: Handle attach etc.
  }
}

void IMDeadCodeElimPass::forwardConstantOutputPort(FModuleOp module) {
  // This tracks constant values of output ports.
  SmallVector<std::pair<unsigned, APSInt>> constantPortIndicesAndValues;
  auto ports = module.getPorts();
  auto *instanceGraphNode = instanceGraph->lookup(module);

  for (const auto &e : llvm::enumerate(ports)) {
    unsigned index = e.index();
    auto port = e.value();
    auto arg = module.getArgument(index);

    // If the port has don't touch, don't propagate the constant value.
    if (!port.isOutput() || hasDontTouch(arg))
      continue;

    // Remember the index and constant value connected to an output port.
    if (auto connect = getSingleConnectUserOf(arg))
      if (auto constant = connect.getSrc().getDefiningOp<ConstantOp>())
        constantPortIndicesAndValues.push_back({index, constant.getValue()});
  }

  // If there is no constant port, abort.
  if (constantPortIndicesAndValues.empty())
    return;

  // Rewrite all uses.
  for (auto *use : instanceGraphNode->uses()) {
    auto instance = cast<InstanceOp>(*use->getInstance());
    ImplicitLocOpBuilder builder(instance.getLoc(), instance);
    for (auto [index, constant] : constantPortIndicesAndValues) {
      auto result = instance.getResult(index);
      assert(ports[index].isOutput() && "must be an output port");

      // Replace the port with the constant.
      result.replaceAllUsesWith(builder.create<ConstantOp>(constant));
    }
  }
}

void IMDeadCodeElimPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "===----- Remove unused ports -----==="
                          << "\n");
  auto circuit = getOperation();
  instanceGraph = &getAnalysis<InstanceGraph>();

  // Forward constant output ports to caller sides so that we can eliminate
  // constant outputs.
  for (auto *node : llvm::post_order(instanceGraph))
    if (auto module = dyn_cast_or_null<FModuleOp>(*node->getModule()))
      forwardConstantOutputPort(module);

  for (auto module : circuit.getBodyBlock()->getOps<FModuleOp>()) {
    // Mark the ports of public modules as alive.
    if (module.isPublic()) {
      markBlockExecutable(module.getBodyBlock());
      for (auto port : module.getBodyBlock()->getArguments())
        markAlive(port);
    }
  }

  // If a value changed liveness then propagate liveness through its users and
  // definition.
  while (!worklist.empty())
    visitValue(worklist.pop_back_val());

  // Rewrite module signatures.
  for (auto module : circuit.getBodyBlock()->getOps<FModuleOp>())
    rewriteModuleSignature(module);

  // Rewrite module bodies parallelly.
  mlir::parallelForEach(circuit.getContext(),
                        circuit.getBodyBlock()->getOps<FModuleOp>(),
                        [&](auto op) { rewriteModuleBody(op); });

  // Erase empty modules. To erase empty modules transitively, it is necessary
  // to visit modules in the post order of instance graph.
  // FIXME: We copy the list of modules into a vector first to avoid iterator
  // invalidation while we mutate the instance graph. See issue 3387.
  SmallVector<FModuleOp, 0> modules(llvm::make_filter_range(
      llvm::map_range(
          llvm::post_order(instanceGraph),
          [](auto *node) { return dyn_cast<FModuleOp>(*node->getModule()); }),
      [](auto module) { return module; }));

  for (auto module : modules)
    eraseEmptyModule(module);
}

void IMDeadCodeElimPass::visitValue(Value value) {
  assert(isKnownAlive(value) && "only alive values reach here");

  // Propagate liveness through users.
  for (Operation *user : value.getUsers())
    visitUser(user);

  // Requiring an input port propagates the liveness to each instance.
  if (auto blockArg = value.dyn_cast<BlockArgument>()) {
    auto module = cast<FModuleOp>(blockArg.getParentBlock()->getParentOp());
    auto portDirection = module.getPortDirection(blockArg.getArgNumber());
    // If the port is input, it's necessary to mark corresponding input ports of
    // instances as alive. We don't have to propagate the liveness of output
    // ports.
    if (portDirection == Direction::In)
      for (auto userOfResultPort : resultPortToInstanceResultMapping[blockArg])
        markAlive(userOfResultPort);
    return;
  }

  // Marking an instance port as alive propagates to the corresponding port of
  // the module.
  if (auto instance = value.getDefiningOp<InstanceOp>()) {
    auto instanceResult = value.cast<mlir::OpResult>();
    // Update the src, when it's an instance op.
    auto module =
        dyn_cast<FModuleOp>(*instanceGraph->getReferencedModule(instance));
    if (!module)
      return;

    BlockArgument modulePortVal =
        module.getArgument(instanceResult.getResultNumber());
    return markAlive(modulePortVal);
  }

  // If a port of a memory is alive, all other ports are.
  if (auto mem = value.getDefiningOp<MemOp>()) {
    for (auto port : mem->getResults())
      markAlive(port);
    return;
  }

  // If op is defined by an operation, mark its operands as alive.
  if (auto op = value.getDefiningOp())
    for (auto operand : op->getOperands())
      markAlive(operand);
}

void IMDeadCodeElimPass::visitConnect(FConnectLike connect) {
  // If the dest is alive, mark the source value as alive.
  if (isKnownAlive(connect.getDest()))
    markAlive(connect.getSrc());
}

void IMDeadCodeElimPass::visitSubelement(Operation *op) {
  if (isKnownAlive(op->getOperand(0)))
    markAlive(op->getResult(0));
}

void IMDeadCodeElimPass::rewriteModuleBody(FModuleOp module) {
  auto *body = module.getBodyBlock();
  // If the module is unreachable, just ignore it.
  // TODO: Erase this module from circuit op.
  if (!isBlockExecutable(body))
    return;

  // Walk the IR bottom-up when deleting operations.
  for (auto &op : llvm::make_early_inc_range(llvm::reverse(*body))) {
    // Connects to values that we found to be dead can be dropped.
    if (auto connect = dyn_cast<FConnectLike>(op)) {
      if (isAssumedDead(connect.getDest())) {
        LLVM_DEBUG(llvm::dbgs() << "DEAD: " << connect << "\n";);
        connect.erase();
        ++numErasedOps;
      }
      continue;
    }

    // Delete dead wires, regs and nodes.
    if (isDeclaration(&op) && isAssumedDead(&op)) {
      LLVM_DEBUG(llvm::dbgs() << "DEAD: " << op << "\n";);
      assert(op.use_empty() && "users should be already removed");
      op.erase();
      ++numErasedOps;
      continue;
    }

    // Remove non-sideeffect op using `isOpTriviallyDead`.
    if (mlir::isOpTriviallyDead(&op)) {
      op.erase();
      ++numErasedOps;
    }
  }
}

void IMDeadCodeElimPass::rewriteModuleSignature(FModuleOp module) {
  // If the module is unreachable, just ignore it.
  // TODO: Erase this module from circuit op.
  if (!isBlockExecutable(module.getBodyBlock()))
    return;

  // Ports of public modules cannot be modified.
  if (module.isPublic())
    return;

  InstanceGraphNode *instanceGraphNode =
      instanceGraph->lookup(module.moduleNameAttr());
  LLVM_DEBUG(llvm::dbgs() << "Prune ports of module: " << module.getName()
                          << "\n");
  unsigned numOldPorts = module.getNumPorts();
  llvm::BitVector deadPortIndexes(numOldPorts);

  ImplicitLocOpBuilder builder(module.getLoc(), module.getContext());
  builder.setInsertionPointToStart(module.getBodyBlock());

  for (auto index : llvm::seq(0u, numOldPorts)) {
    auto argument = module.getArgument(index);
    assert((!hasDontTouch(argument) || isKnownAlive(argument)) &&
           "If the port has don't touch, it should be known alive");

    // If the port has dontTouch, skip.
    if (hasDontTouch(argument))
      continue;

    // If the port is known alive, then we can't delete it except for write-only
    // output ports.
    if (isKnownAlive(argument)) {
      bool deadOutputPortAtAnyInstantiation =
          module.getPortDirection(index) == Direction::Out &&
          llvm::all_of(resultPortToInstanceResultMapping[argument],
                       [&](Value result) { return isAssumedDead(result); });

      if (!deadOutputPortAtAnyInstantiation)
        continue;

      // RefType can't be a wire, especially if it won't be erased.  Skip.
      if (argument.getType().isa<RefType>())
        continue;

      // Ok, this port is used only within its defined module. So we can replace
      // the port with a wire.
      WireOp wire = builder.create<WireOp>(argument.getType());

      // Since `liveSet` contains the port, we have to erase it from the set.
      liveSet.erase(argument);
      liveSet.insert(wire);
      argument.replaceAllUsesWith(wire);
      deadPortIndexes.set(index);
      continue;
    }

    // Replace the port with a dummy wire. This wire should be erased within
    // `rewriteModuleBody`.
    WireOp wire = builder.create<WireOp>(argument.getType());
    argument.replaceAllUsesWith(wire);
    assert(isAssumedDead(wire.getResult()) && "dummy wire must be dead");
    deadPortIndexes.set(index);
  }

  // If there is nothing to remove, abort.
  if (deadPortIndexes.none())
    return;

  // Erase arguments of the old module from liveSet to prevent from creating
  // dangling pointers.
  for (auto arg : module.getArguments())
    liveSet.erase(arg);

  // Delete ports from the module.
  module.erasePorts(deadPortIndexes);

  // Add arguments of the new module to liveSet.
  for (auto arg : module.getArguments())
    liveSet.insert(arg);

  // Rewrite all uses.
  for (auto *use : instanceGraphNode->uses()) {
    auto instance = cast<InstanceOp>(*use->getInstance());
    ImplicitLocOpBuilder builder(instance.getLoc(), instance);
    // Since we will rewrite instance op, it is necessary to remove old instance
    // results from liveSet.
    for (auto oldResult : instance.getResults())
      liveSet.erase(oldResult);

    // Replace old instance results with dummy wires.
    for (auto index : deadPortIndexes.set_bits()) {
      auto result = instance.getResult(index);
      assert(isAssumedDead(result) &&
             "instance results of dead ports must be dead");
      WireOp wire = builder.create<WireOp>(result.getType());
      result.replaceAllUsesWith(wire);
    }

    // Create a new instance op without dead ports.
    auto newInstance = instance.erasePorts(builder, deadPortIndexes);

    // Mark new results as alive.
    for (auto newResult : newInstance.getResults())
      liveSet.insert(newResult);

    instanceGraph->replaceInstance(instance, newInstance);
    // Remove old one.
    instance.erase();
  }

  numRemovedPorts += deadPortIndexes.count();
}

void IMDeadCodeElimPass::eraseEmptyModule(FModuleOp module) {
  // Public modules cannot be erased.
  if (module.isPublic())
    return;

  // If the module doesn't have arguments, operations or annotations, we
  // consider it to be dead.
  if (!module.getBodyBlock()->args_empty() || !module.getBodyBlock()->empty() ||
      !module.getAnnotations().empty())
    return;

  // Ok, the module is empty. Delete instances unless they have symbols.
  LLVM_DEBUG(llvm::dbgs() << "Erase " << module.getName() << "\n");

  InstanceGraphNode *instanceGraphNode =
      instanceGraph->lookup(module.moduleNameAttr());

  bool existsInstanceWithSymbol = false;
  for (auto *use : llvm::make_early_inc_range(instanceGraphNode->uses())) {
    auto instance = cast<InstanceOp>(use->getInstance());
    if (instance.getInnerSym()) {
      existsInstanceWithSymbol = true;
      continue;
    }
    use->erase();
    instance.erase();
  }

  // If there is an instance with a symbol, we don't delete the module itself.
  if (existsInstanceWithSymbol)
    return;

  instanceGraph->erase(instanceGraphNode);
  module.erase();
  ++numErasedModules;
}

std::unique_ptr<mlir::Pass> circt::firrtl::createIMDeadCodeElimPass() {
  return std::make_unique<IMDeadCodeElimPass>();
}
