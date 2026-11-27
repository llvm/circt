//===- ConvertToArcs.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ConvertToArcs.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/LLHDOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Support/ConversionPatternSet.h"
#include "circt/Support/Namespace.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/IntEqClasses.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "convert-to-arcs"

using namespace circt;
using namespace arc;
using namespace hw;
using llvm::SmallSetVector;
using mlir::ConversionConfig;

static bool isArcBreakingOp(Operation *op) {
  if (isa<TapOp>(op))
    return false;
  return op->hasTrait<OpTrait::ConstantLike>() ||
         isa<hw::InstanceOp, seq::CompRegOp, MemoryOp, MemoryReadPortOp,
             ClockedOpInterface, seq::InitialOp, seq::ClockGateOp,
             sim::DPICallOp, llhd::ProbeOp>(op) ||
         op->getNumResults() > 1 || op->getNumRegions() > 0 ||
         !mlir::isMemoryEffectFree(op);
}

static LogicalResult convertInitialValue(seq::CompRegOp reg,
                                         SmallVectorImpl<Value> &values) {
  if (!reg.getInitialValue())
    return values.push_back({}), success();

  // Use from_immutable cast to convert the seq.immutable type to the reg's
  // type.
  OpBuilder builder(reg);
  auto init = seq::FromImmutableOp::create(builder, reg.getLoc(), reg.getType(),
                                           reg.getInitialValue());

  values.push_back(init);
  return success();
}

//===----------------------------------------------------------------------===//
// Conversion
//===----------------------------------------------------------------------===//

namespace {
/// The operation dependency graph used to identify computations that require
/// fixed-point evaluation.
class FixedPointGraph {
public:
  explicit FixedPointGraph(HWModuleOp module);
  ArrayRef<Operation *> getSuccessors(const Operation *op) const;
  ArrayRef<Operation *> getNodes() const { return nodes; }

private:
  void collectNodes(HWModuleOp module);
  void collectEdges(HWModuleOp module);
  void addOrdinarySSAEdges(HWModuleOp module, Operation *op);
  void addDriveProbeEdges(Operation *op);
  void addNode(Operation *op);
  void addEdge(Operation *from, Operation *to);

  SmallVector<Operation *> nodes;
  DenseSet<const Operation *> nodeSet;
  DenseMap<const Operation *, SmallVector<Operation *>> successors;
};

/// Find strongly connected components using Tarjan's algorithm.
/// Singleton components are returned only when they have a self-edge.
class CyclicComponentFinder {
public:
  SmallVector<SmallVector<Operation *>> run(const FixedPointGraph &graph);

private:
  struct Frame {
    Operation *op;
    ArrayRef<Operation *> remainingSuccessors;
  };

  void findComponentsFrom(Operation *root, const FixedPointGraph &graph);
  void discoverNode(Operation *op);
  void extractCyclicComponent(Operation *root, const FixedPointGraph &graph);

  DenseMap<Operation *, size_t> indices;
  DenseMap<Operation *, size_t> lowLinks;
  DenseSet<Operation *> onStack;
  SmallVector<Operation *> stack;
  SmallVector<SmallVector<Operation *>> cyclicComponents;
  size_t nextIndex = 0;
};
} // namespace

FixedPointGraph::FixedPointGraph(HWModuleOp module) {
  collectNodes(module);
  collectEdges(module);
}

ArrayRef<Operation *>
FixedPointGraph::getSuccessors(const Operation *op) const {
  auto iter = successors.find(op);
  if (iter == successors.end())
    return {};
  return iter->second;
}

/// Collect the operations that may participate in an instantaneous dependency
/// cycle. State and scheduling boundaries are excluded, except for the LLHD
/// operations needed to represent zero-delay blocking assignments.
void FixedPointGraph::collectNodes(HWModuleOp module) {
  for (auto &op : module.getOps()) {
    if (isa<hw::OutputOp, FixedPointOp>(op))
      continue;
    if (isa<llhd::ProbeOp, llhd::DriveLikeOp, CoroutineInstanceOp>(op) ||
        !isArcBreakingOp(&op))
      addNode(&op);
  }
}

void FixedPointGraph::collectEdges(HWModuleOp module) {
  for (auto *op : nodes) {
    addOrdinarySSAEdges(module, op);
    addDriveProbeEdges(op);
  }
}

void FixedPointGraph::addOrdinarySSAEdges(HWModuleOp module, Operation *op) {
  for (auto result : op->getResults()) {
    for (auto &use : result.getUses()) {
      auto *user = use.getOwner();
      while (user->getParentOp() != module)
        user = user->getParentOp();
      addEdge(op, user);
    }
  }
}

void FixedPointGraph::addDriveProbeEdges(Operation *op) {
  auto drive = dyn_cast<llhd::DriveLikeOp>(op);
  if (!drive)
    return;
  auto signal = drive.getDriven().getDefiningOp<llhd::SignalOp>();
  if (!signal)
    return;
  for (auto *user : signal->getUsers())
    if (auto probe = dyn_cast<llhd::ProbeOp>(user);
        probe && probe.getSignal() == drive.getDriven())
      addEdge(op, probe);
}

void FixedPointGraph::addNode(Operation *op) {
  nodes.push_back(op);
  nodeSet.insert(op);
}

void FixedPointGraph::addEdge(Operation *from, Operation *to) {
  if (!from || !to || !nodeSet.contains(from) || !nodeSet.contains(to))
    return;
  auto &targets = successors[from];
  if (!llvm::is_contained(targets, to))
    targets.push_back(to);
}

SmallVector<SmallVector<Operation *>>
CyclicComponentFinder::run(const FixedPointGraph &graph) {
  indices.clear();
  lowLinks.clear();
  onStack.clear();
  stack.clear();
  cyclicComponents.clear();
  nextIndex = 0;

  for (auto *op : graph.getNodes())
    if (!indices.contains(op))
      findComponentsFrom(op, graph);
  return std::move(cyclicComponents);
}

/// Explore nodes with an explicit DFS stack and collect completed components.
void CyclicComponentFinder::findComponentsFrom(Operation *root,
                                               const FixedPointGraph &graph) {
  SmallVector<Frame> worklist;
  discoverNode(root);
  worklist.push_back({root, graph.getSuccessors(root)});

  while (!worklist.empty()) {
    auto &frame = worklist.back();
    auto *op = frame.op;
    if (!frame.remainingSuccessors.empty()) {
      auto *successor = frame.remainingSuccessors.front();
      frame.remainingSuccessors = frame.remainingSuccessors.drop_front();
      auto successorIndex = indices.find(successor);
      if (successorIndex == indices.end()) {
        discoverNode(successor);
        worklist.push_back({successor, graph.getSuccessors(successor)});
      } else if (onStack.contains(successor)) {
        lowLinks[op] = std::min(lowLinks[op], successorIndex->second);
      }
      continue;
    }

    if (lowLinks[op] == indices[op])
      extractCyclicComponent(op, graph);
    worklist.pop_back();
    if (!worklist.empty()) {
      auto *parent = worklist.back().op;
      lowLinks[parent] = std::min(lowLinks[parent], lowLinks[op]);
    }
  }
}

/// Assign a discovery index and add the node to the component stack.
void CyclicComponentFinder::discoverNode(Operation *op) {
  auto index = nextIndex++;
  indices[op] = index;
  lowLinks[op] = index;
  stack.push_back(op);
  onStack.insert(op);
}

/// Pop one completed SCC and retain it only if it contains a cycle.
void CyclicComponentFinder::extractCyclicComponent(
    Operation *root, const FixedPointGraph &graph) {
  SmallVector<Operation *> component;
  Operation *member;
  do {
    member = stack.pop_back_val();
    onStack.erase(member);
    component.push_back(member);
  } while (member != root);

  bool hasSelfLoop = component.size() == 1 &&
                     llvm::is_contained(graph.getSuccessors(component.front()),
                                        component.front());
  if (component.size() > 1 || hasSelfLoop)
    cyclicComponents.push_back(std::move(component));
}

/// Include the downstream computation of each signal cycle so that sensitive
/// coroutines observe intermediate signal changes, not just the stable results.
/// Overlapping closures must share a region: a coroutine has one instance and
/// cannot independently participate in two fixed-point evaluations.
static SmallVector<DenseSet<Operation *>>
findSignalEvaluationGroups(const FixedPointGraph &graph) {
  auto components = CyclicComponentFinder().run(graph);
  llvm::erase_if(components, [](const auto &component) {
    return llvm::none_of(
        component, [](Operation *op) { return isa<llhd::DriveLikeOp>(op); });
  });

  llvm::IntEqClasses componentClasses(components.size());
  DenseMap<Operation *, size_t> owners;
  for (auto [index, component] : llvm::enumerate(components)) {
    SmallVector<Operation *> worklist(component);
    while (!worklist.empty()) {
      auto *op = worklist.pop_back_val();
      auto [it, inserted] = owners.try_emplace(op, index);
      if (!inserted) {
        componentClasses.join(index, it->second);
        continue;
      }
      llvm::append_range(worklist, graph.getSuccessors(op));
    }
  }

  componentClasses.compress();
  SmallVector<DenseSet<Operation *>> evaluationGroups(
      componentClasses.getNumClasses());
  for (auto [op, owner] : owners)
    evaluationGroups[componentClasses[owner]].insert(op);
  return evaluationGroups;
}

namespace {
struct Converter {
  LogicalResult run(ModuleOp module);
  LogicalResult runOnModule(HWModuleOp module);
  LogicalResult formFixedPoints(HWModuleOp module);
  LogicalResult formSignalFixedPoint(HWModuleOp module,
                                     DenseSet<Operation *> component);
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
    if (auto sym = dyn_cast<mlir::SymbolOpInterface>(&op))
      globalNamespace.newName(sym.getName());
  for (auto module : module.getOps<HWModuleOp>())
    if (failed(runOnModule(module)))
      return failure();
  return success();
}

LogicalResult Converter::runOnModule(HWModuleOp module) {
  if (failed(formFixedPoints(module)))
    return failure();

  // Find all arc-breaking operations in this module and assign them an index.
  arcBreakers.clear();
  arcBreakerIndices.clear();
  for (Operation &op : *module.getBodyBlock()) {
    if (isa<seq::InitialOp>(&op))
      continue;
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

/// Find strongly connected components in the module operation graph. In
/// addition to ordinary SSA dependencies, an LLHD drive produces the value
/// observed by every probe of the driven signal. Signal cycles and their
/// downstream observers are made explicit as arc.fixed_point regions before
/// the ordinary arc extraction proceeds. Pure SSA cycles remain subject to
/// the existing combinational-loop diagnostics.
LogicalResult Converter::formFixedPoints(HWModuleOp module) {
  if (module.getOps<llhd::DriveLikeOp>().empty())
    return success();

  FixedPointGraph graph(module);
  auto evaluationGroups = findSignalEvaluationGroups(graph);
  for (auto &group : evaluationGroups)
    if (failed(formSignalFixedPoint(module, std::move(group))))
      return failure();
  return success();
}

namespace {
class SignalFixedPointBuilder {
public:
  SignalFixedPointBuilder(HWModuleOp module, DenseSet<Operation *> component)
      : module(module), component(std::move(component)),
        builder(module.getBodyBlock()->getTerminator()) {}

  LogicalResult run();

private:
  LogicalResult collectSignals();
  LogicalResult checkExternalDrives();
  void collectMovedOps();
  void collectMovedOpCaptures();
  void collectDriveCaptures();
  void collectExternalResults();
  LogicalResult createRegion();
  LogicalResult mapSignalAccesses();
  void moveOpsIntoRegion();
  void createOutput();
  void replaceExternalUses();
  void replaceDrivesAndEraseProbes();
  void addCaptureIfExternal(Value value);

  HWModuleOp module;
  DenseSet<Operation *> component;
  OpBuilder builder;
  FixedPointOp fixedPoint;
  MapVector<llhd::SignalOp, llhd::DriveLikeOp> signalDrives;
  SmallVector<Operation *> movedOps;
  SmallSetVector<Value, 8> captures;
  SmallVector<Value> externalResults;
  SmallVector<SmallVector<OpOperand *>> externalUses;
  SmallVector<BlockArgument> feedbackArguments;
  DenseMap<Value, Value> captureToArgument;
  MapVector<Value, BlockArgument> probeResultToFeedbackArgument;
};
} // namespace

LogicalResult SignalFixedPointBuilder::run() {
  if (failed(collectSignals()) || failed(checkExternalDrives()))
    return failure();
  collectMovedOps();
  collectMovedOpCaptures();
  collectDriveCaptures();
  collectExternalResults();
  if (failed(createRegion()))
    return failure();
  moveOpsIntoRegion();
  createOutput();
  // The fixed-point body requires SSA dominance.
  // Signal feedback has now been replaced by block arguments, so
  // the remaining computation must admit a topological ordering.
  if (!mlir::sortTopologically(&fixedPoint.getBody().front()))
    return fixedPoint.emitOpError(
        "contains an SSA dependency cycle after resolving signal feedback");
  replaceExternalUses();
  replaceDrivesAndEraseProbes();
  return success();
}

/// Collect the feedback signals and validate their types and drives.
LogicalResult SignalFixedPointBuilder::collectSignals() {
  for (auto &op : module.getOps()) {
    if (!component.contains(&op))
      continue;
    auto drive = dyn_cast<llhd::DriveLikeOp>(op);
    if (!drive)
      continue;
    auto signalDefOp = drive.getDriven().getDefiningOp<llhd::SignalOp>();
    if (!signalDefOp)
      return drive.emitOpError(
          "fixed-point drives require a locally defined signal");
    if (!drive.getValue().getType().isSignlessInteger())
      return drive.emitOpError(
          "fixed-point signal values must be signless integers");
    if (!signalDrives.insert({signalDefOp, drive}).second)
      return drive.emitOpError(
          "multiple drives to one signal in a fixed-point region are not "
          "supported");

    if (auto time = drive.getTime()) {
      auto delay = time.getDefiningOp<llhd::ConstantTimeOp>();
      if (!delay || delay.getValue().getTime() != 0 ||
          delay.getValue().getDelta() != 0)
        return drive.emitOpError(
            "fixed-point regions support only zero-time, zero-delta drives");
    }
  }

  if (signalDrives.empty())
    return emitError(module.getLoc(),
                     "signal dependency cycle does not contain a drive");
  return success();
}

/// Every feedback signal must be controlled entirely by this component.
LogicalResult SignalFixedPointBuilder::checkExternalDrives() {
  for (auto signal : signalDrives.keys()) {
    for (auto *user : signal->getUsers()) {
      auto drive = dyn_cast<llhd::DriveLikeOp>(user);
      if (drive && drive.getDriven() == signal.getResult() &&
          !component.contains(user))
        return drive.emitOpError(
            "signals in fixed-point regions cannot have external drives");
    }
  }
  return success();
}

void SignalFixedPointBuilder::addCaptureIfExternal(Value value) {
  auto *defOp = value.getDefiningOp();
  if (defOp && component.contains(defOp))
    return;
  captures.insert(value);
}

void SignalFixedPointBuilder::collectMovedOpCaptures() {
  for (auto *op : movedOps) {
    for (auto operand : op->getOperands())
      addCaptureIfExternal(operand);
  }
}

void SignalFixedPointBuilder::collectDriveCaptures() {
  for (auto drive : signalDrives.values()) {
    addCaptureIfExternal(drive.getValue());
    if (drive.getEnable())
      addCaptureIfExternal(drive.getEnable());
  }
}

void SignalFixedPointBuilder::collectMovedOps() {
  for (auto &op : module.getOps())
    if (component.contains(&op) && !isa<llhd::ProbeOp, llhd::DriveLikeOp>(op))
      movedOps.push_back(&op);
}

/// Preserve results of moved operations that are used outside the component.
void SignalFixedPointBuilder::collectExternalResults() {
  for (auto *op : movedOps) {
    for (auto result : op->getResults()) {
      SmallVector<OpOperand *> uses;
      for (auto &use : result.getUses())
        if (!component.contains(use.getOwner()))
          uses.push_back(&use);
      if (uses.empty())
        continue;
      externalResults.push_back(result);
      externalUses.push_back(std::move(uses));
    }
  }
}

/// Create the region and map captures and probes to its block arguments.
LogicalResult SignalFixedPointBuilder::createRegion() {
  auto loc = signalDrives.front().first.getLoc();
  // Seed feedback with the current signal values.
  SmallVector<Value> initials;
  for (auto signal : signalDrives.keys())
    initials.push_back(
        llhd::ProbeOp::create(builder, signal.getLoc(), signal.getResult()));

  SmallVector<Type> resultTypes;
  llvm::append_range(resultTypes, ValueRange(initials).getTypes());
  llvm::append_range(resultTypes, ValueRange(externalResults).getTypes());
  fixedPoint = FixedPointOp::create(builder, loc, resultTypes, initials,
                                    captures.getArrayRef());
  auto &body = fixedPoint.getBody().emplaceBlock();

  // Add feedback arguments followed by invariant capture arguments.
  for (auto signal : signalDrives.keys())
    feedbackArguments.push_back(
        body.addArgument(signal.getInit().getType(), signal.getLoc()));

  for (auto capture : captures)
    captureToArgument[capture] =
        body.addArgument(capture.getType(), capture.getLoc());

  return mapSignalAccesses();
}

/// Map probes to their corresponding feedback arguments.
LogicalResult SignalFixedPointBuilder::mapSignalAccesses() {
  for (auto &op : module.getOps()) {
    if (!component.contains(&op))
      continue;
    if (auto probe = dyn_cast<llhd::ProbeOp>(op)) {
      auto signal = probe.getSignal().getDefiningOp<llhd::SignalOp>();
      auto *signalIt = signalDrives.find(signal);
      if (signalIt == signalDrives.end())
        return probe.emitOpError(
            "probe in fixed-point region has no corresponding drive");
      auto index = signalIt - signalDrives.begin();
      probeResultToFeedbackArgument[probe.getResult()] =
          feedbackArguments[index];
    }
  }
  return success();
}

/// Move the computation and reconnect its operands to the region arguments.
void SignalFixedPointBuilder::moveOpsIntoRegion() {
  auto &body = fixedPoint.getBody().front();
  for (auto *op : movedOps) {
    op->moveBefore(&body, body.end());
    for (auto &operand : op->getOpOperands()) {
      if (auto mapped = probeResultToFeedbackArgument.lookup(operand.get()))
        operand.set(mapped);
      else if (auto mapped = captureToArgument.lookup(operand.get()))
        operand.set(mapped);
    }
  }
}

/// Yield the next signal values and any additional externally visible results.
void SignalFixedPointBuilder::createOutput() {
  auto &body = fixedPoint.getBody().front();
  builder.setInsertionPointToEnd(&body);
  SmallVector<Value> yieldedValues;
  auto remapValue = [&](Value &value) {
    if (auto mapped = probeResultToFeedbackArgument.lookup(value))
      value = mapped;
    else if (auto mapped = captureToArgument.lookup(value))
      value = mapped;
  };

  for (auto [index, drive] : llvm::enumerate(signalDrives.values())) {
    Value value = drive.getValue();
    remapValue(value);

    if (Value enable = drive.getEnable()) {
      remapValue(enable);
      value = comb::MuxOp::create(builder, drive.getLoc(), enable, value,
                                  feedbackArguments[index]);
    }
    yieldedValues.push_back(value);
  }
  llvm::append_range(yieldedValues, externalResults);
  arc::OutputOp::create(builder, fixedPoint.getLoc(), yieldedValues);
}

/// Redirect users outside the component to the stable region results.
void SignalFixedPointBuilder::replaceExternalUses() {
  for (auto [result, feedback] : probeResultToFeedbackArgument) {
    auto stableResult = fixedPoint.getResult(feedback.getArgNumber());
    for (auto &use : llvm::make_early_inc_range(result.getUses()))
      if (!component.contains(use.getOwner()))
        use.set(stableResult);
  }
  for (auto [index, uses] : llvm::enumerate(externalUses))
    for (auto *use : uses)
      use->set(fixedPoint.getResult(signalDrives.size() + index));
}

/// Commit the stable signal values and remove the original signal accesses.
void SignalFixedPointBuilder::replaceDrivesAndEraseProbes() {
  builder.setInsertionPointAfter(fixedPoint);
  for (auto [index, oldDrive] : llvm::enumerate(signalDrives.values())) {
    auto drive =
        cast<llhd::DriveLikeOp>(builder.clone(*oldDrive.getOperation()));
    drive.getValueMutable().assign(fixedPoint.getResult(index));
    drive.getEnableMutable().clear();
  }

  for (auto drive : signalDrives.values())
    drive.erase();
  for (Value result : probeResultToFeedbackArgument.keys())
    result.getDefiningOp<llhd::ProbeOp>().erase();
}

/// Replace a cycle through local LLHD signals with a fixed-point computation.
LogicalResult Converter::formSignalFixedPoint(HWModuleOp module,
                                              DenseSet<Operation *> component) {
  return SignalFixedPointBuilder(module, std::move(component)).run();
}

LogicalResult Converter::analyzeFanIn() {
  SmallVector<std::tuple<Operation *, SmallVector<Value, 2>>> worklist;
  SetVector<Value> seenOperands;
  auto addToWorklist = [&](Operation *op) {
    seenOperands.clear();
    for (auto operand : op->getOperands())
      seenOperands.insert(operand);
    mlir::getUsedValuesDefinedAbove(op->getRegions(), seenOperands);
    worklist.emplace_back(op, seenOperands.getArrayRef());
  };

  // Seed the worklist and fanin masks with the arc breaking operations.
  faninMasks.clear();
  for (auto *op : arcBreakers) {
    unsigned index = arcBreakerIndices.lookup(op);
    auto mask = APInt::getOneBitSet(arcBreakers.size(), index);
    faninMasks[op] = mask;
    addToWorklist(op);
  }

  // Establish a post-order among the operations.
  DenseSet<Operation *> seen;
  DenseSet<Operation *> finished;
  postOrder.clear();
  while (!worklist.empty()) {
    auto &[op, operands] = worklist.back();
    if (operands.empty()) {
      if (!isArcBreakingOp(op) && !isa<hw::OutputOp>(op))
        postOrder.push_back(op);
      finished.insert(op);
      seen.erase(op);
      worklist.pop_back();
      continue;
    }
    auto operand = operands.pop_back_val(); // advance to next operand
    auto *definingOp = operand.getDefiningOp();
    if (!definingOp || isArcBreakingOp(definingOp) ||
        finished.contains(definingOp))
      continue;
    if (!seen.insert(definingOp).second) {
      definingOp->emitError("combinational loop detected");
      return failure();
    }
    addToWorklist(definingOp);
  }
  LLVM_DEBUG(llvm::dbgs() << "- Sorted " << postOrder.size() << " ops\n");

  // Compute fanin masks in reverse post-order, which will compute the mask
  // for an operation's uses before it computes it for the operation itself.
  // This allows us to compute the set of arc breakers an operation
  // contributes to in one pass.
  for (auto *op : llvm::reverse(postOrder)) {
    auto mask = APInt::getZero(arcBreakers.size());
    for (auto *user : op->getUsers()) {
      while (user->getParentOp() != op->getParentOp())
        user = user->getParentOp();
      auto *it = faninMasks.find(user);
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
    arc::OutputOp::create(builder, lastOp->getLoc(), outputs);

    // Create the arc definition.
    builder.setInsertionPoint(module);
    auto defOp =
        DefineOp::create(builder, lastOp->getLoc(),
                         builder.getStringAttr(globalNamespace.newName(
                             module.getModuleName() + "_arc")),
                         builder.getFunctionType(inputTypes, outputTypes));
    defOp.getBody().push_back(block.release());

    // Create the call to the arc definition to replace the operations that
    // we have just extracted.
    builder.setInsertionPoint(module.getBodyBlock()->getTerminator());
    auto arcOp = CallOp::create(builder, lastOp->getLoc(), defOp, inputs);
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

      if (failed(convertInitialValue(regOp, initialValues)))
        return failure();

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
          llvm::cast<SymbolRefAttr>(callOp.getCallableForCallee()),
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
  // The reset values in the `regsByInput` keys may be results of registers
  // that are themselves converted in this loop. Defer replacing and erasing the
  // registers until all shuffling arcs exist, so that no key refers to an
  // erased value when it is assigned as a reset operand.
  SmallVector<std::pair<seq::CompRegOp, Value>> regReplacements;
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
        if (failed(convertInitialValue(regOp, initialValues)))
          return failure();
      }
      regToOutputMapping.push_back(it->second);
    }

    auto loc = regOps.back().getLoc();
    arc::OutputOp::create(builder, loc, outputs);

    builder.setInsertionPoint(module);
    auto defOp = DefineOp::create(builder, loc,
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
        StateOp::create(builder, loc, defOp, std::get<0>(clockAndResetAndOp),
                        /*enable=*/Value{}, 1, inputs, initialValues);
    auto reset = std::get<1>(clockAndResetAndOp);
    if (reset)
      arcOp.getResetMutable().assign(reset);
    if (tapRegisters && llvm::any_of(names, [](auto name) {
          return !cast<StringAttr>(name).getValue().empty();
        }))
      arcOp->setAttr("names", builder.getArrayAttr(names));
    for (auto [reg, resultIdx] : llvm::zip(regOps, regToOutputMapping))
      regReplacements.push_back({reg, arcOp.getResult(resultIdx)});
  }
  for (auto [reg, arcResult] : regReplacements)
    reg.replaceAllUsesWith(arcResult);
  for (auto [reg, arcResult] : regReplacements)
    reg.erase();

  if (numMappedRegs > 0)
    LLVM_DEBUG(llvm::dbgs() << "- Mapped " << numMappedRegs << " regs to "
                            << regsByInput.size() << " shuffling arcs\n");

  return success();
}

//===----------------------------------------------------------------------===//
// LLHD Conversion
//===----------------------------------------------------------------------===//

/// `llhd.combinational` -> `arc.execute`
static LogicalResult convert(llhd::CombinationalOp op,
                             llhd::CombinationalOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter,
                             const TypeConverter &converter) {
  // Convert the result types.
  SmallVector<Type> resultTypes;
  if (failed(converter.convertTypes(op.getResultTypes(), resultTypes)))
    return failure();

  // Collect the SSA values defined outside but used inside the body region.
  auto cloneIntoBody = [](Operation *op) {
    return op->hasTrait<OpTrait::ConstantLike>();
  };
  auto operands =
      mlir::makeRegionIsolatedFromAbove(rewriter, op.getBody(), cloneIntoBody);

  // Create a replacement `arc.execute` op.
  auto executeOp =
      ExecuteOp::create(rewriter, op.getLoc(), resultTypes, operands);
  executeOp.getBody().takeBody(op.getBody());
  rewriter.replaceOp(op, executeOp.getResults());
  return success();
}

/// `llhd.yield` -> `arc.output`
static LogicalResult convert(llhd::YieldOp op, llhd::YieldOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<arc::OutputOp>(op, adaptor.getOperands());
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace circt {
#define GEN_PASS_DEF_CONVERTTOARCSPASS
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

namespace {
struct ConvertToArcsPass
    : public circt::impl::ConvertToArcsPassBase<ConvertToArcsPass> {
  using ConvertToArcsPassBase::ConvertToArcsPassBase;
  void runOnOperation() override;
};
} // namespace

void ConvertToArcsPass::runOnOperation() {
  // Pass-through type converter; this pass does not change any types.
  TypeConverter converter;
  converter.addConversion([](Type type) { return type; });

  // Gather the conversion patterns.
  ConversionPatternSet patterns(&getContext(), converter);
  patterns.add<llhd::CombinationalOp>(convert);
  patterns.add<llhd::YieldOp>(convert);

  // `llhd.combinational` and `llhd.yield` are the only LLHD ops rewritten by
  // this pass; all other ops are left untouched for subsequent lowering
  // passes to handle.
  ConversionTarget target(getContext());
  target.addIllegalOp<llhd::CombinationalOp, llhd::YieldOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  // Disable pattern rollback to use the faster one-shot dialect conversion.
  ConversionConfig config;
  config.allowPatternRollback = false;

  // Apply the dialect conversion patterns.
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns),
                                    config))) {
    emitError(getOperation().getLoc()) << "conversion to arcs failed";
    return signalPassFailure();
  }

  // Outline operations into arcs.
  Converter outliner;
  outliner.tapRegisters = tapRegisters;
  if (failed(outliner.run(getOperation())))
    return signalPassFailure();
}
