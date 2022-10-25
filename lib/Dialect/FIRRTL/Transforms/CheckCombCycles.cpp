//===- CheckCombCycles.cpp - FIRRTL check combinational cycles --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the FIRRTL combintational cycles detection pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallSet.h"
#include <variant>

#define DEBUG_TYPE "check-comb-cycles"

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Node class
//===----------------------------------------------------------------------===//

using ConnectIterator =
    mlir::detail::op_iterator<FConnectLike, Region::OpIterator>;
using ConnectRange = llvm::iterator_range<ConnectIterator>;
// Pair to represent port number and field index.
using PortField = std::pair<long, long>;
// Map to represent the combinational paths from a PortField to any other
// PortFields.
using CombPathsType = DenseMap<PortField, SmallVector<PortField>>;
// Map to represent combinational paths between all the ports of a module.
using CombPathsMap = DenseMap<Operation *, CombPathsType>;

namespace {
/// The graph context containing pointers of the combinational paths map and the
/// instance graph.
struct NodeContext {
  CombPathsMap &map;
  InstanceGraph *graph;
  ConnectRange connects;
  // Record the last visited port that is source to a connect.
  PortField srcPort;
  // Current module being procesed.
  FModuleOp module;
  // All the comb paths between the current module's ports.
  CombPathsType *portPaths;
  // Set true, when the isCycle() is executing. Required to ensure certain data
  // structures are not updated during the Graph traversal inside the isCycle
  // call.
  bool cycleCheck;

  explicit NodeContext(CombPathsMap &map, InstanceGraph *graph,
                       ConnectRange connects, FModuleOp module)
      : map(map), graph(graph), connects(connects), module(module) {
    // Cache the ports path for this module. This ensures, that the current
    // module's paths are updated in `map`.
    portPaths = &map[module];
    // Initialize the last seen input port to invalid.
    srcPort = {-1, -1};
    // Normal traversal.
    cycleCheck = false;
  }
};
} // namespace

namespace {
/// The node class of combinational graph.
struct Node {
  Value value;
  long fieldIndex = 0;
  NodeContext *context;

  explicit Node(Value value = nullptr, NodeContext *context = nullptr)
      : value(value), context(context) {}

  explicit Node(Operation *op, NodeContext *nodeCtxt) : context(nodeCtxt) {
    if (!op) {
      value = nullptr;
      return;
    }
    // Assumption is that the op can either be connect, or with only one result.
    // This is ensured in ChildIterator::skipToNextValidChild.
    value = TypeSwitch<Operation *, Value>(op)
                .Case<FConnectLike>([&](FConnectLike connect) {
                  auto dst = connect.getDest();
                  auto dstRef = getFieldRefFromValue(dst);
                  if (isa<MemOp>(dstRef.getDefiningOp()))
                    return dst;

                  // Record the comb paths that exist between the ports of a
                  // module. If the source or destination is a module port, then
                  // record it.
                  auto addPortPaths = [&](FieldRef &ref, bool src) {
                    auto val = ref.getValue();
                    auto arg = val.dyn_cast_or_null<BlockArgument>();
                    if (!arg)
                      return;
                    auto argNum = arg.getArgNumber();
                    auto fieldId = ref.getFieldID();
                    if (src)
                      // If source, then a new comb path begins from this
                      // port,field pair.
                      context->srcPort = {argNum, fieldId};
                    else
                      // If the srcPort is valid, then a comb path established
                      // between module ports `srcPort` -> <argNum, fieldId>.
                      if (context->srcPort.first != -1 &&
                          context->srcPort.second != -1)
                        (*context->portPaths)[context->srcPort].push_back(
                            {argNum, fieldId});
                  };
                  if (!context->cycleCheck) {
                    // The nodes will be first traversed during DFS, and then
                    // also when `hasCycle` is called. The port paths must be
                    // recorded only during the first DFS, and not when the
                    // `hasCycle` is executing.
                    auto srcFieldRef = getFieldRefFromValue(connect.getSrc());
                    // First record the source port, then dst to ensure the path
                    // from src to dst is recorded correctly.
                    addPortPaths(srcFieldRef, true);
                    addPortPaths(dstRef, false);
                  }
                  fieldIndex = dstRef.getFieldID();
                  return dstRef.getValue();
                })
                .Default([&](auto) { return op->getResult(0); });
  }

  bool operator==(const Node &rhs) const { return value == rhs.value; }
  bool operator!=(const Node &rhs) const { return !(*this == rhs); }
  bool isNull() const { return value == nullptr; }
};
} // namespace

//===----------------------------------------------------------------------===//
// ChildIterator class
//===----------------------------------------------------------------------===//

namespace {
/// This is a child iterator class for FIRRTL values. The base iterator is the
/// `use_iterator` of a value. On top of the `use_iterator`, uses which are
/// 'dest`s of connect ops and uses whose owner has no results are filtered out.
class ChildIterator
    : llvm::iterator_facade_base<ChildIterator, std::forward_iterator_tag,
                                 Value> {
  void skipToNextValidChild() {
    auto isChild = [&]() {
      if (auto connect = dyn_cast<FConnectLike>(childIt->getOwner()))
        return childIt->get() == connect.getSrc();
      // For a SubIndex/Subfield op, next child is valid, only if it includes
      // the fieldIndex.
      if (auto sub = dyn_cast<SubfieldOp>(childIt->getOwner())) {
        auto fRef = sub.getInput().getType().cast<BundleType>().getFieldID(
            sub.getFieldIndex());
        return (fieldIndex < fRef + getBitWidth(sub.getType(), true).value() &&
                fieldIndex >= fRef);
      }
      if (auto sub = dyn_cast<SubindexOp>(childIt->getOwner())) {
        long fRef = sub.getInput().getType().cast<FVectorType>().getFieldID(
            sub.getIndex());
        return (fieldIndex < fRef + getBitWidth(sub.getType(), true).value() &&
                fieldIndex >= fRef);
      }
      return childIt->getOwner()->getNumResults() > 0;
    };
    while (childIt != childEnd && !isChild())
      ++childIt;
  }

public:
  ChildIterator() = default;
  explicit ChildIterator(Value v, long fieldIndex)
      : childEnd(v.use_end()), childIt(v.use_begin()), fieldIndex(fieldIndex) {

    LLVM_DEBUG(if (childIt != childEnd) llvm::dbgs()
               << "\n ChildIterator constructor for uses of:" << v);
    skipToNextValidChild();
  }

  /// The iterator is empty or at the end.
  bool isAtEnd() const { return childIt == nullptr || childIt == childEnd; }

  using llvm::iterator_facade_base<ChildIterator, std::forward_iterator_tag,
                                   Value>::operator++;
  ChildIterator &operator++() {
    assert(!isAtEnd() && "incrementing the end iterator");
    ++childIt;
    skipToNextValidChild();
    return *this;
  }

  Operation *operator*() {
    assert(!isAtEnd() && "dereferencing the end iterator");
    LLVM_DEBUG(llvm::dbgs()
               << "\n ChildIterator dereference :" << *childIt->getOwner());
    return childIt->getOwner();
  }

  bool operator==(const ChildIterator &rhs) const {
    return childIt == rhs.childIt;
  }
  bool operator!=(const ChildIterator &rhs) const { return !(*this == rhs); }

  Value::use_iterator childEnd;
  Value::use_iterator childIt;
  long fieldIndex;
};
} // namespace

//===----------------------------------------------------------------------===//
// NodeIterator class
//===----------------------------------------------------------------------===//

namespace {
/// The default node iterator.
class NodeIterator
    : public llvm::iterator_facade_base<NodeIterator, std::forward_iterator_tag,
                                        Node> {
public:
  explicit NodeIterator(Node node) : node(node) {
    // If value is non-null then set the childIt.
    if (node.value)
      child = ChildIterator(node.value, node.fieldIndex);
  }

  using llvm::iterator_facade_base<NodeIterator, std::forward_iterator_tag,
                                   Node>::operator++;
  NodeIterator &operator++() { return ++child, *this; }
  Node operator*() { return Node(*child, node.context); }

  bool operator==(const NodeIterator &rhs) const { return child == rhs.child; }
  bool operator!=(const NodeIterator &rhs) const { return !(*this == rhs); }

  Value getValue() { return node.value; }
  InstanceGraph *getInstanceGraph() {
    assert(node.context && "invalid node context");
    return node.context->graph;
  }
  bool isAtEnd() const { return child.isAtEnd(); }
  bool isEndIterator() const { return node.isNull(); }

private:
  Node node;

protected:
  ChildIterator child;
};
} // namespace

//===----------------------------------------------------------------------===//
// InstanceNodeIterator class
//===----------------------------------------------------------------------===//

namespace {
class InstanceNodeIterator : public NodeIterator {
  /// Skip instance ports with not child.
  void skipToNextValidPort() {
    ChildIterator newChild;
    while (combPathsIter != combPathsEnd) {
      newChild = ChildIterator(instance.getResult(combPathsIter->first),
                               combPathsIter->second);
      if (newChild.isAtEnd())
        ++combPathsIter;
      else
        break;
    }
    child = combPathsIter == combPathsEnd ? ChildIterator() : newChild;
  }

public:
  explicit InstanceNodeIterator(InstanceOp instance, Node node)
      : NodeIterator(node), instance(instance) {
    assert(instance == getValue().getDefiningOp<InstanceOp>() &&
           "instance must be the defining op of the node value");
    child = ChildIterator();
    combPathsIter = combPathsEnd = nullptr;

    // Query the combinational paths between IOs of the current instance.
    Operation *module = getInstanceGraph()->getReferencedModule(instance);
    if (!isa<FModuleOp>(module))
      return;
    auto mapIter = node.context->map.find(module);
    if (mapIter == node.context->map.end()) {
      return;
    }
    auto portNum = getValue().cast<OpResult>().getResultNumber();
    auto portsIter = mapIter->getSecond().find({portNum, node.fieldIndex});
    if (portsIter == mapIter->getSecond().end())
      return;
    auto &paths = portsIter->getSecond();
    combPathsIter = paths.begin();
    combPathsEnd = paths.end();

    skipToNextValidPort();
  }

  InstanceNodeIterator &operator++() {
    if (!child.isAtEnd())
      ++child;
    else {
      ++combPathsIter;
      skipToNextValidPort();
    }
    return *this;
  }

  long getPortNumber() const { return combPathsIter->first; }

private:
  InstanceOp instance;
  SmallVector<PortField>::iterator combPathsIter, combPathsEnd;
};
} // namespace

//===----------------------------------------------------------------------===//
// SubfieldNodeIterator class
//===----------------------------------------------------------------------===//

namespace {
class SubfieldNodeIterator : public NodeIterator {
public:
  explicit SubfieldNodeIterator(SubfieldOp subfield, Node node)
      : NodeIterator(node) {
    assert(subfield == node.value.getDefiningOp<SubfieldOp>() &&
           "subfield must be the defining op of the node value");

    auto memory = subfield.getInput().getDefiningOp<MemOp>();
    if (!memory) {
      subfield->emitOpError("input must be a port of a MemOp, please run "
                            "-firrtl-lower-types first");
      return;
    }

    if (memory.getReadLatency() != 0)
      return;

    auto portKind = memory.getPortKind(
        subfield.getInput().cast<OpResult>().getResultNumber());
    auto subfieldIndex = subfield.getFieldIndex();
    // Combinational path exists only when the current subfield is `addr`.
    if (!(portKind == MemOp::PortKind::Read &&
          subfieldIndex == (unsigned)ReadPortSubfield::addr) &&
        !(portKind == MemOp::PortKind::ReadWrite &&
          subfieldIndex == (unsigned)ReadWritePortSubfield::addr))
      return;

    // Only `data` or `rdata` subfield is combinationally connected to `addr`
    // subfield. Find the corresponding subfield op.
    for (auto user : subfield.getInput().getUsers()) {
      auto currentSubfield = dyn_cast<SubfieldOp>(user);
      if (!currentSubfield) {
        user->emitOpError("MemOp must be used by SubfieldOp, please run "
                          "-firrtl-lower-types first");
        return;
      }

      auto index = currentSubfield.getFieldIndex();
      if ((portKind == MemOp::PortKind::Read &&
           index == (unsigned)ReadPortSubfield::data) ||
          (portKind == MemOp::PortKind::ReadWrite &&
           index == (unsigned)ReadWritePortSubfield::rdata)) {
        child = ChildIterator(currentSubfield.getResult(), 0);
        dataPort = currentSubfield.getResult();
        return;
      }
    }
  }

  SubfieldNodeIterator &operator++() {
    if (!child.isAtEnd())
      ++child;
    else
      child = ChildIterator();
    return *this;
  }

  Value getDataPort() { return dataPort; }

private:
  Value dataPort;
};
} // namespace

//===----------------------------------------------------------------------===//
// DummySourceNodeIterator class
//===----------------------------------------------------------------------===//

namespace {
/// A dummy source node iterator on the `dest`s of all connect ops.
class DummySourceNodeIterator
    : public llvm::iterator_facade_base<DummySourceNodeIterator,
                                        std::forward_iterator_tag, Node> {
public:
  explicit DummySourceNodeIterator(Node node)
      : node(node), connect(node.context->connects.begin()) {}

  using llvm::iterator_facade_base<DummySourceNodeIterator,
                                   std::forward_iterator_tag, Node>::operator++;
  DummySourceNodeIterator &operator++() {
    assert(connect != node.context->connects.end() &&
           "incrementing the end iterator");
    return ++connect, *this;
  }
  Node operator*() {
    assert(connect != node.context->connects.end() &&
           "dereferencing the end iterator");
    return Node(*connect, node.context);
  }

  bool operator==(const DummySourceNodeIterator &rhs) const {
    return connect == rhs.connect;
  }
  bool operator!=(const DummySourceNodeIterator &rhs) const {
    return !(*this == rhs);
  }
  bool isAtEnd() const { return connect == node.context->connects.end(); }

private:
  Node node;
  ConnectIterator connect;
};
} // namespace

//===----------------------------------------------------------------------===//
// CombGraphIterator class
//===----------------------------------------------------------------------===//

namespace {

class CombGraphIterator
    : public llvm::iterator_facade_base<CombGraphIterator,
                                        std::forward_iterator_tag, Node> {
  using variant_iterator =
      std::variant<NodeIterator, InstanceNodeIterator, SubfieldNodeIterator,
                   DummySourceNodeIterator>;

public:
  explicit CombGraphIterator(Node node, bool end = false)
      : impl(dispatchConstructor(node, end)) {}

  variant_iterator dispatchConstructor(Node node, bool end) {
    if (end)
      return NodeIterator(Node(nullptr, node.context));
    if (!node.value)
      return DummySourceNodeIterator(node);

    auto defOp = node.value.getDefiningOp();
    if (!defOp)
      return NodeIterator(node);

    return TypeSwitch<Operation *, variant_iterator>(defOp)
        .Case<InstanceOp>([&](InstanceOp instance) {
          return InstanceNodeIterator(instance, node);
        })
        .Case<SubfieldOp>([&](SubfieldOp subfield) {
          if (isa_and_nonnull<MemOp>(subfield.getInput().getDefiningOp()))
            return static_cast<variant_iterator>(
                SubfieldNodeIterator(subfield, node));
          // This is required to explicitly ignore self loops of register.
          if (isa_and_nonnull<RegOp, RegResetOp>(
                  getFieldRefFromValue(subfield).getDefiningOp()))
            return static_cast<variant_iterator>(
                NodeIterator(Node(nullptr, node.context)));
          return static_cast<variant_iterator>(
              NodeIterator(Node((Operation *)subfield, node.context)));
        })
        .Case<SubindexOp>([&](SubindexOp sub) {
          // This is required to explicitly ignore self loops of register.
          if (isa_and_nonnull<RegOp, RegResetOp>(
                  getFieldRefFromValue(sub).getDefiningOp()))
            return static_cast<variant_iterator>(
                NodeIterator(Node(nullptr, node.context)));
          return static_cast<variant_iterator>(
              NodeIterator(Node((Operation *)sub, node.context)));
        })
        // The children of reg or regreset op are not iterated.
        .Case<RegOp, RegResetOp>(
            [&](auto) { return NodeIterator(Node(nullptr, node.context)); })
        .Default([&](auto) { return NodeIterator(node); });
  }

  using llvm::iterator_facade_base<CombGraphIterator, std::forward_iterator_tag,
                                   Node>::operator++;
  CombGraphIterator &operator++() {
    switch (impl.index()) {
    case 0:
      return ++std::get<NodeIterator>(impl), *this;
    case 1:
      return ++std::get<InstanceNodeIterator>(impl), *this;
    case 2:
      return ++std::get<SubfieldNodeIterator>(impl), *this;
    case 3:
      return ++std::get<DummySourceNodeIterator>(impl), *this;
    default:
      return llvm_unreachable("invalid iterator variant"), *this;
    }
  }

  const variant_iterator &getIteratorImpl() const { return impl; }

  Node operator*() {
    switch (impl.index()) {
    case 0:
      return *std::get<NodeIterator>(impl);
    case 1:
      return *std::get<InstanceNodeIterator>(impl);
    case 2:
      return *std::get<SubfieldNodeIterator>(impl);
    case 3:
      return *std::get<DummySourceNodeIterator>(impl);
    default:
      return llvm_unreachable("invalid iterator variant"), Node();
    }
  }

  bool operator==(const CombGraphIterator &rhs) const {
    // Comparing with EndIterator, implies just check isAtEnd.
    auto isAtEnd = [](const CombGraphIterator &a,
                      const CombGraphIterator &endIt) {
      switch (a.impl.index()) {
      case 0:
        return std::get<NodeIterator>(a.impl).isAtEnd();
      case 1:
        return std::get<InstanceNodeIterator>(a.impl).isAtEnd();
      case 2:
        return std::get<SubfieldNodeIterator>(a.impl).isAtEnd();
      case 3:
        return std::get<DummySourceNodeIterator>(a.impl).isAtEnd();
      default:
        return llvm_unreachable("invalid iterator variant"), true;
      }
    };

    if (rhs.impl.index() == 0 &&
        std::get<NodeIterator>(rhs.impl).isEndIterator())
      return isAtEnd(*this, rhs);
    if (impl.index() == 0 && std::get<NodeIterator>(impl).isEndIterator())
      return isAtEnd(rhs, *this);
    return impl == rhs.impl;
  }
  bool operator!=(const CombGraphIterator &rhs) const {
    return !(*this == rhs);
  }

private:
  variant_iterator impl;
};
} // namespace

//===----------------------------------------------------------------------===//
// DenseMapInfo on Node
//===----------------------------------------------------------------------===//

namespace llvm {
template <>
struct DenseMapInfo<Node> {
  static Node getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return Node(Value::getFromOpaquePointer(pointer));
  }
  static Node getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return Node(Value::getFromOpaquePointer(pointer));
  }

  static unsigned getHashValue(const Node &node) {
    return mlir::hash_value(node.value);
  }
  static bool isEqual(const Node &lhs, const Node &rhs) { return lhs == rhs; }
};
} // namespace llvm

namespace {
struct NodeDenseSet : public DenseSet<Node> {
  using BaseSet = DenseSet<Node>;
  using iterator = typename BaseSet::iterator;

  std::pair<iterator, bool> insert(Node N) { return BaseSet::insert(N); }
  template <typename IterT>
  void insert(IterT Begin, IterT End) {
    BaseSet::insert(Begin, End);
  }

  void completed(Node) {}
};
} // namespace

//===----------------------------------------------------------------------===//
// GraphTraits on Node
//===----------------------------------------------------------------------===//

namespace llvm {
template <>
struct GraphTraits<Node> {
  using NodeRef = Node;
  using ChildIteratorType = CombGraphIterator;

  static NodeRef getEntryNode(NodeRef node) { return node; }

  static inline ChildIteratorType child_begin(NodeRef node) {
    return ChildIteratorType(node);
  }
  static inline ChildIteratorType child_end(NodeRef node) {
    return ChildIteratorType(node, /*end=*/true);
  }
};
} // namespace llvm

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {

using SCCIterator = llvm::scc_iterator<Node>;
using GT = llvm::GraphTraits<Node>;

// Sample a cycle from SCC.
SmallVector<Node> sampleCycle(SCCIterator &scc) {
  llvm::SmallDenseSet<Node, 4> sccNodes;
  for (auto node : *scc)
    sccNodes.insert(node);

  auto current = *(*scc).begin();
  auto last = current;
  SmallVector<Node> path;
  SmallDenseMap<Node, unsigned> visitOrder;
  while (true) {
    for (auto child :
         llvm::make_range(GT::child_begin(current), GT::child_end(current))) {
      // If the child is out of SCC, we don't explore.
      if (!sccNodes.contains(child))
        continue;

      auto it = visitOrder.find(child);
      // If the child is visited before, path[visitedOrder[c]] ... path.end()
      // forms a cycle.
      if (it != visitOrder.end())
        return SmallVector<Node>(path.begin() + it->second, path.end());

      visitOrder.insert({child, path.size()});
      path.push_back(child);
      current = child;
      break;
    }
    if (last == current)
      return SmallVector<Node>(path.begin(), path.end());
    last = current;
  }
  llvm_unreachable("a cycle must be found in SCC");
}

// This function dumps a (shortest) path from `inputPort` to `outputPort` by
// looking at a module referred by `instance`.
void dumpPathBetweenModulePorts(SmallString<16> &instancePath,
                                InstanceOp instance, unsigned inputPort,
                                unsigned outputPort, NodeContext *context,
                                mlir::InFlightDiagnostic &diag);

// Dump a path to a diagnostic.
void dumpPath(SmallVector<Node> &path, SmallString<16> &instancePath,
              FModuleOp module, bool isCycle, mlir::InFlightDiagnostic &diag) {
  unsigned pathSize = isCycle ? path.size() + 1 : path.size();
  for (unsigned i = 0, e = pathSize; i < e; ++i) {
    Node current = path[i % path.size()];
    Value currentValue = current.value;
    FieldRef fRef(current.value, current.fieldIndex);
    bool isCycleEnd = i == path.size();
    auto attachInfo = [&]() -> mlir::Diagnostic & {
      return diag.attachNote(currentValue.getLoc()) << instancePath << ".";
    };

    // If the currentValue is port, emit its name.
    if (auto arg = currentValue.dyn_cast<BlockArgument>()) {
      attachInfo() << getFieldName(fRef);
      continue;
    }

    TypeSwitch<Operation *>(currentValue.getDefiningOp())
        .Case<WireOp, RegOp, RegResetOp>(
            // For operations which declare signals, we simply print signal
            // names.
            [&](auto op) { attachInfo() << getFieldName(fRef); })
        .Case<InstanceOp, SubfieldOp>([&](auto op) {
          // If the op is InstanceOp or SubfieldOp, it is necessary to
          // investigate the next value since output values do not expilicty
          // appear in the cycle.
          Node next = path[(i + 1) % path.size()];
          for (auto iter = GT::child_begin(current),
                    end = GT::child_end(current);
               iter != end; ++iter) {
            if ((*iter).value != next.value)
              continue;

            auto iterImpl = iter.getIteratorImpl();
            if (std::holds_alternative<InstanceNodeIterator>(iterImpl)) {
              // Instance. Print names of input and output ports.
              auto instance = currentValue.getDefiningOp<InstanceOp>();
              auto inputPort = currentValue.cast<OpResult>().getResultNumber();
              auto outputPort =
                  std::get<InstanceNodeIterator>(iterImpl).getPortNumber();
              if (isCycleEnd)
                attachInfo() << instance.getName() << "."
                             << instance.getPortName(inputPort).str();
              else
                dumpPathBetweenModulePorts(instancePath, instance, inputPort,
                                           outputPort, current.context, diag);

            } else if (std::holds_alternative<SubfieldNodeIterator>(iterImpl)) {
              // SubfieldNodeIterator represents the connection between addr
              // port and data port.
              auto subfieldAddr = currentValue.getDefiningOp<SubfieldOp>();
              auto subfieldData =
                  std::get<SubfieldNodeIterator>(iterImpl).getDataPort();

              attachInfo() << getFieldName(getFieldRefFromValue(subfieldAddr));
              if (!isCycleEnd)
                diag.attachNote(subfieldData.getLoc())
                    << module.getName().str() << "."
                    << getFieldName(getFieldRefFromValue(subfieldData));
            }
            break;
          }
        })
        .Default([&](auto op) {});
  }
}

// This function dumps a (shortest) path from `inputPort` to `outputPort` by
// looking at a module referred by `instance`.
void dumpPathBetweenModulePorts(SmallString<16> &instancePath,
                                InstanceOp instance, unsigned inputPort,
                                unsigned outputPort, NodeContext *context,
                                mlir::InFlightDiagnostic &diag) {
  auto module = dyn_cast<FModuleOp>(*instance.getReferencedModule());
  if (!module)
    return;
  unsigned instancePathSize = instancePath.size();
  instancePath += ".";
  instancePath += instance.getName();
  auto start = Node(module.getArgument(inputPort), context);
  auto end = Node(module.getArgument(outputPort), context);
  llvm::DenseMap<Node, Node> previousNode;
  // Find a shortest path from input port to output port by BFS.
  std::queue<Node> que;
  que.push(start);
  while (!que.empty()) {
    Node current = que.front();
    if (current == end)
      break;
    for (auto child :
         llvm::make_range(GT::child_begin(current), GT::child_end(current))) {
      // If a child is marked before, skip it.
      if (previousNode.count(child))
        continue;
      // Record a previous node.
      previousNode.insert({child, current});
      que.push(child);
    }

    que.pop();
  }

  SmallVector<Node> path;
  Node current = end;
  // Reconstruct a path backwardly.
  while (current.value) {
    path.push_back(current);
    current = previousNode[current];
  }
  std::reverse(path.begin(), path.end());

  diag.attachNote(instance.getLoc())
      << "Instance " << instancePath << ": "
      << instance.getPortName(inputPort).str() << " ---> "
      << instance.getPortName(outputPort).str();
  dumpPath(path, instancePath, module, /*isCycle=*/false, diag);
  instancePath.resize(instancePathSize);
}

void dumpSimpleCycle(SCCIterator &scc, FModuleOp module,
                     mlir::InFlightDiagnostic &diag) {
  // Sample a cycle to print.
  SmallVector<Node> cycle = sampleCycle(scc);
  SmallString<16> instancePath;
  instancePath += module.getName();
  dumpPath(cycle, instancePath, module, /*isCycle=*/true, diag);
}

/// This pass constructs a local graph for each module to detect combinational
/// cycles. To capture the cross-module combinational cycles, this pass inlines
/// the combinational paths between IOs of its subinstances into a subgraph and
/// encodes them in a `combPathsMap`.
class CheckCombCyclesPass : public CheckCombCyclesBase<CheckCombCyclesPass> {
  void runOnOperation() override {
    auto &instanceGraph = getAnalysis<InstanceGraph>();
    bool detectedCycle = false;

    // Traverse modules in a post order to make sure the combinational paths
    // between IOs of a module have been detected and recorded in `combPathsMap`
    // before we handle its parent modules.
    for (auto node : llvm::post_order<InstanceGraph *>(&instanceGraph)) {
      if (auto module = dyn_cast<FModuleOp>(*node->getModule())) {
        NodeContext context(map, &instanceGraph, module.getOps<FConnectLike>(),
                            module);
        auto dummyNode = Node(nullptr, &context);

        // Traversing SCCs in the combinational graph to detect cycles. As
        // FIRRTL module is an SSA region, all cycles must contain at least one
        // connect op. Thus we introduce a dummy source node to iterate on the
        // `dest`s of all connect ops in the module.
        for (auto combSCC = SCCIterator::begin(dummyNode); !combSCC.isAtEnd();
             ++combSCC) {
          context.srcPort = {-1, -1};
          context.cycleCheck = true;
          if (combSCC.hasCycle()) {
            detectedCycle = true;
            auto errorDiag = mlir::emitError(
                module.getLoc(),
                "detected combinational cycle in a FIRRTL module");
            if (printSimpleCycle)
              dumpSimpleCycle(combSCC, module, errorDiag);
            else {
              for (auto node : *combSCC) {
                auto &noteDiag = errorDiag.attachNote(node.value.getLoc());
                noteDiag << "this operation is part of the combinational cycle";
              }
            }
          }
          context.cycleCheck = false;
        }

        LLVM_DEBUG({
          llvm::dbgs() << "\n printing the port paths :"
                       << module.moduleNameAttr();
          for (auto p : map[module]) {
            llvm::dbgs() << "\n port pair:"
                         << module.getPortNameAttr(p.first.first) << ","
                         << p.first.second;
            for (auto dst : p.second)
              llvm::dbgs() << "\n connected to :"
                           << module.getPortNameAttr(dst.first) << ","
                           << dst.second;
          }
        });

        continue;
      }
      if (auto extModule = dyn_cast<FExtModuleOp>(*node->getModule())) {
        // TODO: Handle FExtModuleOp with `ExtModulePathAnnotation`s.
        continue;
      }
      llvm_unreachable("invalid instance graph node");
    }

    if (detectedCycle)
      signalPassFailure();
    markAllAnalysesPreserved();
  }

private:
  /// A global map from FIRRTL modules to their combinational paths between IOs.
  CombPathsMap map;
};
} // namespace

std::unique_ptr<mlir::Pass> circt::firrtl::createCheckCombCyclesPass() {
  return std::make_unique<CheckCombCyclesPass>();
}
