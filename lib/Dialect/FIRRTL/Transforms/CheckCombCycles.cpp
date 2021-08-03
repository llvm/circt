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
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/InstanceGraph.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include <variant>

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Node class
//===----------------------------------------------------------------------===//

using CombPathsType = SmallVector<SmallVector<size_t, 2>>;
using CombPathsMap = DenseMap<Operation *, CombPathsType>;

using ConnectIterator =
    mlir::detail::op_iterator<ConnectOp, Region::OpIterator>;
using ConnectRange = llvm::iterator_range<ConnectIterator>;

namespace {
/// The graph context containing pointers of the combinational paths map and the
/// instance graph.
struct NodeContext {
  CombPathsMap *map;
  InstanceGraph *graph;
  ConnectRange connects;

  explicit NodeContext(CombPathsMap *map, InstanceGraph *graph,
                       ConnectRange connects)
      : map(map), graph(graph), connects(connects) {}
};
} // namespace

namespace {
/// The node class of combinational graph.
struct Node {
  Value value;
  NodeContext *context;

  explicit Node(Value value = nullptr, NodeContext *context = nullptr)
      : value(value), context(context) {}

  bool operator==(const Node &rhs) const { return value == rhs.value; }
  bool operator!=(const Node &rhs) const { return !(*this == rhs); }
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
      if (auto connect = dyn_cast<ConnectOp>(childIt->getOwner()))
        return childIt->get() == connect.src();
      return childIt->getOwner()->getNumResults() > 0;
    };
    while (childIt != childEnd && !isChild())
      ++childIt;
  }

public:
  ChildIterator() = default;
  explicit ChildIterator(Value v, bool end = false)
      : childEnd(v.use_end()), childIt(end ? childEnd : v.use_begin()) {
    skipToNextValidChild();
  }

  /// The iterator is empty or at the end.
  bool isAtEnd() { return childIt == nullptr || childIt == childEnd; }

  using llvm::iterator_facade_base<ChildIterator, std::forward_iterator_tag,
                                   Value>::operator++;
  ChildIterator &operator++() {
    assert(!isAtEnd() && "incrementing the end iterator");
    ++childIt;
    skipToNextValidChild();
    return *this;
  }

  Value operator*() {
    assert(!isAtEnd() && "dereferencing the end iterator");
    if (auto connect = dyn_cast<ConnectOp>(childIt->getOwner()))
      return connect.dest();
    return childIt->getOwner()->getResult(0);
  }

  bool operator==(const ChildIterator &rhs) const {
    return childIt == rhs.childIt;
  }
  bool operator!=(const ChildIterator &rhs) const { return !(*this == rhs); }

private:
  Value::use_iterator childEnd;
  Value::use_iterator childIt;
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
  explicit NodeIterator(Node node, bool end = false)
      : node(node), child(ChildIterator(node.value, end)) {}

  using llvm::iterator_facade_base<NodeIterator, std::forward_iterator_tag,
                                   Node>::operator++;
  NodeIterator &operator++() { return ++child, *this; }
  Node operator*() { return Node(*child, node.context); }

  bool operator==(const NodeIterator &rhs) const { return child == rhs.child; }
  bool operator!=(const NodeIterator &rhs) const { return !(*this == rhs); }

  Value getValue() { return node.value; }
  CombPathsMap *getCombPathsMap() {
    assert(node.context && "invalid node context");
    return node.context->map;
  }
  InstanceGraph *getInstanceGraph() {
    assert(node.context && "invalid node context");
    return node.context->graph;
  }

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
    while (portIt != portEnd) {
      newChild = ChildIterator(instance.getResult(*portIt));
      if (newChild.isAtEnd())
        ++portIt;
      else
        break;
    }
    child = portIt == portEnd ? ChildIterator() : newChild;
  }

public:
  explicit InstanceNodeIterator(InstanceOp instance, Node node,
                                bool end = false)
      : NodeIterator(node, true), instance(instance) {
    assert(instance == getValue().getDefiningOp<InstanceOp>() &&
           "instance must be the defining op of the node value");
    if (end)
      return;

    // Query the combinational paths between IOs of the current instance.
    auto module = getInstanceGraph()->getReferencedModule(instance);
    auto &combPaths = getCombPathsMap()->FindAndConstruct(module).second;
    auto &ports = combPaths[getValue().cast<OpResult>().getResultNumber()];

    portEnd = ports.end();
    portIt = ports.begin();
    skipToNextValidPort();
  }

  InstanceNodeIterator &operator++() {
    if (!child.isAtEnd())
      ++child;
    if (child.isAtEnd()) {
      ++portIt;
      skipToNextValidPort();
    }
    return *this;
  }

private:
  InstanceOp instance;
  SmallVectorImpl<size_t>::iterator portEnd;
  SmallVectorImpl<size_t>::iterator portIt;
};
} // namespace

//===----------------------------------------------------------------------===//
// SubfieldNodeIterator class
//===----------------------------------------------------------------------===//

namespace {
class SubfieldNodeIterator : public NodeIterator {
public:
  explicit SubfieldNodeIterator(SubfieldOp subfield, Node node,
                                bool end = false)
      : NodeIterator(node, true) {
    assert(subfield == node.value.getDefiningOp<SubfieldOp>() &&
           "subfield must be the defining op of the node value");
    if (end)
      return;

    auto memory = subfield.input().getDefiningOp<MemOp>();
    if (!memory) {
      subfield->emitOpError("input must be a port of a MemOp, please run "
                            "-firrtl-lower-types first");
      return;
    }

    auto portKind =
        memory.getPortKind(subfield.input().cast<OpResult>().getResultNumber());
    // Combinational path exists only when the current subfield is `addr` and
    // the read latency is zero.
    if (subfield.fieldIndex() != 0 || memory.readLatency() != 0 ||
        portKind == MemOp::PortKind::Write)
      return;

    // Only `data` or `rdata` subfield is combinationally connected to `addr`
    // subfield. Find the corresponding subfield op.
    auto users = subfield.input().getUsers();
    for (auto it = users.begin(), e = users.end(); it != e; ++it) {
      auto currentSubfield = dyn_cast<SubfieldOp>(*it);
      if (!currentSubfield) {
        it->emitOpError("MemOp must be used by SubfieldOp, please run "
                        "-firrtl-lower-types first");
        return;
      }

      if ((currentSubfield.fieldIndex() == 3 &&
           portKind == MemOp::PortKind::Read) ||
          (currentSubfield.fieldIndex() == 4 &&
           portKind == MemOp::PortKind::ReadWrite)) {
        child = ChildIterator(currentSubfield.result());
        return;
      }
    }
  }

  SubfieldNodeIterator &operator++() {
    if (!child.isAtEnd())
      ++child;
    if (child.isAtEnd())
      child = ChildIterator();
    return *this;
  }
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
  explicit DummySourceNodeIterator(Node node, bool end = false)
      : node(node), connect(end ? node.context->connects.end()
                                : node.context->connects.begin()) {}

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
    return Node((*connect).dest(), node.context);
  }

  bool operator==(const DummySourceNodeIterator &rhs) const {
    return connect == rhs.connect;
  }
  bool operator!=(const DummySourceNodeIterator &rhs) const {
    return !(*this == rhs);
  }

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
    if (!node.value)
      return DummySourceNodeIterator(node, end);

    auto defOp = node.value.getDefiningOp();
    if (!defOp)
      return NodeIterator(node, end);

    return TypeSwitch<Operation *, variant_iterator>(defOp)
        .Case<InstanceOp>([&](InstanceOp instance) {
          return InstanceNodeIterator(instance, node, end);
        })
        .Case<SubfieldOp>([&](SubfieldOp subfield) {
          return SubfieldNodeIterator(subfield, node, end);
        })
        // The children of reg or regreset op are not iterated.
        .Case<RegOp, RegResetOp>([&](auto) { return NodeIterator(node, true); })
        .Default([&](auto) { return NodeIterator(node, end); });
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
/// This pass constructs a local graph for each module to detect combinational
/// cycles. To capture the cross-module combinational cycles, this pass inlines
/// the combinational paths between IOs of its subinstances into a subgraph and
/// encodes them in a `combPathsMap`.
class CheckCombCyclesPass : public CheckCombCyclesBase<CheckCombCyclesPass> {
  void runOnOperation() override {
    auto &instanceGraph = getAnalysis<InstanceGraph>();

    // Traverse modules in a post order to make sure the combinational paths
    // between IOs of a module have been detected and recorded in `combPathsMap`
    // before we handle its parent modules.
    for (auto node : llvm::post_order<InstanceGraph *>(&instanceGraph)) {
      if (auto module = dyn_cast<FModuleOp>(node->getModule())) {
        NodeContext context(&map, &instanceGraph, module.getOps<ConnectOp>());
        auto dummyNode = Node(nullptr, &context);

        // Traversing SCCs in the combinational graph to detect cycles. As
        // FIRRTL module is an SSA region, all cycles must contain at least one
        // connect op. Thus we introduce a dummy source node to iterate on the
        // `dest`s of all connect ops in the module.
        using SCCIterator = llvm::scc_iterator<Node>;
        for (auto combSCC = SCCIterator::begin(dummyNode); !combSCC.isAtEnd();
             ++combSCC) {
          if (combSCC.hasCycle()) {
            auto errorDiag = mlir::emitError(
                module.getLoc(),
                "detected combinational cycle in a FIRRTL module");
            for (auto node : *combSCC) {
              auto &noteDiag = errorDiag.attachNote(node.value.getLoc());
              noteDiag << "this operation is part of the combinational cycle";
            }
          }
        }

        SmallVector<bool, 8> directionVec;
        for (auto &port : module.getPorts())
          directionVec.push_back(port.isOutput());

        auto &combPaths = map[module];
        NodeDenseSet nodeSet;
        SmallVector<size_t, 2> outputVec;
        unsigned index = 0;

        // Record all combinational paths.
        for (auto &port : module.getPorts()) {
          nodeSet.clear();
          outputVec.clear();
          auto arg = module.getPortArgument(index++);
          if (port.isOutput()) {
            combPaths.push_back(outputVec);
            continue;
          }
          Node inputNode(arg, &context);
          for (auto node : llvm::depth_first_ext<Node>(inputNode, nodeSet)) {
            if (auto output = node.value.dyn_cast<BlockArgument>())
              if (directionVec[output.getArgNumber()])
                outputVec.push_back(output.getArgNumber());
          }
          combPaths.push_back(outputVec);
        }
      } else if (auto extModule = dyn_cast<FExtModuleOp>(node->getModule())) {
        // TODO: Handle FExtModuleOp with `ExtModulePathAnnotation`s.
        auto &combPaths = map[extModule];
        combPaths.resize(extModule.getNumArguments());
      }
    }
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
