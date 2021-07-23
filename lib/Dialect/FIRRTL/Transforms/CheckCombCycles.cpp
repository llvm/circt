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

using namespace circt;
using namespace firrtl;

namespace {
class CombNodeImpl;
}

using CombNode = CombNodeImpl *;
using ValueOrOpType = PointerUnion<Value, Operation *>;
using CombNodeKeyType = std::pair<ValueOrOpType, unsigned>;

//===----------------------------------------------------------------------===//
// CombNodeImpl class
//===----------------------------------------------------------------------===//

namespace {
/// This is the implementation of a combinational node.
class CombNodeImpl {
public:
  CombNodeImpl(ValueOrOpType valOrOp, unsigned fieldID = 0)
      : valOrOp(valOrOp), fieldID(fieldID) {}

  bool operator==(const CombNodeImpl &other) const {
    return valOrOp == other.valOrOp && fieldID == other.fieldID;
  }

  CombNode *child_begin() { return children.begin(); }
  CombNode *child_end() { return children.end(); }

  void addChild(CombNode child) { children.emplace_back(child); }
  void clearChilds() { children.clear(); }

  ValueOrOpType getValOrOp() const { return valOrOp; }
  unsigned getFieldID() const { return fieldID; }

  llvm::hash_code hash_value() const {
    return llvm::hash_combine(valOrOp.getOpaqueValue(), fieldID);
  }

private:
  /// The underneath value or operation of this node.
  ValueOrOpType valOrOp;

  /// The field ID this node is referencing to.
  unsigned fieldID;

  /// Store all children of this node in the graph.
  SmallVector<CombNode, 4> children;
};
} // namespace

//===----------------------------------------------------------------------===//
// CombGraph class
//===----------------------------------------------------------------------===//

namespace {
// This is a combinational graph contains a set of nodes.
class CombGraph {
public:
  ~CombGraph() {
    for (auto node : nodes)
      node->~CombNodeImpl();
  }

  /// Get a node from the graph if it exists. Otherwise, allocate a new node and
  /// add to the graph.
  CombNode getOrAddNode(ValueOrOpType valOrOp, unsigned fieldID = 0) {
    auto it = nodes.find_as<CombNodeKeyType>({valOrOp, fieldID});
    if (it != nodes.end())
      return *it;
    auto newNode = new (allocator) CombNodeImpl(valOrOp, fieldID);
    nodes.insert(newNode);
    return newNode;
  }

  /// Get a node from the graph if it exists.
  CombNode getNode(ValueOrOpType valOrOp, unsigned fieldID = 0) {
    auto it = nodes.find_as<CombNodeKeyType>({valOrOp, fieldID});
    return it != nodes.end() ? *it : nullptr;
  }

private:
  /// For allocating memories for combinational nodes.
  llvm::BumpPtrAllocator allocator;

  /// A set of combinational nodes.
  DenseSet<CombNode> nodes;
};
} // namespace

//===----------------------------------------------------------------------===//
// GraphTraits on CombNode
//===----------------------------------------------------------------------===//

namespace llvm {
template <>
struct GraphTraits<CombNode> {
  using NodeRef = CombNode;
  using ChildIteratorType = NodeRef *;

  static NodeRef getEntryNode(NodeRef node) { return node; }

  static inline ChildIteratorType child_begin(NodeRef node) {
    return node->child_begin();
  }
  static inline ChildIteratorType child_end(NodeRef node) {
    return node->child_end();
  }
};
} // namespace llvm

//===----------------------------------------------------------------------===//
// CycleDetector class
//===----------------------------------------------------------------------===//

/// This is used for storing combinational paths between IOs of a FIRRTL module.
/// The two elements of the pair are the argument index of the input port and
/// output port, respectively.
using CombPaths = SmallVector<std::pair<unsigned, unsigned>>;

namespace {
class CycleDetector : public FIRRTLVisitor<CycleDetector, bool> {
public:
  CycleDetector(FModuleOp module,
                DenseMap<Operation *, CombPaths> &combPathsMap,
                InstanceGraph &instanceGraph)
      : module(module), combPathsMap(combPathsMap),
        instanceGraph(instanceGraph) {}

  /// Each operation visitor is supposed to finish the following tasks:
  ///   1. Create a node for each of its results if the result is not a
  ///   sequential value, such as register and memory write port;
  ///   2. Create edges between its arguments and results if applicable.
  using FIRRTLVisitor<CycleDetector, bool>::visitStmt;
  using FIRRTLVisitor<CycleDetector, bool>::visitDecl;
  using FIRRTLVisitor<CycleDetector, bool>::visitExpr;

  /// Statement handlers.
  template <typename ConnectOpType>
  bool handleConnectOp(ConnectOpType op);
  bool visitStmt(ConnectOp op) { return handleConnectOp(op); }
  bool visitStmt(PartialConnectOp op) { return handleConnectOp(op); }
  bool visitStmt(WhenOp op) {
    op->emitOpError("unsupported, please run -firrtl-expand-whens first");
    return false;
  }

  /// Declaration handlers.
  bool visitDecl(InstanceOp op);
  bool visitDecl(MemOp op);
  bool visitDecl(RegOp op) { return true; }
  bool visitDecl(RegResetOp op) { return true; }

  /// Expression handlers.
  bool visitExpr(SubfieldOp op);
  bool visitExpr(SubindexOp op) {
    op->emitOpError("unsupported, please run -firrtl-lower-types first");
    return false;
  };
  bool visitExpr(SubaccessOp op) {
    op->emitOpError("unsupported, please run -firrtl-lower-types first");
    return false;
  };

  /// Handle other operations that don't have concrete visitors.
  bool visitUnhandledOp(Operation *op);

  /// Build combinational graph of the module with the visitor dispatcher.
  void buildCombGraph();

  /// Detect combinational cycles in the module. Also detect combinational paths
  /// between IOs of the module and record these paths in the `combPathsMap`.
  void detect();

private:
  /// The module detected on.
  FModuleOp module;

  /// The global combinational paths map.
  DenseMap<Operation *, CombPaths> &combPathsMap;

  /// The global instance graph.
  InstanceGraph &instanceGraph;

  /// The local combinational graph.
  CombGraph combGraph;
};
} // namespace

template <typename ConnectOpType>
bool CycleDetector::handleConnectOp(ConnectOpType op) {
  // Create an edge between the `source` value and `destination` value if both
  // exist in the graph.
  auto srcNode = combGraph.getNode(op.src());
  auto dstNode = combGraph.getNode(op.dest());
  if (srcNode && dstNode) {
    auto connectNode = combGraph.getOrAddNode(op.getOperation());
    srcNode->addChild(connectNode);
    connectNode->addChild(dstNode);
  }
  return true;
}

/// Declaration handlers.
bool CycleDetector::visitDecl(InstanceOp op) {
  SmallVector<CombNode, 8> nodes;
  for (auto result : op.getResults())
    nodes.push_back(combGraph.getOrAddNode(result));

  // Query the combinational paths between IOs and create edges.
  auto const &combPaths =
      combPathsMap.lookup(instanceGraph.getReferencedModule(op));
  for (auto const &combPath : combPaths) {
    auto srcNode = nodes[combPath.first];
    auto dstNode = nodes[combPath.second];
    srcNode->addChild(dstNode);
  }
  return true;
}

bool CycleDetector::visitDecl(MemOp op) {
  // Combinational paths only exist when reading latency is zero.
  if (op.readLatency() != 0)
    return true;

  // Traverse every ports of the memory.
  for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
    // Combinational paths only exist in `read` or `readwrite` ports.
    auto portKind = op.getPortKind(i);
    if (portKind == MemOp::PortKind::Write)
      continue;

    auto portType = op.getPortType(i).cast<BundleType>();

    auto enIndex = portType.getElementIndex("en");
    auto addrIndex = portType.getElementIndex("addr");
    auto dataIndex = portKind == MemOp::PortKind::ReadWrite
                         ? portType.getElementIndex("rdata")
                         : portType.getElementIndex("data");

    // Travers all subfields of the memory port. Combinational paths can only
    // start from `en` or `addr`, and propagate to `rdata` (for `readwrite`
    // port) or `data` (for `read` port).
    SmallVector<CombNode, 2> srcNodes;
    CombNode dstNode = nullptr;
    for (auto &use : op.getResult(i).getUses()) {
      auto subField = dyn_cast<SubfieldOp>(use.getOwner());
      if (!subField) {
        use.getOwner()->emitOpError(
            "should only be used by SubFieldOp, please run "
            "-firrtl-lower-types first");
        return false;
      }

      // Record source nodes and destination node of combinational paths.
      auto fieldIndex = subField.fieldIndex();
      if (fieldIndex == enIndex || fieldIndex == addrIndex)
        srcNodes.push_back(combGraph.getOrAddNode(use.get(), fieldIndex + 1));
      else if (fieldIndex == dataIndex)
        dstNode = combGraph.getOrAddNode(use.get(), fieldIndex + 1);
    }

    // Now, we can create edges.
    if (dstNode) {
      for (auto srcNode : srcNodes)
        srcNode->addChild(dstNode);
    }
  }
  return true;
}

/// Expression handlers.
bool CycleDetector::visitExpr(SubfieldOp op) {
  auto definingOp = op.input().getDefiningOp();
  if (!definingOp || !isa<MemOp>(definingOp)) {
    op->emitOpError("input must be a port of a MemOp, please run "
                    "-firrtl-lower-types first");
    return false;
  }

  auto portType = op.input().getType().cast<BundleType>();
  auto fieldIndex = op.fieldIndex();
  // If the memory port is not on a combinational path, it was not be inserted
  // into the graph as a node in the MemOp handler.
  if (auto memPortNode = combGraph.getNode(op.input(), fieldIndex + 1)) {
    auto node = combGraph.getOrAddNode(op.result());

    // We use `isFlip` to determine the direction of the edge.
    if (portType.getElement(fieldIndex).isFlip)
      memPortNode->addChild(node);
    else
      node->addChild(memPortNode);
  }
  return true;
}

/// Handle other operations that don't have concrete visitors.
bool CycleDetector::visitUnhandledOp(Operation *op) {
  // Create an edge between each exist input node and each output node.
  SmallVector<CombNode, 4> dstNodes;
  for (auto result : op->getResults())
    dstNodes.push_back(combGraph.getOrAddNode(result));

  for (auto operand : op->getOperands()) {
    if (auto srcNode = combGraph.getNode(operand)) {
      for (auto dstNode : dstNodes)
        srcNode->addChild(dstNode);
    }
  }
  return true;
}

/// Build combinational graph of the module with the visitor dispatcher.
void CycleDetector::buildCombGraph() {
  // Each module port should be a node in the graph.
  for (auto arg : module.getArguments())
    combGraph.getOrAddNode(arg);

  // Traverse all operations in the module and create combinational nodes and
  // edges.
  module.walk([&](Operation *op) {
    if (!dispatchVisitor(op))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
}

/// Detect combinational cycles in the module. Also detect combinational paths
/// between IOs of the module and record these paths in the `combPathsMap`.
void CycleDetector::detect() {
  buildCombGraph();

  // Add a dummy node in the begining to launch a Tarjan's SCC algorithm.
  auto dummyNode = combGraph.getOrAddNode(nullptr);
  for (auto connect : module.getOps<ConnectOp>()) {
    if (auto node = combGraph.getNode(connect.dest()))
      dummyNode->addChild(node);
  }

  // Lauch SCC iteration and report diagnostics.
  using SCCIterator = llvm::scc_iterator<CombNode>;
  for (auto SCC = SCCIterator::begin(dummyNode); !SCC.isAtEnd(); ++SCC) {
    if (SCC.hasCycle()) {
      auto errorDiag = mlir::emitError(
          module.getLoc(), "detected combinational cycle in a FIRRTL module");
      for (auto it = SCC->rbegin(), e = SCC->rend(); it != e; ++it) {
        auto node = *it;
        auto nodeVal = node->getValOrOp().dyn_cast<Value>();
        auto nodeOp = node->getValOrOp().dyn_cast<Operation *>();

        auto &noteDiag =
            errorDiag.attachNote(nodeVal ? nodeVal.getLoc() : nodeOp->getLoc());
        noteDiag << "this operation is part of the combinational cycle";

        // Indicate field ID if it is not zero.
        if (auto fieldID = node->getFieldID())
          noteDiag << ", field ID is " << fieldID;

        // Indicate result number if the definining op has multiple results.
        if (nodeVal) {
          auto definingOp = nodeVal.getDefiningOp();
          if (definingOp && definingOp->getNumResults() > 1) {
            auto resultNumber = nodeVal.dyn_cast<OpResult>().getResultNumber();
            noteDiag << ", result number is " << resultNumber;
          }
        }
      }
    }
  }
  dummyNode->clearChilds();

  // Collect all inputs of the module and a port direction list.
  SmallVector<BlockArgument, 4> inputs;
  SmallVector<bool, 8> directionList;
  unsigned index = 0;
  for (auto &port : module.getPorts()) {
    directionList.push_back(port.isOutput());
    if (!port.isOutput())
      inputs.push_back(module.getPortArgument(index));
    ++index;
  }

  // Launch a DFS for each input port to find the output ports that are
  // combinationally connected to it.
  auto &combPaths = combPathsMap[module];
  for (auto input : inputs) {
    for (auto node : llvm::depth_first<CombNode>(combGraph.getNode(input))) {
      if (auto val = node->getValOrOp().dyn_cast<Value>())
        if (auto output = val.dyn_cast<BlockArgument>())
          if (directionList[output.getArgNumber()])
            combPaths.push_back({input.getArgNumber(), output.getArgNumber()});
    }
  }
}

//===----------------------------------------------------------------------===//
// Hashing CombNode
//===----------------------------------------------------------------------===//

namespace llvm {
template <>
struct DenseMapInfo<CombNode> {
  static CombNode getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return static_cast<CombNode>(pointer);
  }
  static CombNode getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return static_cast<CombNode>(pointer);
  }

  static unsigned getHashValue(const CombNodeImpl *node) {
    return node->hash_value();
  }
  static bool isEqual(const CombNodeImpl *lhs, const CombNodeImpl *rhs) {
    auto empty = getEmptyKey();
    auto tombstone = getTombstoneKey();
    if (lhs == empty || lhs == tombstone || rhs == empty || rhs == tombstone)
      return lhs == rhs;
    return *lhs == *rhs;
  }

  // Allow lookup via find_as(), without constructing a temporary CombNodeImpl.
  static unsigned getHashValue(const CombNodeKeyType key) {
    return llvm::hash_combine(key.first.getOpaqueValue(), key.second);
  }
  static bool isEqual(const CombNodeKeyType lhs, const CombNodeImpl *rhs) {
    auto empty = getEmptyKey();
    auto tombstone = getTombstoneKey();
    if (rhs == empty || rhs == tombstone)
      return false;
    return lhs.first == rhs->getValOrOp() && lhs.second == rhs->getFieldID();
  }
};
} // namespace llvm

//===----------------------------------------------------------------------===//
// Fast implementation
//===----------------------------------------------------------------------===//

using CombPathsType = SmallVector<SmallVector<unsigned, 2>>;
using CombPathsMapType = DenseMap<Operation *, CombPathsType>;

namespace {
struct Node {
  Value impl;
  CombPathsMapType *map;
  InstanceGraph *graph;

  explicit Node(Value impl = nullptr, CombPathsMapType *map = nullptr,
                InstanceGraph *graph = nullptr)
      : impl(impl), map(map), graph(graph) {}

  bool operator==(const Node &rhs) const { return impl == rhs.impl; }
  bool operator!=(const Node &rhs) const { return !(*this == rhs); }
};
} // namespace

namespace {
class ChildIterator
    : public llvm::iterator_facade_base<ChildIterator,
                                        std::forward_iterator_tag, Node> {
public:
  explicit ChildIterator(Node node, bool end = false);

  using llvm::iterator_facade_base<ChildIterator, std::forward_iterator_tag,
                                   Node>::operator++;
  ChildIterator &operator++();
  Node operator*() { return Node(*child, node.map, node.graph); }

  bool operator==(const ChildIterator &rhs) const { return child == rhs.child; }
  bool operator!=(const ChildIterator &rhs) const { return !(*this == rhs); }

private:
  Node node;

  static bool isChild(OpOperand &operand) {
    if (auto connect = dyn_cast<ConnectOp>(operand.getOwner()))
      return operand.get() == connect.src();
    return operand.getOwner()->getNumResults() == 1;
  }
  static Value getChildValue(OpOperand &operand) {
    if (auto connect = dyn_cast<ConnectOp>(operand.getOwner()))
      return connect.dest();
    return operand.getOwner()->getResult(0);
  }
  using filter_use_iterator =
      llvm::filter_iterator<Value::use_iterator, decltype(&isChild)>;
  using child_iterator =
      llvm::mapped_iterator<filter_use_iterator, decltype(&getChildValue)>;

  child_iterator child_begin(Value target) {
    return child_iterator(
        filter_use_iterator(target.use_begin(), target.use_end(), &isChild),
        &getChildValue);
  }
  child_iterator child_end(Value target) {
    return child_iterator(
        filter_use_iterator(target.use_end(), target.use_end(), &isChild),
        &getChildValue);
  }
  child_iterator getNullChildIterator() {
    return child_iterator(filter_use_iterator(nullptr, nullptr, &isChild),
                          &getChildValue);
  }
  bool child_empty(Value target) {
    return child_begin(target) == child_end(target);
  }

  using child_range = llvm::iterator_range<child_iterator>;
  child_range getChilds(Value target) {
    return {child_begin(target), child_end(target)};
  }

  unsigned counter = 0;
  child_iterator child;

  void getInstOutputIndices(InstanceOp instance,
                            SmallVectorImpl<unsigned> &indices);
  void skipInstOutputsWithNoChild(InstanceOp instance,
                                  SmallVectorImpl<unsigned> &outputIndices);

  bool isMemReadAddr(SubfieldOp subfield);
  void skipToMemReadData(SubfieldOp subfield);
};
} // namespace

void ChildIterator::getInstOutputIndices(InstanceOp instance,
                                         SmallVectorImpl<unsigned> &indices) {
  auto &map = *node.map;
  auto &graph = *node.graph;
  auto inputIndex = node.impl.cast<OpResult>().getResultNumber();
  indices = map[graph.getReferencedModule(instance)][inputIndex];
}

void ChildIterator::skipInstOutputsWithNoChild(
    InstanceOp instance, SmallVectorImpl<unsigned> &indices) {
  while (counter < indices.size() &&
         child_empty(instance.getResult(indices[counter])))
    ++counter;

  if (counter == indices.size())
    child = getNullChildIterator();
  else
    child = child_begin(instance.getResult(indices[counter]));
}

bool ChildIterator::isMemReadAddr(SubfieldOp subfield) {
  auto inputDefOp = subfield.input().getDefiningOp();
  if (!inputDefOp || !isa<MemOp>(inputDefOp)) {
    subfield->emitOpError("input must be a port of a MemOp, please run "
                          "-firrtl-lower-types first");
    return false;
  }

  auto mem = dyn_cast<MemOp>(inputDefOp);
  auto portKind =
      mem.getPortKind(subfield.input().cast<OpResult>().getResultNumber());

  if (mem.readLatency() != 0 || portKind == MemOp::PortKind::Write)
    return false;

  auto portType = subfield.input().getType().cast<BundleType>();
  if (portType.getElementName(subfield.fieldIndex()) != "addr")
    return false;

  return true;
}

void ChildIterator::skipToMemReadData(SubfieldOp subfield) {
  for (auto user : subfield.input().getUsers()) {
    auto output = dyn_cast<SubfieldOp>(user);
    if (!output) {
      user->emitOpError("MemOp must be used by SubfieldOp, please run "
                        "-firrtl-lower-types first");
      return;
    }

    auto portType = subfield.input().getType().cast<BundleType>();
    auto outputName = portType.getElementName(output.fieldIndex());
    if (outputName == "data" || outputName == "rdata") {
      child = child_begin(output.result());
      return;
    }
    ++counter;
  }

  child = getNullChildIterator();
}

ChildIterator::ChildIterator(Node node, bool end)
    : node(node), child(getNullChildIterator()) {
  if (auto defOp = node.impl.getDefiningOp()) {
    if (auto instance = dyn_cast<InstanceOp>(defOp)) {
      if (end)
        return;
      SmallVector<unsigned> indices;
      getInstOutputIndices(instance, indices);
      skipInstOutputsWithNoChild(instance, indices);
      return;
    }
    if (auto subfield = dyn_cast<SubfieldOp>(defOp)) {
      if (end || !isMemReadAddr(subfield))
        return;
      skipToMemReadData(subfield);
      return;
    }
    if (isa<RegOp, RegResetOp>(defOp))
      return;
  }
  child = end ? child_end(node.impl) : child_begin(node.impl);
}

ChildIterator &ChildIterator::operator++() {
  if (auto defOp = node.impl.getDefiningOp()) {
    if (auto instance = dyn_cast<InstanceOp>(defOp)) {
      SmallVector<unsigned> indices;
      getInstOutputIndices(instance, indices);

      auto childEnd = child_end(instance.getResult(indices[counter]));
      if (child != childEnd)
        ++child;
      if (child == childEnd) {
        ++counter;
        skipInstOutputsWithNoChild(instance, indices);
      }
      return *this;
    }
    if (auto subfield = dyn_cast<SubfieldOp>(defOp)) {
      assert(isMemReadAddr(subfield) &&
             "must be addr subfield of a read port of MemOp");

      auto memReadData = dyn_cast<SubfieldOp>(
          *std::next(subfield.input().getUsers().begin(), counter));
      auto childEnd = child_end(memReadData.result());
      if (child != childEnd)
        ++child;
      if (child == childEnd)
        child = getNullChildIterator();
      return *this;
    }
  }
  ++child;
  return *this;
}

//===----------------------------------------------------------------------===//
// GraphTraits on Node
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
    return mlir::hash_value(node.impl);
  }
  static bool isEqual(const Node &lhs, const Node &rhs) { return lhs == rhs; }
};
} // namespace llvm

namespace llvm {
template <>
struct GraphTraits<Node> {
  using NodeRef = Node;
  using ChildIteratorType = ChildIterator;

  static NodeRef getEntryNode(NodeRef node) { return node; }

  static inline ChildIteratorType child_begin(NodeRef node) {
    return ChildIterator(node);
  }
  static inline ChildIteratorType child_end(NodeRef node) {
    return ChildIterator(node, /*end=*/true);
  }
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
      // TODO: Handle FExtModuleOp with `ExtModulePathAnnotation`s.
      if (auto module = dyn_cast<FModuleOp>(node->getModule())) {
        DenseSet<Value> visited;
        for (auto connect : module.getOps<ConnectOp>()) {
          if (visited.count(connect.dest()))
            continue;
          Node destNode(connect.dest(), &map, &instanceGraph);
          using SCCIterator = llvm::scc_iterator<Node>;
          for (auto SCC = SCCIterator::begin(destNode); !SCC.isAtEnd(); ++SCC) {
            auto hasCycle = SCC.hasCycle();
            for (auto it = SCC->begin(), e = SCC->end(); it != e; ++it) {
              visited.insert((*it).impl);
              if (hasCycle)
                mlir::emitError((*it).impl.getLoc(), "detected cycle!");
            }
          }
        }

        SmallVector<bool, 8> directionList;
        for (auto &port : module.getPorts())
          directionList.push_back(port.isOutput());

        auto &combPaths = map[module];
        NodeDenseSet nodeSet;
        SmallVector<unsigned, 2> outputVec;
        unsigned index = 0;
        for (auto &port : module.getPorts()) {
          nodeSet.clear();
          outputVec.clear();
          auto arg = module.getPortArgument(index++);
          if (port.isOutput()) {
            combPaths.push_back(outputVec);
            continue;
          }
          Node inputNode(arg, &map, &instanceGraph);
          for (auto node : llvm::depth_first_ext<Node>(inputNode, nodeSet)) {
            if (auto output = node.impl.dyn_cast<BlockArgument>())
              if (directionList[output.getArgNumber()])
                outputVec.push_back(output.getArgNumber());
          }
          combPaths.push_back(outputVec);
        }
        // CycleDetector(module, combPathsMap, instanceGraph).detect();

      } else if (auto extModule = dyn_cast<FExtModuleOp>(node->getModule())) {
        auto &combPaths = map[extModule];
        combPaths.resize(extModule.getNumArguments());
      }
    }
    markAllAnalysesPreserved();
  }

private:
  /// A global map from FIRRTL modules to their combinational paths between IOs.
  DenseMap<Operation *, CombPaths> combPathsMap;
  CombPathsMapType map;
};
} // namespace

std::unique_ptr<mlir::Pass> circt::firrtl::createCheckCombCyclesPass() {
  return std::make_unique<CheckCombCyclesPass>();
}
