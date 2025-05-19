//===- InstanceGraph.h - Instance graph -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a generic instance graph for module- and instance-likes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_INSTANCEGRAPH_H
#define CIRCT_SUPPORT_INSTANCEGRAPH_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/DOTGraphTraits.h"

/// The InstanceGraph op interface, see InstanceGraphInterface.td for more
/// details.
#include "circt/Support/InstanceGraphInterface.h"

namespace circt {
namespace igraph {

namespace detail {
/// This just maps a iterator of references to an iterator of addresses.
template <typename It>
struct AddressIterator
    : public llvm::mapped_iterator<It, typename It::pointer (*)(
                                           typename It::reference)> {
  // This using statement is to get around a bug in MSVC.  Without it, it
  // tries to look up "It" as a member type of the parent class.
  using Iterator = It;
  static typename Iterator::value_type *
  addrOf(typename Iterator::value_type &v) noexcept {
    return std::addressof(v);
  }
  /* implicit */ AddressIterator(Iterator iterator)
      : llvm::mapped_iterator<It, typename Iterator::pointer (*)(
                                      typename Iterator::reference)>(iterator,
                                                                     addrOf) {}
};
} // namespace detail

class InstanceGraphNode;

/// This is an edge in the InstanceGraph. This tracks a specific instantiation
/// of a module.
class InstanceRecord
    : public llvm::ilist_node_with_parent<InstanceRecord, InstanceGraphNode> {
public:
  /// Get the instance-like op that this is tracking.
  template <typename TTarget = InstanceOpInterface>
  auto getInstance() {
    if constexpr (std::is_same<TTarget, InstanceOpInterface>::value)
      return instance;
    return dyn_cast_or_null<TTarget>(instance.getOperation());
  }

  /// Get the module where the instantiation lives.
  InstanceGraphNode *getParent() const { return parent; }

  /// Get the module which the instance-like is instantiating.
  InstanceGraphNode *getTarget() const { return target; }

  /// Erase this instance record, removing it from the parent module and the
  /// target's use-list.
  void erase();

private:
  friend class InstanceGraph;
  friend class InstanceGraphNode;

  InstanceRecord(InstanceGraphNode *parent, InstanceOpInterface instance,
                 InstanceGraphNode *target)
      : parent(parent), instance(instance), target(target) {}
  InstanceRecord(const InstanceRecord &) = delete;

  /// This is the module where the instantiation lives.
  InstanceGraphNode *parent;

  /// The InstanceLike that this is tracking.
  InstanceOpInterface instance;

  /// This is the module which the instance-like is instantiating.
  InstanceGraphNode *target;
  /// Intrusive linked list for other uses.
  InstanceRecord *nextUse = nullptr;
  InstanceRecord *prevUse = nullptr;
};

/// This is a Node in the InstanceGraph.  Each node represents a Module in a
/// Circuit.  Both external modules and regular modules can be represented by
/// this class. It is possible to efficiently iterate all modules instantiated
/// by this module, as well as all instantiations of this module.
class InstanceGraphNode : public llvm::ilist_node<InstanceGraphNode> {
  using InstanceList = llvm::iplist<InstanceRecord>;

public:
  InstanceGraphNode() : module(nullptr) {}

  /// Get the module that this node is tracking.
  template <typename TTarget = ModuleOpInterface>
  auto getModule() {
    if constexpr (std::is_same<TTarget, ModuleOpInterface>::value)
      return module;
    return cast<TTarget>(module.getOperation());
  }

  /// Iterate the instance records in this module.
  using iterator = detail::AddressIterator<InstanceList::iterator>;
  iterator begin() { return instances.begin(); }
  iterator end() { return instances.end(); }

  /// Return true if there are no more instances of this module.
  bool noUses() { return !firstUse; }

  /// Return true if this module has exactly one use.
  bool hasOneUse() { return llvm::hasSingleElement(uses()); }

  /// Get the number of direct instantiations of this module.
  size_t getNumUses() { return std::distance(usesBegin(), usesEnd()); }

  /// Iterator for module uses.
  struct UseIterator
      : public llvm::iterator_facade_base<
            UseIterator, std::forward_iterator_tag, InstanceRecord *> {
    UseIterator() : current(nullptr) {}
    UseIterator(InstanceGraphNode *node) : current(node->firstUse) {}
    InstanceRecord *operator*() const { return current; }
    using llvm::iterator_facade_base<UseIterator, std::forward_iterator_tag,
                                     InstanceRecord *>::operator++;
    UseIterator &operator++() {
      assert(current && "incrementing past end");
      current = current->nextUse;
      return *this;
    }
    bool operator==(const UseIterator &other) const {
      return current == other.current;
    }

  private:
    InstanceRecord *current;
  };

  /// Iterate the instance records which instantiate this module.
  UseIterator usesBegin() { return {this}; }
  UseIterator usesEnd() { return {}; }
  llvm::iterator_range<UseIterator> uses() {
    return llvm::make_range(usesBegin(), usesEnd());
  }

  /// Record a new instance op in the body of this module. Returns a newly
  /// allocated InstanceRecord which will be owned by this node.
  InstanceRecord *addInstance(InstanceOpInterface instance,
                              InstanceGraphNode *target);

private:
  friend class InstanceRecord;

  InstanceGraphNode(const InstanceGraphNode &) = delete;

  /// Record that a module instantiates this module.
  void recordUse(InstanceRecord *record);

  /// The module.
  ModuleOpInterface module;

  /// List of instance operations in this module.  This member owns the
  /// InstanceRecords, which may be pointed to by other InstanceGraphNode's use
  /// lists.
  InstanceList instances;

  /// List of instances which instantiate this module.
  InstanceRecord *firstUse = nullptr;

  // Provide access to the constructor.
  friend class InstanceGraph;
};

/// This graph tracks modules and where they are instantiated. This is intended
/// to be used as a cached analysis on circuits.  This class can be used
/// to walk the modules efficiently in a bottom-up or top-down order.
///
/// To use this class, retrieve a cached copy from the analysis manager:
///   auto &instanceGraph = getAnalysis<InstanceGraph>(getOperation());
class InstanceGraph {
  /// This is the list of InstanceGraphNodes in the graph.
  using NodeList = llvm::iplist<InstanceGraphNode>;

public:
  /// Create a new module graph of a circuit.  Must be called on the parent
  /// operation of ModuleOpInterface ops.
  InstanceGraph(Operation *parent);
  InstanceGraph(const InstanceGraph &) = delete;
  virtual ~InstanceGraph() = default;

  /// Lookup an module by name. Returns null if no module with the given name
  /// exists in the instance graph.
  InstanceGraphNode *lookupOrNull(StringAttr name);

  /// Look up an InstanceGraphNode for a module. Returns null if the module has
  /// not been added to the instance graph.
  InstanceGraphNode *lookupOrNull(ModuleOpInterface op) {
    return lookup(op.getModuleNameAttr());
  }

  /// Look up an InstanceGraphNode for a module. Aborts if the module does not
  /// exist.
  InstanceGraphNode *lookup(ModuleOpInterface op) {
    auto *node = lookupOrNull(op);
    assert(node != nullptr && "Module not in InstanceGraph!");
    return node;
  }

  /// Lookup an module by name. Aborts if the module does not exist.
  InstanceGraphNode *lookup(StringAttr name) {
    auto *node = lookupOrNull(name);
    assert(node != nullptr && "Module not in InstanceGraph!");
    return node;
  }

  /// Lookup an InstanceGraphNode for a module.
  InstanceGraphNode *operator[](ModuleOpInterface op) { return lookup(op); }

  /// Check if child is instantiated by a parent.
  bool isAncestor(
      ModuleOpInterface child, ModuleOpInterface parent,
      llvm::function_ref<bool(InstanceRecord *)> skipInstance =
          [](InstanceRecord *_) { return false; });

  /// Get the node corresponding to the top-level module of a circuit.
  virtual InstanceGraphNode *getTopLevelNode() { return nullptr; }

  /// Get the nodes corresponding to the inferred top-level modules of a
  /// circuit.
  FailureOr<llvm::ArrayRef<InstanceGraphNode *>> getInferredTopLevelNodes();

  /// Return the parent under which all nodes are nested.
  Operation *getParent() { return parent; }

  /// Returns pointer to member of operation list.
  static NodeList InstanceGraph::*getSublistAccess(Operation *) {
    return &InstanceGraph::nodes;
  }

  /// Iterate through all modules.
  using iterator = detail::AddressIterator<NodeList::iterator>;
  iterator begin() { return nodes.begin(); }
  iterator end() { return nodes.end(); }

  //===-------------------------------------------------------------------------
  // Methods to keep an InstanceGraph up to date.
  //
  // These methods are not thread safe.  Make sure that modifications are
  // properly synchronized or performed in a serial context.  When the
  // InstanceGraph is used as an analysis, this is only safe when the pass is
  // on a CircuitOp or a ModuleOp.

  /// Add a newly created module to the instance graph.
  virtual InstanceGraphNode *addModule(ModuleOpInterface module);

  /// Remove this module from the instance graph. This will also remove all
  /// InstanceRecords in this module.  All instances of this module must have
  /// been removed from the graph.
  virtual void erase(InstanceGraphNode *node);

  /// Replaces an instance of a module with another instance. The target module
  /// of both InstanceOps must be the same.
  virtual void replaceInstance(InstanceOpInterface inst,
                               InstanceOpInterface newInst);

protected:
  ModuleOpInterface getReferencedModuleImpl(InstanceOpInterface op);

  /// Get the node corresponding to the module.  If the node has does not exist
  /// yet, it will be created.
  InstanceGraphNode *getOrAddNode(StringAttr name);

  /// The node under which all modules are nested.
  Operation *parent;

  /// The storage for graph nodes, with deterministic iteration.
  NodeList nodes;

  /// This maps each operation to its graph node.
  llvm::DenseMap<Attribute, InstanceGraphNode *> nodeMap;

  /// A caching of the inferred top level module(s).
  llvm::SmallVector<InstanceGraphNode *> inferredTopLevelNodes;
};

struct InstancePathCache;

/**
 * An instance path composed of a series of instances.
 */
class InstancePath final {
public:
  InstancePath() = default;

  InstanceOpInterface top() const {
    assert(!empty() && "instance path is empty");
    return path[0];
  }

  InstanceOpInterface leaf() const {
    assert(!empty() && "instance path is empty");
    return path.back();
  }

  InstancePath dropFront() const { return InstancePath(path.drop_front()); }

  InstancePath dropBack() const { return InstancePath(path.drop_back()); }

  InstanceOpInterface operator[](size_t idx) const { return path[idx]; }
  ArrayRef<InstanceOpInterface>::iterator begin() const { return path.begin(); }
  ArrayRef<InstanceOpInterface>::iterator end() const { return path.end(); }
  size_t size() const { return path.size(); }
  bool empty() const { return path.empty(); }

  bool operator==(const InstancePath &that) const { return path == that.path; }

  /// Print the path to any stream-like object.
  void print(llvm::raw_ostream &into) const;

private:
  // Only the path cache is allowed to create paths.
  friend struct InstancePathCache;
  InstancePath(ArrayRef<InstanceOpInterface> path) : path(path) {}

  ArrayRef<InstanceOpInterface> path;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const InstancePath &path) {
  path.print(os);
  return os;
}

/// A data structure that caches and provides paths to module
/// instances in the IR.
struct InstancePathCache {
  /// The instance graph of the IR.
  InstanceGraph &instanceGraph;

  explicit InstancePathCache(InstanceGraph &instanceGraph)
      : instanceGraph(instanceGraph) {}

  // Return all absolute paths from the top-level node to the given module.
  ArrayRef<InstancePath> getAbsolutePaths(ModuleOpInterface op);

  // Return all relative paths from the given node to the given module.
  ArrayRef<InstancePath> getRelativePaths(ModuleOpInterface op,
                                          InstanceGraphNode *node);

  /// Replace an InstanceOp. This is required to keep the cache updated.
  void replaceInstance(InstanceOpInterface oldOp, InstanceOpInterface newOp);

  /// Append an instance to a path.
  InstancePath appendInstance(InstancePath path, InstanceOpInterface inst);

  /// Prepend an instance to a path.
  InstancePath prependInstance(InstanceOpInterface inst, InstancePath path);

private:
  using PathsCache = DenseMap<Operation *, ArrayRef<InstancePath>>;
  ArrayRef<InstancePath> getPaths(ModuleOpInterface op, InstanceGraphNode *top,
                                  PathsCache &cache);

  /// An allocator for individual instance paths and entire path lists.
  llvm::BumpPtrAllocator allocator;

  /// Cached absolute instance paths.
  PathsCache absolutePathsCache;

  /// Cached relative instance paths.
  DenseMap<InstanceGraphNode *, PathsCache> relativePathsCache;
};

} // namespace igraph
} // namespace circt

// Graph traits for modules.
template <>
struct llvm::GraphTraits<circt::igraph::InstanceGraphNode *> {
  using NodeType = circt::igraph::InstanceGraphNode;
  using NodeRef = NodeType *;

  // Helper for getting the module referenced by the instance op.
  static NodeRef getChild(const circt::igraph::InstanceRecord *record) {
    return record->getTarget();
  }

  using ChildIteratorType =
      llvm::mapped_iterator<NodeType::iterator, decltype(&getChild)>;

  static NodeRef getEntryNode(NodeRef node) { return node; }
  static ChildIteratorType child_begin(NodeRef node) {
    return {node->begin(), &getChild};
  }
  static ChildIteratorType child_end(NodeRef node) {
    return {node->end(), &getChild};
  }
};

// Provide graph traits for iterating the modules in inverse order.
template <>
struct llvm::GraphTraits<llvm::Inverse<circt::igraph::InstanceGraphNode *>> {
  using NodeType = circt::igraph::InstanceGraphNode;
  using NodeRef = NodeType *;

  // Helper for getting the module containing the instance op.
  static NodeRef getParent(const circt::igraph::InstanceRecord *record) {
    return record->getParent();
  }

  using ChildIteratorType =
      llvm::mapped_iterator<NodeType::UseIterator, decltype(&getParent)>;

  static NodeRef getEntryNode(Inverse<NodeRef> inverse) {
    return inverse.Graph;
  }
  static ChildIteratorType child_begin(NodeRef node) {
    return {node->usesBegin(), &getParent};
  }
  static ChildIteratorType child_end(NodeRef node) {
    return {node->usesEnd(), &getParent};
  }
};

// Graph traits for the common instance graph.
template <>
struct llvm::GraphTraits<circt::igraph::InstanceGraph *>
    : public llvm::GraphTraits<circt::igraph::InstanceGraphNode *> {
  using nodes_iterator = circt::igraph::InstanceGraph::iterator;

  static NodeRef getEntryNode(circt::igraph::InstanceGraph *graph) {
    return graph->getTopLevelNode();
  }
  // NOLINTNEXTLINE(readability-identifier-naming)
  static nodes_iterator nodes_begin(circt::igraph::InstanceGraph *graph) {
    return graph->begin();
  }
  // NOLINTNEXTLINE(readability-identifier-naming)
  static nodes_iterator nodes_end(circt::igraph::InstanceGraph *graph) {
    return graph->end();
  }
};

// Graph traits for DOT labeling.
template <>
struct llvm::DOTGraphTraits<circt::igraph::InstanceGraph *>
    : public llvm::DefaultDOTGraphTraits {
  using DefaultDOTGraphTraits::DefaultDOTGraphTraits;

  static std::string getNodeLabel(circt::igraph::InstanceGraphNode *node,
                                  circt::igraph::InstanceGraph *) {
    // The name of the graph node is the module name.
    return node->getModule().getModuleName().str();
  }

  template <typename Iterator>
  static std::string
  getEdgeAttributes(const circt::igraph::InstanceGraphNode *node, Iterator it,
                    circt::igraph::InstanceGraph *) {
    // Set an edge label that is the name of the instance.
    auto *instanceRecord = *it.getCurrent();
    auto instanceOp = instanceRecord->getInstance();
    return ("label=" + instanceOp.getInstanceName()).str();
  }
};

#endif // CIRCT_SUPPORT_INSTANCEGRAPH_H
