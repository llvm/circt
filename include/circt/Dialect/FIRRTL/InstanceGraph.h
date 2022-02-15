//===- InstanceGraph.h - Instance graph -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the FIRRTL InstanceGraph, which is similar to a CallGraph.
//
//===----------------------------------------------------------------------===//
#ifndef CIRCT_DIALECT_FIRRTL_INSTANCEGRAPH_H
#define CIRCT_DIALECT_FIRRTL_INSTANCEGRAPH_H

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Support/LLVM.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator.h"

namespace circt {
namespace firrtl {

namespace detail {
/// This just maps a iterator of references to an iterator of addresses.
template <typename It>
struct AddressIterator
    : public llvm::mapped_iterator<It, typename It::pointer (*)(
                                           typename It::reference)> {
  // This using statement is to get around a bug in MSVC.  Without it, it
  // tries to look up "It" as a member type of the parent class.
  using Iterator = It;
  /* implicit */ AddressIterator(Iterator iterator)
      : llvm::mapped_iterator<It, typename Iterator::pointer (*)(
                                      typename Iterator::reference)>(
            iterator, &std::addressof<typename Iterator::value_type>) {}
};
} // namespace detail

class InstanceGraphNode;

/// This is an edge in the InstanceGraph. This tracks a specific instantiation
/// of a module.
class InstanceRecord
    : public llvm::ilist_node_with_parent<InstanceRecord, InstanceGraphNode> {
public:
  /// Get the InstanceOp that this is tracking.
  InstanceOp getInstance() const { return instance; }

  /// Get the module where the InstanceOp lives.
  InstanceGraphNode *getParent() const { return parent; }

  /// Get the module which the InstanceOp is instantiating.
  InstanceGraphNode *getTarget() const { return target; }

  /// Erase this instance record, removing it from the parent module and the
  /// target's use-list.
  void erase();

private:
  friend class InstanceGraph;
  friend class InstanceGraphNode;

  InstanceRecord(InstanceGraphNode *parent, InstanceOp instance,
                 InstanceGraphNode *target)
      : parent(parent), instance(instance), target(target) {}
  InstanceRecord(const InstanceRecord &) = delete;

  /// This is the module where the InstanceOp lives.
  InstanceGraphNode *parent;

  /// The InstanceOp that this is tracking.
  InstanceOp instance;

  /// This is the module which the InstanceOp is instantiating.
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
  /// Get the module that this node is tracking.
  Operation *getModule() const { return module; }

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
  InstanceRecord *addInstance(InstanceOp instance, InstanceGraphNode *target);

private:
  friend class InstanceRecord;

  InstanceGraphNode() : module(nullptr) {}
  InstanceGraphNode(const InstanceGraphNode &) = delete;

  /// Record that a module instantiates this module.
  void recordUse(InstanceRecord *record);

  /// The module.
  Operation *module;

  /// List of instance operations in this module.  This member owns the
  /// InstanceRecords, which may be pointed to by other InstanceGraohNode's use
  /// lists.
  InstanceList instances;

  /// List of instances which instantiate this module.
  InstanceRecord *firstUse = nullptr;

  // Provide access to the constructor.
  friend class InstanceGraph;
};

/// This graph tracks modules and where they are instantiated. This is intended
/// to be used as a cached analysis on FIRRTL circuits.  This class can be used
/// to walk the modules efficiently in a bottom-up or top-down order.
///
/// To use this class, retrieve a cached copy from the analysis manager:
///   auto &instanceGraph = getAnalysis<InstanceGraph>(getOperation());
class InstanceGraph {
  /// This is the list of InstanceGraphNodes in the graph.
  using NodeList = llvm::iplist<InstanceGraphNode>;

public:
  /// Create a new module graph of a circuit.  This must be called on a FIRRTL
  /// CircuitOp or MLIR ModuleOp.
  explicit InstanceGraph(Operation *operation);

  /// Get the node corresponding to the top-level module of a circuit.
  InstanceGraphNode *getTopLevelNode();

  /// Get the module corresponding to the top-lebel module of a circuit.
  FModuleLike getTopLevelModule();

  /// Look up an InstanceGraphNode for a module. Operation must be an FModuleOp
  /// or an FExtModuleOp.
  InstanceGraphNode *lookup(Operation *op);

  /// Lookup an module by name.
  InstanceGraphNode *lookup(StringAttr name);

  /// Lookup an InstanceGraphNode for a module. Operation must be an FModuleOp
  /// or an FExtModuleOp.
  InstanceGraphNode *operator[](Operation *op) { return lookup(op); }

  /// Look up the referenced module from an InstanceOp. This will use a
  /// hashtable lookup to find the module, where
  /// InstanceOp.getReferencedModule() will be a linear search through the IR.
  Operation *getReferencedModule(InstanceOp op);

  /// Check if child is instantiated by a parent.
  bool isAncestor(FModuleLike child, FModuleOp parent);

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
  // on a CircuitOp.

  /// Add a newly created module to the instance graph.
  InstanceGraphNode *addModule(Operation *op);

  /// Remove this module from the instance graph. This will also remove all
  /// InstanceRecords in this module.  All instances of this module must have
  /// been removed from the graph.
  void erase(InstanceGraphNode *node);

  /// Replaces an instance of a module with another instance. The target module
  /// of both InstanceOps must be the same.
  void replaceInstance(InstanceOp inst, InstanceOp newInst);

  /// Returns pointer to member of operation list.
  static NodeList InstanceGraph::*
  getSublistAccessgetSublistAccess(Operation *) {
    return &InstanceGraph::nodes;
  }

private:
  InstanceGraph(const InstanceGraph &) = delete;

  InstanceGraphNode *topLevelNode;

  /// Get the node corresponding to the module.  If the node has does not exist
  /// yet, it will be created.
  InstanceGraphNode *getOrAddNode(StringAttr name);

  /// The storage for graph nodes, with deterministic iteration.
  NodeList nodes;

  /// This maps each operation to its graph node.
  llvm::DenseMap<Attribute, InstanceGraphNode *> nodeMap;
};

/// An absolute instance path.
using InstancePath = ArrayRef<InstanceOp>;

template <typename T>
inline static T &formatInstancePath(T &into, const InstancePath &path) {
  into << "$root";
  for (auto inst : path)
    into << "/" << inst.name() << ":" << inst.moduleName();
  return into;
}

template <typename T>
static T &operator<<(T &os, const InstancePath &path) {
  return formatInstancePath(os, path);
}

/// A data structure that caches and provides absolute paths to module instances
/// in the IR.
struct InstancePathCache {
  /// The instance graph of the IR.
  InstanceGraph &instanceGraph;

  explicit InstancePathCache(InstanceGraph &instanceGraph)
      : instanceGraph(instanceGraph) {}
  ArrayRef<InstancePath> getAbsolutePaths(Operation *op);

private:
  /// An allocator for individual instance paths and entire path lists.
  llvm::BumpPtrAllocator allocator;

  /// Cached absolute instance paths.
  DenseMap<Operation *, ArrayRef<InstancePath>> absolutePathsCache;

  /// Append an instance to a path.
  InstancePath appendInstance(InstancePath path, InstanceOp inst);
};

} // namespace firrtl
} // namespace circt

namespace llvm {
template <>
struct GraphTraits<circt::firrtl::InstanceGraphNode *> {
  using NodeType = circt::firrtl::InstanceGraphNode;
  using NodeRef = NodeType *;

  // Helper for getting the module referenced by the instance op.
  static NodeRef getChild(const circt::firrtl::InstanceRecord *record) {
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

template <>
struct GraphTraits<circt::firrtl::InstanceGraph *>
    : public GraphTraits<circt::firrtl::InstanceGraphNode *> {
  using nodes_iterator = circt::firrtl::InstanceGraph::iterator;

  static NodeRef getEntryNode(circt::firrtl::InstanceGraph *instanceGraph) {
    return instanceGraph->getTopLevelNode();
  }
  static nodes_iterator
  nodes_begin(circt::firrtl::InstanceGraph *instanceGraph) {
    return instanceGraph->begin();
  }
  static nodes_iterator nodes_end(circt::firrtl::InstanceGraph *instanceGraph) {
    return instanceGraph->end();
  }
};

// Provide graph traits for iterating the modules in inverse order.
template <>
struct GraphTraits<Inverse<circt::firrtl::InstanceGraphNode *>> {
  using NodeType = circt::firrtl::InstanceGraphNode;
  using NodeRef = NodeType *;

  // Helper for getting the module containing the instance op.
  static NodeRef getParent(const circt::firrtl::InstanceRecord *record) {
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

} // end namespace llvm
#endif // CIRCT_DIALECT_FIRRTL_INSTANCEGRAPH_H
