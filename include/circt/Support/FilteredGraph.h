//===- FilteredGraph.h - Filtered Graph -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Edge-filtering graph that wraps an underlying Graph that defines GraphTraits
// exposing the edges.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_FILTEREDGRAPH_H
#define CIRCT_SUPPORT_FILTEREDGRAPH_H

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"

namespace circt {
// Graph that filters out edges according to a predicate.
template <typename G>
struct FilteredGraph {
  using edge_type = typename llvm::GraphTraits<G>::EdgeRef;
  std::function<bool(edge_type)> filter;
  const G graph;
};

template <typename G, typename Callable>
// NOLINTNEXTLINE(readability-identifier-naming)
FilteredGraph<G> make_filtered_graph(const G &graph, Callable &&fn) {
  return FilteredGraph<G>{std::forward<Callable>(fn), graph};
}

} // namespace circt

// GraphTraits using edge-filtering and an underlying GraphTraits
// specialization which must support edge-related methods.
// NodeRef and EdgeRef are wrappers around underlying graph's versions,
// with references to the filter function.
// This is slightly less indirect than pointing to the FilteredGraph itself.
template <typename G>
struct llvm::GraphTraits<circt::FilteredGraph<G>> {
  using GT = GraphTraits<G>;
  using FGraphTy = circt::FilteredGraph<G>;
  using FilterFnTy = llvm::function_ref<bool(typename FGraphTy::edge_type)>;

  struct NodeRef {
    FilterFnTy filter;
    typename GT::NodeRef node;

    // Compare underlying node, filter can't be compared and should be same.
    bool operator==(const NodeRef &rhs) const { return node == rhs.node; }
    bool operator!=(const NodeRef &rhs) const { return !operator==(rhs); }
    bool operator<(const NodeRef &rhs) const { return node < rhs.node; }

    // Dereference for underlying node.
    auto operator*() const { return node; }
  };

  struct EdgeRef {
    FilterFnTy filter;
    typename GT::EdgeRef edge;
  };

  // NOLINTNEXTLINE(readability-identifier-naming)
  static NodeRef edge_dest(EdgeRef edge) {
    return NodeRef{edge.filter, GT::edge_dest(edge.edge)};
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  static auto child_edge_range(NodeRef node) {
    return llvm::map_range(llvm::make_filter_range(
                               llvm::make_range(GT::child_edge_begin(node.node),
                                                GT::child_edge_end(node.node)),
                               node.filter),
                           [fn = node.filter](auto edge) {
                             return EdgeRef{fn, edge};
                           });
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  static auto child_edge_begin(NodeRef node) {
    return child_edge_range(node).begin();
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  static auto child_edge_end(NodeRef node) {
    return child_edge_range(node).end();
  }

  using ChildEdgeIteratorType =
      std::invoke_result_t<decltype(child_edge_begin), NodeRef>;

  using ChildIteratorType =
      llvm::mapped_iterator<ChildEdgeIteratorType, decltype(&edge_dest)>;

  static NodeRef getEntryNode(const FGraphTy &fGraph) {
    return NodeRef{fGraph.filter, GT::getEntryNode(fGraph.graph)};
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  static ChildIteratorType child_begin(NodeRef node) {
    return {child_edge_begin(node), &edge_dest};
  }
  // NOLINTNEXTLINE(readability-identifier-naming)
  static ChildIteratorType child_end(NodeRef node) {
    return {child_edge_end(node), &edge_dest};
  }
};

template <typename G>
struct llvm::GraphTraits<llvm::Inverse<circt::FilteredGraph<G>>>
    : llvm::GraphTraits<circt::FilteredGraph<llvm::Inverse<G>>> {};

// DF iterator support.
namespace circt {
namespace detail {
template <typename NodeRef, typename SetTy = llvm::SmallSet<NodeRef, 8>>
struct SmallSetForDF {
  SetTy set;

  auto insert(const NodeRef &item) { return set.insert(item); }

  // Callback df_iterator requires to be implemented.
  void completed(NodeRef) {}
};
} // namespace detail

template <typename G>
// NOLINTNEXTLINE(readability-identifier-naming)
auto make_df_range(const FilteredGraph<G> &graph) {
  using GT = llvm::GraphTraits<FilteredGraph<G>>;
  using DFI = llvm::df_iterator<FilteredGraph<G>,
                                detail::SmallSetForDF<typename GT::NodeRef>>;
  return llvm::make_range(DFI::begin(graph), DFI::end(graph));
}

} // namespace circt

#endif // CIRCT_SUPPORT_FILTEREDGRAPH_H
