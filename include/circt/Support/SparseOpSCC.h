//===- SparseOpSCCs.h - SCC analysis on sparse op subgraphs ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Collect strongly connected components (SCCs) in the (filtered) def-use graph
// of MLIR operations, starting from a sparse set of seed operations.
//
// Graph model
// -----------
// Each operation is a node.  A directed edge runs from op A to op B if B uses
// one of A's results.  The traversal direction is configurable:
//   - OpSCCDirection::Forward   -- follow edges from defining ops to uses.
//   - OpSCCDirection::Backward  -- follow edges from uses to defining op.
//
// SCC classification
// ------------------
// An SCC is either:
//   - Trivial: a single op with no self-loop.  Represented as
//     mlir::Operation * inside an OpSCC value.
//   - Cyclic: a group of mutually-reachable ops, or a single op with a
//     self-loop.  Represented as a CyclicOpSCC inside an OpSCC value.
//
// Filtering
// ---------
// An optional OpSCCFilter predicate can be supplied to the constructor to
// prevent the traversal over certain edges of the graph. The first argument
// contains the operation into which the traversal would lead. The second
// argument contains the edge's destination operand. For forward traversal
// the operand's owner is identical to the first argument. For reverse
// traversal the first argument is identical to the operand's defining
// operation.
//
// Output ordering
// ---------------
// SCCs are available in topological order of the condensation DAG (seeds /
// sources first, leaves last) via topological(), and in the reverse via
// reverseTopological().
//
// Blocks and Regions
// ------------------
// The traversal does not follow through block arguments. It does not consider
// control flow. It will descend into / ascend from regions without considering
// the parent operation. The filter predicate can be used to restrict the
// traversal to certain blocks or regions.
//
// Operation Graph Mutation
// ------------------------
// The SparseOpSCC class internally stores the result of the SCC analysis
// and is only updated when visit(...) is called. It is not recommended
// to mutate the IR between visit calls. Calling visit invalidates all
// iterators. It is safe to mutate the IR while iterating. However, the
// iteration sequence may contain invalid operation pointers, if the underlying
// operation is erased after visiting the graph. To reflect the changes in the
// analysis, reset() must be called and the graph must be re-visited.
//
// Usage example
// -------------
// Collect all ops reachable from seedOp, excluding register ops, and process
// them in topological order:
//
//   auto regFilter = [](Operation *op, OpOperand&) {
//     return !isa<seq::FirRegOp>(op);
//   };
//   SparseOpSCC<OpSCCDirection::Forward> sccs(regFilter);
//   sccs.visit(seedOp);
//
//   for (OpSCC entry : sccs.topological()) {
//     if (Operation *op = llvm::dyn_cast<mlir::Operation *>(entry)) {
//       // Trivial SCC: a single op with no cycle.
//       processSingle(op);
//     } else {
//       // Cyclic SCC: a group of mutually-reachable ops (or a self-loop).
//       for (Operation *op : llvm::cast<CyclicOpSCC>(entry))
//         processInCycle(op);
//     }
//   }
//
// Alternative filter that traverses registers through their clock and reset
// values but not the "next" data values:
//
//   auto regEdgeFilter = [](Operation*, OpOperand& operand) {
//     if (auto regOp = dyn_cast<seq::FirRegOp>(operand.getOwner()))
//       return operand != regOp.getNextMutable();
//     return true;
//   };
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_SPARSEOPSCC_H
#define CIRCT_SUPPORT_SPARSEOPSCC_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerEmbeddedInt.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include <type_traits>

namespace circt {

/// Filter predicate passed to the SparseOpSCC constructor.  Return `true` to
/// include an edge in the traversal, `false` to skip it. The first argument is
/// the operation the traversal would enter. The second argument is the
/// `OpOperand` being followed: for forward traversal its owner equals the
/// first argument; for backward traversal its defining op equals the first
/// argument.
using OpSCCFilter = std::function<bool(mlir::Operation *, mlir::OpOperand &)>;

/// Sentinel type representing "no SCC" in an OpSCC PointerUnion.
/// It is placed first in the union so that the all-zero (default-constructed)
/// state identifies unambiguously as NullOpSCC, not as a null Operation*.
struct NullOpSCC {
  void *getAsVoidPointer() const { return nullptr; }
  static NullOpSCC getFromVoidPointer(void *) { return NullOpSCC{}; }
  static constexpr int NumLowBitsAvailable =
      llvm::PointerLikeTypeTraits<mlir::Operation *>::NumLowBitsAvailable;
};

namespace detail {
/// Backing storage for a cyclic SCC (implementation detail).
using CyclicOpSCCStorage = llvm::SmallVector<mlir::Operation *, 4>;
} // namespace detail

/// A cyclic SCC: a pointer-sized, directly-iterable reference to a group of
/// mutually-reachable operations (or a single op with a self-loop).
///
/// Instances are obtained via llvm::cast<CyclicOpSCC> on an OpSCC entry.
/// The referenced storage is owned by the SparseOpSCC that produced the entry.
class CyclicOpSCC {
public:
  using iterator = detail::CyclicOpSCCStorage::const_iterator;

  CyclicOpSCC() : storage(nullptr) {}
  CyclicOpSCC(const detail::CyclicOpSCCStorage *storage) : storage(storage) {}

  iterator begin() const { return storage->begin(); }
  iterator end() const { return storage->end(); }
  size_t size() const { return storage->size(); }
  mlir::Operation *const *data() const { return storage->data(); }
  mlir::Operation *operator[](size_t i) const { return (*storage)[i]; }

  operator bool() const { return storage != nullptr; }

  bool operator==(CyclicOpSCC other) const { return storage == other.storage; }
  bool operator!=(CyclicOpSCC other) const { return storage != other.storage; }

  // Interface for PointerLikeTypeTraits.
  void *getAsVoidPointer() const {
    return const_cast<detail::CyclicOpSCCStorage *>(storage);
  }
  static CyclicOpSCC getFromVoidPointer(void *p) {
    return CyclicOpSCC(static_cast<const detail::CyclicOpSCCStorage *>(p));
  }
  static constexpr int NumLowBitsAvailable = llvm::PointerLikeTypeTraits<
      const detail::CyclicOpSCCStorage *>::NumLowBitsAvailable;

private:
  const detail::CyclicOpSCCStorage *storage;
};

} // namespace circt

namespace llvm {
template <>
struct PointerLikeTypeTraits<circt::NullOpSCC> {
  static void *getAsVoidPointer(circt::NullOpSCC) { return nullptr; }
  static circt::NullOpSCC getFromVoidPointer(void *) {
    return circt::NullOpSCC{};
  }
  static constexpr int NumLowBitsAvailable =
      circt::NullOpSCC::NumLowBitsAvailable;
};

template <>
struct PointerLikeTypeTraits<circt::CyclicOpSCC> {
  static void *getAsVoidPointer(circt::CyclicOpSCC scc) {
    return scc.getAsVoidPointer();
  }
  static circt::CyclicOpSCC getFromVoidPointer(void *p) {
    return circt::CyclicOpSCC::getFromVoidPointer(p);
  }
  static constexpr int NumLowBitsAvailable =
      circt::CyclicOpSCC::NumLowBitsAvailable;
};
} // namespace llvm

namespace circt {

/// One entry in the SCC output: a null sentinel, a trivial (non-cyclic)
/// operation, or a cyclic group.  Use llvm::isa / llvm::cast / llvm::dyn_cast
/// to distinguish.  The null state (isa<NullOpSCC>) is returned by getSCC()
/// when the queried operation has not been visited.
using OpSCC = llvm::PointerUnion<NullOpSCC, mlir::Operation *, CyclicOpSCC>;

/// Traversal direction for SparseOpSCC.
///   - Forward: follow def-use edges forward (defining op -> users).
///   - Backward:  follow def-use edges backward (user -> defining op).
enum class OpSCCDirection { Forward, Backward };

template <OpSCCDirection, unsigned>
class SparseOpSCC;

namespace detail {
using OpSccEmbeddedIndex = llvm::PointerEmbeddedInt<unsigned, 31>;
using OpOrIndex = llvm::PointerUnion<mlir::Operation *, OpSccEmbeddedIndex>;

// Iterator template resolving indices to CyclicOpSCC
template <typename BaseIteratorT>
class OpSCCIterator final
    : public llvm::mapped_iterator_base<OpSCCIterator<BaseIteratorT>,
                                        BaseIteratorT, OpSCC> {
public:
  using llvm::mapped_iterator_base<OpSCCIterator<BaseIteratorT>, BaseIteratorT,
                                   OpSCC>::mapped_iterator_base;

  OpSCC mapElement(OpOrIndex opOrIndex) const {
    if (llvm::isa<mlir::Operation *>(opOrIndex))
      return llvm::cast<mlir::Operation *>(opOrIndex);
    unsigned index = llvm::cast<OpSccEmbeddedIndex>(opOrIndex);
    return CyclicOpSCC(&cyclicSccs[index]);
  }

private:
  template <OpSCCDirection, unsigned>
  friend class circt::SparseOpSCC;

  OpSCCIterator(BaseIteratorT it,
                const llvm::SmallVectorImpl<CyclicOpSCCStorage> &cyclicSccs)
      : llvm::mapped_iterator_base<OpSCCIterator<BaseIteratorT>, BaseIteratorT,
                                   OpSCC>(it),
        cyclicSccs(cyclicSccs) {}

  const llvm::SmallVectorImpl<CyclicOpSCCStorage> &cyclicSccs;
};

} // namespace detail

/// Iterative Tarjan SCC analysis on a sparse subgraph of MLIR operations.
///
/// Call visit() with one or more seed operations to trigger the DFS.  Results
/// accumulate across multiple visit() calls, so the visited subgraph can be
/// expanded incrementally.
///
/// The optional filter passed to the constructor is applied to every candidate
/// operation before it is visited.  An operation that fails the filter is
/// treated as if it did not exist in the graph.
///
/// Iterators obtained from topological() and reverseTopological() hold a
/// reference into this object and are invalidated by calling visit() or
//  reset().
template <OpSCCDirection Direction, unsigned NumInlineElts = 32>
class SparseOpSCC {
public:
  explicit SparseOpSCC(OpSCCFilter shouldTraverseFn = {})
      : shouldTraverseFn(shouldTraverseFn) {}

  /// Clear all accumulated state.
  void reset() {
    sccStack.clear();
    dfsStack.clear();
    opToSccIndex.clear();
    sccs.clear();
    cyclicSccs.clear();
  }

  /// Visit `op` if it passes the shouldTraverseFn and has not been visited yet.
  void visit(mlir::Operation *op) {
    if (!opToSccIndex.contains(op))
      tarjanImpl(op);
  }

  /// Visit each operation in `ops`, skipping already-visited or filtered ones.
  void visit(llvm::ArrayRef<mlir::Operation *> ops) {
    for (auto *op : ops)
      visit(op);
  }

  /// Return true if `op` was reached by any previous visit() call.
  bool hasVisited(mlir::Operation *op) const {
    return opToSccIndex.contains(op);
  }

  /// Return the SCC that `op` belongs to, or an OpSCC holding NullOpSCC if it
  /// has not been visited.  Check with isa<NullOpSCC> or the bool conversion
  /// before dispatching with isa/cast.
  OpSCC getSCC(mlir::Operation *op) const {
    auto it = opToSccIndex.find(op);
    if (it == opToSccIndex.end())
      return OpSCC(NullOpSCC{});
    detail::OpOrIndex entry = sccs[it->second];
    if (llvm::isa<mlir::Operation *>(entry))
      return OpSCC(llvm::cast<mlir::Operation *>(entry));
    unsigned cyclicIdx = llvm::cast<detail::OpSccEmbeddedIndex>(entry);
    return OpSCC(CyclicOpSCC(&cyclicSccs[cyclicIdx]));
  }

  /// Number of operations visited so far.
  unsigned getNumVisited() const { return opToSccIndex.size(); }
  /// Total number of SCC entries emitted (trivial ops + cyclic groups).
  unsigned getNumSCCs() const { return sccs.size(); }
  /// Number of cyclic SCC groups (excludes trivial ops).
  unsigned getNumCyclicSCCs() const { return cyclicSccs.size(); }

  /// Iterate over SCCs in topological order (sources/seeds first, leaves last).
  auto topological() const {
    return llvm::iterator_range(topological_begin(), topological_end());
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  auto topological_begin() const {
    if constexpr (Direction == OpSCCDirection::Backward)
      return detail::OpSCCIterator<typename decltype(sccs)::const_iterator>(
          sccs.begin(), cyclicSccs);
    else
      return detail::OpSCCIterator<
          typename decltype(sccs)::const_reverse_iterator>(sccs.rbegin(),
                                                           cyclicSccs);
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  auto topological_end() const {
    if constexpr (Direction == OpSCCDirection::Backward)
      return detail::OpSCCIterator<typename decltype(sccs)::const_iterator>(
          sccs.end(), cyclicSccs);
    else
      return detail::OpSCCIterator<
          typename decltype(sccs)::const_reverse_iterator>(sccs.rend(),
                                                           cyclicSccs);
  }

  /// Iterate over SCCs in reverse topological order (leaves first).
  auto reverseTopological() const {
    return llvm::iterator_range(reverseTopological_begin(),
                                reverseTopological_end());
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  auto reverseTopological_begin() const {
    if constexpr (Direction == OpSCCDirection::Forward)
      return detail::OpSCCIterator<typename decltype(sccs)::const_iterator>(
          sccs.begin(), cyclicSccs);
    else
      return detail::OpSCCIterator<
          typename decltype(sccs)::const_reverse_iterator>(sccs.rbegin(),
                                                           cyclicSccs);
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  auto reverseTopological_end() const {
    if constexpr (Direction == OpSCCDirection::Forward)
      return detail::OpSCCIterator<typename decltype(sccs)::const_iterator>(
          sccs.end(), cyclicSccs);
    else
      return detail::OpSCCIterator<
          typename decltype(sccs)::const_reverse_iterator>(sccs.rend(),
                                                           cyclicSccs);
  }

private:
  // DFS stack frame for forward traversal. Skips over unused results.
  struct ForwardFrame {
    mlir::Operation *op;
    unsigned resultIdx;
    std::optional<mlir::Value::use_iterator> useIt;

    explicit ForwardFrame(mlir::Operation *op) : op(op), resultIdx(0) {
      if (op->getNumResults() > 0)
        useIt = op->getResult(0).use_begin();
    }

    mlir::Operation *nextChild(OpSCCFilter shouldTraverseFn) {
      while (resultIdx < op->getNumResults()) {
        auto useEnd = op->getResult(resultIdx).use_end();
        while (*useIt != useEnd) {
          mlir::OpOperand &use = **useIt;
          ++(*useIt);
          if (!shouldTraverseFn || shouldTraverseFn(use.getOwner(), use))
            return use.getOwner();
        }
        ++resultIdx;
        if (resultIdx < op->getNumResults())
          useIt = op->getResult(resultIdx).use_begin();
      }
      return nullptr;
    }
  };

  // DFS stack frame for backward traversal. Skips over block arguments.
  struct InverseFrame {
    mlir::Operation *op;
    unsigned operandIdx;

    explicit InverseFrame(mlir::Operation *op) : op(op), operandIdx(0) {}

    mlir::Operation *nextChild(OpSCCFilter shouldTraverseFn) {
      while (operandIdx < op->getNumOperands()) {
        mlir::OpOperand &operand = op->getOpOperand(operandIdx++);
        auto *defOp = operand.get().getDefiningOp();
        if (defOp && (!shouldTraverseFn || shouldTraverseFn(defOp, operand)))
          return defOp;
      }
      return nullptr;
    }
  };

  using FrameT = std::conditional_t<Direction == OpSCCDirection::Forward,
                                    ForwardFrame, InverseFrame>;

  bool hasSelfLoop(mlir::Operation *op) const {
    for (mlir::OpOperand &operand : op->getOpOperands())
      if (operand.get().getDefiningOp() == op &&
          (!shouldTraverseFn || shouldTraverseFn(op, operand)))
        return true;
    return false;
  }

  void tarjanImpl(mlir::Operation *startOp) {
    // Per-call Tarjan state: {index, lowlink} for each op discovered in this
    // DFS. Discarded when the call returns.
    llvm::DenseMap<mlir::Operation *, std::pair<unsigned, unsigned>> tarjanData;
    unsigned nextIdx = 0;

    auto pushFrame = [&](mlir::Operation *op) {
      tarjanData[op] = {nextIdx, nextIdx};
      ++nextIdx;
      sccStack.insert(op);
      dfsStack.push_back(FrameT(op));
    };

    pushFrame(startOp);

    while (!dfsStack.empty()) {
      FrameT &frame = dfsStack.back();
      mlir::Operation *op = frame.op;

      if (auto *child = frame.nextChild(shouldTraverseFn)) {
        auto it = tarjanData.find(child);
        if (it != tarjanData.end()) {
          // Already seen in this DFS.
          if (sccStack.contains(child))
            // Back edge — update lowlink.
            tarjanData[op].second =
                std::min(tarjanData[op].second, it->second.first);
          // else: forward/cross edge within this DFS — ignore.
        } else if (!opToSccIndex.contains(child)) {
          // Not yet seen in any DFS — recurse.
          pushFrame(child);
        }
        // else: completed in a previous visit() call — cross edge, ignore.
        continue;
      }

      // All children processed — backtrack.
      auto [opIndex, opLowLink] = tarjanData.at(op);
      dfsStack.pop_back();

      // If op is the root of its SCC, pop and emit it.
      if (opLowLink == opIndex) {
        detail::CyclicOpSCCStorage sccOps;
        do {
          sccOps.push_back(sccStack.pop_back_val());
        } while (sccOps.back() != op);

        unsigned sccIdx = sccs.size();
        for (auto *sccOp : sccOps)
          opToSccIndex[sccOp] = sccIdx;

        if (sccOps.size() == 1 && !hasSelfLoop(sccOps.front())) {
          sccs.push_back(detail::OpOrIndex(sccOps.front()));
        } else {
          unsigned cyclicIdx = cyclicSccs.size();
          cyclicSccs.emplace_back(std::move(sccOps));
          sccs.push_back(detail::OpOrIndex(cyclicIdx));
        }
        continue;
      }

      // Not an SCC root — back-propagate lowlink to the parent frame.
      if (!dfsStack.empty()) {
        auto &parentLowLink = tarjanData.at(dfsStack.back().op).second;
        parentLowLink = std::min(parentLowLink, opLowLink);
      }
    }
  }

  OpSCCFilter shouldTraverseFn;

  llvm::SmallSetVector<mlir::Operation *, NumInlineElts> sccStack;
  llvm::SmallVector<FrameT, NumInlineElts> dfsStack;
  llvm::SmallDenseMap<mlir::Operation *, unsigned, NumInlineElts> opToSccIndex;
  llvm::SmallVector<detail::OpOrIndex, NumInlineElts> sccs;
  llvm::SmallVector<detail::CyclicOpSCCStorage, 0> cyclicSccs;
};

} // namespace circt

#endif // CIRCT_SUPPORT_SPARSEOPSCC_H
