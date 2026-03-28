//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header defines an abstract incremental SAT interface
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_SATSOLVER_H
#define CIRCT_SUPPORT_SATSOLVER_H

#include "llvm/ADT/ArrayRef.h"
#include <memory>

namespace circt {
template <typename T>
struct HeapScore;

/// A max-heap of ids into caller-owned storage.
///
/// The heap stores only integer ids and keeps a reverse map from each id to its
/// heap slot. That lets callers update an existing entry without creating
/// duplicates. Scores come from a stateless policy type. If a caller changes an
/// element in `storage` in a way that raises its score, it must call
/// `increase(id)` to restore heap order.
///
/// This data structure design is inspired by MiniSat's `minisat/mtl/Heap.h`.
template <typename T, typename ScoreFn = HeapScore<T>>
class IndexedMaxHeap {
public:
  // Create an empty heap referencing caller-owned storage. Caller must keep
  // storage alive; heap is empty until insert() is used.
  explicit IndexedMaxHeap(llvm::SmallVectorImpl<T> &storage)
      : storage(storage) {}

  void resize(unsigned size) {
    if (positions.size() < size)
      positions.resize(size, kInvalidIndex);
  }

  bool empty() const { return heap.empty(); }

  bool contains(unsigned id) const {
    return id < positions.size() && positions[id] != kInvalidIndex;
  }

  /// Remove all heap entries while keeping the underlying storage untouched.
  void clear() {
    heap.clear();
    std::fill(positions.begin(), positions.end(), kInvalidIndex);
  }

  /// Insert `id` if it is not already present.
  ///
  /// This heap treats ids as stable identities; duplicates are ignored instead
  /// of creating multiple entries for the same object.
  void insert(unsigned id) {
    if (contains(id))
      return;
    resize(id + 1);
    positions[id] = heap.size();
    heap.push_back(id);
    siftUp(positions[id]);
  }

  /// Restore heap order after the score for an existing `id` increased.
  ///
  /// This only moves the entry upward. Callers should use this when the score
  /// policy is monotonic in the "better" direction, such as SAT variable
  /// activity bumps. Mutating a stored element without calling `increase(id)`
  /// leaves the heap order stale.
  void increase(unsigned id) {
    if (contains(id))
      siftUp(positions[id]);
  }

  /// Remove and return the highest-scoring id.
  unsigned pop() {
    assert(!empty() && "cannot pop from empty heap");
    unsigned top = heap.front();
    positions[top] = kInvalidIndex;
    if (heap.size() == 1) {
      heap.pop_back();
      return top;
    }

    // Move the last entry to the root and sift down to restore heap order.
    heap.front() = heap.back();
    positions[heap.front()] = 0;
    heap.pop_back();
    siftDown(0);
    return top;
  }

private:
  static constexpr unsigned kInvalidIndex = ~0u;

  /// Compute the ordering score for one heap entry.
  double score(unsigned id) const { return ScoreFn{}(storage[id]); }

  /// Swap two heap slots and keep the reverse index in sync.
  void swapHeapEntries(unsigned lhs, unsigned rhs) {
    std::swap(heap[lhs], heap[rhs]);
    positions[heap[lhs]] = lhs;
    positions[heap[rhs]] = rhs;
  }

  /// Bubble one entry toward the root until the heap property is restored.
  void siftUp(unsigned index) {
    unsigned elem = heap[index];
    double elemScore = score(elem);
    while (index > 0) {
      unsigned parent = (index - 1) / 2;
      if (score(heap[parent]) >= elemScore)
        break;
      swapHeapEntries(index, parent);
      index = parent;
    }
    assert(heap[index] == elem && "siftUp must preserve the promoted element");
  }

  /// Push one entry toward the leaves until the heap property is restored.
  void siftDown(unsigned index) {
    unsigned elem = heap[index];
    double elemScore = score(elem);
    unsigned heapSize = heap.size();
    while (true) {
      unsigned child = 2 * index + 1;
      if (child >= heapSize)
        break;
      if (child + 1 < heapSize && score(heap[child + 1]) > score(heap[child]))
        ++child;
      if (elemScore >= score(heap[child]))
        break;
      swapHeapEntries(index, child);
      index = child;
    }
    assert(heap[index] == elem && "siftDown must preserve the demoted element");
  }

  /// The caller-owned objects indexed by heap ids.
  llvm::SmallVectorImpl<T> &storage;

  /// Binary heap of ids ordered by descending score.
  llvm::SmallVector<unsigned, 0> heap;

  /// Reverse map from id to heap slot, or `InvalidIndex` if absent.
  llvm::SmallVector<unsigned, 0> positions;
};

/// Abstract interface for incremental SAT solvers with an IPASIR-style API.
class IncrementalSATSolver {
public:
  enum Result : int { kSAT = 10, kUNSAT = 20, kUNKNOWN = 0 };

  virtual ~IncrementalSATSolver() = default;

  /// Add one literal to the clause currently under construction. A `0`
  /// literal terminates the clause and submits it to the solver.
  virtual void add(int lit) = 0;
  /// Add an assumption literal for the next `solve()` call only.
  virtual void assume(int lit) = 0;
  /// Solve under the previously added clauses and current assumptions.
  virtual Result solve() = 0;
  virtual Result solve(llvm::ArrayRef<int> assumptions) {
    for (int lit : assumptions)
      assume(lit);
    return solve();
  };
  /// Return the satisfying assignment for variable `v` from the last SAT
  /// result. The sign of the returned literal encodes the Boolean value.
  virtual int val(int v) const = 0;

  // These helpers are not part of the standard IPASIR interface.
  /// Set the per-`solve()` conflict budget. Negative values restore the
  /// backend default of no explicit conflict limit. The backend may choose to
  /// ignore this if it does not support conflict limits.
  virtual void setConflictLimit(int limit) {}
  /// Reserve storage for variables in the range `[1, maxVar]`.
  virtual void reserveVars(int maxVar) {}
  /// Add a complete clause in one call.
  virtual void addClause(llvm::ArrayRef<int> lits) {
    for (int lit : lits)
      add(lit);
    add(0);
  }
};

/// Construct a Z3-backed incremental IPASIR-style SAT solver.
std::unique_ptr<IncrementalSATSolver> createZ3SATSolver();
struct CadicalSATSolverOptions {
  enum class CadicalSolverConfig {
    Default, // Default.
    Plain,   // Disable preprocessing.
    Sat,     // Target satisfiable instances.
    Unsat,   // Target unsatisfiable instances.
  };
  CadicalSolverConfig config = CadicalSolverConfig::Default;
};
/// Construct a CaDiCaL-backed incremental IPASIR-style SAT solver.
std::unique_ptr<IncrementalSATSolver>
createCadicalSATSolver(const CadicalSATSolverOptions &options = {});
/// Return true when at least one incremental SAT backend is available.
bool hasIncrementalSATSolverBackend();

} // namespace circt

#endif // CIRCT_SUPPORT_SATSOLVER_H
