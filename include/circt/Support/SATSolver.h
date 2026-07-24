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
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace circt {
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
  /// Add a fresh variable for safe incremental SAT solving.
  virtual int newVar() = 0;
};

/// Emit clauses encoding `outVar <=> and(inputLits)`.
void addAndClauses(int outVar, llvm::ArrayRef<int> inputLits,
                   llvm::function_ref<void(llvm::ArrayRef<int>)> addClause);

/// Emit clauses encoding `outVar <=> or(inputLits)`.
void addOrClauses(int outVar, llvm::ArrayRef<int> inputLits,
                  llvm::function_ref<void(llvm::ArrayRef<int>)> addClause);

/// Emit clauses encoding `outVar <=> (lhsLit xor rhsLit)`.
void addXorClauses(int outVar, int lhsLit, int rhsLit,
                   llvm::function_ref<void(llvm::ArrayRef<int>)> addClause);

/// Emit clauses encoding `outVar <=> parity(inputLits)`.
void addParityClauses(int outVar, llvm::ArrayRef<int> inputLits,
                      llvm::function_ref<void(llvm::ArrayRef<int>)> addClause,
                      llvm::function_ref<int()> newVar);

/// Emit clauses encoding that at most one literal in `inputLits` can be true.
/// Unlike the Tseitin-style gate helpers above, this helper does not
/// take an `outVar`; it only emits the cardinality constraint itself.
void addAtMostOneClauses(
    llvm::ArrayRef<int> inputLits,
    llvm::function_ref<void(llvm::ArrayRef<int>)> addClause,
    llvm::function_ref<int()> newVar);

/// Emit clauses encoding that exactly one literal in `inputLits` is true.
void addExactlyOneClauses(
    llvm::ArrayRef<int> inputLits,
    llvm::function_ref<void(llvm::ArrayRef<int>)> addClause,
    llvm::function_ref<int()> newVar);

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

/// Construct an incremental SAT solver using the requested backend. The
/// `auto` backend prefers CaDiCaL and falls back to Z3.
std::unique_ptr<IncrementalSATSolver>
createSATSolver(llvm::StringRef backend = "auto");

} // namespace circt

#endif // CIRCT_SUPPORT_SATSOLVER_H
