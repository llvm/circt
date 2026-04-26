//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines incremental SAT solvers with an IPASIR-style interface.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/SATSolver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SMTAPI.h"

#ifdef CIRCT_CADICAL_ENABLED
#include "third_party/cadical/cadical.hpp"
#endif

#include <cassert>
#include <cstdlib>
#include <string>

namespace circt {

namespace {

//===----------------------------------------------------------------------===//
// CaDiCaL Backend
//===----------------------------------------------------------------------===//

#ifdef CIRCT_CADICAL_ENABLED

const char *
toCadicalConfigName(CadicalSATSolverOptions::CadicalSolverConfig config) {
  switch (config) {
  case CadicalSATSolverOptions::CadicalSolverConfig::Default:
    return "default";
  case CadicalSATSolverOptions::CadicalSolverConfig::Plain:
    return "plain";
  case CadicalSATSolverOptions::CadicalSolverConfig::Sat:
    return "sat";
  case CadicalSATSolverOptions::CadicalSolverConfig::Unsat:
    return "unsat";
  }
  llvm_unreachable("unknown CaDiCaL configuration");
}

class CadicalSATSolver : public IncrementalSATSolver {
public:
  explicit CadicalSATSolver(const CadicalSATSolverOptions &options) {
    if (options.config !=
        CadicalSATSolverOptions::CadicalSolverConfig::Default) {
      bool configured = solver.configure(toCadicalConfigName(options.config));
      assert(configured && "invalid CaDiCaL configuration");
      (void)configured;
    }
  }
  void add(int lit) override { solver.add(lit); }
  void assume(int lit) override {
    if (lit != 0)
      solver.assume(lit);
  }
  Result solve() override {
    if (conflictLimit >= 0)
      solver.limit("conflicts", conflictLimit);
    switch (solver.solve()) {
    case CaDiCaL::SATISFIABLE:
      return kSAT;
    case CaDiCaL::UNSATISFIABLE:
      return kUNSAT;
    default:
      return kUNKNOWN;
    }
  }
  int val(int v) const override {
    if (v <= 0 || v > maxVariable)
      return 0;
    return solver.val(v);
  }
  void setConflictLimit(int limit) override { conflictLimit = limit; }
  void reserveVars(int maxVar) override {
    if (maxVar <= maxVariable)
      return;
    solver.resize(maxVar);
    maxVariable = maxVar;
  }
  void addClause(llvm::ArrayRef<int> lits) override {
    if (lits.empty()) {
      solver.add(0);
      return;
    }
    solver.clause(lits.data(), lits.size());
  }
  int newVar() override {
    int var = solver.declare_one_more_variable();
    if (var > maxVariable)
      maxVariable = var;
    return var;
  }

private:
  mutable CaDiCaL::Solver solver;
  int maxVariable = 0;
  int conflictLimit = -1;
};

#endif // CIRCT_CADICAL_ENABLED

//===----------------------------------------------------------------------===//
// Z3 Backend
//===----------------------------------------------------------------------===//

#if LLVM_WITH_Z3

class Z3SATSolver : public IncrementalSATSolver {
public:
  Z3SATSolver();
  ~Z3SATSolver() override;

  void add(int lit) override;
  void assume(int lit) override;
  Result solve() override;
  Result solve(llvm::ArrayRef<int> assumptions) override;
  int val(int v) const override;
  void reserveVars(int maxVar) override;
  int newVar() override;

private:
  void clearSolveScope();
  int newVariable();
  llvm::SMTExprRef literalToExpr(int lit);
  void addClauseInternal(llvm::ArrayRef<int> lits);

  llvm::SMTSolverRef solver;
  llvm::SmallVector<llvm::SMTExprRef> variables;
  llvm::SmallVector<int> assumptions;
  llvm::SmallVector<int> clauseBuffer;
  int maxVariable = 0;
  Result lastResult = kUNKNOWN;
  bool solveScopeActive = false;
};

Z3SATSolver::Z3SATSolver() : solver(llvm::CreateZ3Solver()) {}

Z3SATSolver::~Z3SATSolver() { clearSolveScope(); }

void Z3SATSolver::add(int lit) {
  clearSolveScope();
  if (lit == 0) {
    addClauseInternal(clauseBuffer);
    clauseBuffer.clear();
    return;
  }

  reserveVars(std::abs(lit));
  clauseBuffer.push_back(lit);
}

void Z3SATSolver::assume(int lit) {
  clearSolveScope();
  if (lit == 0)
    return;
  assumptions.push_back(lit);
}

IncrementalSATSolver::Result Z3SATSolver::solve() {
  auto localAssumptions = assumptions;
  assumptions.clear();
  return solve(localAssumptions);
}

IncrementalSATSolver::Result
Z3SATSolver::solve(llvm::ArrayRef<int> assumptions) {
  clearSolveScope();
  solver->push();
  solveScopeActive = true;
  for (int lit : assumptions)
    solver->addConstraint(literalToExpr(lit));
  auto result = solver->check();
  if (!result)
    return lastResult = kUNKNOWN;
  if (*result)
    return lastResult = kSAT;
  return lastResult = kUNSAT;
}

int Z3SATSolver::val(int v) const {
  if (lastResult != kSAT || v <= 0 || v > maxVariable)
    return 0;
  llvm::APSInt value(llvm::APInt(1, 0), true);
  // Z3 returns an interpretation for all variables, even those not involved
  // in the problem. If the variable is not involved, return 0 to indicate
  // "undefined" rather than a potentially misleading true/false value.
  if (!solver->getInterpretation(variables[v - 1], value))
    return 0;
  return value != 0 ? v : -v;
}

void Z3SATSolver::reserveVars(int maxVar) {
  if (maxVar <= maxVariable)
    return;
  while (static_cast<int>(variables.size()) < maxVar)
    newVariable();
  maxVariable = maxVar;
}

int Z3SATSolver::newVar() {
  int var = newVariable();
  if (var > maxVariable)
    maxVariable = var;
  return var;
}

void Z3SATSolver::clearSolveScope() {
  if (!solveScopeActive)
    return;
  solver->pop();
  solveScopeActive = false;
  lastResult = kUNKNOWN;
}

int Z3SATSolver::newVariable() {
  int varIndex = static_cast<int>(variables.size()) + 1;
  std::string name = "v" + std::to_string(varIndex);
  variables.push_back(solver->mkSymbol(name.c_str(), solver->getBoolSort()));
  return varIndex;
}

llvm::SMTExprRef Z3SATSolver::literalToExpr(int lit) {
  int absLit = std::abs(lit);
  // Ensure variable exists for this literal.
  reserveVars(absLit);
  auto *variable = variables[absLit - 1];
  return lit > 0 ? variable : solver->mkNot(variable);
}

void Z3SATSolver::addClauseInternal(llvm::ArrayRef<int> lits) {
  if (lits.empty()) {
    solver->addConstraint(solver->mkBoolean(false));
    return;
  }

  llvm::SMTExprRef clause = nullptr;
  for (int lit : lits) {
    if (lit == 0)
      continue;
    auto *expr = literalToExpr(lit);
    clause = clause ? solver->mkOr(clause, expr) : expr;
  }

  if (!clause) {
    solver->addConstraint(solver->mkBoolean(false));
    return;
  }
  solver->addConstraint(clause);
}

#endif // LLVM_WITH_Z3

} // namespace

void addAndClauses(int outVar, llvm::ArrayRef<int> inputLits,
                   llvm::function_ref<void(llvm::ArrayRef<int>)> addClause) {
  for (int lit : inputLits)
    addClause({-outVar, lit});

  llvm::SmallVector<int> clause;
  clause.reserve(inputLits.size() + 1);
  for (int lit : inputLits)
    clause.push_back(-lit);
  clause.push_back(outVar);
  addClause(clause);
}

void addOrClauses(int outVar, llvm::ArrayRef<int> inputLits,
                  llvm::function_ref<void(llvm::ArrayRef<int>)> addClause) {
  for (int lit : inputLits)
    addClause({-lit, outVar});

  llvm::SmallVector<int> clause;
  clause.reserve(inputLits.size() + 1);
  clause.push_back(-outVar);
  clause.append(inputLits.begin(), inputLits.end());
  addClause(clause);
}

void addXorClauses(int outVar, int lhsLit, int rhsLit,
                   llvm::function_ref<void(llvm::ArrayRef<int>)> addClause) {
  addClause({-lhsLit, -rhsLit, -outVar});
  addClause({lhsLit, rhsLit, -outVar});
  addClause({lhsLit, -rhsLit, outVar});
  addClause({-lhsLit, rhsLit, outVar});
}

void addParityClauses(int outVar, llvm::ArrayRef<int> inputLits,
                      llvm::function_ref<void(llvm::ArrayRef<int>)> addClause,
                      llvm::function_ref<int()> newVar) {
  assert(!inputLits.empty() && "parity requires at least one input");
  if (inputLits.size() == 1) {
    addClause({-outVar, inputLits.front()});
    addClause({outVar, -inputLits.front()});
    return;
  }

  int accumulatedLit = inputLits.front();
  for (auto [index, lit] : llvm::enumerate(inputLits.drop_front())) {
    bool isLast = index + 2 == inputLits.size();
    int outLit = isLast ? outVar : newVar();
    addXorClauses(outLit, accumulatedLit, lit, addClause);
    accumulatedLit = outLit;
  }
}

void addAtMostOneClauses(
    llvm::ArrayRef<int> inputLits,
    llvm::function_ref<void(llvm::ArrayRef<int>)> addClause,
    llvm::function_ref<int()> newVar) {
  if (inputLits.size() < 2)
    return;

  // Emit the clause encoding `lhs => rhs`.
  auto imply = [&](int lhs, int rhs) { addClause({-lhs, rhs}); };

  // Use a sequential-ladder encoding for the at-most-one constraint; see
  // Carsten Sinz, "Towards an Optimal CNF Encoding of Boolean Cardinality
  // Constraints", CP 2005.
  //
  // `ladder[i]` means "at least one of `inputLits[0]` through `inputLits[i]`
  // is true".
  // We use these helper variables to remember when an earlier choice was made.
  llvm::SmallVector<int, 8> ladder(inputLits.size() - 1);
  for (int &var : ladder)
    var = newVar();

  imply(inputLits.front(), ladder.front());
  for (unsigned i = 1, e = inputLits.size() - 1; i < e; ++i) {
    // If `inputLits[i]` is true, remember that one of `inputLits[0..i]` is
    // true.
    imply(inputLits[i], ladder[i]);
    // If an earlier variable was already true, keep that fact true for the
    // larger range `inputLits[0..i]`.
    imply(ladder[i - 1], ladder[i]);
    // If `ladder[i - 1]` is true, then some earlier variable was true, so
    // `inputLits[i]` must be false since at most one variable can be true.
    imply(ladder[i - 1], -inputLits[i]);
  }

  // If `ladder.back()` is true, then some earlier variable was true, so the
  // last variable must be false.
  imply(ladder.back(), -inputLits.back());
}

void addExactlyOneClauses(
    llvm::ArrayRef<int> inputLits,
    llvm::function_ref<void(llvm::ArrayRef<int>)> addClause,
    llvm::function_ref<int()> newVar) {
  addClause(inputLits);
  addAtMostOneClauses(inputLits, addClause, newVar);
}

std::unique_ptr<IncrementalSATSolver> createZ3SATSolver() {
#if LLVM_WITH_Z3
  return std::make_unique<Z3SATSolver>();
#else
  return {};
#endif
}

std::unique_ptr<IncrementalSATSolver>
createCadicalSATSolver(const CadicalSATSolverOptions &options) {
#ifdef CIRCT_CADICAL_ENABLED
  return std::make_unique<CadicalSATSolver>(options);
#else
  return {};
#endif
}

bool hasIncrementalSATSolverBackend() {
  return static_cast<bool>(createCadicalSATSolver()) ||
         static_cast<bool>(createZ3SATSolver());
}

} // namespace circt
