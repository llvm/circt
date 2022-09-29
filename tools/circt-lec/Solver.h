//===-- Solver.h - SMT solver interface -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file defines a SMT solver interface for the `circt-lec` tool.
///
//===----------------------------------------------------------------------===//

#ifndef LEC_SOLVER_H
#define LEC_SOLVER_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Value.h"
#include <string>
#include <z3++.h>

/// A satisfiability checker for circuit equivalence
///
/// This class interfaces with an external SMT solver acting as a logical
/// engine. First spawn two circuits through `addCircuit`; after collecting
/// their logical constraints, the `solve` method will compare them and report
/// whether they result to be equivalent or, when not, also printing a model
/// acting as a counterexample.
class Solver {
public:
  Solver() : circuits{}, context(), solver(context){};
  ~Solver();

  /// Solve the equivalence problem between the two circuits, then present the
  /// results to the user.
  mlir::LogicalResult solve();

  class Circuit;
  /// Create a new circuit to be compared and return it.
  Circuit *addCircuit(circt::StringRef name, bool firstCircuit);

private:
  /// Prints a model satisfying the solved constraints.
  void printModel();
  /// Prints the constraints which were added to the solver.
  /// Compared to solver.assertions().to_string() this method exposes each
  /// assertion as a z3::expression for eventual in-depth debugging.
  void printAssertions();
  /// Prints the internal statistics of the SMT solver for benchmarking purposes
  /// and operational insight.
  void printStatistics();

  /// Formulates additional constraints which are satisfiable if only if the
  /// two circuits which are being compared are NOT equivalent, in which case
  /// there would be a model acting as a counterexample.
  /// The procedure fails when detecting a mismatch of arity or type between
  /// the inputs and outputs of the circuits.
  mlir::LogicalResult constrainCircuits();

  /// A map from internal solver symbols to the IR values they represent.
  llvm::DenseMap<std::string, mlir::Value> symbolTable;
  /// The two circuits to be compared.
  llvm::SmallVector<Circuit *, 2> circuits;
  /// The Z3 context of reference, owning all the declared values, constants
  /// and expressions.
  z3::context context;
  /// The Z3 solver acting as the logical engine backend.
  z3::solver solver;
};

namespace llvm {
/// Implementation of DenseMapInfo for std::string
/// needed by Solver::symbolTable.
template <>
struct DenseMapInfo<std::string> {
  static inline std::string getEmptyKey() {
    std::string empty("<<<EMPTY KEY>>>");
    return empty;
  }

  static inline std::string getTombstoneKey() {
    std::string tombstone("<<<TOMBSTONE KEY>>>");
    return tombstone;
  }

  static unsigned getHashValue(const std::string &val) {
    return std::hash<std::string>{}(val);
  }

  static bool isEqual(const std::string &lhs, const std::string &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

#endif // LEC_SOLVER_H
