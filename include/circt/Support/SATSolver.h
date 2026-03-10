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

#include <memory>

namespace circt {

/// Abstract interface for incremental SAT solvers with an IPASIR-style API.
class IncrementalSATSolver {
public:
  enum Result : int { kSAT = 10, kUNSAT = 20, kUNKNOWN = 0 };

  virtual ~IncrementalSATSolver() = default;

  virtual void add(int lit) = 0;
  virtual void assume(int lit) = 0;
  virtual Result solve() = 0;
  virtual int val(int v) const = 0;

  virtual void reserveVars(int maxVar) {}
};

/// Construct a Z3-backed incremental IPASIR-style SAT solver.
std::unique_ptr<IncrementalSATSolver> createZ3SATSolver();

} // namespace circt

#endif // CIRCT_SUPPORT_SATSOLVER_H
