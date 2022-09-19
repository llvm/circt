//===- SignalTracingAnalysis.h - signal tracing dataflow analysis ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file declares the SignalTracingAnalysis using the dataflow
// analysis framework.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_ANALYSIS_SIGNAL_TRACING_ANALYSIS_H
#define CIRCT_ANALYSIS_SIGNAL_TRACING_ANALYSIS_H

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
// #include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::dataflow;

namespace circt {
namespace analysis {

struct SignalState : public AbstractDenseLattice {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SignalState)

  using AbstractDenseLattice::AbstractDenseLattice;

  ChangeResult reset();

  ChangeResult join(const AbstractDenseLattice &lattice) override;

  ChangeResult setTraced(bool isTraced);

  bool getTraced() const;

  void print(llvm::raw_ostream &os) const override;

private:
  bool traced = false;
};

struct SignalTracingAnalysis : public DenseDataFlowAnalysis<SignalState> {
  SignalTracingAnalysis(DataFlowSolver &solver, DenseSet<Operation *> &sources,
                        DenseSet<Operation *> &sinks)
      : DenseDataFlowAnalysis(solver), sources(sources), sinks(sinks) {}

  void visitOperation(Operation *op, const SignalState &before,
                      SignalState *after) override;

  void setToEntryState(SignalState *lattice) override;

private:
  DenseSet<Operation *> &sources;
  DenseSet<Operation *> &sinks;
};

// struct SignalState {
//   void print(llvm::raw_ostream &os) const;
//   bool operator==(const SignalState &rhs) const;
//   static SignalState join(const SignalState &lhs, const SignalState &rhs);
// };

// struct SignalTracingAnalysis
//     : public SparseDataFlowAnalysis<Lattice<SignalState>> {
//   using SparseDataFlowAnalysis::SparseDataFlowAnalysis;

//   void visitOperation(Operation *op,
//                       ArrayRef<const Lattice<SignalState> *> operands,
//                       ArrayRef<Lattice<SignalState> *> results) override;

//   void setToEntryState(Lattice<SignalState> *lattice) override;
// };

} // namespace analysis
} // namespace circt

#endif // CIRCT_ANALYSIS_SIGNAL_TRACING_ANALYSIS_H
