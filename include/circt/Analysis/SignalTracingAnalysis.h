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

#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::dataflow;

namespace circt {
namespace analysis {

struct SignalState {
  explicit SignalState() = default;
  explicit SignalState(bool isTraced) : traced(isTraced) {}

  ChangeResult reset();

  bool operator==(const SignalState &rhs) const;

  static SignalState join(const SignalState &lhs, const SignalState &rhs);

  ChangeResult setTraced(bool isTraced);

  bool getTraced() const;

  void print(llvm::raw_ostream &os) const;

private:
  bool traced = false;
};

struct SignalTracingAnalysis
    : public SparseDataFlowAnalysis<Lattice<SignalState>> {
  SignalTracingAnalysis(DataFlowSolver &solver, ModuleOp top,
                        DenseSet<Operation *> &sources,
                        DenseSet<Operation *> &sinks)
      : SparseDataFlowAnalysis(solver), instanceGraph(hw::InstanceGraph(top)),
        sources(sources), sinks(sinks) {}

  void visitOperation(Operation *op,
                      ArrayRef<const Lattice<SignalState> *> operands,
                      ArrayRef<Lattice<SignalState> *> results) override;

  void setToEntryState(Lattice<SignalState> *lattice) override;

  bool shouldVisitZeroResultOperations() override { return true; }

private:
  hw::InstanceGraph instanceGraph;
  DenseSet<Operation *> &sources;
  DenseSet<Operation *> &sinks;
};

} // namespace analysis
} // namespace circt

#endif // CIRCT_ANALYSIS_SIGNAL_TRACING_ANALYSIS_H
