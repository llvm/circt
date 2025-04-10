//===- RangeAnalysisComb.cpp - Lower some ops in comb -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Comb/CombPasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"

using namespace circt;
using namespace circt::comb;
using namespace mlir::dataflow;

namespace circt {
namespace comb {
#define GEN_PASS_DEF_RANGEANALYSISCOMB
#include "circt/Dialect/Comb/Passes.h.inc"
} // namespace comb
} // namespace circt

namespace {
class RangeAnalysisCombPass : public circt::comb::impl::RangeAnalysisCombBase<RangeAnalysisCombPass> {
  public:
  using RangeAnalysisCombBase::RangeAnalysisCombBase;
  void runOnOperation() override;
};
} // namespace

void RangeAnalysisCombPass::runOnOperation() {
  Operation *op = getOperation();
  MLIRContext *ctx = op->getContext();
  mlir::DataFlowSolver solver;
  solver.load<DeadCodeAnalysis>();
  solver.load<IntegerRangeAnalysis>();
  if (failed(solver.initializeAndRun(op)))
    return signalPassFailure();

  // RewritePatternSet patterns(ctx);
  // populateIntRangeNarrowingPatterns(patterns, solver);
  
}