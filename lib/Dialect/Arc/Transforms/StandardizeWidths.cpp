//===- StandardizeWidths.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-standardize-widths"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_STANDARDIZEWIDTHS
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using mlir::DataFlowSolver;
using namespace mlir::dataflow;
using namespace circt;
using namespace arc;

struct SubwordValue {
  /// Print the constant value.
  void print(raw_ostream &os) const {
    if (isUninitialized()) {
      os << "<UNINITIALIZED>";
      return;
    }
    if (isUnknown()) {
      os << "<UNKNOWN>";
      return;
    }
    if (isConstant) {
      os << "<CONST>";
      return;
    }
    os << "@" << offset;
  }

  bool operator==(const SubwordValue &rhs) const {
    return uninitialized == rhs.uninitialized && unknown == rhs.unknown &&
           offset == rhs.offset;
  }

  /// Whether the state is uninitialized.
  bool isUninitialized() const { return uninitialized; }

  /// Whether the state is unknown.
  bool isUnknown() const { return unknown; }

  /// The state where the constant value is uninitialized. This happens when the
  /// state hasn't been set during the analysis.
  static SubwordValue getUninitialized() { return SubwordValue{}; }

  /// The state where the constant value is unknown.
  static SubwordValue getUnknown() {
    SubwordValue value;
    value.uninitialized = false;
    value.unknown = true;
    return value;
  }

  static SubwordValue getConstant(hw::ConstantOp op) {
    SubwordValue value;
    value.uninitialized = false;
    value.isConstant = true;
    return value;
  }

  static SubwordValue join(const SubwordValue &lhs, const SubwordValue &rhs) {
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;
    if (lhs == rhs)
      return lhs;
    return getUnknown();
  }

  bool uninitialized = true;
  bool unknown = false;
  bool isConstant = false;
  unsigned offset = 0;
};

struct SubwordAnalysis
    : public SparseForwardDataFlowAnalysis<Lattice<SubwordValue>> {
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;
  void visitOperation(Operation *op,
                      ArrayRef<const Lattice<SubwordValue> *> operands,
                      ArrayRef<Lattice<SubwordValue> *> results) override;
  void setToEntryState(Lattice<SubwordValue> *lattice) override;
};

void SubwordAnalysis::visitOperation(
    Operation *op, ArrayRef<const Lattice<SubwordValue> *> operands,
    ArrayRef<Lattice<SubwordValue> *> results) {

  if (!isa<comb::ConcatOp>(op))
    for (auto [index, operand] : llvm::enumerate(operands))
      if (!operand->getValue().isUninitialized() &&
          !operand->getValue().isUnknown())
        LLVM_DEBUG(llvm::dbgs() << "- Operand " << index << " of " << *op
                                << " is " << *operand << "\n");

  if (auto constOp = dyn_cast<hw::ConstantOp>(op)) {
    propagateIfChanged(results[0],
                       results[0]->join(SubwordValue::getConstant(constOp)));
    return;
  }

  if (auto extractOp = dyn_cast<comb::ExtractOp>(op)) {
    auto *lattice = results[0];
    // LLVM_DEBUG(llvm::dbgs() << "- Visiting " << *op << "\n");
    SubwordValue value;
    value.uninitialized = false;
    value.offset = extractOp.getLowBit();
    propagateIfChanged(results[0], results[0]->join(value));
    return;
  }

  // for (auto *operand : operands)
  //   if (operand->getValue().isUninitialized())
  //     return;

  // LLVM_DEBUG(llvm::dbgs() << "- Visiting " << *op << "\n");
}

void SubwordAnalysis::setToEntryState(Lattice<SubwordValue> *lattice) {
  propagateIfChanged(lattice, lattice->join(SubwordValue::getUnknown()));
}

namespace {
struct StandardizeWidthsPass
    : public impl::StandardizeWidthsBase<StandardizeWidthsPass> {
  void runOnOperation() override;
};
} // namespace

void StandardizeWidthsPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "Standardizing widths\n");
  Operation *op = getOperation();

  DataFlowSolver solver;
  solver.load<DeadCodeAnalysis>();
  solver.load<SubwordAnalysis>();
  if (failed(solver.initializeAndRun(op)))
    return signalPassFailure();
}
