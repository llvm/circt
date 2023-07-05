//===- EggSynthesis.cpp - Egg synthesis pass ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include "egg_netlist_synthesizer.h"

using namespace mlir;
using namespace circt;

namespace {
struct EggSynthesisPass : public EggSynthesisBase<EggSynthesisPass> {
  void runOnOperation() override;
};
} // namespace

static bool isBoolean(Type type) {
  return hw::isHWIntegerType(type) && hw::getBitWidth(type) == 1;
}

void EggSynthesisPass::runOnOperation() {
  arc::DefineOp defineOp = getOperation();

  OpPrintingFlags flags;

  // Initialize an EGraph.
  rust::Box<BooleanEGraph> egraph = egraph_new();

  // Print the wrapper module.
  llvm::outs() << "(module\n";

  for (auto &op : defineOp.getBodyBlock()) {
    // Handle output specially.
    if (isa<arc::OutputOp>(op))
      continue;

    // Invariants.
    assert(op.getNumResults() == 1);
    assert(llvm::all_of(op.getResultTypes(), isBoolean));
    assert(llvm::all_of(op.getOperandTypes(), isBoolean));

    // Print a let binding for this expression.
    llvm::outs() << "  (let ";

    // Print the symbol name for this expression.
    op.getResult(0).printAsOperand(llvm::outs(), flags);
    llvm::outs() << ' ';

    // TODO: add a helper to recursively build expressions with egg.

    // Print this expression.
    llvm::TypeSwitch<Operation *>(&op)
        .Case([&](hw::ConstantOp c) {
          if (c.getValue().isIntN(0)) {
            llvm::outs() << '0';
            return;
          }
          assert(c.getValue().isIntN(1));
          llvm::outs() << '1';
        })
        .Case([&](comb::AndOp a) {
          assert(a.getNumOperands() >= 2);
          for (size_t i = 0, e = a.getNumOperands(); i < e; ++i) {
            if (i < e - 1)
              llvm::outs() << "(& ";
            a.getOperand(i).printAsOperand(llvm::outs(), flags);
            if (i < e - 1)
              llvm::outs() << ' ';
          }
          for (size_t i = 0, e = a.getNumOperands(); i < e; ++i)
            if (i < e - 1)
              llvm::outs() << ')';
        })
        .Case([&](comb::OrOp o) {
          assert(o.getNumOperands() >= 2);
          for (size_t i = 0, e = o.getNumOperands(); i < e; ++i) {
            if (i < e - 1)
              llvm::outs() << "(| ";
            o.getOperand(i).printAsOperand(llvm::outs(), flags);
            if (i < e - 1)
              llvm::outs() << ' ';
          }
          for (size_t i = 0, e = o.getNumOperands(); i < e; ++i)
            if (i < e - 1)
              llvm::outs() << ')';
        })
        .Case([&](comb::XorOp n) {
          assert(n.isBinaryNot());
          llvm::outs() << "(! ";
          n.getOperand(0).printAsOperand(llvm::outs(), flags);
          llvm::outs() << ')';
        });

    llvm::outs() << ")\n";
  }

  llvm::outs() << ")\n";
}

std::unique_ptr<mlir::Pass> circt::createEggSynthesisPass() {
  return std::make_unique<EggSynthesisPass>();
}
