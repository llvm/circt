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
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include "egg_netlist_synthesizer.h"

#define DEBUG_TYPE "egg-synthesis"

using namespace mlir;
using namespace circt;

namespace {
struct EggSynthesisPass : public EggSynthesisBase<EggSynthesisPass> {
  void runOnOperation() override;

private:
  rust::Box<BooleanId> getExpr(Operation *op, BooleanEGraph &egraph);
  llvm::SmallString<8> getSymbol(Value value);

  // Value to Symbol mapping state.
  uint64_t valueId = 0;
  DenseMap<Value, llvm::SmallString<8>> valueToSymbol;
  llvm::StringMap<Value> symbolToValue;
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

  // Populate the EGraph with each operation.
  rust::Vec<BooleanId> stmts;
  for (auto &op : defineOp.getBodyBlock()) {
    // Handle output specially.
    if (isa<arc::OutputOp>(op))
      continue;

    // Invariants.
    assert(op.getNumResults() == 1);
    assert(llvm::all_of(op.getResultTypes(), isBoolean));
    assert(llvm::all_of(op.getOperandTypes(), isBoolean));

    // Create an egraph expression.
    rust::Box<BooleanId> expr = getExpr(&op, *egraph);

    // Create a let statement.
    rust::Box<BooleanId> let = build_let(
        *egraph, getSymbol(op.getResult(0)).str().str(), std::move(expr));

    append_expr(stmts, std::move(let));
  }

  // Create the module to synthesize.
  rust::Box<BooleanExpression> expression =
      build_module(*egraph, std::move(stmts));

  // Create the synthesizer.
  rust::Box<Synthesizer> synthesizer = synthesizer_new(
      "/Users/mikeu/skywater-preparation/sky130_fd_sc_hd_tt_100C_1v80.json",
      "Area");

  LLVM_DEBUG({
    llvm::dbgs() << "Expression before synthesis:\n";
    print_expr(*expression);
  });

  // Run the synthesizer.
  rust::Box<BooleanExpression> result = synthesizer_run(
      std::move(egraph), std::move(synthesizer), std::move(expression));

  LLVM_DEBUG({
    llvm::dbgs() << "Expression after synthesis:\n";
    print_expr(*result);
  });

  // Clean up state.
  valueId = 0;
  valueToSymbol.clear();
  symbolToValue.clear();
}

rust::Box<BooleanId> EggSynthesisPass::getExpr(Operation *op,
                                               BooleanEGraph &egraph) {
  return llvm::TypeSwitch<Operation *, rust::Box<BooleanId>>(op)
      .Case([&](hw::ConstantOp c) {
        return build_num(egraph, c.getValue().getZExtValue());
      })
      .Case([&](comb::AndOp a) {
        auto numOperands = a.getNumOperands();
        assert(numOperands >= 2);

        SmallVector<rust::Box<BooleanId>> symbols;
        for (Value operand : a.getOperands())
          symbols.push_back(
              build_symbol(egraph, getSymbol(operand).str().str()));

        rust::Box<BooleanId> expr =
            build_and(egraph, std::move(symbols[numOperands - 2]),
                      std::move(symbols[numOperands - 1]));

        for (size_t i = 3, e = numOperands; i <= e; ++i)
          expr = build_and(egraph, std::move(symbols[numOperands - i]),
                           std::move(expr));

        return expr;
      })
      .Case([&](comb::OrOp o) {
        auto numOperands = o.getNumOperands();
        assert(numOperands >= 2);

        SmallVector<rust::Box<BooleanId>> symbols;
        for (Value operand : o.getOperands())
          symbols.push_back(
              build_symbol(egraph, getSymbol(operand).str().str()));

        rust::Box<BooleanId> expr =
            build_and(egraph, std::move(symbols[numOperands - 2]),
                      std::move(symbols[numOperands - 1]));

        for (size_t i = 3, e = numOperands; i <= e; ++i)
          expr = build_and(egraph, std::move(symbols[numOperands - i]),
                           std::move(expr));

        return expr;
      })
      .Case([&](comb::XorOp n) {
        assert(n.isBinaryNot());
        rust::Box<BooleanId> symbol =
            build_symbol(egraph, getSymbol(n.getOperand(0)).str().str());

        rust::Box<BooleanId> expr = build_not(egraph, std::move(symbol));

        return expr;
      });
}

llvm::SmallString<8> EggSynthesisPass::getSymbol(Value value) {
  auto it = valueToSymbol.find(value);
  if (it != valueToSymbol.end())
    return it->getSecond();

  SmallString<8> symbol("v");
  symbol += std::to_string(valueId);
  valueId += 1;

  valueToSymbol[value] = symbol;
  symbolToValue[symbol] = value;

  return symbol;
}

std::unique_ptr<mlir::Pass> circt::createEggSynthesisPass() {
  return std::make_unique<EggSynthesisPass>();
}
