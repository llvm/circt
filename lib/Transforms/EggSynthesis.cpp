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
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
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
  void generateCellInstance(llvm::SmallString<8> symbol,
                            BooleanExpression &expr, OpBuilder &builder,
                            BackedgeBuilder &backedgeBuilder);
  hw::HWModuleExternOp generateCellDeclaration(BooleanExpression &expr,
                                               Location loc,
                                               OpBuilder &builder);
  llvm::SmallString<8> getSymbol(Value value);

  // Value to Symbol mapping state.
  uint64_t valueId = 0;
  DenseMap<Value, llvm::SmallString<8>> valueToSymbol;
  llvm::StringMap<Value> symbolToValue;

  // Symbol to expression and backedge mapping state.
  llvm::StringMap<BooleanExpression *> symbolToExpr;
  llvm::StringMap<Backedge> symbolToBackedge;
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

  // Create the OpBuilder and BackedgeBuilder.
  auto builder = OpBuilder(defineOp);
  auto backedgeBuilder = BackedgeBuilder(builder, defineOp.getLoc());

  // Build a mapping from symbol to synthesized BooleanExpression.
  rust::Vec<BooleanExpression> lets = expr_get_module_body(std::move(result));
  for (BooleanExpression &let : llvm::make_range(lets.begin(), lets.end())) {
    // Get the symbol defined by the let.
    rust::String letSymbol = expr_get_let_symbol(let);
    llvm::SmallString<8> symbol;
    for (char c : llvm::make_range(letSymbol.begin(), letSymbol.end()))
      symbol += c;

    // Get the expression bound to the symbol.
    rust::Box<BooleanExpression> expr = expr_get_let_expr(let);

    // Map from symbol to a new backedge.
    symbolToBackedge[symbol] = backedgeBuilder.get(
        builder.getI1Type(), symbolToValue[symbol].getLoc());

    // Map from symbol to the expression.
    symbolToExpr[symbol] = expr.into_raw();
  }

  // Generate external module declarations and instantiations from the netlist.
  for (auto output : defineOp.getBodyBlock().getTerminator()->getOperands()) {
    llvm::SmallString<8> symbol = valueToSymbol[output];
    BooleanExpression *expr = symbolToExpr[symbol];
    auto exprBox = rust::Box<BooleanExpression>::from_raw(expr);
    generateCellInstance(symbol, *exprBox, builder, backedgeBuilder);

    // Replace output with resolved backedge for symbol.
  }

  // Run DCE to clean up.

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

void EggSynthesisPass::generateCellInstance(llvm::SmallString<8> symbol,
                                            BooleanExpression &expr,
                                            OpBuilder &builder,
                                            BackedgeBuilder &backedgeBuilder) {
  // Declare extern module for the gate.
  hw::HWModuleExternOp decl =
      generateCellDeclaration(expr, symbolToValue[symbol].getLoc(), builder);

  // Get list of input names from declaration.
  // Get list of operands:
  //  if symbol, use generated value (might be a backedge)
  //  if nested instance, generate a symbol and backedge, and recurse
  // Make an instance of the gate.
  // Replace backedge with instance result.
}

hw::HWModuleExternOp
EggSynthesisPass::generateCellDeclaration(BooleanExpression &expr, Location loc,
                                          OpBuilder &builder) {
  auto moduleOp = getOperation().getParentOp<ModuleOp>();

  // Get the gate name.
  StringAttr gateName =
      builder.getStringAttr(std::string(expr_get_gate_name(expr)));

  // Check if the declaration exists, and just return it.
  if (auto decl = moduleOp.lookupSymbol<hw::HWModuleExternOp>(gateName))
    return decl;

  // Get the gate ports.
  rust::Vec<rust::String> gateInputNames = expr_get_gate_input_names(expr);
  SmallVector<hw::PortInfo> ports;
  for (auto gateInputName :
       llvm::make_range(gateInputNames.begin(), gateInputNames.end())) {
    auto portName = builder.getStringAttr(std::string(gateInputName));
    auto portDirection = hw::PortDirection::INPUT;
    auto portType = builder.getI1Type();
    ports.push_back(hw::PortInfo{portName, portDirection, portType});
  }

  // Build the gate declaration.
  builder.setInsertionPointToStart(moduleOp.getBody());
  auto decl = builder.create<hw::HWModuleExternOp>(loc, gateName, ports);

  return decl;
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
