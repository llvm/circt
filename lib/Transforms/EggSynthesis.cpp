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
#include "llvm/Support/Debug.h"

#include "egg_netlist_synthesizer.h"

#define DEBUG_TYPE "egg-synthesis"

using namespace mlir;
using namespace circt;

namespace {
struct EggSynthesisPass : public EggSynthesisBase<EggSynthesisPass> {
  void runOnOperation() override;

private:
  rust::Box<BooleanId> getExpr(Operation *op, BooleanEGraph &egraph);
  void generateCellInstance(SmallString<8> symbol, BooleanExpression &expr,
                            OpBuilder &builder,
                            BackedgeBuilder &backedgeBuilder);
  hw::HWModuleExternOp generateCellDeclaration(BooleanExpression &expr,
                                               Location loc,
                                               OpBuilder &builder);
  SmallString<8> getSymbol(Value value);

  // Value to Symbol mapping state.
  uint64_t valueId = 0;
  DenseMap<Value, SmallString<8>> valueToSymbol;
  llvm::StringMap<Value> symbolToValue;

  // Symbol to expression and backedge mapping state.
  llvm::StringMap<Backedge> symbolToBackedge;

  // Operations that are converted.
  SmallVector<Operation *> opsToErase;
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

  // Ensure block arguments' symbols have backedges, which just return the arg.
  for (auto arg : defineOp.getArguments()) {
    SmallString<8> symbol = valueToSymbol[arg];
    symbolToBackedge[symbol] = backedgeBuilder.get(
        builder.getI1Type(), symbolToValue[symbol].getLoc());
    symbolToBackedge[symbol].setValue(arg);
  }

  // Iterate over each let, which binds a symbol to an expression.
  rust::Vec<BooleanExpression> lets = expr_get_module_body(std::move(result));
  SmallVector<SmallString<8>> letSymbols;
  SmallVector<BooleanExpression *> letExprs;
  for (BooleanExpression &let : llvm::make_range(lets.begin(), lets.end())) {
    // Get the symbol defined by the let.
    rust::String letSymbol = expr_get_let_symbol(let);
    SmallString<8> symbol;
    for (char c : llvm::make_range(letSymbol.begin(), letSymbol.end()))
      symbol += c;

    // We generate expressions for constants used in our formulation of not as
    // xor, but we don't actually need to instantiate gates for the constant.
    if (symbolToValue[symbol].getDefiningOp<hw::ConstantOp>())
      continue;

    // Get the expression bound to the symbol.
    rust::Box<BooleanExpression> expr = expr_get_let_expr(let);

    // Map from the symbol to a new backedge.
    symbolToBackedge[symbol] = backedgeBuilder.get(
        builder.getI1Type(), symbolToValue[symbol].getLoc());

    // Replace uses of the original value with the backedge.
    symbolToValue[symbol].replaceAllUsesWith(symbolToBackedge[symbol]);

    // Generate the instance for this cell.
    generateCellInstance(symbol, *expr, builder, backedgeBuilder);
  }

  // Clean up replaced ops.
  for (auto *op : opsToErase) {
    op->dropAllUses();
    op->erase();
  }

  // Clean up state.
  valueId = 0;
  valueToSymbol.clear();
  symbolToValue.clear();
}

rust::Box<BooleanId> EggSynthesisPass::getExpr(Operation *op,
                                               BooleanEGraph &egraph) {
  opsToErase.push_back(op);

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

void EggSynthesisPass::generateCellInstance(SmallString<8> symbol,
                                            BooleanExpression &expr,
                                            OpBuilder &builder,
                                            BackedgeBuilder &backedgeBuilder) {
  // If the symbol is mapped to a know Value, get is Location and set insertion
  // point to keep things in the same order.
  Location symbolLoc = builder.getUnknownLoc();
  auto symbolValue = symbolToValue.find(symbol);
  if (symbolValue != symbolToValue.end()) {
    symbolLoc = symbolValue->getValue().getLoc();
    builder.setInsertionPoint(symbolValue->getValue().getDefiningOp());
  }

  // Declare extern module for the gate.
  hw::HWModuleExternOp decl = generateCellDeclaration(expr, symbolLoc, builder);

  // Get list of operands to the instance.
  SmallVector<Value> operands;
  rust::Vec<BooleanExpression> inputExprs = expr_get_gate_input_exprs(expr);
  for (BooleanExpression &inputExpr :
       llvm::make_range(inputExprs.begin(), inputExprs.end())) {
    // If symbol, lookup the symbol and use its backedge.
    if (expr_is_symbol(inputExpr)) {
      rust::String inputExprSymbol = expr_get_symbol(inputExpr);
      SmallString<8> inputSymbol;
      for (char c :
           llvm::make_range(inputExprSymbol.begin(), inputExprSymbol.end()))
        inputSymbol += c;

      operands.push_back(symbolToBackedge[inputSymbol]);

      continue;
    }

    // If nested instance, generate a symbol and backedge, then recurse.
    SmallString<8> inputSymbol("v");
    inputSymbol += std::to_string(valueId);
    valueId += 1;

    symbolToBackedge[inputSymbol] =
        backedgeBuilder.get(builder.getI1Type(), symbolLoc);

    generateCellInstance(inputSymbol, inputExpr, builder, backedgeBuilder);

    operands.push_back(symbolToBackedge[inputSymbol]);
  }

  // Make an instance of the gate.
  auto instance =
      builder.create<hw::InstanceOp>(symbolLoc, decl, symbol, operands);

  // Replace backedge with instance result.
  symbolToBackedge[symbol].setValue(instance.getResult(0));
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
  for (auto &gateInputName :
       llvm::make_range(gateInputNames.begin(), gateInputNames.end())) {
    auto portName = builder.getStringAttr(std::string(gateInputName));
    auto portDirection = hw::PortDirection::INPUT;
    auto portType = builder.getI1Type();
    ports.push_back(hw::PortInfo{portName, portDirection, portType});
  }

  // TODO: this probably needs to match the gate's actual output name.
  ports.push_back(hw::PortInfo{builder.getStringAttr("OUT"),
                               hw::PortDirection::OUTPUT, builder.getI1Type()});

  // Build the gate declaration.
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(moduleOp.getBody());
  auto decl = builder.create<hw::HWModuleExternOp>(loc, gateName, ports);

  return decl;
}

SmallString<8> EggSynthesisPass::getSymbol(Value value) {
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
