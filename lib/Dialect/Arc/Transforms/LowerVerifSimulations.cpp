//===- LowerVerifSimulations.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-lower-verif-simulations"

using namespace mlir;
using namespace circt;
using namespace arc;

using hw::PortInfo;

namespace circt {
namespace arc {
#define GEN_PASS_DEF_LOWERVERIFSIMULATIONSPASS
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

namespace {
struct LowerVerifSimulationsPass
    : public arc::impl::LowerVerifSimulationsPassBase<
          LowerVerifSimulationsPass> {
  void runOnOperation() override;
  void lowerSimulation(verif::SimulationOp op, SymbolTable &symbolTable);
};
} // namespace

void LowerVerifSimulationsPass::runOnOperation() {
  SymbolTableCollection symbolTables;

  // Declare the `exit` function if it does not yet exist.
  auto &symbolTable = symbolTables.getSymbolTable(getOperation());
  if (!symbolTable.lookup("exit")) {
    auto builder = OpBuilder::atBlockBegin(getOperation().getBody());
    auto func = builder.create<func::FuncOp>(
        getOperation().getLoc(), builder.getStringAttr("exit"),
        builder.getFunctionType({builder.getI32Type()}, {}));
    SymbolTable::setSymbolVisibility(func, SymbolTable::Visibility::Private);
  }

  SmallVector<StringAttr> symbols;
  getOperation().walk([&](verif::SimulationOp op) {
    auto *symbolTableOp = SymbolTable::getNearestSymbolTable(op);
    assert(symbolTableOp);
    if (symbolTableOp == getOperation())
      symbols.push_back(op.getSymNameAttr());
    lowerSimulation(op, symbolTables.getSymbolTable(symbolTableOp));
  });
}

void LowerVerifSimulationsPass::lowerSimulation(verif::SimulationOp op,
                                                SymbolTable &symbolTable) {
  LLVM_DEBUG(llvm::dbgs() << "Lowering " << op.getSymName() << "\n");
  auto *context = &getContext();
  auto i1Type = IntegerType::get(context, 1);

  // Assemble the ports of the implementation module.
  auto &body = *op.getBody();
  auto *yieldOp = body.getTerminator();
  std::array<PortInfo, 4> implPorts;

  auto clockName = StringAttr::get(context, "clock");
  implPorts[0].name = clockName;
  implPorts[0].type = seq::ClockType::get(context);
  implPorts[0].dir = PortInfo::Input;
  implPorts[0].loc = body.getArgument(0).getLoc();

  auto initName = StringAttr::get(context, "init");
  implPorts[1].name = initName;
  implPorts[1].type = i1Type;
  implPorts[1].dir = PortInfo::Input;
  implPorts[1].loc = body.getArgument(1).getLoc();

  auto doneName = StringAttr::get(context, "done");
  implPorts[2].name = doneName;
  implPorts[2].type = i1Type;
  implPorts[2].dir = PortInfo::Output;
  implPorts[2].loc = yieldOp->getOperand(0).getLoc();

  auto exitCodeName = StringAttr::get(context, "exit_code");
  auto exitCodeType = yieldOp->getOperand(1).getType();
  implPorts[3].name = exitCodeName;
  implPorts[3].type = exitCodeType;
  implPorts[3].dir = PortInfo::Output;
  implPorts[3].loc = yieldOp->getOperand(1).getLoc();

  // Replace the `verif.yield` operation with an `hw.output`.
  OpBuilder builder(yieldOp);
  builder.create<hw::OutputOp>(yieldOp->getLoc(), yieldOp->getOperands());
  yieldOp->erase();

  // Move the body of the simulation into a separate HW module.
  builder.setInsertionPoint(op);
  auto implName = StringAttr::get(context, Twine("verif.simulation.impl.") +
                                               op.getSymName());
  auto implOp =
      builder.create<hw::HWModuleOp>(op.getLoc(), implName, implPorts);
  symbolTable.insert(implOp);
  implOp.getBody().takeBody(op.getBodyRegion());

  // Create a new function for the verification op.
  auto funcType = builder.getFunctionType({}, {});
  auto funcOp =
      builder.create<func::FuncOp>(op.getLoc(), op.getSymNameAttr(), funcType);
  auto &funcBody = funcOp.getBody().emplaceBlock();
  builder.setInsertionPointToEnd(&funcBody);

  auto falseOp = builder.create<hw::ConstantOp>(op.getLoc(), i1Type, 0);
  auto trueOp = builder.create<hw::ConstantOp>(op.getLoc(), i1Type, 1);
  auto lowOp = builder.create<seq::ToClockOp>(op.getLoc(), falseOp);
  auto highOp = builder.create<seq::ToClockOp>(op.getLoc(), trueOp);

  // Instantiate the implementation module.
  auto instType = SimModelInstanceType::get(
      context, FlatSymbolRefAttr::get(implOp.getSymNameAttr()));
  auto instOp = builder.create<SimInstantiateOp>(op.getLoc());
  auto &instBody = instOp.getBody().emplaceBlock();
  auto instArg = instBody.addArgument(instType, op.getLoc());
  builder.setInsertionPointToEnd(&instBody);

  // Create an `scf.execute_region` op inside such that we can use simple
  // control flow in the `arc.sim.instantiate` op body. This is simpler than
  // setting up an `scf.while` op.
  auto execOp = builder.create<scf::ExecuteRegionOp>(op.getLoc(), TypeRange{});
  builder.setInsertionPointToEnd(&execOp.getRegion().emplaceBlock());

  // Apply the initial clock tick to the design.
  builder.create<SimSetInputOp>(op.getLoc(), instArg, clockName, lowOp);
  builder.create<SimSetInputOp>(op.getLoc(), instArg, initName, trueOp);
  builder.create<SimStepOp>(op.getLoc(), instArg);
  builder.create<SimSetInputOp>(op.getLoc(), instArg, clockName, highOp);
  builder.create<SimStepOp>(op.getLoc(), instArg);
  builder.create<SimSetInputOp>(op.getLoc(), instArg, clockName, lowOp);
  builder.create<SimSetInputOp>(op.getLoc(), instArg, initName, falseOp);
  builder.create<SimStepOp>(op.getLoc(), instArg);

  // Create the block that will perform a single clock tick.
  auto &loopBlock = execOp.getRegion().emplaceBlock();
  builder.create<cf::BranchOp>(op.getLoc(), &loopBlock);
  builder.setInsertionPointToEnd(&loopBlock);

  // Sample the done and exit code signals.
  auto doneSample =
      builder.create<SimGetPortOp>(op.getLoc(), i1Type, instArg, doneName);
  auto exitCodeSample = builder.create<SimGetPortOp>(op.getLoc(), exitCodeType,
                                                     instArg, exitCodeName);

  // Apply a full clock cycle to the design.
  builder.create<SimSetInputOp>(op.getLoc(), instArg, clockName, highOp);
  builder.create<SimStepOp>(op.getLoc(), instArg);
  builder.create<SimSetInputOp>(op.getLoc(), instArg, clockName, lowOp);
  builder.create<SimStepOp>(op.getLoc(), instArg);

  // If done, exit the loop.
  auto &exitBlock = execOp.getRegion().emplaceBlock();
  builder.create<cf::CondBranchOp>(op.getLoc(), doneSample, &exitBlock,
                                   &loopBlock);
  builder.setInsertionPointToEnd(&exitBlock);

  // Compute `exit_code | (exit_code != 0)` as a way of guaranteeing that
  // the exit code will be non-zero even if bits are truncated by the operating
  // system.
  auto i32Type = builder.getI32Type();
  auto nonZeroI32 = builder.create<arith::ExtUIOp>(
      op.getLoc(), i32Type,
      builder.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ne, exitCodeSample,
          builder.create<hw::ConstantOp>(op.getLoc(), exitCodeType, 0)));

  Value codeI32 = exitCodeSample;
  if (exitCodeType.getIntOrFloatBitWidth() < 32)
    codeI32 = builder.create<arith::ExtUIOp>(op.getLoc(), i32Type, codeI32);
  else if (exitCodeType.getIntOrFloatBitWidth() > 32)
    codeI32 = builder.create<arith::TruncIOp>(op.getLoc(), i32Type, codeI32);
  codeI32 = builder.create<arith::OrIOp>(op.getLoc(), codeI32, nonZeroI32);

  // Call exit with the computed exit code.
  builder.create<func::CallOp>(op.getLoc(), TypeRange{},
                               builder.getStringAttr("exit"),
                               ValueRange{codeI32});
  builder.create<scf::YieldOp>(op.getLoc());

  // Create the final function return.
  builder.setInsertionPointToEnd(&funcBody);
  builder.create<func::ReturnOp>(op.getLoc());

  // Get rid of the original simulation op.
  op.erase();
}
