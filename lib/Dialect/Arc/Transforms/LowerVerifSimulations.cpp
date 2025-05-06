//===- LowerVerifSimulations.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
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
  auto builder = OpBuilder::atBlockBegin(getOperation().getBody());
  auto exitFuncType = builder.getFunctionType({builder.getI32Type()}, {});
  auto &symbolTable = symbolTables.getSymbolTable(getOperation());
  if (auto *exitOp = symbolTable.lookup("exit")) {
    auto func = dyn_cast<func::FuncOp>(exitOp);
    if (!func) {
      exitOp->emitOpError() << "expected to be a `func.func`";
      return signalPassFailure();
    }
    if (func.getFunctionType() != exitFuncType) {
      func.emitOpError() << "expected to have function type " << exitFuncType
                         << ", got " << func.getFunctionType() << " instead";
      return signalPassFailure();
    }
  } else {
    auto func = builder.create<func::FuncOp>(
        getOperation().getLoc(), builder.getStringAttr("exit"), exitFuncType);
    SymbolTable::setSymbolVisibility(func, SymbolTable::Visibility::Private);
  }

  getOperation().walk([&](verif::SimulationOp op) {
    auto *symbolTableOp = SymbolTable::getNearestSymbolTable(op);
    assert(symbolTableOp);
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

  auto successName = StringAttr::get(context, "success");
  implPorts[3].name = successName;
  implPorts[3].type = i1Type;
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
  auto loc = op.getLoc();
  auto implOp = builder.create<hw::HWModuleOp>(loc, implName, implPorts);
  symbolTable.insert(implOp);
  implOp.getBody().takeBody(op.getBodyRegion());

  // Create a new function for the verification op.
  auto funcType = builder.getFunctionType({}, {});
  auto funcOp =
      builder.create<func::FuncOp>(loc, op.getSymNameAttr(), funcType);
  auto *funcBody = builder.createBlock(&funcOp.getBody());

  auto falseOp = builder.create<hw::ConstantOp>(loc, i1Type, 0);
  auto trueOp = builder.create<hw::ConstantOp>(loc, i1Type, 1);
  auto lowOp = builder.create<seq::ToClockOp>(loc, falseOp);
  auto highOp = builder.create<seq::ToClockOp>(loc, trueOp);

  // Instantiate the implementation module.
  auto instType = SimModelInstanceType::get(
      context, FlatSymbolRefAttr::get(implOp.getSymNameAttr()));
  auto instOp = builder.create<SimInstantiateOp>(loc);
  auto *instBody =
      builder.createBlock(&instOp.getBody(), {}, {instType}, {loc});
  auto instArg = instBody->getArgument(0);

  // Create an `scf.execute_region` op inside such that we can use simple
  // control flow in the `arc.sim.instantiate` op body. This is simpler than
  // setting up an `scf.while` op.
  auto execOp = builder.create<scf::ExecuteRegionOp>(loc, TypeRange{});
  builder.setInsertionPointToEnd(&execOp.getRegion().emplaceBlock());

  // Apply the initial clock tick to the design.
  builder.create<SimSetInputOp>(loc, instArg, clockName, lowOp);
  builder.create<SimSetInputOp>(loc, instArg, initName, trueOp);
  builder.create<SimStepOp>(loc, instArg);
  builder.create<SimSetInputOp>(loc, instArg, clockName, highOp);
  builder.create<SimStepOp>(loc, instArg);
  builder.create<SimSetInputOp>(loc, instArg, clockName, lowOp);
  builder.create<SimSetInputOp>(loc, instArg, initName, falseOp);
  builder.create<SimStepOp>(loc, instArg);

  // Create the block that will perform a single clock tick.
  auto &loopBlock = execOp.getRegion().emplaceBlock();
  builder.create<cf::BranchOp>(loc, &loopBlock);
  builder.setInsertionPointToEnd(&loopBlock);

  // Sample the done and success signals.
  auto doneSample =
      builder.create<SimGetPortOp>(loc, i1Type, instArg, doneName);
  auto successSample =
      builder.create<SimGetPortOp>(loc, i1Type, instArg, successName);

  // Apply a full clock cycle to the design.
  builder.create<SimSetInputOp>(loc, instArg, clockName, highOp);
  builder.create<SimStepOp>(loc, instArg);
  builder.create<SimSetInputOp>(loc, instArg, clockName, lowOp);
  builder.create<SimStepOp>(loc, instArg);

  // If done, exit the loop.
  auto &exitBlock = execOp.getRegion().emplaceBlock();
  builder.create<cf::CondBranchOp>(loc, doneSample, &exitBlock, &loopBlock);
  builder.setInsertionPointToEnd(&exitBlock);

  // Convert the i1 success signal into an i32 failure signal that can be used
  // as an exit code.
  auto i32Type = builder.getI32Type();
  auto failureI32 = builder.create<arith::ExtUIOp>(
      loc, i32Type,
      builder.create<arith::XOrIOp>(
          loc, successSample, builder.create<hw::ConstantOp>(loc, i1Type, 1)));

  // Call exit with the computed exit code.
  builder.create<func::CallOp>(loc, TypeRange{}, builder.getStringAttr("exit"),
                               ValueRange{failureI32});
  builder.create<scf::YieldOp>(loc);

  // Create the final function return.
  builder.setInsertionPointToEnd(funcBody);
  builder.create<func::ReturnOp>(loc);

  // Get rid of the original simulation op.
  op.erase();
}
