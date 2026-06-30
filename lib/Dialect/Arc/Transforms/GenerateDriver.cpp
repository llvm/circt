//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass generates a self-contained simulation driver for every `arc.model`
// in the module, so that the model can be run directly through the arcilator
// JIT without a hand-written driver loop.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Arc/ArcTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_GENERATEDRIVER
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace arc;

//===----------------------------------------------------------------------===//
// Generate Simulation Drivers
//===----------------------------------------------------------------------===//

namespace {
struct GenerateDriverPass
    : public arc::impl::GenerateDriverBase<GenerateDriverPass> {
  void runOnOperation() override;
  void generateDriver(ModelOp modelOp, SymbolTable &symbolTable);
};
} // namespace

void GenerateDriverPass::runOnOperation() {
  auto &symbolTable = getAnalysis<SymbolTable>();
  for (auto modelOp : getOperation().getOps<ModelOp>())
    generateDriver(modelOp, symbolTable);
}

/// Generate a `func.func @<model>_main` that runs the given model to
/// completion.
void GenerateDriverPass::generateDriver(ModelOp modelOp,
                                        SymbolTable &symbolTable) {
  auto *context = &getContext();
  auto loc = modelOp.getLoc();
  auto i64Type = IntegerType::get(context, 64);
  auto instanceType = SimModelInstanceType::get(
      context, FlatSymbolRefAttr::get(modelOp.getSymNameAttr()));

  // Create the driver function as a sibling placed before the model. The symbol
  // table uniquifies the name if it happens to collide.
  OpBuilder builder(modelOp);
  auto funcName = builder.getStringAttr(modelOp.getSymName() + "_main");
  auto funcType = builder.getFunctionType({}, {});
  auto funcOp = func::FuncOp::create(builder, loc, funcName, funcType);
  symbolTable.insert(funcOp);
  builder.setInsertionPointToStart(funcOp.addEntryBlock());

  // Instantiate the model. Its lifetime spans the body region, in which the
  // model storage is available as the instance block argument.
  auto instanceOp = SimInstantiateOp::create(builder, loc);
  func::ReturnOp::create(builder, loc);
  auto *instanceBlock = builder.createBlock(&instanceOp.getBody());
  auto instance = instanceBlock->addArgument(instanceType, loc);

  // Loop until no further wakeup is scheduled. The next wakeup slot uses
  // `UINT64_MAX` to signal that the model has gone quiescent.
  auto never = arith::ConstantOp::create(builder, loc,
                                         builder.getIntegerAttr(i64Type, -1));
  auto firstWakeup = SimGetNextWakeupOp::create(builder, loc, instance);

  auto whileOp =
      scf::WhileOp::create(builder, loc, i64Type, ValueRange{firstWakeup});

  // The "before" region tests whether a wakeup is still pending and forwards
  // the wakeup time into the loop body.
  auto *beforeBlock = builder.createBlock(&whileOp.getBefore());
  auto wakeup = beforeBlock->addArgument(i64Type, loc);
  auto pending = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne,
                                       wakeup, never);
  scf::ConditionOp::create(builder, loc, pending, ValueRange{wakeup});

  // The "after" region advances time to the scheduled wakeup, evaluates the
  // model, and reads the next wakeup time to feed back into the condition.
  auto *afterBlock = builder.createBlock(&whileOp.getAfter());
  auto resumeTime = afterBlock->addArgument(i64Type, loc);
  SimSetTimeOp::create(builder, loc, instance, resumeTime);
  SimStepOp::create(builder, loc, instance);
  auto nextWakeup = SimGetNextWakeupOp::create(builder, loc, instance);
  scf::YieldOp::create(builder, loc, ValueRange{nextWakeup});
}
