//===- Schedule.cpp - Schedule pass -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the Schedule pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Scheduling/Algorithms.h"

#include "llvm/ADT/StringExtras.h"

using namespace circt;
using namespace scheduling;
using namespace ssp;

//===----------------------------------------------------------------------===//
// Simplex schedulers
//===----------------------------------------------------------------------===//

template <typename ProblemT>
static InstanceOp scheduleProblemTWithSimplex(InstanceOp instOp,
                                              Operation *lastOp,
                                              OpBuilder &builder) {
  auto prob = loadProblem<ProblemT>(instOp);
  if (failed(prob.check()) ||
      failed(scheduling::scheduleSimplex(prob, lastOp)) ||
      failed(prob.verify()))
    return {};
  return saveProblem(prob, builder);
}

static InstanceOp scheduleChainingProblemWithSimplex(InstanceOp instOp,
                                                     Operation *lastOp,
                                                     float cycleTime,
                                                     OpBuilder &builder) {
  auto prob = loadProblem<scheduling::ChainingProblem>(instOp);
  if (failed(prob.check()) ||
      failed(scheduling::scheduleSimplex(prob, lastOp, cycleTime)) ||
      failed(prob.verify()))
    return {};
  return saveProblem(prob, builder);
}

static InstanceOp scheduleWithSimplex(InstanceOp instOp, StringRef options,
                                      OpBuilder &builder) {
  // Parse options.
  StringRef lastOpName = "";
  float cycleTime = 0.0f;
  for (StringRef option : llvm::split(options, ',')) {
    if (option.empty())
      continue;
    if (option.consume_front("last-op-name=")) {
      lastOpName = option;
      continue;
    }
    if (option.consume_front("cycle-time=")) {
      cycleTime = std::stof(option.str());
      continue;
    }
    llvm::errs() << "ssp-schedule: Ignoring option '" << option
                 << "' for simplex scheduler\n";
  }

  // Determine "last" operation, i.e. the one whose start time is minimized by
  // the simplex scheduler.
  auto graph = instOp.getDependenceGraph();
  OperationOp lastOp;
  if (lastOpName.empty() && !graph.getBodyBlock()->empty())
    lastOp = cast<OperationOp>(graph.getBodyBlock()->back());
  else
    lastOp = graph.lookupSymbol<OperationOp>(lastOpName);

  if (!lastOp) {
    auto instName = instOp.getSymName().value_or("unnamed");
    llvm::errs()
        << "ssp-schedule: Ambiguous objective for simplex scheduler: Instance '"
        << instName << "' has no designated last operation\n";
    return {};
  }

  // Dispatch for known problems.
  auto problemName = instOp.getProblemName();
  if (problemName.equals("Problem"))
    return scheduleProblemTWithSimplex<Problem>(instOp, lastOp, builder);
  if (problemName.equals("CyclicProblem"))
    return scheduleProblemTWithSimplex<CyclicProblem>(instOp, lastOp, builder);
  if (problemName.equals("SharedOperatorsProblem"))
    return scheduleProblemTWithSimplex<SharedOperatorsProblem>(instOp, lastOp,
                                                               builder);
  if (problemName.equals("ModuloProblem"))
    return scheduleProblemTWithSimplex<ModuloProblem>(instOp, lastOp, builder);
  if (problemName.equals("ChainingProblem"))
    return scheduleChainingProblemWithSimplex(instOp, lastOp, cycleTime,
                                              builder);

  llvm::errs() << "ssp-schedule: Unsupported problem '" << problemName
               << "' for simplex scheduler\n";
  return {};
}

static InstanceOp scheduleWith(InstanceOp instOp, StringRef scheduler,
                               StringRef options, OpBuilder &builder) {
  if (scheduler.empty() || scheduler.equals("simplex"))
    return scheduleWithSimplex(instOp, options, builder);

  llvm::errs() << "ssp-schedule: Unsupported scheduler '" << scheduler
               << "' requested\n";
  return {};
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

namespace {
struct SchedulePass : public ScheduleBase<SchedulePass> {
  void runOnOperation() override;
};
} // end anonymous namespace

void SchedulePass::runOnOperation() {
  auto moduleOp = getOperation();

  SmallVector<InstanceOp> instanceOps;
  OpBuilder builder(&getContext());
  for (auto instOp : moduleOp.getOps<InstanceOp>()) {
    builder.setInsertionPoint(instOp);
    auto scheduledOp = scheduleWith(instOp, scheduler.getValue(),
                                    schedulerOptions.getValue(), builder);
    if (!scheduledOp)
      return signalPassFailure();
    instanceOps.push_back(instOp);
  }

  llvm::for_each(instanceOps, [](InstanceOp op) { op.erase(); });
}

std::unique_ptr<mlir::Pass> circt::ssp::createSchedulePass() {
  return std::make_unique<SchedulePass>();
}
