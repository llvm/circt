//===- LowerHWTriggered.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Proof-of-concept lowering of top-level hw.triggered ops to arc.clock_domain
// ops that contain a single arc.execute op.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace circt;
using namespace arc;

namespace circt {
namespace arc {
#define GEN_PASS_DEF_LOWERHWTRIGGERED
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

namespace {

static void cloneBodyOperations(Block *source, Block *dest, OpBuilder &builder,
                                IRMapping &mapping) {
  for (Operation &op : *source) {
    if (dest->mightHaveTerminator()) {
      if (auto *terminator = dest->getTerminator())
        builder.setInsertionPoint(terminator);
      else
        builder.setInsertionPointToEnd(dest);
    } else {
      builder.setInsertionPointToEnd(dest);
    }
    auto *cloned = builder.clone(op, mapping);
    for (auto [original, result] :
         llvm::zip(op.getResults(), cloned->getResults()))
      mapping.map(original, result);
  }
}

struct LowerHWTriggeredPass
    : public arc::impl::LowerHWTriggeredBase<LowerHWTriggeredPass> {
  using LowerHWTriggeredBase::LowerHWTriggeredBase;

  void runOnOperation() override;

private:
  LogicalResult lowerTriggered(hw::TriggeredOp op);
};

} // namespace

LogicalResult LowerHWTriggeredPass::lowerTriggered(hw::TriggeredOp op) {
  if (op.getEvent() != hw::EventControl::AtPosEdge)
    return op.emitOpError(
        "only posedge hw.triggered is supported by this Arc lowering "
        "proof-of-concept");

  auto trigger = op.getTrigger().getDefiningOp<seq::FromClockOp>();
  if (!trigger)
    return op.emitOpError(
        "expected trigger to be produced by seq.from_clock for Arc lowering");

  WalkResult containsUnsupportedSimOp = op.walk([&](Operation *nested) {
    if (nested == op)
      return WalkResult::advance();
    if (!isa_and_nonnull<circt::sim::SimDialect>(nested->getDialect()))
      return WalkResult::advance();
    nested->emitOpError(
        "Sim dialect ops inside hw.triggered are unsupported by this "
        "Arc lowering proof-of-concept");
    return WalkResult::interrupt();
  });
  if (containsUnsupportedSimOp.wasInterrupted())
    return failure();

  OpBuilder builder(op);
  SmallVector<Value> inputs(op.getInputs());
  auto domain = ClockDomainOp::create(builder, op.getLoc(), TypeRange{}, inputs,
                                      trigger.getInput());
  if (domain.getBody().empty())
    builder.createBlock(&domain.getBody(), {}, ValueRange(inputs).getTypes(),
                        SmallVector<Location>(inputs.size(), op.getLoc()));
  auto &domainBlock = domain.getBody().front();

  builder.setInsertionPointToStart(&domainBlock);
  SmallVector<Value> domainArgs(domainBlock.getArguments());
  auto execute =
      ExecuteOp::create(builder, op.getLoc(), TypeRange{}, domainArgs);
  if (execute.getBody().empty())
    builder.createBlock(&execute.getBody(), {},
                        ValueRange(domainArgs).getTypes(),
                        SmallVector<Location>(domainArgs.size(), op.getLoc()));
  auto &executeBlock = execute.getBody().front();

  IRMapping mapping;
  for (auto [oldArg, newArg] :
       llvm::zip(op.getInnerInputs(), executeBlock.getArguments()))
    mapping.map(oldArg, newArg);

  cloneBodyOperations(op.getBodyBlock(), &executeBlock, builder, mapping);
  builder.setInsertionPointToEnd(&executeBlock);
  arc::OutputOp::create(builder, op.getLoc(), ValueRange{});
  builder.setInsertionPointToEnd(&domainBlock);
  arc::OutputOp::create(builder, op.getLoc(), ValueRange{});

  op.erase();
  if (trigger->use_empty())
    trigger.erase();
  return success();
}

void LowerHWTriggeredPass::runOnOperation() {
  hw::HWModuleOp module = getOperation();
  SmallVector<hw::TriggeredOp> triggeredOps;
  for (Operation &op : module.getBodyBlock()->without_terminator())
    if (auto triggered = dyn_cast<hw::TriggeredOp>(op))
      triggeredOps.push_back(triggered);

  if (triggeredOps.empty()) {
    markAllAnalysesPreserved();
    return;
  }

  for (auto triggered : triggeredOps)
    if (failed(lowerTriggered(triggered))) {
      signalPassFailure();
      return;
    }
}
