//===- IbisPrepareScheduling.cpp - Prepares ibis static blocks for scheduling //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Dialect/Ibis/IbisDialect.h"
#include "circt/Dialect/Ibis/IbisOps.h"
#include "circt/Dialect/Ibis/IbisPasses.h"
#include "circt/Dialect/Ibis/IbisTypes.h"
#include "circt/Dialect/Pipeline/PipelineOps.h"

#include "circt/Support/Namespace.h"
#include "circt/Support/SymCache.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace ibis;

namespace {

struct PrepareSchedulingPass
    : public IbisPrepareSchedulingBase<PrepareSchedulingPass> {
  void runOnOperation() override;

  void prepareSBlock(IsolatedStaticBlockOp sblock);
};
} // anonymous namespace

void PrepareSchedulingPass::runOnOperation() { prepareSBlock(getOperation()); }

void PrepareSchedulingPass::prepareSBlock(IsolatedStaticBlockOp sblock) {
  Location loc = sblock.getLoc();
  Block *bodyBlock = sblock.getBodyBlock();
  auto b = OpBuilder::atBlockBegin(bodyBlock);
  auto ph = b.create<ibis::PipelineHeaderOp>(loc);

  // Create a pipeline.unscheduled operation which returns the same types
  // as that returned by the sblock.
  auto sblockRet = cast<BlockReturnOp>(bodyBlock->getTerminator());
  auto retTypes = sblockRet.getOperandTypes();

  // Generate in- and output names.
  SmallVector<Attribute> inNames, outNames;
  for (size_t i = 0, e = bodyBlock->getNumArguments(); i < e; ++i)
    inNames.push_back(b.getStringAttr("in" + std::to_string(i)));
  for (size_t i = 0, e = retTypes.size(); i < e; ++i)
    outNames.push_back(b.getStringAttr("out" + std::to_string(i)));

  auto pipeline = b.create<pipeline::UnscheduledPipelineOp>(
      loc, retTypes, bodyBlock->getArguments(), b.getArrayAttr(inNames),
      b.getArrayAttr(outNames), ph.getClock(), ph.getReset(), ph.getGo(),
      ph.getStall());
  b.setInsertionPointToEnd(pipeline.getEntryStage());

  // First, we replace all of the operands of the return op with the values
  // generated by the pipeline. This ensures that argument of the sblock that
  // is directly returned without being modified by an operation inside the
  // sblock is still being passed through the pipeline. While doing so, we
  // sneakily also set the pipeline return values so that it will reflect the
  // later value replacements.
  auto pipelineRet = b.create<pipeline::ReturnOp>(loc, sblockRet.getOperands());
  for (size_t i = 0, e = retTypes.size(); i < e; ++i)
    sblockRet.setOperand(i, pipeline.getResult(i));

  // Next, we can replace all of the sblock argument uses within the pipeline
  // with the pipeline arguments.
  for (auto [sbArg, plArg] :
       llvm::zip(bodyBlock->getArguments(),
                 pipeline.getEntryStage()->getArguments())) {
    sbArg.replaceAllUsesExcept(plArg, pipeline);
  }

  // And now we're safe to move the body of the sblock into the pipeline.
  // Drop the 2 first ops (pipeline, pipeline header) and the back (the return
  // op). Block::getOperations doesn't play nicely with ArrayRef's so have to
  // copy it...
  llvm::SmallVector<Operation *> opsToMove;
  llvm::transform(bodyBlock->getOperations(), std::back_inserter(opsToMove),
                  [&](Operation &op) { return &op; });
  for (Operation *op :
       ArrayRef(opsToMove.begin(), opsToMove.end()).drop_front(2).drop_back())
    op->moveBefore(pipelineRet);
}

std::unique_ptr<Pass> circt::ibis::createPrepareSchedulingPass() {
  return std::make_unique<PrepareSchedulingPass>();
}
