//===- InlineSequencesPass.cpp - RTG InlineSequencesPass implementation ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass inlines sequences and computes sequence interleavings.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "circt/Dialect/RTG/Transforms/RTGPasses.h"
#include "mlir/IR/IRMapping.h"

namespace circt {
namespace rtg {
#define GEN_PASS_DEF_INLINESEQUENCESPASS
#include "circt/Dialect/RTG/Transforms/RTGPasses.h.inc"
} // namespace rtg
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::rtg;

//===----------------------------------------------------------------------===//
// Inline Sequences Pass
//===----------------------------------------------------------------------===//

namespace {
struct InlineSequencesPass
    : public rtg::impl::InlineSequencesPassBase<InlineSequencesPass> {
  using Base::Base;

  void runOnOperation() override;
  LogicalResult inlineSequences(TestOp testOp, SymbolTable &table);
};
} // namespace

LogicalResult InlineSequencesPass::inlineSequences(TestOp testOp,
                                                   SymbolTable &table) {
  OpBuilder builder(testOp);
  for (auto iter = testOp.getBody()->begin();
       iter != testOp.getBody()->end();) {
    auto embedOp = dyn_cast<EmbedSequenceOp>(&*iter);
    if (!embedOp) {
      ++iter;
      continue;
    }

    auto randSeqOp = embedOp.getSequence().getDefiningOp<RandomizeSequenceOp>();
    if (!randSeqOp)
      return embedOp->emitError("sequence operand not directly defined by "
                                "'rtg.randomize_sequence' op");
    auto getSeqOp = randSeqOp.getSequence().getDefiningOp<GetSequenceOp>();
    if (!getSeqOp)
      return randSeqOp->emitError(
          "sequence operand not directly defined by 'rtg.get_sequence' op");

    auto seqOp = table.lookup<SequenceOp>(getSeqOp.getSequenceAttr());

    builder.setInsertionPointAfter(embedOp);
    IRMapping mapping;
    for (auto &op : *seqOp.getBody())
      builder.clone(op, mapping);

    (iter++)->erase();

    if (randSeqOp->use_empty())
      randSeqOp->erase();

    if (getSeqOp->use_empty())
      getSeqOp->erase();

    ++numSequencesInlined;
  }

  return success();
}

void InlineSequencesPass::runOnOperation() {
  auto moduleOp = getOperation();
  SymbolTable table(moduleOp);

  // Inline all sequences and remove the operations that place the sequences.
  for (auto testOp : moduleOp.getOps<TestOp>())
    if (failed(inlineSequences(testOp, table)))
      return;

  // Remove all sequences since they are not accessible from the outside and
  // are not needed anymore since we fully inlined them.
  for (auto seqOp : llvm::make_early_inc_range(moduleOp.getOps<SequenceOp>()))
    seqOp->erase();
}
