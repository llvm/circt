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

#include "circt/Dialect/RTG/IR/RTGISAAssemblyOpInterfaces.h"
#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "circt/Dialect/RTG/IR/RTGVisitors.h"
#include "circt/Dialect/RTG/Transforms/RTGPasses.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/Debug.h"

namespace circt {
namespace rtg {
#define GEN_PASS_DEF_INLINESEQUENCESPASS
#include "circt/Dialect/RTG/Transforms/RTGPasses.h.inc"
} // namespace rtg
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::rtg;

#define DEBUG_TYPE "rtg-inline-sequences"

//===----------------------------------------------------------------------===//
// Inline Sequences Pass
//===----------------------------------------------------------------------===//

namespace {
struct InlineSequencesPass
    : public rtg::impl::InlineSequencesPassBase<InlineSequencesPass> {
  using Base::Base;

  void runOnOperation() override;
};

/// Enum to indicate to the visitor driver whether the operation should be
/// deleted.
enum class DeletionKind { Delete, Keep };

/// The SequenceInliner computes sequence interleavings and inlines them.
struct SequenceInliner
    : public RTGOpVisitor<SequenceInliner, FailureOr<DeletionKind>> {
  using RTGOpVisitor<SequenceInliner, FailureOr<DeletionKind>>::visitOp;

  SequenceInliner(ModuleOp moduleOp) : table(moduleOp) {}

  LogicalResult inlineSequences(TestOp testOp);
  void materializeInterleavedSequence(Value value, ArrayRef<Block *> blocks,
                                      uint32_t batchSize);

  // Visitor methods

  FailureOr<DeletionKind> visitOp(InterleaveSequencesOp op) {
    SmallVector<Block *> blocks;
    for (auto [i, seq] : llvm::enumerate(op.getSequences())) {
      auto iter = materializedSequences.find(seq);
      if (iter == materializedSequences.end())
        return op->emitError()
               << "sequence operand #" << i
               << " could not be resolved; it was likely produced by an op or "
                  "block argument not supported by this pass";

      blocks.push_back(iter->getSecond().first);
    }

    LLVM_DEBUG(llvm::dbgs()
               << "  - Computing sequence interleaving: " << op << "\n");

    materializeInterleavedSequence(op.getInterleavedSequence(), blocks,
                                   op.getBatchSize());
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(GetSequenceOp op) {
    auto seqOp = table.lookup<SequenceOp>(op.getSequenceAttr());
    if (!seqOp)
      return op->emitError() << "referenced sequence not found";

    LLVM_DEBUG(llvm::dbgs() << "  - Registering existing sequence: "
                            << op.getSequence() << "\n");

    materializedSequences[op.getResult()] =
        std::make_pair(seqOp.getBody(), IRMapping());
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(SubstituteSequenceOp op) {
    LLVM_DEBUG(llvm::dbgs() << "  - Substitute sequence: " << op << "\n");

    auto iter = materializedSequences.find(op.getSequence());
    if (iter == materializedSequences.end())
      return op->emitError() << "sequence operand could not be resolved; it "
                                "was likely produced by an op or block "
                                "argument not supported by this pass";

    IRMapping mapping = iter->getSecond().second;
    Block *block = iter->getSecond().first;
    for (auto [arg, repl] :
         llvm::zip(block->getArguments(), op.getReplacements()))
      mapping.map(arg, repl);
    materializedSequences[op.getResult()] = std::make_pair(block, mapping);
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(RandomizeSequenceOp op) {
    LLVM_DEBUG(llvm::dbgs() << "  - Randomize sequence: " << op << "\n");

    auto iter = materializedSequences.find(op.getSequence());
    if (iter == materializedSequences.end())
      return op->emitError() << "sequence operand could not be resolved; it "
                                "was likely produced by an op or block "
                                "argument not supported by this pass";

    materializedSequences[op.getResult()] = iter->getSecond();
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(EmbedSequenceOp op) {
    LLVM_DEBUG(llvm::dbgs() << "  - Inlining sequence: " << op << "\n");

    auto iter = materializedSequences.find(op.getSequence());
    if (iter == materializedSequences.end())
      return op->emitError() << "sequence operand could not be resolved; it "
                                "was likely produced by an op or block "
                                "argument not supported by this pass";

    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);
    IRMapping mapping = iter->getSecond().second;
    for (auto &op : *iter->getSecond().first)
      builder.clone(op, mapping);

    ++numSequencesInlined;

    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitUnhandledOp(Operation *op) {
    return DeletionKind::Keep;
  }

  FailureOr<DeletionKind> visitExternalOp(Operation *op) {
    return DeletionKind::Keep;
  }

  SymbolTable table;
  DenseMap<Value, std::pair<Block *, IRMapping>> materializedSequences;
  SmallVector<std::unique_ptr<Block>> blockStorage;
  size_t numSequencesInlined = 0;
  size_t numSequencesInterleaved = 0;
};

} // namespace

void SequenceInliner::materializeInterleavedSequence(Value value,
                                                     ArrayRef<Block *> blocks,
                                                     uint32_t batchSize) {
  auto *interleavedBlock =
      blockStorage.emplace_back(std::make_unique<Block>()).get();
  IRMapping mapping;
  OpBuilder builder(value.getContext());
  builder.setInsertionPointToStart(interleavedBlock);

  SmallVector<Block::iterator> iters(blocks.size());
  for (auto [i, block] : llvm::enumerate(blocks))
    iters[i] = block->begin();

  llvm::BitVector finishedBlocks(blocks.size());
  for (unsigned i = 0; !finishedBlocks.all(); i = (i + 1) % blocks.size()) {
    if (finishedBlocks[i])
      continue;
    for (unsigned k = 0; k < batchSize;) {
      if (iters[i] == blocks[i]->end()) {
        finishedBlocks.set(i);
        break;
      }
      auto *op = builder.clone(*iters[i], mapping);
      if (isa<InstructionOpInterface>(op))
        ++k;
      ++iters[i];
    }
  }

  materializedSequences[value] = std::make_pair(interleavedBlock, IRMapping());
  numSequencesInterleaved += blocks.size();
}

LogicalResult SequenceInliner::inlineSequences(TestOp testOp) {
  LLVM_DEBUG(llvm::dbgs() << "\n=== Processing test @" << testOp.getSymName()
                          << "\n\n");

  SmallVector<Operation *> toDelete;
  for (auto &op : *testOp.getBody()) {
    auto result = dispatchOpVisitor(&op);
    if (failed(result))
      return failure();

    if (*result == DeletionKind::Delete)
      toDelete.push_back(&op);
  }

  for (auto *op : llvm::reverse(toDelete))
    op->erase();

  return success();
}

void InlineSequencesPass::runOnOperation() {
  auto moduleOp = getOperation();
  SequenceInliner inliner(moduleOp);

  // Inline all sequences and remove the operations that place the sequences.
  for (auto testOp : moduleOp.getOps<TestOp>())
    if (failed(inliner.inlineSequences(testOp)))
      return signalPassFailure();

  // Remove all sequences since they are not accessible from the outside and
  // are not needed anymore since we fully inlined them.
  for (auto seqOp : llvm::make_early_inc_range(moduleOp.getOps<SequenceOp>()))
    seqOp->erase();

  numSequencesInlined = inliner.numSequencesInlined;
  numSequencesInterleaved = inliner.numSequencesInterleaved;
}
