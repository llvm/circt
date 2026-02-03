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

  SequenceInliner(ModuleOp moduleOp, bool failOnRemaining)
      : table(moduleOp), failOnRemaining(failOnRemaining) {}

  LogicalResult inlineSequences(Block &block);
  void materializeInterleavedSequence(Value value, ArrayRef<Block *> blocks,
                                      uint32_t batchSize);

  FailureOr<std::pair<Block *, IRMapping>>
  getMaterializedSequence(Value seq, Location loc) {
    auto iter = materializedSequences.find(seq);
    if (iter == materializedSequences.end()) {
      StringLiteral msg = "sequence operand could not be resolved; it "
                          "was likely produced by an op or block "
                          "argument not supported by this pass";
      if (failOnRemaining)
        return mlir::emitError(loc, msg);

      LLVM_DEBUG(llvm::dbgs() << msg << "\n");
      return failure();
    }

    return iter->getSecond();
  }

  // Visitor methods

  FailureOr<DeletionKind> visitOp(InterleaveSequencesOp op) {
    SmallVector<Block *> blocks;
    for (auto [i, seq] : llvm::enumerate(op.getSequences())) {
      auto res = getMaterializedSequence(seq, op.getLoc());
      if (failed(res))
        return failure();

      blocks.push_back(res->first);
    }

    LLVM_DEBUG(llvm::dbgs()
               << "  - Computing sequence interleaving: " << op << "\n");

    materializeInterleavedSequence(op.getInterleavedSequence(), blocks,
                                   op.getBatchSize());
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(GetSequenceOp op) {
    auto seqOp = table.lookup<SequenceOp>(op.getSequenceAttr().getAttr());
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

    auto res = getMaterializedSequence(op.getSequence(), op.getLoc());
    if (failed(res))
      return failure();

    IRMapping mapping = res->second;
    Block *block = res->first;
    for (auto [arg, repl] :
         llvm::zip(block->getArguments(), op.getReplacements())) {
      LLVM_DEBUG(llvm::dbgs()
                 << "    - Mapping " << arg << " to " << repl << "\n");
      mapping.map(arg, repl);
    }

    materializedSequences[op.getResult()] = std::make_pair(block, mapping);
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(RandomizeSequenceOp op) {
    LLVM_DEBUG(llvm::dbgs() << "  - Randomize sequence: " << op << "\n");

    auto res = getMaterializedSequence(op.getSequence(), op.getLoc());
    if (failed(res))
      return failure();

    // It's important to force a copy here. Without the temporary variable, we'd
    // assign an lvalue.
    materializedSequences[op.getResult()] = *res;
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(EmbedSequenceOp op) {
    LLVM_DEBUG(llvm::dbgs() << "  - Inlining sequence: " << op << "\n");

    auto res = getMaterializedSequence(op.getSequence(), op.getLoc());
    if (failed(res))
      return failure();

    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);
    IRMapping mapping = res->second;

    LLVM_DEBUG({
      for (auto [k, v] : mapping.getValueMap())
        llvm::dbgs() << "    - Maps " << k << " to " << v << "\n";
    });

    for (auto &op : *res->first) {
      Operation *o = builder.clone(op, mapping);
      (void)o;
      LLVM_DEBUG(llvm::dbgs() << "    - Inlined " << *o << "\n");
    }

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
  bool failOnRemaining;
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

// NOLINTNEXTLINE(misc-no-recursion)
LogicalResult SequenceInliner::inlineSequences(Block &block) {
  // Make sure we inline sequences in nested regions. Walk doesn't work here
  // because it uses 'early_inc_range' which means we'd skip sequence
  // embeddings that we added with the previous inlining.
  SmallVector<Operation *> toDelete;
  for (auto &op : block) {
    for (auto &region : op.getRegions())
      for (auto &block : region)
        if (failOnRemaining && failed(inlineSequences(block)))
          return failure();

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
  SequenceInliner inliner(moduleOp, failOnRemaining);

  // Fast-path: no sequences are defined.
  if (moduleOp.getOps<SequenceOp>().empty())
    return;

  // Fast-path: no sequences are defined.
  if (moduleOp.getOps<SequenceOp>().empty())
    return;

  // Inline all sequences and remove the operations that place the sequences.
  for (auto testOp : moduleOp.getOps<TestOp>()) {
    auto res = inliner.inlineSequences(*testOp.getBody());
    if (failOnRemaining && failed(res))
      return signalPassFailure();
  }

  numSequencesInlined = inliner.numSequencesInlined;
  numSequencesInterleaved = inliner.numSequencesInterleaved;
}
