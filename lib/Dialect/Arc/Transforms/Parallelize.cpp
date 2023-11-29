//===- Parallelize.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GraphWriter.h"

#define DEBUG_TYPE "arc-parallelize"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_PARALLELIZE
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace circt;
using namespace arc;
using namespace mlir;

using llvm::MapVector;
using llvm::SmallMapVector;
using llvm::SmallSetVector;

/// Check whether a block is an ancestor of another block.
static bool isAncestorBlock(Block *ancestor, Block *of) {
  while (of) {
    if (of == ancestor)
      return true;
    of = of->getParentOp()->getBlock();
  }
  return false;
}

static bool isProperAncestorBlock(Block *ancestor, Block *of) {
  return ancestor != of && isAncestorBlock(ancestor, of);
}

/// Check whether an `op` is defined outside of the given `block`.
static bool isOutsideOfBlock(Operation *op, Block *block) {
  Block *opBlock = op->getBlock();
  while (opBlock) {
    if (opBlock == block)
      return false;
    opBlock = opBlock->getParentOp()->getBlock();
  }
  return true;
}

/// Check whether a `value` has uses outside of the given `block`.
static bool isUsedOutsideOfBlock(Value value, Block *block) {
  for (auto *user : value.getUsers())
    if (isOutsideOfBlock(user, block))
      return true;
  return false;
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct ParallelizePass : public arc::impl::ParallelizeBase<ParallelizePass> {
  using ParallelizeBase::ParallelizeBase;

  void runOnOperation() override;
  LogicalResult runOnClockTree(ClockTreeOp clockTreeOp);
  void dumpGraph(ClockTreeOp clockTreeOp);
  unsigned getComplexity(Operation *op);

  SymbolTableCollection *symbolTableCollection;
  DenseMap<Operation *, unsigned> complexityCache;
};
} // namespace

void ParallelizePass::runOnOperation() {
  complexityCache.clear();
  SymbolTableCollection symtbls;
  symbolTableCollection = &symtbls;

  getOperation().walk<WalkOrder::PreOrder>([&](ClockTreeOp op) {
    if (failed(runOnClockTree(op))) {
      signalPassFailure();
      return WalkResult::interrupt();
    }
    return WalkResult::skip();
  });
}

LogicalResult ParallelizePass::runOnClockTree(ClockTreeOp clockTreeOp) {
  // Inline group operations with large complexity such that we get a graph of
  // operations with reasonably-sized chunks of work.
  LLVM_DEBUG(llvm::dbgs() << "Inline complex groups\n");
  SmallVector<Operation *> opsToDelete;
  for (auto &op : clockTreeOp.getBodyBlock()) {
    auto groupOp = dyn_cast<GroupOp>(&op);
    if (!groupOp)
      continue;
    unsigned complexity = getComplexity(groupOp);
    if (complexity < maxTaskComplexity)
      continue;
    LLVM_DEBUG(llvm::dbgs()
               << "- Inlining arc.group of complexity " << complexity << "\n");

    auto *terminator = groupOp.getBody().front().getTerminator();
    groupOp.replaceAllUsesWith(terminator->getOperands());
    terminator->erase();
    clockTreeOp.getBodyBlock().getOperations().splice(
        ++groupOp->getIterator(), groupOp.getBody().front().getOperations());
    opsToDelete.push_back(groupOp);
  }
  for (auto *op : opsToDelete)
    op->erase();

  // Compute the topological order over the operations in the block.
  LLVM_DEBUG(llvm::dbgs() << "Compute topological order\n");
  MapVector<Operation *, unsigned> topoOrder;
  SmallMapVector<Operation *, Operation::user_iterator, 16> worklist;
  auto opsIt = clockTreeOp.getBodyBlock().begin();
  auto opsEnd = clockTreeOp.getBodyBlock().end();
  while (!worklist.empty() || opsIt != opsEnd) {
    if (worklist.empty()) {
      auto *op = &*(opsIt++);
      if (!topoOrder.contains(op))
        worklist.insert({op, op->user_begin()});
      continue;
    }
    auto &[op, userIt] = worklist.back();
    if (userIt != op->user_end()) {
      auto *user = *(userIt++);
      while (user->getBlock() != op->getBlock())
        user = user->getParentOp();
      if (!topoOrder.contains(user))
        worklist.insert({user, user->user_begin()});
      continue;
    }
    unsigned rank = isa<MemoryWriteOp>(op) ? 0 : 1;
    for (auto *user : op->getUsers()) {
      while (user->getBlock() != op->getBlock())
        user = user->getParentOp();
      rank = std::max(rank, topoOrder.lookup(user) + 1);
    }
    topoOrder.insert({op, rank});
    worklist.pop_back();
  }

  // Group operations by rank.
  MapVector<unsigned, SmallVector<Operation *>> ranks;
  for (auto [op, rank] : topoOrder)
    ranks[rank].push_back(op);
  dumpGraph(clockTreeOp);

  // Move grouped operations into parallel sections and order ops by rank.
  LLVM_DEBUG(llvm::dbgs() << "Organizing ops by rank\n");
  OpBuilder builder(clockTreeOp);
  builder.setInsertionPointToEnd(&clockTreeOp.getBodyBlock());
  for (auto &[rank, ops] : llvm::reverse(ranks)) {
    LLVM_DEBUG(llvm::dbgs()
               << "- Rank " << rank << " has " << ops.size() << " ops\n");

    // Don't form a parallel region if there aren't enough ops in the group, or
    // this is special rank 0 where all memory writes live.
    if (rank == 0 || ops.size() == 1) {
      for (auto *op : ops) {
        op->remove();
        builder.insert(op);
      }
      continue;
    }

    auto parallelOp = builder.create<omp::ParallelOp>(clockTreeOp.getLoc());
    auto &parallelBlock = parallelOp.getRegion().emplaceBlock();
    OpBuilder sectionBuilder(parallelOp);
    OpBuilder allocaBuilder(parallelOp);
    sectionBuilder.setInsertionPointToEnd(&parallelBlock);
    auto sectionsOp = sectionBuilder.create<omp::SectionsOp>(
        clockTreeOp.getLoc(), TypeRange{}, ValueRange{});
    auto &sectionsBlock = sectionsOp.getRegion().emplaceBlock();
    sectionBuilder.create<omp::TerminatorOp>(clockTreeOp.getLoc(),
                                             ValueRange{});
    sectionBuilder.setInsertionPointToEnd(&sectionsBlock);

    for (auto *op : ops) {
      auto sectionOp = sectionBuilder.create<omp::SectionOp>(op->getLoc());
      OpBuilder::InsertionGuard guard(sectionBuilder);
      sectionBuilder.setInsertionPointToEnd(
          &sectionOp.getRegion().emplaceBlock());
      op->remove();
      sectionBuilder.insert(op);
      for (auto result : op->getResults()) {
        auto allocaOp = allocaBuilder.create<LLVM::AllocaOp>(
            result.getLoc(), LLVM::LLVMPointerType::get(builder.getContext()),
            result.getType(),
            allocaBuilder.create<hw::ConstantOp>(result.getLoc(),
                                                 builder.getI32Type(), 1));
        auto loadOp = builder.create<LLVM::LoadOp>(result.getLoc(),
                                                   result.getType(), allocaOp);
        result.replaceAllUsesWith(loadOp);
        sectionBuilder.create<LLVM::StoreOp>(result.getLoc(), result, allocaOp);
      }
      sectionBuilder.create<omp::TerminatorOp>(op->getLoc(), ValueRange{});
    }
    sectionBuilder.create<omp::TerminatorOp>(clockTreeOp.getLoc(),
                                             ValueRange{});
  }

  return success();
}

void ParallelizePass::dumpGraph(ClockTreeOp clockTreeOp) {
  // Dump the computation structure as a graph.
  std::error_code ec;
  llvm::raw_fd_ostream os("graph.dot", ec);
  os << "digraph {\n";

  SmallVector<Operation *> ops;
  SmallVector<Operation *> joinOps;
  DenseMap<std::pair<Operation *, Operation *>, unsigned> weights;
  clockTreeOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op == clockTreeOp)
      return WalkResult::advance();
    ops.push_back(op);
    return WalkResult::skip();
    // if (isa<arc::OutputOp>(op))
    //   weights[{op, op->getParentOp()}] += op->getNumOperands();
    // if (getComplexity(op) < 1000) {
    //   ops.push_back(op);
    //   return WalkResult::skip();
    // }
    // joinOps.push_back(op);
    // return WalkResult::advance();
  });

  for (auto *parentOp : ops) {
    parentOp->walk([&](Operation *op) {
      for (auto operand : op->getOperands()) {
        auto *defOp = operand.getDefiningOp();
        if (!defOp)
          continue;
        if (isProperAncestorBlock(operand.getParentBlock(),
                                  &clockTreeOp.getBodyBlock()))
          continue;
        if (isProperAncestorBlock(parentOp->getBlock(),
                                  operand.getParentBlock()))
          continue;
        ++weights[{defOp, parentOp}];
      }
    });
  }
  for (auto *op : joinOps)
    ops.push_back(op);

  for (auto *op : ops) {
    os << "op" << op << " [label=\"" << op->getName() << " ("
       << getComplexity(op) << ")";
    if (auto loc = dyn_cast_or_null<FileLineColLoc>(op->getLoc()))
      os << " " << loc.getLine();
    os << "\"];\n";
  }
  for (auto [ops, weight] : weights) {
    os << "op" << ops.first << " -> op" << ops.second << " [label=\"" << weight
       << "\"];\n";
  }

  os << "}\n";
}

unsigned ParallelizePass::getComplexity(Operation *op) {
  if (auto it = complexityCache.find(op); it != complexityCache.end())
    return it->second;
  unsigned complexity = 1;
  for (auto &region : op->getRegions())
    for (auto &block : region)
      for (auto &subOp : block)
        complexity += getComplexity(&subOp);
  if (auto callOp = dyn_cast<CallOpInterface>(op))
    complexity += getComplexity(callOp.resolveCallable(symbolTableCollection));
  complexityCache.insert({op, complexity});
  return complexity;
}

std::unique_ptr<Pass>
arc::createParallelizePass(const ParallelizeOptions &options) {
  return std::make_unique<ParallelizePass>(options);
}
