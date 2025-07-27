//===- FindInitialVectors.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements a simple SLP vectorizer for Arc, the pass starts with
// `arc.state` operations as seeds in every new vector, then following the
// dependency graph nodes computes a rank to every operation in the module
// and assigns a rank to each one of them. After that it groups isomorphic
// operations together and put them in a vector.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

#define DEBUG_TYPE "find-initial-vectors"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_FINDINITIALVECTORS
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace circt;
using namespace arc;
using llvm::SmallMapVector;

namespace {
struct FindInitialVectorsPass
    : public impl::FindInitialVectorsBase<FindInitialVectorsPass> {
  void runOnOperation() override;

  struct StatisticVars {
    size_t vecOps{0};
    size_t savedOps{0};
    size_t bigSeedVec{0};
    size_t vecCreated{0};
  };

  StatisticVars stat;
};
} // namespace

namespace {
struct TopologicalOrder {
  /// An integer rank assigned to each operation.
  SmallMapVector<Operation *, unsigned, 32> opRanks;
  LogicalResult compute(Block *block);
  unsigned get(Operation *op) const {
    const auto *it = opRanks.find(op);
    assert(it != opRanks.end() && "op has no rank");
    return it->second;
  }
};
} // namespace

/// Assign each operation in the given block a topological rank. Stateful
/// elements are assigned rank 0. All other operations receive the maximum rank
/// of their users, plus one.
LogicalResult TopologicalOrder::compute(Block *block) {
  LLVM_DEBUG(llvm::dbgs() << "- Computing topological order in block " << block
                          << "\n");
  struct WorklistItem {
    WorklistItem(Operation *op) : userIt(op->user_begin()) {}
    Operation::user_iterator userIt;
    unsigned rank = 0;
  };
  SmallMapVector<Operation *, WorklistItem, 16> worklist;
  for (auto &op : *block) {
    if (opRanks.contains(&op))
      continue;
    worklist.insert({&op, WorklistItem(&op)});
    while (!worklist.empty()) {
      auto &[op, item] = worklist.back();
      if (auto stateOp = dyn_cast<StateOp>(op)) {
        if (stateOp.getLatency() > 0)
          item.userIt = op->user_end();
      } else if (auto writeOp = dyn_cast<MemoryWritePortOp>(op)) {
        item.userIt = op->user_end();
      }
      if (item.userIt == op->user_end()) {
        opRanks.insert({op, item.rank});
        worklist.pop_back();
        continue;
      }
      if (auto *rankIt = opRanks.find(*item.userIt); rankIt != opRanks.end()) {
        item.rank = std::max(item.rank, rankIt->second + 1);
        ++item.userIt;
        continue;
      }
      if (!worklist.insert({*item.userIt, WorklistItem(*item.userIt)}).second)
        return op->emitError("dependency cycle");
    }
  }
  return success();
}

namespace {
using Key = std::tuple<unsigned, StringRef, SmallVector<Type>,
                       SmallVector<Type>, DictionaryAttr>;

Key computeKey(Operation *op, unsigned rank) {
  // The key = concat(op_rank, op_name, op_operands_types, op_result_types,
  //                  op_attrs)
  return std::make_tuple(
      rank, op->getName().getStringRef(),
      SmallVector<Type>(op->operand_type_begin(), op->operand_type_end()),
      SmallVector<Type>(op->result_type_begin(), op->result_type_end()),
      op->getAttrDictionary());
}

struct Vectorizer {
  Vectorizer(Block *block) : block(block) {}
  LogicalResult collectSeeds(Block *block) {
    if (failed(order.compute(block)))
      return failure();

    for (auto &[op, rank] : order.opRanks)
      candidates[computeKey(op, rank)].push_back(op);

    return success();
  }

  LogicalResult vectorize(FindInitialVectorsPass::StatisticVars &stat);
  // Store Isomorphic ops together
  SmallMapVector<Key, SmallVector<Operation *>, 16> candidates;
  TopologicalOrder order;
  Block *block;
};
} // namespace

namespace llvm {
template <>
struct DenseMapInfo<Key> {
  static inline Key getEmptyKey() {
    return Key(0, StringRef(), SmallVector<Type>(), SmallVector<Type>(),
               DictionaryAttr());
  }

  static inline Key getTombstoneKey() {
    static StringRef tombStoneKeyOpName =
        DenseMapInfo<StringRef>::getTombstoneKey();
    return Key(1, tombStoneKeyOpName, SmallVector<Type>(), SmallVector<Type>(),
               DictionaryAttr());
  }

  static unsigned getHashValue(const Key &key) {
    return hash_value(std::get<0>(key)) ^ hash_value(std::get<1>(key)) ^
           hash_value(std::get<2>(key)) ^ hash_value(std::get<3>(key)) ^
           hash_value(std::get<4>(key));
  }

  static bool isEqual(const Key &lhs, const Key &rhs) { return lhs == rhs; }
};
} // namespace llvm

// When calling this function we assume that we have the candidate groups of
// isomorphic ops so we need to feed them to the `VectorizeOp`
LogicalResult
Vectorizer::vectorize(FindInitialVectorsPass::StatisticVars &stat) {
  LLVM_DEBUG(llvm::dbgs() << "- Vectorizing the ops in block" << block << "\n");

  if (failed(collectSeeds(block)))
    return failure();

  // Unachievable?! just in case!
  if (candidates.empty())
    return success();

  // Iterate over every group of isomorphic ops
  for (const auto &[key, ops] : candidates) {
    // If the group has only one scalar then it doesn't worth vectorizing,
    // We skip also ops with more than one result as `arc.vectorize` supports
    // only one result in its body region. Ignore zero-result and zero operands
    // ops as well.
    if (ops.size() == 1 || ops[0]->getNumResults() != 1 ||
        ops[0]->getNumOperands() == 0)
      continue;

    // Collect Statistics
    stat.vecOps += ops.size();
    stat.savedOps += ops.size() - 1;
    stat.bigSeedVec = std::max(ops.size(), stat.bigSeedVec);
    ++stat.vecCreated;

    // Here, we have a bunch of isomorphic ops, we need to extract the operands
    // results and attributes of every op and store them in a vector
    // Holds the operands
    SmallVector<SmallVector<Value, 4>> vectorOperands;
    vectorOperands.resize(ops[0]->getNumOperands());
    for (auto *op : ops)
      for (auto [into, operand] : llvm::zip(vectorOperands, op->getOperands()))
        into.push_back(operand);
    SmallVector<ValueRange> operandValueRanges;
    operandValueRanges.assign(vectorOperands.begin(), vectorOperands.end());
    // Holds the results
    SmallVector<Type> resultTypes(ops.size(), ops[0]->getResult(0).getType());

    // Now construct the `VectorizeOp`
    ImplicitLocOpBuilder builder(ops[0]->getLoc(), ops[0]);
    auto vectorizeOp =
        VectorizeOp::create(builder, resultTypes, operandValueRanges);

    // Now we have the operands, results and attributes, now we need to get
    // the blocks.

    // There was no blocks so we need to create one and set the insertion point
    // at the first of this region
    auto &vectorizeBlock = vectorizeOp.getBody().emplaceBlock();
    builder.setInsertionPointToStart(&vectorizeBlock);

    // Add the block arguments
    // comb.and %x, %y
    // comb.and %u, %v
    // at this point the operands vector will be {{x, u}, {y, v}}
    // we need to create an th block args, so we need the type and the location
    // the type is a vector type
    IRMapping argMapping;
    for (auto [vecOperand, origOpernad] :
         llvm::zip(vectorOperands, ops[0]->getOperands())) {
      auto arg = vectorizeBlock.addArgument(vecOperand[0].getType(),
                                            origOpernad.getLoc());
      argMapping.map(origOpernad, arg);
    }

    auto *clonedOp = builder.clone(*ops[0], argMapping);
    // `VectorizeReturnOp`
    VectorizeReturnOp::create(builder, clonedOp->getResult(0));

    // Now replace the original ops with the vectorized ops
    for (auto [op, result] : llvm::zip(ops, vectorizeOp->getResults())) {
      op->getResult(0).replaceAllUsesWith(result);
      op->erase();
    }
  }
  return success();
}

void FindInitialVectorsPass::runOnOperation() {
  for (auto moduleOp : getOperation().getOps<hw::HWModuleOp>()) {
    auto result = moduleOp.walk([&](Block *block) {
      if (!mayHaveSSADominance(*block->getParent()))
        if (failed(Vectorizer(block).vectorize(stat)))
          return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return signalPassFailure();
  }

  numOfVectorizedOps = stat.vecOps;
  numOfSavedOps = stat.savedOps;
  biggestSeedVector = stat.bigSeedVec;
  numOfVectorsCreated = stat.vecCreated;
}

std::unique_ptr<Pass> arc::createFindInitialVectorsPass() {
  return std::make_unique<FindInitialVectorsPass>();
}
