//===- GroupComputation.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GraphWriter.h"

#define DEBUG_TYPE "arc-group-computation"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_GROUPCOMPUTATION
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace circt;
using namespace arc;
using namespace mlir;

using llvm::SmallMapVector;
using llvm::SmallSetVector;

/// Returns whether an operation can trivially be copied at no additional cost.
static bool isTriviallyClonable(Operation *op) {
  // if (op->getNumOperands() == 0)
  //   return true;
  return op->hasTrait<OpTrait::ConstantLike>() || isa<StateReadOp>(op);
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
// Communities
//===----------------------------------------------------------------------===//
namespace {

struct Community;

struct Node : llvm::ilist_node<Node> {
  Operation *op;
  Community *community;
  SmallMapVector<Node *, unsigned, 2> incomingEdges;
  SmallMapVector<Node *, unsigned, 2> outgoingEdges;
  unsigned edges = 0;
};

struct Community : llvm::ilist_node<Community> {
  llvm::simple_ilist<Node> nodes;
  Operation *seedOp;
  // SmallDenseMap<Community *, unsigned, 2> incomingEdges;
  // SmallDenseMap<Community *, unsigned, 2> outgoingEdges;
  unsigned edgesWithin = 0;
  unsigned edgesTotal = 0;

  // Two topological orders. One where order decreases towards the operands
  // (operands have lower rank than this community), and one where order
  // decreases towards results (results have lower rank than this community).
  unsigned minRankOperand = UINT_MAX;
  unsigned maxRankOperand = UINT_MAX;
  unsigned minRankResult = UINT_MAX;
  unsigned maxRankResult = UINT_MAX;

  unsigned id = 0;

  // void debugVerifyCounts() {
  //   unsigned within = 0;
  //   unsigned total = 0;
  //   for (auto [other, count] : llvm::concat<std::pair<Community *,
  //   unsigned>>(
  //            incomingEdges, outgoingEdges)) {
  //     total += count;
  //     if (other == this)
  //       within += count;
  //   }
  //   assert(edgesWithin == within);
  //   assert(edgesTotal == total);
  // }
};

} // namespace

template <typename T>
static void maximize(T &x, T y) {
  if (x < y)
    x = y;
}

template <typename T>
static void minimize(T &x, T y) {
  if (x > y)
    x = y;
}

/// Recompute the topological order of the given communities.
static LogicalResult updateCommunityOrder(ArrayRef<Community *> communities) {
  // LLVM_DEBUG(llvm::dbgs() << "- Updating order of " << communities.size()
  //                         << " communities\n");

  SmallPtrSet<Community *, 16> updatedCommunities;
  SmallSetVector<Community *, 8> countedIncoming, countedOutgoing;

  using WorklistItem = SmallSetVector<Community *, 8>;
  SmallMapVector<Community *, WorklistItem, 16> worklist;
  SmallPtrSet<Community *, 16> finalizeWorklist;

  // Update the operand rank of the communities.
  for (unsigned dir = 0; dir < 2; ++dir) {
    auto primaryWorklist = communities;
    updatedCommunities.clear();
    while (!worklist.empty() || !primaryWorklist.empty()) {
      if (worklist.empty()) {
        auto *community = primaryWorklist.front();
        primaryWorklist = primaryWorklist.drop_front();
        if (updatedCommunities.contains(community))
          continue;
        worklist.insert({community, {}});
      }
      auto &[community, remaining] = worklist.back();

      // If this is the first time we visit this community, indicated by
      // `remaining` being empty, update the community's rank and collect the
      // incoming or outgoing edges to other communities.
      if (remaining.empty()) {
        unsigned oldRankOperand = community->minRankOperand;
        unsigned oldRankResult = community->minRankResult;
        unsigned newRankOperand = 0;
        unsigned newRankResult = 0;
        for (auto &node : community->nodes) {
          for (auto [otherNode, count] : node.incomingEdges)
            if (otherNode->community != community)
              if (countedIncoming.insert(otherNode->community))
                maximize(newRankOperand,
                         otherNode->community->minRankOperand + 1);
          for (auto [otherNode, count] : node.outgoingEdges)
            if (otherNode->community != community)
              if (countedOutgoing.insert(otherNode->community))
                maximize(newRankResult,
                         otherNode->community->minRankResult + 1);
        }

        // If the operand rank changed, update the communities along the
        // outgoing edges.
        if (dir == 0 && newRankOperand != oldRankOperand) {
          // LLVM_DEBUG(llvm::dbgs() << "  - Community " << community->id
          //                         << " rankOperand=" << newRankOperand
          //                         << " (before " << oldRankOperand << ")\n");
          community->minRankOperand = newRankOperand;
          remaining = countedOutgoing;
        }

        // If the reuslt rank changed, update the communities along the incoming
        // edges.
        if (dir == 1 && newRankResult != oldRankResult) {
          // LLVM_DEBUG(llvm::dbgs() << "  - Community " << community->id
          //                         << " rankResult=" << newRankResult
          //                         << " (before " << oldRankResult << ")\n");
          community->minRankResult = newRankResult;
          remaining = countedIncoming;
        }

        finalizeWorklist.insert(community);
        for (auto *community : countedIncoming)
          finalizeWorklist.insert(community);
        for (auto *community : countedOutgoing)
          finalizeWorklist.insert(community);

        countedIncoming.clear();
        countedOutgoing.clear();
      }

      if (remaining.empty()) {
        updatedCommunities.insert(community);
        worklist.pop_back();
        continue;
      }
      auto *next = remaining.pop_back_val();
      if (!worklist.insert({next, {}}).second) {
        llvm::errs() << "ERROR: community loop through " << next->id << "\n";
        for (auto &[community, ignored] : worklist)
          llvm::errs() << "- " << community->id << "\n";
        return failure();
      }
    }
  }

  // Update the maximum ranks in the updated communities.
  for (auto *community : finalizeWorklist) {
    unsigned newRankOperand = UINT_MAX;
    unsigned newRankResult = UINT_MAX;
    for (auto &node : community->nodes) {
      for (auto [otherNode, count] : node.incomingEdges)
        if (otherNode->community != community)
          if (countedIncoming.insert(otherNode->community))
            minimize(newRankResult, otherNode->community->minRankResult);
      for (auto [otherNode, count] : node.outgoingEdges)
        if (otherNode->community != community)
          if (countedOutgoing.insert(otherNode->community))
            minimize(newRankOperand, otherNode->community->minRankOperand);
    }
    community->maxRankOperand = newRankOperand;
    community->maxRankResult = newRankResult;
    countedIncoming.clear();
    countedOutgoing.clear();
    // if (community->maxRankOperand - community->minRankOperand > 2 ||
    //     community->maxRankResult - community->minRankResult > 2)
    //   LLVM_DEBUG(llvm::dbgs()
    //              << "  - Final rank " << community << " operand=["
    //              << community->minRankOperand << ", "
    //              << community->rankOperand() << ", "
    //              << community->maxRankOperand << "] result=["
    //              << community->minRankResult << ", " <<
    //              community->rankResult()
    //              << ", " << community->maxRankResult << "]\n");
  }

  return success();
}

static bool canGroupNode(Node &node) {
  // Never group up memory operations.
  // TODO: Remove this once LegalizeStateUpdate can properly handle
  // memories.
  if (isa<MemoryReadOp, MemoryWriteOp>(node.op))
    return false;

  return true;
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct GroupComputationPass
    : public arc::impl::GroupComputationBase<GroupComputationPass> {
  void runOnOperation() override;
  LogicalResult runOnClockTree(ClockTreeOp op);
};
} // namespace

void GroupComputationPass::runOnOperation() {
  getOperation().walk<WalkOrder::PreOrder>([&](ClockTreeOp op) {
    if (failed(runOnClockTree(op))) {
      signalPassFailure();
      return WalkResult::interrupt();
    }
    return WalkResult::skip();
  });
}

LogicalResult GroupComputationPass::runOnClockTree(ClockTreeOp clockTreeOp) {
  LLVM_DEBUG(llvm::dbgs() << "Processing clock tree\n");

  // Put operations into initial communities.
  llvm::SpecificBumpPtrAllocator<Community> communityAlloc;
  llvm::SpecificBumpPtrAllocator<Node> nodeAlloc;
  SmallVector<Node *, 0> nodes;
  DenseMap<Operation *, Node *> nodeByOp;
  SmallVector<Community *, 0> communities;
  auto i64 = IntegerType::get(&getContext(), 64);

  for (auto &op : clockTreeOp.getBodyBlock()) {
    if (isTriviallyClonable(&op))
      continue;
    auto *community = new (communityAlloc.Allocate()) Community;
    auto *node = new (nodeAlloc.Allocate()) Node;
    community->seedOp = &op;
    community->nodes.push_back(*node);
    communities.push_back(community);
    community->id = communities.size();
    node->op = &op;
    node->community = community;
    nodes.push_back(node);
    nodeByOp[&op] = node;

    // node->op->setAttr("community", IntegerAttr::get(i64,
    // node->community->id));
  }
  LLVM_DEBUG(llvm::dbgs() << "- " << nodes.size() << " initial communities\n");

  // Establish the edges between the nodes.
  unsigned totalEdges = 0;
  for (auto *node : nodes) {
    node->op->walk([&](Operation *op) {
      for (auto operand : op->getOperands()) {
        if (operand.getParentBlock() != &clockTreeOp.getBodyBlock())
          continue;
        auto *defOp = operand.getDefiningOp();
        if (!defOp || defOp == node->op)
          continue;
        auto *otherNode = nodeByOp.lookup(defOp);
        if (!otherNode)
          continue;
        totalEdges += 1;
        node->edges += 1;
        otherNode->edges += 1;
        node->incomingEdges[otherNode] += 1;
        otherNode->outgoingEdges[node] += 1;
        // node->community->incomingEdges[otherNode->community] += 1;
        // otherNode->community->outgoingEdges[node->community] += 1;
        node->community->edgesTotal += 1;
        otherNode->community->edgesTotal += 1;
      }
    });
  }
  LLVM_DEBUG(llvm::dbgs() << "- " << totalEdges << " total edges\n");
  if (failed(updateCommunityOrder(communities)))
    return failure();

  // // Annotate the modularity of each community.
  // auto f64 = FloatType::getF64(&getContext());
  // for (auto *node : nodes) {
  //   node->op->setAttr("edgesWithin",
  //                     IntegerAttr::get(i64, node->community->edgesWithin));
  //   node->op->setAttr("edgesTotal",
  //                     IntegerAttr::get(i64, node->community->edgesTotal));
  //   double modularity =
  //       (double)node->community->edgesWithin / (2 * totalEdges) -
  //       std::pow((double)node->community->edgesTotal / (2 * totalEdges), 2);
  //   node->op->setAttr("modularity", FloatAttr::get(f64, modularity));
  // }

  // Move nodes into neighboring communities to improve modularity.
  // TODO: Check topo order to prevent loops from forming.
  SmallPtrSet<Community *, 8> checkedCommunities;
  unsigned iterMoves = 1;
  int64_t iterDelta = 0;
  unsigned moveIdx = 0;
  for (unsigned iter = 0; iter < 1000 && iterMoves > 0; ++iter) {
    LLVM_DEBUG(llvm::dbgs() << "- Iteration " << iter << "\n");
    iterMoves = 0;
    iterDelta = 0;
    for (auto *node : nodes) {
      if (!canGroupNode(*node))
        continue;
      // LLVM_DEBUG(llvm::dbgs() << "  - Moving " << *node->op << "\n");

      // Old community without the node.
      unsigned oldEdgesWithin = node->community->edgesWithin;
      unsigned oldEdgesTotal = node->community->edgesTotal;
      oldEdgesTotal -= node->edges;
      for (auto [otherNode, count] : llvm::concat<std::pair<Node *, unsigned>>(
               node->incomingEdges, node->outgoingEdges)) {
        if (otherNode->community == node->community)
          oldEdgesWithin -= 2 * count;
      }
      int64_t oldDelta =
          (int64_t)oldEdgesWithin * totalEdges - pow((int64_t)oldEdgesTotal, 2);
      oldDelta -= (int64_t)node->community->edgesWithin * totalEdges -
                  pow((int64_t)node->community->edgesTotal, 2);

      // LLVM_DEBUG(llvm::dbgs()
      //            << "    - Old community {in=" <<
      //            node->community->edgesWithin
      //            << " tot=" << node->community->edgesTotal
      //            << "} to {in=" << oldEdgesWithin << " tot=" << oldEdgesTotal
      //            << "} dM=" << oldDelta << "\n");

      // Compute the operand and result rank of the operation once removed from
      // its community.
      unsigned isolatedRankOperand = 0;
      unsigned isolatedRankResult = 0;
      for (auto [otherNode, count] : node->incomingEdges)
        maximize(isolatedRankOperand, otherNode->community->minRankOperand + 1);
      for (auto [otherNode, count] : node->outgoingEdges)
        maximize(isolatedRankResult, otherNode->community->minRankResult + 1);

      // LLVM_DEBUG(llvm::dbgs()
      //            << "    - Isolated rankOperand=" << isolatedRankOperand
      //            << ", rankResult=" << isolatedRankResult << "\n");

      // New communities with the node.
      int64_t bestDelta = 0;
      unsigned bestEdgesWithin = 0;
      unsigned bestEdgesTotal = 0;
      Community *bestCommunity = nullptr;

      SmallVector<std::pair<Node *, unsigned>> candidates;
      for (auto [otherNode, count] : node->incomingEdges)
        if (node->community != otherNode->community)
          if (otherNode->community->maxRankOperand >= isolatedRankOperand)
            candidates.push_back({otherNode, count});
      for (auto [otherNode, count] : node->outgoingEdges)
        if (node->community != otherNode->community)
          if (otherNode->community->maxRankResult >= isolatedRankResult)
            candidates.push_back({otherNode, count});

      for (auto [otherNode, count] : candidates) {
        if (!canGroupNode(*otherNode))
          continue;
        auto *community = otherNode->community;
        if (community == node->community)
          continue;
        // if (node->community->minRankOperand > community->rankOperand() ||
        //     node->community->maxRankOperand < community->rankOperand())
        //   continue;
        // if (node->community->minRankResult > community->rankResult() ||
        //     node->community->maxRankResult < community->rankResult())
        //   continue;
        if (!checkedCommunities.insert(community).second)
          continue;
        community->seedOp = otherNode->op;

        unsigned newEdgesWithin = community->edgesWithin;
        unsigned newEdgesTotal = community->edgesTotal;
        newEdgesTotal += node->edges;
        for (auto [otherNode, count] :
             llvm::concat<std::pair<Node *, unsigned>>(node->incomingEdges,
                                                       node->outgoingEdges)) {
          if (otherNode->community == community)
            newEdgesWithin += 2 * count;
        }
        int64_t newDelta = (int64_t)newEdgesWithin * totalEdges -
                           pow((int64_t)newEdgesTotal, 2);
        newDelta -= (int64_t)community->edgesWithin * totalEdges -
                    pow((int64_t)community->edgesTotal, 2);

        // LLVM_DEBUG(llvm::dbgs()
        //            << "    - Trying " << *community->seedOp << "\n");
        // LLVM_DEBUG(llvm::dbgs()
        //            << "      New community {in=" << community->edgesWithin
        //            << " tot=" << community->edgesTotal
        //            << "} to {in=" << newEdgesWithin << " tot=" <<
        //            newEdgesTotal
        //            << "} dM=" << newDelta << "\n");

        int64_t delta = oldDelta + newDelta;
        if (delta > bestDelta) {
          bestDelta = delta;
          bestEdgesWithin = newEdgesWithin;
          bestEdgesTotal = newEdgesTotal;
          bestCommunity = community;
        }
      }
      checkedCommunities.clear();
      if (!bestCommunity)
        continue;

      // Remove from old community and add to new community.
      // LLVM_DEBUG(llvm::dbgs() << "  - Moving " << *node->op << "\n");
      // LLVM_DEBUG(llvm::dbgs()
      //            << "    next to " << *bestCommunity->seedOp << "\n");
      // LLVM_DEBUG(llvm::dbgs() << "    dM = " << bestDelta << "\n");
      // LLVM_DEBUG(llvm::dbgs() << "    move #" << moveIdx << "\n");
      // LLVM_DEBUG(llvm::dbgs()
      //            << "    current rankOperand=["
      //            << node->community->minRankOperand << ", "
      //            << node->community->maxRankOperand << "], rankResult=["
      //            << node->community->minRankResult << ", "
      //            << node->community->maxRankResult << "]\n");
      SmallSetVector<Community *, 8> involvedCommunities;
      involvedCommunities.insert(node->community);
      involvedCommunities.insert(bestCommunity);
      for (auto [otherNode, count] : llvm::concat<std::pair<Node *, unsigned>>(
               node->incomingEdges, node->outgoingEdges)) {
        involvedCommunities.insert(otherNode->community);
      }
      // std::array<Community *, 2> involvedCommunities = {node->community,
      //                                                   bestCommunity};
      node->community->edgesWithin = oldEdgesWithin;
      node->community->edgesTotal = oldEdgesTotal;
      node->community->nodes.remove(*node);
      node->community = bestCommunity;
      node->community->nodes.push_back(*node);
      node->community->edgesWithin = bestEdgesWithin;
      node->community->edgesTotal = bestEdgesTotal;
      // node->op->setAttr("community",
      //                   IntegerAttr::get(i64, node->community->id));
      if (failed(updateCommunityOrder(involvedCommunities.getArrayRef())) /*||
          moveIdx == 6*/) {
        // llvm::outs() << "digraph G {\n";
        // for (auto *community : communities) {
        //   if (community->nodes.empty())
        //     continue;
        //   llvm::outs() << "subgraph cluster_" << community->id << " {\n";
        //   llvm::outs() << "label=\"community " << community->id
        //                << ": rankOperand=[" << community->minRankOperand <<
        //                ", "
        //                << community->maxRankOperand << "], rankResult=["
        //                << community->minRankResult << ", "
        //                << community->maxRankResult << "]\";\n";
        //   for (auto &node : community->nodes) {
        //     std::string opName;
        //     llvm::raw_string_ostream opNameStream(opName);
        //     opNameStream << *node.op;
        //     llvm::outs() << "op" << node.op << " [label=\""
        //                  << llvm::DOT::EscapeString(opName) << "\"];\n";
        //   }
        //   llvm::outs() << "}\n";
        //   for (auto &node : community->nodes) {
        //     for (auto [otherNode, count] : node.incomingEdges)
        //       llvm::outs() << "op" << otherNode->op << " -> "
        //                    << "op" << node.op << ";\n";
        //   }
        // }
        // llvm::outs() << "}\n";
        return failure();
      }
      ++iterMoves;
      iterDelta += bestDelta;
      ++moveIdx;
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  - " << iterMoves << " moves (dM = " << iterDelta << ")\n");
  }

  // // Annotate communities.
  // llvm::MapVector<Community *, SmallVector<Operation *>> communitySizes;
  // for (auto *node : nodes) {
  //   node->op->setAttr("community", IntegerAttr::get(i64,
  //   node->community->id));
  //   communitySizes[node->community].push_back(node->op);
  // }
  // LLVM_DEBUG(llvm::dbgs() << "- " << communitySizes.size()
  //                         << " final communities\n");

  // llvm::sort(communitySizes, [&](auto &a, auto &b) {
  //   return a.second.size() > b.second.size();
  // });
  // unsigned i = 0;
  // for (auto &[community, ops] : communitySizes) {
  //   if (i++ >= 20)
  //     break;
  //   LLVM_DEBUG(llvm::dbgs() << "- Community " << community->id << " has "
  //                           << ops.size() << " ops\n");
  // }

  // Group operations into their communities.
  LLVM_DEBUG(llvm::dbgs() << "Grouping operations into communities\n");

  auto collectDeps =
      [&](Community *community) -> SmallSetVector<Community *, 4> {
    SmallSetVector<Community *, 4> deps;
    for (auto &node : community->nodes)
      for (auto [otherNode, count] : node.incomingEdges)
        if (otherNode->community != community)
          deps.insert(otherNode->community);
    return deps;
  };

  SmallPtrSet<Community *, 8> handledCommunities;
  SmallMapVector<Community *, SmallSetVector<Community *, 4>, 16> worklist;

  auto groupBuilder = OpBuilder::atBlockEnd(&clockTreeOp.getBodyBlock());

  while (!worklist.empty() || !communities.empty()) {
    if (worklist.empty()) {
      auto *community = communities.pop_back_val();
      if (community->nodes.empty() || handledCommunities.contains(community))
        continue;
      worklist.insert({community, collectDeps(community)});
    }
    auto &[community, deps] = worklist.back();

    // If all dependencies have been handled, group the operations for this
    // community and pop it off the worklist.
    if (deps.empty()) {
      assert(!community->nodes.empty());
      handledCommunities.insert(community);
      auto loc = community->nodes.front().op->getLoc();

      // Create a block and move the operations into it.
      auto block = std::make_unique<Block>();
      OpBuilder builder(&getContext());
      builder.setInsertionPointToEnd(block.get());
      for (auto &node : community->nodes) {
        node.op->remove();
        builder.insert(node.op);
      }
      sortTopologically(block.get());

      // Eagerly clone any trivial operations into the block.
      IRMapping trivialClones;
      for (auto &op : block->getOperations()) {
        builder.setInsertionPoint(&op);
        op.walk([&](Operation *op) {
          for (auto &operand : op->getOpOperands()) {
            auto *defOp = operand.get().getDefiningOp();
            if (!defOp || !isTriviallyClonable(defOp) ||
                !isOutsideOfBlock(defOp, block.get()))
              continue;
            if (!trivialClones.contains(operand.get())) {
              auto *clonedOp = builder.clone(*defOp);
              trivialClones.map(defOp->getResults(), clonedOp->getResults());
            }
            operand.set(trivialClones.lookup(operand.get()));
            if (defOp->use_empty())
              defOp->erase();
          }
        });
      }
      builder.setInsertionPointToEnd(block.get());

      // If this is a trivial community, don't bother moving ops into a separate
      // group op. Just inline them where that group op would be created.
      bool isTrivial = ++community->nodes.begin() == community->nodes.end();
      if (isTrivial) {
        groupBuilder.getBlock()->getOperations().splice(
            groupBuilder.getInsertionPoint(), block->getOperations());
        worklist.pop_back();
        continue;
      }

      // Find values that are used outside of the block.
      SmallSetVector<Value, 8> outputValues;
      for (auto &op : block->getOperations())
        for (auto result : op.getResults())
          if (isUsedOutsideOfBlock(result, block.get()))
            outputValues.insert(result);

      // Add the terminator.
      auto outputOp =
          builder.create<arc::OutputOp>(loc, outputValues.getArrayRef());

      // Create the group op.
      auto groupOp =
          groupBuilder.create<arc::GroupOp>(loc, outputOp.getOperandTypes());
      groupOp.getRegion().push_back(block.release());
      // groupOp->setAttr("community", IntegerAttr::get(i64, community->id));

      // Replace external uses of block values with the corresponding result of
      // the group op.
      for (auto [externalResult, internalValue] :
           llvm::zip(groupOp.getResults(), outputValues))
        for (auto &use : llvm::make_early_inc_range(internalValue.getUses()))
          if (isOutsideOfBlock(use.getOwner(), &groupOp.getRegion().front()))
            use.set(externalResult);

      worklist.pop_back();
      continue;
    }

    // If we arrive here there are still dependencies left to be handled. Pop
    // the next off the `deps` list and add it to the worklist.
    auto *dep = deps.pop_back_val();
    if (!handledCommunities.contains(dep))
      if (!worklist.insert({dep, collectDeps(dep)}).second)
        return mlir::emitError(UnknownLoc::get(&getContext()),
                               "communities have cyclic dependency");
  }

  return success();
}

std::unique_ptr<Pass> arc::createGroupComputationPass() {
  return std::make_unique<GroupComputationPass>();
}
