//===- Partition.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-partition"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_PARTITION
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace arc;
using llvm::SmallSetVector;

namespace {
struct PartitionPass : public arc::impl::PartitionBase<PartitionPass> {
  void runOnOperation() override;
};

class CostModel {
public:
  CostModel(SymbolTableCollection &symbolTables) : symbolTables(symbolTables) {}

  unsigned getOperationCost(Operation *op) {
    if (auto it = costs.find(op); it != costs.end())
      return it->second;

    unsigned cost = 1;
    if (auto callOp = dyn_cast<CallOpInterface>(op))
      cost += getOperationCost(callOp.resolveCallableInTable(&symbolTables));
    for (auto &region : op->getRegions())
      for (auto &block : region)
        for (auto &subOp : block)
          cost += getOperationCost(&subOp);
    costs.insert({op, cost});
    return cost;
  }

private:
  SymbolTableCollection &symbolTables;
  DenseMap<Operation *, unsigned> costs;
};

struct Group {
  unsigned cost;
  SmallPtrSet<Operation *, 1> ops;
  SmallPtrSet<Group *, 2> children;
};

class Partitioner {
public:
  Partitioner(CostModel &costModel, Block &rootBlock)
      : costModel(costModel), rootBlock(rootBlock) {}
  void initialize();
  void updateCandidates();
  void mergeNext();
  bool done() { return candidates.empty(); }
  Group *finalize(unsigned numTasks);

private:
  CostModel &costModel;
  Block &rootBlock;
  llvm::SpecificBumpPtrAllocator<Group> groupAllocator;
  DenseMap<Operation *, Group *> opGroups;
  SetVector<Operation *> worklist;
  DenseMap<Operation *, std::pair<unsigned, unsigned>> candidates;
  unsigned candidateId = 1;
};
} // namespace

void Partitioner::initialize() {
  for (auto &op : rootBlock) {
    if (op.getNumResults() == 0 || isa<StateOp>(op)) {
      auto *group = new (groupAllocator.Allocate()) Group();
      group->cost = costModel.getOperationCost(&op);
      group->ops.insert(&op);
      opGroups[&op] = group;
      for (auto operand : op.getOperands())
        if (auto *defOp = operand.getDefiningOp())
          worklist.insert(defOp);
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "Created " << opGroups.size()
                          << " initial groups\n");
  LLVM_DEBUG(llvm::dbgs() << "Worklist has " << worklist.size()
                          << " initial ops to check\n");
}

void Partitioner::updateCandidates() {
  SmallSetVector<Group *, 4> userGroups;

  for (auto [op, key] : candidates)
    worklist.insert(op);

  while (!worklist.empty()) {
    auto *op = worklist.pop_back_val();
    auto allGrouped = llvm::all_of(
        op->getUsers(), [&](auto *user) { return opGroups.contains(user); });
    if (!allGrouped)
      continue;

    userGroups.clear();
    for (auto *user : op->getUsers())
      userGroups.insert(opGroups.lookup(user));

    // Handle trivial cases where the op feeds into just a single group.
    if (userGroups.size() == 1) {
      auto *group = userGroups[0];
      group->cost += costModel.getOperationCost(op);
      group->ops.insert(op);
      opGroups[op] = group;
      for (auto operand : op->getOperands())
        if (auto *defOp = operand.getDefiningOp())
          if (!opGroups.contains(defOp))
            worklist.insert(defOp);
      candidates.erase(op);
      continue;
    }

    unsigned cost = costModel.getOperationCost(op);
    for (auto *group : userGroups)
      cost += group->cost;
    auto &candidate = candidates[op];
    candidate.first = cost;
    if (!candidate.second)
      candidate.second = candidateId++;
  }
  // LLVM_DEBUG(llvm::dbgs() << "- Found " << candidates.size()
  //                         << " candidates\n");
}

void Partitioner::mergeNext() {
  Operation *op = nullptr;
  std::pair<unsigned, unsigned> key = {-1, -1};
  for (auto [candidateOp, candidateKey] : candidates) {
    if (!op || candidateKey < key) {
      op = candidateOp;
      key = candidateKey;
    }
  }
  if (!op)
    return;
  candidates.erase(op);
  // LLVM_DEBUG(llvm::dbgs() << "- Picked candidate " << *op << "\n");

  // Find all child groups.
  SmallSetVector<Group *, 4> userGroups;
  unsigned cost = costModel.getOperationCost(op);
  // LLVM_DEBUG(llvm::dbgs() << "  - Op cost " << cost << "\n");
  for (auto *user : op->getUsers()) {
    auto *group = opGroups.lookup(user);
    if (userGroups.insert(group))
      cost += group->cost;
  }
  assert(userGroups.size() > 1);

  // Create a new group for this candidate.
  auto *group = new (groupAllocator.Allocate()) Group();
  group->cost = cost;
  group->ops.insert(op);
  opGroups[op] = group;
  for (auto *child : userGroups)
    group->children.insert(child);

  SmallSetVector<Group *, 8> groupWorklist;
  groupWorklist.insert(group);
  while (!groupWorklist.empty()) {
    auto *child = groupWorklist.pop_back_val();
    for (auto *childOp : child->ops)
      opGroups[childOp] = group;
    for (auto *childGroup : child->children)
      groupWorklist.insert(childGroup);
  }
  for (auto operand : op->getOperands())
    if (auto *defOp = operand.getDefiningOp())
      if (!opGroups.contains(defOp))
        worklist.insert(defOp);
  // for (auto *child : userGroups) {
  //   LLVM_DEBUG(llvm::dbgs() << "  - Feeds group with " << child->ops.size()
  //                           << " ops, cost " << child->cost << "\n");
  //   // group->children.insert(child);
  //   // for (auto *childOp : child->ops)
  //   //   opGroups[childOp] = child;
  // }
  // LLVM_DEBUG(llvm::dbgs() << "  - Total cost " << group->cost << "\n");
  // LLVM_DEBUG(llvm::dbgs() << "  - Worklist now has " << worklist.size()
  //                         << " ops to check\n");
  LLVM_DEBUG(llvm::dbgs() << "- Merged group cost " << group->cost << " ("
                          << candidates.size() << " candidates left)\n");
}

Group *Partitioner::finalize(unsigned numTasks) {
  auto *root = new (groupAllocator.Allocate()) Group();
  root->cost = 0;
  for (auto [op, group] : opGroups)
    if (root->children.insert(group).second)
      root->cost += group->cost;
  LLVM_DEBUG(llvm::dbgs() << "Found " << root->children.size()
                          << " root groups, total cost " << root->cost << "\n");

  // Inline large groups until all of them are below a target size.
  unsigned maxCost = root->cost / numTasks;
  LLVM_DEBUG(llvm::dbgs() << "- Targeting " << maxCost << " max cost\n");

  SmallSetVector<Group *, 8> worklist;
  for (auto *group : root->children)
    if (group->cost > maxCost)
      worklist.insert(group);

  while (!worklist.empty()) {
    auto *group = worklist.pop_back_val();
    LLVM_DEBUG(llvm::dbgs()
               << "- Inlining group with cost " << group->cost << "\n");
    for (auto *op : group->ops)
      root->ops.insert(op);
    for (auto *child : group->children) {
      root->children.insert(child);
      if (child->cost > maxCost)
        worklist.insert(child);
    }
    root->children.erase(group);
  }

  // Determine the maximum group cost.
  maxCost = 0;
  for (auto *group : root->children)
    if (group->cost > maxCost)
      maxCost = group->cost;
  LLVM_DEBUG(llvm::dbgs() << "- Targeting " << maxCost
                          << " max cost for combining small groups\n");

  // Combine small groups up to the maximum cost.
  while (true) {
    // Find the smallest.
    Group *group1 = nullptr;
    for (auto *group : root->children)
      if (!group1 || group->cost < group1->cost)
        group1 = group;
    if (!group1)
      break;

    // Find the second smallest.
    Group *group2 = nullptr;
    for (auto *group : root->children)
      if (group != group1)
        if (!group2 || group->cost < group2->cost)
          group2 = group;
    if (!group2)
      break;

    if (group1->cost + group2->cost > maxCost)
      break;

    // Combine the two groups.
    auto *merged = new (groupAllocator.Allocate()) Group();
    merged->cost = group1->cost + group2->cost;
    merged->children.insert(group1);
    merged->children.insert(group2);
    root->children.erase(group1);
    root->children.erase(group2);
    root->children.insert(merged);
    LLVM_DEBUG(llvm::dbgs()
               << "- Merging groups with cost " << merged->cost << "\n");
  }

  return root;
}

void PartitionPass::runOnOperation() {
  SymbolTableCollection symbolTables;
  CostModel costModel(symbolTables);
  for (auto module : getOperation().getOps<hw::HWModuleOp>()) {
    LLVM_DEBUG(llvm::dbgs() << "Partitioning " << module.getName() << "\n");
    Partitioner partitioner(costModel, *module.getBodyBlock());
    partitioner.initialize();
    partitioner.updateCandidates();
    while (!partitioner.done()) {
      partitioner.mergeNext();
      partitioner.updateCandidates();
    }
    auto *root = partitioner.finalize(taskCount);

    LLVM_DEBUG(llvm::dbgs()
               << "Found " << root->children.size() << " root groups:\n");
    unsigned groupIdx = 0;
    for (auto *op : root->ops)
      op->setAttr("group", IntegerAttr::get(IntegerType::get(&getContext(), 32),
                                            groupIdx));
    ++groupIdx;

    unsigned remainingCost = root->cost;
    for (auto *group : root->children) {
      remainingCost -= group->cost;
      SmallSetVector<Group *, 8> worklist;
      worklist.insert(group);
      unsigned size = 0;
      while (!worklist.empty()) {
        auto *group = worklist.pop_back_val();
        size += group->ops.size();
        for (auto *op : group->ops)
          op->setAttr(
              "group",
              IntegerAttr::get(IntegerType::get(&getContext(), 32), groupIdx));
        for (auto *subgroup : group->children)
          worklist.insert(subgroup);
      }
      LLVM_DEBUG(llvm::dbgs() << "- Group " << groupIdx << " has cost "
                              << group->cost << " (" << size << " ops)\n");
      ++groupIdx;
    }
    LLVM_DEBUG(llvm::dbgs() << "- Remaining cost " << remainingCost << " ("
                            << root->ops.size() << " ops)\n");
  }
}

std::unique_ptr<Pass> arc::createPartitionPass() {
  return std::make_unique<PartitionPass>();
}
