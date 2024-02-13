//===- FirRegLowering.h - FirReg lowering utilities ===========--*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef CONVERSION_SEQTOSV_FIRREGLOWERING_H
#define CONVERSION_SEQTOSV_FIRREGLOWERING_H

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/Namespace.h"
#include "circt/Support/SymCache.h"
#include "mlir/IR/Visitors.h"
#include <mlir/IR/ValueRange.h>
#include <stack>
#include <unordered_set>

namespace circt {

using namespace hw;
// This class computes the set of muxes that are reachable from an op.
// The heuristic propagates the reachability only through the 3 ops, mux,
// array_create and array_get. All other ops block the reachability.
// This analysis is built lazily on every query.
// The query: is a mux is reachable from a reg, results in a DFS traversal
// of the IR rooted at the register. This traversal is completed and the result
// is cached in a Map, for faster retrieval on any future query of any op in
// this subgraph.
class ReachableMuxes {
public:
  ReachableMuxes(HWModuleOp m) : module(m) {}

  bool isMuxReachableFrom(Operation *regOp, Operation *muxOp) {
    return llvm::any_of(regOp->getResult(0).getUsers(), [&](Operation *user) {
      if (opBlocksReachability(user))
        return false;
      buildReachabilityFrom(user);
      return reachableMuxes[user].contains(muxOp);
    });
  }

private:
  static inline bool opBlocksReachability(Operation *op) {
    return (!isa<comb::MuxOp, ArrayGetOp, ArrayCreateOp>(op));
  }
  void buildReachabilityFrom(Operation *startNode) {
    // This is a backward dataflow analysis.
    // First build a graph rooted at the `startNode`. Every user of an operation
    // that doesnot block the reachability is a child node. Then, the ops that
    // are reachable from a node is computed as the union of the Reachability of
    // all its child nodes.
    // for all child in the Children(node)
    // Reachability(node) = node + Union{Reachability(child)}
    if (visited.find(startNode) != visited.end())
      return;
    // The op and its users information that needs to be tracked on the stack
    // for an iterative DFS traversal.
    struct OpUserInfo {
      Operation *op;
      const mlir::ResultRange::user_range userRange;
      mlir::ResultRange::user_iterator userIter;

      OpUserInfo(Operation *op)
          : op(op), userRange(op->getUsers()), userIter(userRange.begin()) {}

      // Increments the itertor to the next valid user op and returns false if
      // the iterator reaches the end of the range.
      auto getNextValid(mlir::ResultRange::user_iterator &iter) const {
        for (; iter != userRange.end(); ++iter)
          if (!opBlocksReachability(*iter))
            return true;
        return false;
      }
    };
    // The stack to record enough information for an iterative post-order
    // traversal.
    std::stack<OpUserInfo> stk;

    stk.emplace(startNode);

    while (!stk.empty()) {
      auto &info = stk.top();
      Operation *currentNode = info.op;

      // Node is being visited for the first time.
      if (info.userIter == info.userRange.begin())
        visited.insert(currentNode);
      if (info.getNextValid(info.userIter)) {
        Operation *child = *info.userIter;
        ++info.userIter;
        if (visited.find(child) == visited.end())
          stk.emplace(child);

      } else { // All children of the node have been visited
        // Any op is reachable from itself.
        reachableMuxes[currentNode].insert(currentNode);
        auto userIterator = info.userRange.begin();
        while (info.getNextValid(userIterator)) {
          Operation *childOp = *userIterator;
          reachableMuxes[currentNode].insert(childOp);
          // Propagate the reachability backwards from m to currentNode.
          auto iter = reachableMuxes.find(childOp);

          if (iter != reachableMuxes.end())
            reachableMuxes[currentNode].insert(iter->getSecond().begin(),
                                               iter->getSecond().end());

          ++userIterator;
        }
        stk.pop();
      }
    }
  }
  HWModuleOp module;
  llvm::DenseMap<Operation *, llvm::SmallDenseSet<Operation *>> reachableMuxes;
  std::unordered_set<Operation *> visited;
};

/// Lower FirRegOp to `sv.reg` and `sv.always`.
class FirRegLowering {
public:
  FirRegLowering(TypeConverter &typeConverter, hw::HWModuleOp module,
                 bool disableRegRandomization = false,
                 bool emitSeparateAlwaysBlocks = false);

  void lower();
  bool needsRegRandomization() const { return needsRandom; }

  unsigned numSubaccessRestored = 0;

private:
  struct RegLowerInfo {
    sv::RegOp reg;
    IntegerAttr preset;
    Value asyncResetSignal;
    Value asyncResetValue;
    int64_t randStart;
    size_t width;
  };

  RegLowerInfo lower(seq::FirRegOp reg);

  void initialize(OpBuilder &builder, RegLowerInfo reg, ArrayRef<Value> rands);
  void initializeRegisterElements(Location loc, OpBuilder &builder, Value reg,
                                  Value rand, unsigned &pos);

  void createTree(OpBuilder &builder, Value reg, Value term, Value next);
  std::optional<std::tuple<Value, Value, Value>>
  tryRestoringSubaccess(OpBuilder &builder, Value reg, Value term,
                        hw::ArrayCreateOp nextRegValue);

  void addToAlwaysBlock(Block *block, sv::EventControl clockEdge, Value clock,
                        const std::function<void(OpBuilder &)> &body,
                        ResetType resetStyle = {},
                        sv::EventControl resetEdge = {}, Value reset = {},
                        const std::function<void(OpBuilder &)> &resetBody = {});

  void addToIfBlock(OpBuilder &builder, Value cond,
                    const std::function<void()> &trueSide,
                    const std::function<void()> &falseSide);

  hw::ConstantOp getOrCreateConstant(Location loc, const APInt &value) {
    OpBuilder builder(module.getBody());
    auto &constant = constantCache[value];
    if (constant) {
      constant->setLoc(builder.getFusedLoc({constant->getLoc(), loc}));
      return constant;
    }

    constant = builder.create<hw::ConstantOp>(loc, value);
    return constant;
  }

  using AlwaysKeyType = std::tuple<Block *, sv::EventControl, Value, ResetType,
                                   sv::EventControl, Value>;
  llvm::SmallDenseMap<AlwaysKeyType, std::pair<sv::AlwaysOp, sv::IfOp>>
      alwaysBlocks;

  using IfKeyType = std::pair<Block *, Value>;
  llvm::SmallDenseMap<IfKeyType, sv::IfOp> ifCache;

  llvm::SmallDenseMap<APInt, hw::ConstantOp> constantCache;
  llvm::SmallDenseMap<std::pair<Value, unsigned>, Value> arrayIndexCache;
  std::unique_ptr<ReachableMuxes> reachableMuxes;

  TypeConverter &typeConverter;
  hw::HWModuleOp module;

  bool disableRegRandomization;
  bool emitSeparateAlwaysBlocks;

  bool needsRandom = false;
};
} // namespace circt

#endif // CONVERSION_SEQTOSV_FIRREGLOWERING_H
