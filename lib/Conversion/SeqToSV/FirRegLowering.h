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
#include "llvm/ADT/SmallPtrSet.h"

namespace circt {

using namespace hw;
// This class computes the set of muxes that are reachable from an op.
// The heuristic propagates the reachability only through the 3 ops, mux,
// array_create and array_get. All other ops block the reachability.
// This analysis is built lazily on every query.
// The query: is a mux is reachable from a reg, results in a DFS traversal
// of the IR rooted at the register. This traversal is completed and the
// result is cached in a Map, for faster retrieval on any future query of any
// op in this subgraph.
class ReachableMuxes {
public:
  ReachableMuxes(HWModuleOp m) : module(m) {}

  bool isMuxReachableFrom(seq::FirRegOp regOp, comb::MuxOp muxOp);

private:
  void buildReachabilityFrom(Operation *startNode);
  HWModuleOp module;
  llvm::DenseMap<Operation *, llvm::SmallDenseSet<Operation *>> reachableMuxes;
  llvm::SmallPtrSet<Operation *, 16> visited;
};

// The op and its users information that needs to be tracked on the stack
// for an iterative DFS traversal.
struct OpUserInfo {
  Operation *op;
  using ValidUsersIterator =
      llvm::filter_iterator<Operation::user_iterator,
                            std::function<bool(const Operation *)>>;

  ValidUsersIterator userIter, userEnd;
  static std::function<bool(const Operation *op)> opAllowsReachability;

  OpUserInfo(Operation *op)
      : op(op), userIter(op->getUsers().begin(), op->getUsers().end(),
                         opAllowsReachability),
        userEnd(op->getUsers().end(), op->getUsers().end(),
                opAllowsReachability) {}

  bool getAndSetUnvisited() {
    if (unvisited) {
      unvisited = false;
      return true;
    }
    return false;
  }

private:
  bool unvisited = true;
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
                        sv::ResetType resetStyle = {},
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

  using AlwaysKeyType = std::tuple<Block *, sv::EventControl, Value,
                                   sv::ResetType, sv::EventControl, Value>;
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
