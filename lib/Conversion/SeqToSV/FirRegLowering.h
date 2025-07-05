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
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/Namespace.h"
#include "circt/Support/SymCache.h"
#include "mlir/IR/Attributes.h"
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
  /// A map sending registers to their paths.
  using PathTable = DenseMap<seq::FirRegOp, hw::HierPathOp>;

  /// When a register is buried under an ifdef op, the initialization code at
  /// the footer of the HW module will refer to the register using a
  /// hierarchical path op. This function creates any necessary hier paths and
  /// returns a map from buried registers to their hier path ops. HierPathOps
  /// creation needs to be serialized to keep the symbol creation deterministic,
  /// so this is done as a pre-pass on the entire MLIR module. The result should
  /// be passed in to each invocation of FirRefLowering.
  static PathTable createPaths(mlir::ModuleOp top);

  FirRegLowering(TypeConverter &typeConverter, hw::HWModuleOp module,
                 const PathTable &pathTable,
                 bool disableRegRandomization = false,
                 bool emitSeparateAlwaysBlocks = false);

  void lower();
  bool needsRegRandomization() const { return needsRandom; }

  unsigned numSubaccessRestored = 0;

private:
  /// The conditions under which a register is defined.
  struct RegCondition {
    enum Kind {
      /// The register is under an ifdef "then" branch.
      IfDefThen,
      /// The register is under an ifdef "else" branch.
      IfDefElse,
    };
    RegCondition(Kind kind, sv::MacroIdentAttr macro) : data(macro, kind) {}
    Kind getKind() const { return data.getInt(); }
    sv::MacroIdentAttr getMacro() const {
      return cast<sv::MacroIdentAttr>(data.getPointer());
    }
    llvm::PointerIntPair<Attribute, 1, Kind> data;
  };

  struct RegLowerInfo {
    sv::RegOp reg;
    hw::HierPathOp path;
    IntegerAttr preset;
    Value asyncResetSignal;
    Value asyncResetValue;
    int64_t randStart;
    size_t width;
  };

  void lowerUnderIfDef(sv::IfDefOp ifDefOp);
  void lowerInBlock(Block *block);
  void lowerReg(seq::FirRegOp reg);
  void createInitialBlock();

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

  SmallVector<Value> createRandomizationVector(OpBuilder &builder,
                                               Location loc);
  void createRandomInitialization(ImplicitLocOpBuilder &builder);
  void createPresetInitialization(ImplicitLocOpBuilder &builder);
  void createAsyncResetInitialization(ImplicitLocOpBuilder &builder);

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

  /// Recreate the ifdefs under which `reg` was defined. Leave the builder with
  /// its insertion point inside the created ifdef guards.
  void buildRegConditions(OpBuilder &b, sv::RegOp reg);

  using AlwaysKeyType = std::tuple<Block *, sv::EventControl, Value,
                                   sv::ResetType, sv::EventControl, Value>;
  llvm::SmallDenseMap<AlwaysKeyType, std::pair<sv::AlwaysOp, sv::IfOp>>
      alwaysBlocks;

  using IfKeyType = std::pair<Block *, Value>;
  llvm::SmallDenseMap<IfKeyType, sv::IfOp> ifCache;

  llvm::SmallDenseMap<APInt, hw::ConstantOp> constantCache;
  llvm::SmallDenseMap<std::pair<Value, unsigned>, Value> arrayIndexCache;

  /// The ambient ifdef conditions we have encountered while lowering.
  std::vector<RegCondition> conditions;

  /// A list of registers discovered, bucketed by initialization style.
  SmallVector<RegLowerInfo> randomInitRegs, presetInitRegs;

  /// A map from RegOps to the ifdef conditions under which they are defined.
  /// We only bother recording a list of conditions if there is at least one.
  DenseMap<sv::RegOp, std::vector<RegCondition>> regConditionTable;

  /// A map from async reset signal to the registers that use it.
  llvm::MapVector<Value, SmallVector<RegLowerInfo>> asyncResets;

  std::unique_ptr<ReachableMuxes> reachableMuxes;

  const PathTable &pathTable;
  TypeConverter &typeConverter;
  hw::HWModuleOp module;

  bool disableRegRandomization;
  bool emitSeparateAlwaysBlocks;

  bool needsRandom = false;
};
} // namespace circt

#endif // CONVERSION_SEQTOSV_FIRREGLOWERING_H
