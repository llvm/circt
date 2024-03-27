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

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/Namespace.h"
#include "circt/Support/SymCache.h"

namespace circt {
/// Lower FirRegOp to `sv.reg` and `sv.always`.
class FirRegLowering {
public:
  FirRegLowering(TypeConverter &typeConverter, hw::HWModuleOp module,
                 bool disableRegRandomization = false,
                 bool emitSeparateAlwaysBlocks = false)
      : typeConverter(typeConverter), module(module),
        disableRegRandomization(disableRegRandomization),
        emitSeparateAlwaysBlocks(emitSeparateAlwaysBlocks){};

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

  TypeConverter &typeConverter;
  hw::HWModuleOp module;

  bool disableRegRandomization;
  bool emitSeparateAlwaysBlocks;

  bool needsRandom = false;
};
} // namespace circt

#endif // CONVERSION_SEQTOSV_FIRREGLOWERING_H
