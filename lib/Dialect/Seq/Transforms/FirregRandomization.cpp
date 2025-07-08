//===- FirregRandomization.cpp - Randomize initial values of registers --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace seq {
#define GEN_PASS_DEF_FIRREGRANDOMIZATION
#include "circt/Dialect/Seq/SeqPasses.h.inc"
} // namespace seq
} // namespace circt

using namespace circt;
using namespace seq;
using namespace hw;
using namespace mlir;
using namespace func;

namespace {
struct FirregRandomizationPass
    : public circt::seq::impl::FirregRandomizationBase<
          FirregRandomizationPass> {
  using FirregRandomizationBase<
      FirregRandomizationPass>::FirregRandomizationBase;
  void runOnOperation() override;
  void runOnModule(hw::HWModuleOp module, Operation *randomDecl);
  using FirregRandomizationBase<FirregRandomizationPass>::emitSV;
};
} // anonymous namespace

struct RegLowerInfo {
  CompRegOp compReg;
  int64_t randStart;
  size_t width;
};

static Value initialize(OpBuilder &builder, RegLowerInfo reg,
                        ArrayRef<Value> rands) {

  auto loc = reg.compReg.getLoc();
  SmallVector<Value> nibbles;
  if (reg.width == 0)
    return builder.create<hw::ConstantOp>(loc, APInt(reg.width, 0));

  uint64_t width = reg.width;
  uint64_t offset = reg.randStart;
  while (width) {
    auto index = offset / 32;
    auto start = offset % 32;
    auto nwidth = std::min(32 - start, width);
    auto elemVal = rands[index];
    auto elem =
        builder.createOrFold<comb::ExtractOp>(loc, elemVal, start, nwidth);
    nibbles.push_back(elem);
    offset += nwidth;
    width -= nwidth;
  }
  auto concat = builder.createOrFold<comb::ConcatOp>(loc, nibbles);
  auto bitcast = builder.createOrFold<hw::BitcastOp>(
      loc, reg.compReg.getResult().getType(), concat);

  // Initialize register elements.
  return bitcast;
}

void FirregRandomizationPass::runOnOperation() {
  auto module = getOperation();
  OpBuilder builder(module);

  builder.setInsertionPointToStart(module.getBody());
  Operation *randomDecl;
  if (emitSV) {
    randomDecl =
        builder.create<sv::MacroDeclOp>(builder.getUnknownLoc(), "RANDOM");
  } else {
    auto funcType = builder.getFunctionType({}, {builder.getIntegerType(32)});
    auto randomFunc = builder.create<func::FuncOp>(builder.getUnknownLoc(),
                                                   "random", funcType);
    randomFunc.setPrivate();
    randomDecl = randomFunc;
  }

  for (auto hwModule : module.getBody()->getOps<hw::HWModuleOp>())
    runOnModule(hwModule, randomDecl);
}

void FirregRandomizationPass::runOnModule(hw::HWModuleOp module,
                                          Operation *randomDecl) {
  SmallVector<RegLowerInfo> regs;
  for (auto reg : module.getOps<seq::CompRegOp>()) {
    // If it has an initial value, we don't randomize it.
    if (reg.getInitialValue())
      continue;

    RegLowerInfo info;
    info.compReg = reg;
    info.width = hw::getBitWidth(reg.getType());

    // If it has a random init start attribute, we randomize it.
    if (auto attr = reg->getAttrOfType<IntegerAttr>("firrtl.random_init_start"))
      info.randStart = attr.getInt();
    else
      info.randStart = -1;

    regs.push_back(info);
  }

  // Compute total width of random space.  Place non-chisel registers at the end
  // of the space.  The Random space is unique to the initial block, due to
  // verilog thread rules, so we can drop trailing random calls if they are
  // unused.
  uint64_t maxBit = 0;
  for (auto reg : regs)
    if (reg.randStart >= 0)
      maxBit = std::max(maxBit, (uint64_t)reg.randStart + reg.width);

  for (auto &reg : regs) {
    if (reg.randStart == -1) {
      reg.randStart = maxBit;
      maxBit += reg.width;
    }
  }

  auto builder = ImplicitLocOpBuilder::atBlockTerminator(module.getLoc(),
                                                         module.getBodyBlock());

  SmallVector<Type> resultTypes;
  for (auto reg : regs)
    resultTypes.push_back(reg.compReg.getResult().getType());

  auto loc = module.getLoc();

  auto init = builder.create<seq::InitialOp>(resultTypes, [&] {
    SmallVector<Value> initValues;

    // Create randomization vector
    SmallVector<Value> randValues;
    auto numRandomCalls = (maxBit + 31) / 32;
    if (emitSV) {
      if (!regs.empty()) {
        builder.create<sv::VerbatimOp>("`INIT_RANDOM_PROLOG_");
        for (uint64_t x = 0; x < numRandomCalls; ++x) {
          auto rand = builder.create<sv::MacroRefExprSEOp>(
              loc, builder.getIntegerType(32), "RANDOM");
          randValues.push_back(rand);
        }
      };
    } else {
      // Create a function call for `random`.
      for (uint64_t x = 0; x < numRandomCalls; ++x) {
        randValues.push_back(
            builder
                .create<mlir::func::CallOp>(loc, cast<func::FuncOp>(randomDecl))
                .getResult(0));
      }
    }
    // Create initialisers for all registers.
    for (auto &svReg : regs)
      initValues.push_back(::initialize(builder, svReg, randValues));
    builder.create<seq::YieldOp>(initValues);
  });

  for (auto [reg, init] : llvm::zip(regs, init.getResults())) {
    reg.compReg.getInitialValueMutable().assign(init);
  }
}

std::unique_ptr<Pass> circt::seq::createFirregRandomizationPass(
    const FirregRandomizationOptions &options) {
  return std::make_unique<FirregRandomizationPass>(options);
}
