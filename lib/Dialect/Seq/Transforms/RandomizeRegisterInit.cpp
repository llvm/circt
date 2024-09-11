//===- RandomizeRegisterInit.cpp - Randomize initial values of registers --===//
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
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace seq {
#define GEN_PASS_DEF_RANDOMIZEREGISTERINIT
#include "circt/Dialect/Seq/SeqPasses.h.inc"
} // namespace seq
} // namespace circt

using namespace circt;
using namespace seq;
using namespace hw;

namespace {
struct RandomizeRegisterInitPass
    : public circt::seq::impl::RandomizeRegisterInitBase<
          RandomizeRegisterInitPass> {
  using RandomizeRegisterInitBase<
      RandomizeRegisterInitPass>::RandomizeRegisterInitBase;
  void runOnOperation() override;
  void runOnModule(hw::HWModuleOp module);
};
} // anonymous namespace

struct RegLowerInfo {
  CompRegOp compReg;
  IntegerAttr preset;
  Value asyncResetSignal;
  Value asyncResetValue;
  int64_t randStart;
  size_t width;
};

void RandomizeRegisterInitPass::runOnOperation() {
  auto module = getOperation();
  OpBuilder builder(module);
  builder.setInsertionPointToStart(module.getBody());
  builder.create<sv::MacroDeclOp>(builder.getUnknownLoc(), "RANDOM");
  for (auto hwModule : module.getBody()->getOps<hw::HWModuleOp>())
    runOnModule(hwModule);
}

void RandomizeRegisterInitPass::runOnModule(hw::HWModuleOp module) {
  SmallVector<RegLowerInfo> regs;
  for (auto reg : module.getOps<seq::CompRegOp>()) {

    // If it has an initial value, we don't randomize it.
    if (reg.getInitialValue())
      continue;

    // If it has a random init start attribute, we randomize it.
    if (auto attr =
            reg->getAttrOfType<IntegerAttr>("firrtl.random_init_start")) {
      RegLowerInfo info;
      info.compReg = reg;
      info.randStart = attr.getInt();
      info.width = hw::getBitWidth(reg.getType());

      regs.push_back(info);
    } else {
      RegLowerInfo info;
      info.compReg = reg;
      info.randStart = -1;
      info.width = hw::getBitWidth(reg.getType());

      regs.push_back(info);

      // Skip if there is no random init start attribute.
      // Do we want to enforce randomized initialization not only for initial
      // ops, but for all comp regs?
    }
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

  bool enableSV = true;

  auto builder = ImplicitLocOpBuilder::atBlockTerminator(module.getLoc(),
                                                         module.getBodyBlock());

  SmallVector<Type> resultTypes;
  for (auto reg : regs)
    resultTypes.push_back(reg.compReg.getResult().getType());

  auto initOp = builder.create<seq::InitialOp>(resultTypes, [&]() {
    builder.create<sv::VerbatimOp>("`INIT_RANDOM_PROLOG_");
    // Create randomization vector
    SmallVector<Value> randValues;
    auto numRandomCalls = (maxBit + 31) / 32;

    for (size_t i = 0; i < numRandomCalls; i++) {
      randValues.push_back(builder.create<sv::MacroRefExprSEOp>(
          module.getLoc(), builder.getIntegerType(32), "RANDOM"));
    }

    auto randomVector = builder.create<comb::ConcatOp>(randValues);
    SmallVector<Value> results;
    for (auto reg : regs) {
      // Extract the random value from the random vector
      auto randomValue = builder.create<comb::ExtractOp>(
          module.getLoc(), randomVector, reg.randStart, reg.width);
      results.push_back(randomValue);
    }

    builder.create<seq::YieldOp>(results);
  });

  for (auto [reg, init] : llvm::zip(regs, initOp.getResults())) {
    reg.compReg.getInitialValueMutable().assign(init);
  }
}

std::unique_ptr<Pass> circt::seq::createRandomizeRegisterInitPass() {
  return std::make_unique<RandomizeRegisterInitPass>();
}
