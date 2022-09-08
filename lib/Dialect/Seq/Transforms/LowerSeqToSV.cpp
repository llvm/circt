//===- LowerSeqToSV.cpp - Seq to SV lowering ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transform translate Seq ops to SV.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/Namespace.h"
#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace seq;

namespace {
struct SeqToSVPass : public LowerSeqToSVBase<SeqToSVPass> {
  void runOnOperation() override;
};
struct SeqFIRRTLToSVPass : public LowerSeqFIRRTLToSVBase<SeqFIRRTLToSVPass> {
  void runOnOperation() override;
};
} // anonymous namespace

namespace {
/// Lower CompRegOp to `sv.reg` and `sv.alwaysff`. Use a posedge clock and
/// synchronous reset.
struct CompRegLower : public OpConversionPattern<CompRegOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CompRegOp reg, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = reg.getLoc();

    auto svReg = rewriter.create<sv::RegOp>(loc, reg.getResult().getType(),
                                            reg.getNameAttr());
    svReg->setDialectAttrs(reg->getDialectAttrs());

    // If the seq::CompRegOp has an inner_sym attribute, set this for the
    // sv::RegOp inner_sym attribute.
    if (reg.getSymName().has_value())
      svReg.setInnerSymAttr(reg.getSymNameAttr());

    if (auto attribute = circt::sv::getSVAttributes(reg))
      circt::sv::setSVAttributes(svReg, attribute);

    auto regVal = rewriter.create<sv::ReadInOutOp>(loc, svReg);
    if (reg.getReset() && reg.getResetValue()) {
      rewriter.create<sv::AlwaysFFOp>(
          loc, sv::EventControl::AtPosEdge, reg.getClk(), ResetType::SyncReset,
          sv::EventControl::AtPosEdge, reg.getReset(),
          [&]() { rewriter.create<sv::PAssignOp>(loc, svReg, reg.getInput()); },
          [&]() {
            rewriter.create<sv::PAssignOp>(loc, svReg, reg.getResetValue());
          });
    } else {
      rewriter.create<sv::AlwaysFFOp>(
          loc, sv::EventControl::AtPosEdge, reg.getClk(), [&]() {
            rewriter.create<sv::PAssignOp>(loc, svReg, reg.getInput());
          });
    }

    rewriter.replaceOp(reg, {regVal});
    return success();
  }
};
} // namespace

namespace {
/// Lower FirRegOp to `sv.reg` and `sv.always`.
class FirRegLower {
public:
  FirRegLower() = default;

  void lower(hw::HWModuleOp module);

private:
  struct RegLowerInfo {
    sv::RegOp reg;
    Value asyncResetSignal;
    Value asyncResetValue;
    int64_t randStart;
    size_t width;
  };

  RegLowerInfo lower(hw::HWModuleOp module, FirRegOp reg);

  void initialize(OpBuilder &builder, RegLowerInfo reg, ArrayRef<Value> rands);

  void addToAlwaysBlock(hw::HWModuleOp module, sv::EventControl clockEdge,
                        Value clock, std::function<void(OpBuilder &)> body,
                        ResetType resetStyle = {},
                        sv::EventControl resetEdge = {}, Value reset = {},
                        std::function<void(OpBuilder &)> resetBody = {});

  using AlwaysKeyType = std::tuple<Block *, sv::EventControl, Value, ResetType,
                                   sv::EventControl, Value>;
  llvm::SmallDenseMap<AlwaysKeyType, std::pair<sv::AlwaysOp, sv::IfOp>>
      alwaysBlocks;
};
} // namespace

void FirRegLower::lower(hw::HWModuleOp module) {
  // Find all registers to lower in the module.
  auto regs = module.getOps<seq::FirRegOp>();
  if (regs.empty())
    return;

  // Lower the regs to SV regs.
  SmallVector<RegLowerInfo> toInit;
  for (auto reg : llvm::make_early_inc_range(regs))
    toInit.push_back(lower(module, reg));

  // Compute total width of random space.  Place non-chisel registers at the end
  // of the space.  The Random space is unique to the initial block, due to
  // verilog thread rules, so we can drop trailing random calls if they are
  // unused.
  uint64_t maxBit = 0;
  for (auto reg : toInit)
    if (reg.randStart >= 0)
      maxBit = std::max(maxBit, (uint64_t)reg.randStart + reg.width);
  for (auto &reg : toInit)
    if (reg.randStart == -1) {
      reg.randStart = maxBit;
      maxBit += reg.width;
    }

  // Create an initial block at the end of the module where random
  // initialisation will be inserted.  Create two builders into the two
  // `ifdef` ops where the registers will be placed.
  //
  // `ifndef SYNTHESIS
  //   `ifdef RANDOMIZE_REG_INIT
  //      ... regBuilder ...
  //   `endif
  //   initial
  //     `INIT_RANDOM_PROLOG_
  //     ... initBuilder ..
  // `endif
  if (toInit.empty())
    return;

  auto loc = module.getLoc();
  MLIRContext *context = module.getContext();
  auto randInitRef = sv::MacroIdentAttr::get(context, "RANDOMIZE_REG_INIT");

  auto builder =
      ImplicitLocOpBuilder::atBlockTerminator(loc, module.getBodyBlock());
  builder.create<sv::IfDefOp>(
      "SYNTHESIS", [] {},
      [&] {
        builder.create<sv::OrderedOutputOp>([&] {
          builder.create<sv::IfDefOp>("FIRRTL_BEFORE_INITIAL", [&] {
            builder.create<sv::VerbatimOp>("`FIRRTL_BEFORE_INITIAL");
          });

          builder.create<sv::InitialOp>([&] {
            builder.create<sv::IfDefProceduralOp>("INIT_RANDOM_PROLOG_", [&] {
              builder.create<sv::VerbatimOp>("`INIT_RANDOM_PROLOG_");
            });
            llvm::MapVector<Value, SmallVector<RegLowerInfo>> resets;
            builder.create<sv::IfDefProceduralOp>(randInitRef, [&] {
              // Create randomization vector
              SmallVector<Value> randValues;
              for (uint64_t x = 0; x < (maxBit + 31) / 32; ++x) {
                auto lhs =
                    builder.create<sv::LogicOp>(loc, builder.getIntegerType(32),
                                                "_RANDOM_" + llvm::utostr(x));
                auto rhs = builder.create<sv::MacroRefExprSEOp>(
                    loc, builder.getIntegerType(32), "RANDOM");
                builder.create<sv::BPAssignOp>(loc, lhs, rhs);
                randValues.push_back(lhs.getResult());
              }

              // Create initialisers for all registers.
              for (auto &svReg : toInit) {
                initialize(builder, svReg, randValues);

                if (svReg.asyncResetSignal)
                  resets[svReg.asyncResetSignal].emplace_back(svReg);
              }
            });

            if (!resets.empty()) {
              builder.create<sv::IfDefProceduralOp>("RANDOMIZE", [&] {
                // If the register is async reset, we need to insert extra
                // initialization in post-randomization so that we can set the
                // reset value to register if the reset signal is enabled.
                for (auto &reset : resets) {
                  // Create a block guarded by the RANDOMIZE macro and the
                  // reset: `ifdef RANDOMIZE
                  //   if (reset) begin
                  //     ..
                  //   end
                  // `endif
                  builder.create<sv::IfOp>(reset.first, [&] {
                    for (auto &reg : reset.second)
                      builder.create<sv::BPAssignOp>(reg.reg.getLoc(), reg.reg,
                                                     reg.asyncResetValue);
                  });
                }
              });
            }
          });

          builder.create<sv::IfDefOp>("FIRRTL_AFTER_INITIAL", [&] {
            builder.create<sv::VerbatimOp>("`FIRRTL_AFTER_INITIAL");
          });
        });
      });

  module->removeAttr("firrtl.random_init_width");
}

FirRegLower::RegLowerInfo FirRegLower::lower(hw::HWModuleOp module,
                                             FirRegOp reg) {
  Location loc = reg.getLoc();

  ImplicitLocOpBuilder builder(reg.getLoc(), reg);
  RegLowerInfo svReg{nullptr, nullptr, nullptr, -1, 0};
  svReg.reg = builder.create<sv::RegOp>(loc, reg.getType(), reg.getNameAttr());
  svReg.width = hw::getBitWidth(reg.getResult().getType());
  if (auto attr = reg->getAttrOfType<IntegerAttr>("firrtl.random_init_start"))
    svReg.randStart = attr.getUInt();

  // Don't move these over
  reg->removeAttr("firrtl.random_init_start");

  // Move Attributes
  svReg.reg->setDialectAttrs(reg->getDialectAttrs());

  if (auto innerSymAttr = reg.getInnerSymAttr())
    svReg.reg.setInnerSymAttr(innerSymAttr);

  auto regVal = builder.create<sv::ReadInOutOp>(loc, svReg.reg);

  auto setInput = [&](OpBuilder &builder) {
    if (reg.getNext() != reg)
      builder.create<sv::PAssignOp>(loc, svReg.reg, reg.getNext());
  };

  if (reg.hasReset()) {
    addToAlwaysBlock(
        module, sv::EventControl::AtPosEdge, reg.getClk(), setInput,
        reg.getIsAsync() ? ResetType::AsyncReset : ResetType::SyncReset,
        sv::EventControl::AtPosEdge, reg.getReset(), [&](OpBuilder &builder) {
          builder.create<sv::PAssignOp>(loc, svReg.reg, reg.getResetValue());
        });
    if (reg.getIsAsync()) {
      svReg.asyncResetSignal = reg.getReset();
      svReg.asyncResetValue = reg.getResetValue();
    }
  } else {
    addToAlwaysBlock(module, sv::EventControl::AtPosEdge, reg.getClk(),
                     setInput);
  }

  reg.replaceAllUsesWith(regVal.getResult());
  reg.erase();

  return svReg;
}

void FirRegLower::initialize(OpBuilder &builder, RegLowerInfo reg,
                             ArrayRef<Value> rands) {
  auto loc = reg.reg.getLoc();
  SmallVector<Value> nibbles;
  if (reg.width == 0)
    return;

  uint64_t width = reg.width;
  uint64_t offset = reg.randStart;
  while (width) {
    auto index = offset / 32;
    auto start = offset % 32;
    auto nwidth = std::min(32 - start, width);
    auto elemVal = builder.create<sv::ReadInOutOp>(loc, rands[index]);
    auto elem =
        builder.createOrFold<comb::ExtractOp>(loc, elemVal, start, nwidth);
    nibbles.push_back(elem);
    offset += nwidth;
    width -= nwidth;
  }
  auto concat = builder.createOrFold<comb::ConcatOp>(loc, nibbles);
  auto bitcast = builder.createOrFold<hw::BitcastOp>(
      loc, reg.reg.getElementType(), concat);
  builder.create<sv::BPAssignOp>(loc, reg.reg, bitcast);
}

void FirRegLower::addToAlwaysBlock(hw::HWModuleOp module,
                                   sv::EventControl clockEdge, Value clock,
                                   std::function<void(OpBuilder &)> body,
                                   ::ResetType resetStyle,
                                   sv::EventControl resetEdge, Value reset,
                                   std::function<void(OpBuilder &)> resetBody) {
  auto loc = clock.getLoc();
  auto builder =
      ImplicitLocOpBuilder::atBlockTerminator(loc, module.getBodyBlock());

  auto &op = alwaysBlocks[{builder.getBlock(), clockEdge, clock, resetStyle,
                           resetEdge, reset}];
  auto &alwaysOp = op.first;
  auto &insideIfOp = op.second;

  if (!alwaysOp) {
    if (reset) {
      assert(resetStyle != ::ResetType::NoReset);
      // Here, we want to create the following structure with sv.always and
      // sv.if. If `reset` is async, we need to add `reset` to a sensitivity
      // list.
      //
      // sv.always @(clockEdge or reset) {
      //   sv.if (reset) {
      //     resetBody
      //   } else {
      //     body
      //   }
      // }

      auto createIfOp = [&]() {
        // It is weird but intended. Here we want to create an empty sv.if
        // with an else block.
        insideIfOp = builder.create<sv::IfOp>(
            reset, []() {}, []() {});
      };
      if (resetStyle == ::ResetType::AsyncReset) {
        sv::EventControl events[] = {clockEdge, resetEdge};
        Value clocks[] = {clock, reset};

        alwaysOp = builder.create<sv::AlwaysOp>(events, clocks, [&]() {
          if (resetEdge == sv::EventControl::AtNegEdge)
            llvm_unreachable("negative edge for reset is not expected");
          createIfOp();
        });
      } else {
        alwaysOp = builder.create<sv::AlwaysOp>(clockEdge, clock, createIfOp);
      }
    } else {
      assert(!resetBody);
      alwaysOp = builder.create<sv::AlwaysOp>(clockEdge, clock);
      insideIfOp = nullptr;
    }
  }

  if (reset) {
    assert(insideIfOp && "reset body must be initialized before");
    auto resetBuilder =
        ImplicitLocOpBuilder::atBlockEnd(loc, insideIfOp.getThenBlock());
    resetBody(resetBuilder);

    auto bodyBuilder =
        ImplicitLocOpBuilder::atBlockEnd(loc, insideIfOp.getElseBlock());
    body(bodyBuilder);
  } else {
    auto bodyBuilder =
        ImplicitLocOpBuilder::atBlockEnd(loc, alwaysOp.getBodyBlock());
    body(bodyBuilder);
  }
}

void SeqToSVPass::runOnOperation() {
  ModuleOp top = getOperation();

  MLIRContext &ctxt = getContext();
  ConversionTarget target(ctxt);
  target.addIllegalDialect<SeqDialect>();
  target.addLegalDialect<sv::SVDialect>();
  RewritePatternSet patterns(&ctxt);
  patterns.add<CompRegLower>(&ctxt);

  if (failed(applyPartialConversion(top, target, std::move(patterns))))
    signalPassFailure();
}

void SeqFIRRTLToSVPass::runOnOperation() {
  hw::HWModuleOp module = getOperation();
  FirRegLower().lower(module);
}

std::unique_ptr<Pass> circt::seq::createSeqLowerToSVPass() {
  return std::make_unique<SeqToSVPass>();
}

std::unique_ptr<Pass> circt::seq::createSeqFIRRTLLowerToSVPass() {
  return std::make_unique<SeqFIRRTLToSVPass>();
}
