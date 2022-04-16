//===- SeqPasses.cpp - Implement Seq passes -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace circt;
using namespace seq;

namespace circt {
namespace seq {
#define GEN_PASS_CLASSES
#include "circt/Dialect/Seq/SeqPasses.h.inc"
} // namespace seq
} // namespace circt

namespace {
struct SeqToSVPass : public LowerSeqToSVBase<SeqToSVPass> {
  void runOnOperation() override;
};
} // anonymous namespace

namespace {
class RegisterLowering {
public:
  RegisterLowering(Block *block) : block(block) {}

  /// Lower all registers in a block to a single `sv.always` block for each
  /// clock and reset signal combination placed after the last op.
  void lower() {
    for (auto reg : llvm::make_early_inc_range(block->getOps<CompRegOp>()))
      lower(reg);
  }

  /// Lower CompRegOp to `sv.reg` and `sv.alwaysff`. Use a posedge clock and
  /// synchronous reset.
  void lower(CompRegOp reg);

private:
  using AlwaysFFKeyType =
      std::tuple<sv::EventControl, Value, ResetType, sv::EventControl, Value>;
  llvm::SmallDenseMap<AlwaysFFKeyType, sv::AlwaysFFOp> alwaysBlocks;

  void addToAlwaysFFBlock(Operation *reg, sv::EventControl clockEdge,
                          Value clock, std::function<void(OpBuilder &)> body,
                          ResetType resetStyle = {},
                          sv::EventControl resetEdge = {}, Value reset = {},
                          std::function<void(OpBuilder &)> resetBody = {});

  Block *block;
};
} // namespace

void RegisterLowering::lower(CompRegOp reg) {
  Location loc = reg.getLoc();

  ImplicitLocOpBuilder builder(loc, block->getParentOp()->getContext());
  builder.setInsertionPoint(reg);

  auto svReg =
      builder.create<sv::RegOp>(reg.getResult().getType(), reg.nameAttr());
  svReg->setDialectAttrs(reg->getDialectAttrs());

  // If the seq::CompRegOp has an inner_sym attribute, set this for the
  // sv::RegOp inner_sym attribute.
  if (reg.sym_name().hasValue())
    svReg.inner_symAttr(reg.sym_nameAttr());

  auto regVal = builder.create<sv::ReadInOutOp>(loc, svReg);
  reg.replaceAllUsesWith(regVal.getResult());

  if (reg.reset() && reg.resetValue()) {
    addToAlwaysFFBlock(
        reg, sv::EventControl::AtPosEdge, reg.clk(),
        [&](OpBuilder &builder) {
          builder.create<sv::PAssignOp>(loc, svReg, reg.input());
        },
        ResetType::SyncReset, sv::EventControl::AtPosEdge, reg.reset(),
        [&](OpBuilder &builder) {
          builder.create<sv::PAssignOp>(loc, svReg, reg.resetValue());
        });
  } else {
    addToAlwaysFFBlock(reg, sv::EventControl::AtPosEdge, reg.clk(),
                       [&](OpBuilder &builder) {
                         builder.create<sv::PAssignOp>(loc, svReg, reg.input());
                       });
  }

  reg.erase();
}

void RegisterLowering::addToAlwaysFFBlock(
    Operation *reg, sv::EventControl clockEdge, Value clock,
    std::function<void(OpBuilder &)> body, ResetType resetStyle,
    sv::EventControl resetEdge, Value reset,
    std::function<void(OpBuilder &)> resetBody) {

  // Fetch an existing block with the same triggers or create a new one.
  auto &op = alwaysBlocks[{clockEdge, clock, resetStyle, resetEdge, reset}];
  if (!op) {
    ImplicitLocOpBuilder builder(block->getParentOp()->getLoc(), reg);
    if (reset) {
      assert(resetStyle != ::ResetType::NoReset);

      op = builder.create<sv::AlwaysFFOp>(
          clockEdge, clock, resetStyle, resetEdge, reset, [&] {}, [&] {});
    } else {
      assert(!resetBody);
      op = builder.create<sv::AlwaysFFOp>(clockEdge, clock);
    }
  }

  // Extend the reset & body blocks.
  if (reset) {
    auto resetBuilder = OpBuilder::atBlockEnd(op.getResetBlock());
    resetBody(resetBuilder);
  }
  auto bodyBuilder = OpBuilder::atBlockEnd(op.getBodyBlock());
  body(bodyBuilder);

  // Move the op to the last register using it to ensure that values
  // it might use are defined beforehand in the exported Verilog.
  op->moveBefore(reg);
}

void SeqToSVPass::runOnOperation() {
  getOperation()->walk([&](Block *block) { RegisterLowering(block).lower(); });
}

namespace circt {
namespace seq {
std::unique_ptr<Pass> createSeqLowerToSVPass() {
  return std::make_unique<SeqToSVPass>();
}
} // namespace seq
} // namespace circt

namespace {
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Seq/SeqPasses.h.inc"
} // namespace

void circt::seq::registerSeqPasses() { registerPasses(); }
