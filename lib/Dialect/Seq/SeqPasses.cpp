//===- SeqPasses.cpp - Implement Seq passes -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

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
/// Lower CompRegOp to `sv.reg` and `sv.alwaysff`. Use a posedge clock and
/// synchronous reset.
struct CompRegLower : public OpConversionPattern<CompRegOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CompRegOp reg, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = reg.getLoc();

    auto svReg = rewriter.create<sv::RegOp>(loc, reg.getResult().getType());
    DictionaryAttr regAttrs = reg->getAttrDictionary();
    if (!regAttrs.empty())
      svReg->setAttrs(regAttrs);
    if (!svReg->hasAttrOfType<StringAttr>("name"))
      // sv.reg requires a name attribute.
      svReg->setAttr("name", rewriter.getStringAttr(""));
    auto regVal = rewriter.create<sv::ReadInOutOp>(loc, svReg);

    if (reg.reset() && reg.resetValue()) {
      rewriter.create<sv::AlwaysFFOp>(
          loc, sv::EventControl::AtPosEdge, reg.clk(), ResetType::SyncReset,
          sv::EventControl::AtPosEdge, reg.reset(),
          [&]() { rewriter.create<sv::PAssignOp>(loc, svReg, reg.input()); },
          [&]() {
            rewriter.create<sv::PAssignOp>(loc, svReg, reg.resetValue());
          });
    } else {
      rewriter.create<sv::AlwaysFFOp>(
          loc, sv::EventControl::AtPosEdge, reg.clk(),
          [&]() { rewriter.create<sv::PAssignOp>(loc, svReg, reg.input()); });
    }

    rewriter.replaceOp(reg, {regVal});
    return success();
  }
};

class RegLower : public OpConversionPattern<RegOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RegOp reg, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final;
};

} // namespace

static bool isI1Constant(::mlir::Value value, size_t n)
{
  assert (value.getType().isSignlessInteger(1));
  if (auto constantOp = value.getDefiningOp<::circt::hw::ConstantOp>()) {
    return constantOp.value() == n;
  }

  return false;
}

static bool isI1Gnd(::mlir::Value value) { return isI1Constant(value, 0); }
static bool isI1Vdd(::mlir::Value value) { return isI1Constant(value, 1); }

LogicalResult
RegLower::matchAndRewrite(RegOp reg,
                          ArrayRef<Value> operands,
                          ConversionPatternRewriter &rewriter) const
{
  Location loc = reg.getLoc();

  auto svReg = rewriter.create<sv::RegOp>(loc, reg.getResult().getType());

  DictionaryAttr regAttrs = reg->getAttrDictionary();
  for (auto & attr : regAttrs) {
    if (attr.first != "clockEdge" && attr.first != "resetEdge" && attr.first != "resetType") {
      svReg->setAttr(attr.first, attr.second);
    }
  }
  if (!svReg->hasAttrOfType<StringAttr>("name"))
    // sv.reg requires a name attribute.
    svReg->setAttr("name", rewriter.getStringAttr(""));

  auto onAssignInput = [&]() {
    rewriter.create<sv::PAssignOp>(loc, svReg, reg.input());
  };

  auto onReset = [&]() {
    rewriter.create<sv::PAssignOp>(loc, svReg, reg.resetValue());
  };

  auto onRegularOperation = [&]() {
    if (isI1Vdd(reg.enable())) {
      return onAssignInput();
    }

    rewriter.create<sv::IfOp>(loc,
        reg.enable(),
        onAssignInput,
        nullptr
        );
  };

  if (!reg.reset() || isI1Gnd(reg.reset())) {
    rewriter.create<sv::AlwaysFFOp>(
        loc, reg.clockEdge(), reg.clk(),
        onRegularOperation);
  } else {
    // reg.resetEdge() will always be defined, when reg.reset() is set.
    rewriter.create<sv::AlwaysFFOp>(
        loc, reg.clockEdge(), reg.clk(),
        reg.resetType(), reg.resetEdge().getValue(),
        reg.reset(), onRegularOperation,
        onReset);
  }

  auto svRegValue = rewriter.create<sv::ReadInOutOp>(loc, svReg);
  rewriter.replaceOp(reg, { svRegValue });

  return success();
}

void SeqToSVPass::runOnOperation() {
  ModuleOp top = getOperation();
  MLIRContext &ctxt = getContext();

  ConversionTarget target(ctxt);
  target.addIllegalDialect<SeqDialect>();
  target.addLegalDialect<sv::SVDialect>();
  RewritePatternSet patterns(&ctxt);
  patterns.add<CompRegLower>(&ctxt);
  patterns.add<RegLower>(&ctxt);

  if (failed(applyPartialConversion(top, target, std::move(patterns))))
    signalPassFailure();
}

namespace circt {
namespace seq {
std::unique_ptr<OperationPass<ModuleOp>> createSeqLowerToSVPass() {
  return std::make_unique<SeqToSVPass>();
}
} // namespace seq
} // namespace circt

namespace {
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Seq/SeqPasses.h.inc"
} // namespace

void circt::seq::registerSeqPasses() { registerPasses(); }
