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
struct SeqToSVPass : public impl::LowerSeqToSVBase<SeqToSVPass> {
  void runOnOperation() override;
};
struct SeqFIRRTLToSVPass
    : public impl::LowerSeqFIRRTLToSVBase<SeqFIRRTLToSVPass> {
  void runOnOperation() override;
  using LowerSeqFIRRTLToSVBase<SeqFIRRTLToSVPass>::disableRegRandomization;
  using LowerSeqFIRRTLToSVBase<
      SeqFIRRTLToSVPass>::addVivadoRAMAddressConflictSynthesisBugWorkaround;
  using LowerSeqFIRRTLToSVBase<SeqFIRRTLToSVPass>::LowerSeqFIRRTLToSVBase;
  using LowerSeqFIRRTLToSVBase<SeqFIRRTLToSVPass>::numSubaccessRestored;
};
} // anonymous namespace

/// Create the assign.
static void createAssign(ConversionPatternRewriter &rewriter, sv::RegOp svReg,
                         CompRegOp reg) {
  rewriter.create<sv::PAssignOp>(reg.getLoc(), svReg, reg.getInput());
}
/// Create the assign inside of an if block.
static void createAssign(ConversionPatternRewriter &rewriter, sv::RegOp svReg,
                         CompRegClockEnabledOp reg) {
  Location loc = reg.getLoc();
  rewriter.create<sv::IfOp>(loc, reg.getClockEnable(), [&]() {
    rewriter.create<sv::PAssignOp>(reg.getLoc(), svReg, reg.getInput());
  });
}

namespace {
/// Lower CompRegOp to `sv.reg` and `sv.alwaysff`. Use a posedge clock and
/// synchronous reset.
template <typename OpTy>
struct CompRegLower : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<OpTy>::OpAdaptor;

  LogicalResult
  matchAndRewrite(OpTy reg, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = reg.getLoc();

    auto svReg = rewriter.create<sv::RegOp>(loc, reg.getResult().getType(),
                                            reg.getNameAttr());
    svReg->setDialectAttrs(reg->getDialectAttrs());

    // If the seq::CompRegOp has an inner_sym attribute, set this for the
    // sv::RegOp inner_sym attribute.
    if (reg.getSymName().has_value())
      svReg.setInnerSymAttr(reg.getSymNameAttr());

    circt::sv::setSVAttributes(svReg, circt::sv::getSVAttributes(reg));

    auto regVal = rewriter.create<sv::ReadInOutOp>(loc, svReg);
    if (reg.getReset() && reg.getResetValue()) {
      rewriter.create<sv::AlwaysFFOp>(
          loc, sv::EventControl::AtPosEdge, reg.getClk(), ResetType::SyncReset,
          sv::EventControl::AtPosEdge, reg.getReset(),
          [&]() { createAssign(rewriter, svReg, reg); },
          [&]() {
            rewriter.create<sv::PAssignOp>(loc, svReg, reg.getResetValue());
          });
    } else {
      rewriter.create<sv::AlwaysFFOp>(
          loc, sv::EventControl::AtPosEdge, reg.getClk(),
          [&]() { createAssign(rewriter, svReg, reg); });
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
  FirRegLower(hw::HWModuleOp module, bool disableRegRandomization = false,
              bool addVivadoRAMAddressConflictSynthesisBugWorkaround = false)
      : module(module), disableRegRandomization(disableRegRandomization),
        addVivadoRAMAddressConflictSynthesisBugWorkaround(
            addVivadoRAMAddressConflictSynthesisBugWorkaround){};

  void lower();

  unsigned numSubaccessRestored = 0;

private:
  struct RegLowerInfo {
    sv::RegOp reg;
    Value asyncResetSignal;
    Value asyncResetValue;
    int64_t randStart;
    size_t width;
  };

  RegLowerInfo lower(FirRegOp reg);

  void initialize(OpBuilder &builder, RegLowerInfo reg, ArrayRef<Value> rands);
  void initializeRegisterElements(Location loc, OpBuilder &builder, Value reg,
                                  Value rand, unsigned &pos);

  void createTree(OpBuilder &builder, Value reg, Value term, Value next);
  std::optional<std::tuple<Value, Value, Value>>
  tryRestoringSubaccess(OpBuilder &builder, Value reg, Value term,
                        hw::ArrayCreateOp nextArray);

  void addToAlwaysBlock(Block *block, sv::EventControl clockEdge, Value clock,
                        std::function<void(OpBuilder &)> body,
                        ResetType resetStyle = {},
                        sv::EventControl resetEdge = {}, Value reset = {},
                        std::function<void(OpBuilder &)> resetBody = {});

  void addToIfBlock(OpBuilder &builder, Value cond,
                    const std::function<void()> &trueSide,
                    const std::function<void()> &falseSide);

  hw::ConstantOp getOrCreateConstant(Location loc, const APInt &value) {
    OpBuilder builder(module.getBody());
    auto &constant = constantCache[value];
    if (constant) {
      constant->setLoc(builder.getFusedLoc(constant->getLoc(), loc));
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

  hw::HWModuleOp module;

  bool disableRegRandomization;
  bool addVivadoRAMAddressConflictSynthesisBugWorkaround;
};
} // namespace

void FirRegLower::addToIfBlock(OpBuilder &builder, Value cond,
                               const std::function<void()> &trueSide,
                               const std::function<void()> &falseSide) {
  auto op = ifCache.lookup({builder.getBlock(), cond});
  // Always build both sides of the if, in case we want to use an empty else
  // later. This way we don't have to build a new if and replace it.
  if (!op) {
    auto newIfOp =
        builder.create<sv::IfOp>(cond.getLoc(), cond, trueSide, falseSide);
    ifCache.insert({{builder.getBlock(), cond}, newIfOp});
  } else {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(op.getThenBlock());
    trueSide();
    builder.setInsertionPointToEnd(op.getElseBlock());
    falseSide();
  }
}

void FirRegLower::lower() {
  // Find all registers to lower in the module.
  auto regs = module.getOps<seq::FirRegOp>();
  if (regs.empty())
    return;

  // Lower the regs to SV regs.
  SmallVector<RegLowerInfo> toInit;
  for (auto reg : llvm::make_early_inc_range(regs))
    toInit.push_back(lower(reg));

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
  if (toInit.empty() || disableRegRandomization)
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

// Return true if two arguments are equivalent, or if both of them are the same
// array indexing.
// NOLINTNEXTLINE(misc-no-recursion)
static bool areEquivalentValues(Value term, Value next) {
  if (term == next)
    return true;
  // Check whether these values are equivalent array accesses with constant
  // index. We have to check the equivalence recursively because they might not
  // be CSEd.
  if (auto t1 = term.getDefiningOp<hw::ArrayGetOp>())
    if (auto t2 = next.getDefiningOp<hw::ArrayGetOp>())
      if (auto c1 = t1.getIndex().getDefiningOp<hw::ConstantOp>())
        if (auto c2 = t2.getIndex().getDefiningOp<hw::ConstantOp>())
          return c1.getType() == c2.getType() &&
                 c1.getValue() == c2.getValue() &&
                 areEquivalentValues(t1.getInput(), t2.getInput());
  // Otherwise, regard as different.
  // TODO: Handle struct if necessary.
  return false;
}

static llvm::SetVector<Value> extractConditions(Value value) {
  auto andOp = value.getDefiningOp<comb::AndOp>();
  // If the value is not AndOp with a bin flag, use it as a condition.
  if (!andOp || !andOp.getTwoState()) {
    llvm::SetVector<Value> ret;
    ret.insert(value);
    return ret;
  }

  return llvm::SetVector<Value>(andOp.getOperands().begin(),
                                andOp.getOperands().end());
}

static std::optional<APInt> getConstantValue(Value value) {
  auto constantIndex = value.template getDefiningOp<hw::ConstantOp>();
  if (constantIndex)
    return constantIndex.getValue();
  return {};
}

// Return a tuple <cond, idx, val> if the array register update can be
// represented with a dynamic index assignment:
// if (cond)
//   reg[idx] <= val;
//
std::optional<std::tuple<Value, Value, Value>>
FirRegLower::tryRestoringSubaccess(OpBuilder &builder, Value reg, Value term,
                                   hw::ArrayCreateOp nextRegValue) {
  Value trueVal;
  SmallVector<Value> muxConditions;
  if (!llvm::all_of(
          llvm::enumerate(llvm::reverse(nextRegValue.getOperands())),
          [&](auto idxAndValue) {
            // Check that `nextRegValue[i]` is `cond_i ? val : reg[i]`.
            auto [i, value] = idxAndValue;
            auto mux = value.template getDefiningOp<comb::MuxOp>();
            // Ensure that mux has binary flag.
            if (!mux || !mux.getTwoState())
              return false;
            // The next value must be same.
            if (trueVal && trueVal != mux.getTrueValue())
              return false;
            if (!trueVal)
              trueVal = mux.getTrueValue();
            muxConditions.push_back(mux.getCond());
            // Check that ith element is an element of the register we are
            // currently lowering.
            auto arrayGet =
                mux.getFalseValue().template getDefiningOp<hw::ArrayGetOp>();
            if (!arrayGet)
              return false;
            return areEquivalentValues(arrayGet.getInput(), term) &&
                   getConstantValue(arrayGet.getIndex()) == i;
          }))
    return {};

  // Extract common expressions among mux conditions.
  llvm::SetVector<Value> commonConditions =
      extractConditions(muxConditions.front());
  for (auto condition : ArrayRef(muxConditions).drop_front()) {
    auto cond = extractConditions(condition);
    commonConditions.remove_if([&](auto v) { return !cond.contains(v); });
  }
  Value indexValue;
  for (auto [idx, condition] : llvm::enumerate(muxConditions)) {
    llvm::SetVector<Value> extractedConditions = extractConditions(condition);
    // Remove common conditions and check the remaining condition is only an
    // index comparision.
    extractedConditions.remove_if(
        [&](auto v) { return commonConditions.contains(v); });
    if (extractedConditions.size() != 1)
      return {};

    auto indexCompare =
        (*extractedConditions.begin()).getDefiningOp<comb::ICmpOp>();
    if (!indexCompare || !indexCompare.getTwoState() ||
        indexCompare.getPredicate() != comb::ICmpPredicate::eq)
      return {};
    // `IndexValue` must be same.
    if (indexValue && indexValue != indexCompare.getLhs())
      return {};
    if (!indexValue)
      indexValue = indexCompare.getLhs();
    if (getConstantValue(indexCompare.getRhs()) != idx)
      return {};
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfterValue(reg);
  Value commonConditionValue;
  if (commonConditions.empty())
    commonConditionValue = getOrCreateConstant(reg.getLoc(), APInt(1, 1));
  else
    commonConditionValue = builder.createOrFold<comb::AndOp>(
        reg.getLoc(), builder.getI1Type(), commonConditions.takeVector(), true);
  return std::make_tuple(commonConditionValue, indexValue, trueVal);
}

void FirRegLower::createTree(OpBuilder &builder, Value reg, Value term,
                             Value next) {
  // If term and next values are equivalent, we don't have to create an
  // assignment.
  if (areEquivalentValues(term, next))
    return;
  auto mux = next.getDefiningOp<comb::MuxOp>();
  if (mux && mux.getTwoState()) {
    addToIfBlock(
        builder, mux.getCond(),
        [&]() { createTree(builder, reg, term, mux.getTrueValue()); },
        [&]() { createTree(builder, reg, term, mux.getFalseValue()); });
  } else {
    // If the next value is an array creation, split the value into
    // invidial elements and construct trees recursively.
    if (auto array = next.getDefiningOp<hw::ArrayCreateOp>()) {
      // First, try restoring subaccess assignments.
      if (auto matchResultOpt =
              tryRestoringSubaccess(builder, reg, term, array)) {
        Value cond, index, trueValue;
        std::tie(cond, index, trueValue) = *matchResultOpt;
        addToIfBlock(
            builder, cond,
            [&]() {
              auto nextReg = builder.create<sv::ArrayIndexInOutOp>(reg.getLoc(),
                                                                   reg, index);
              auto termElement =
                  builder.create<hw::ArrayGetOp>(term.getLoc(), term, index);
              createTree(builder, nextReg, termElement, trueValue);
              termElement.erase();
            },
            []() {});
        ++numSubaccessRestored;
        return;
      }
      for (auto [idx, value] :
           llvm::enumerate(llvm::reverse(array.getOperands()))) {

        // Create an index constant.
        auto idxVal = getOrCreateConstant(
            array.getLoc(),
            APInt(std::max(1u, llvm::Log2_64_Ceil(array.getOperands().size())),
                  idx));

        auto &index = arrayIndexCache[{reg, idx}];
        if (!index) {
          // Create an array index op just after `reg`.
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointAfterValue(reg);
          index =
              builder.create<sv::ArrayIndexInOutOp>(reg.getLoc(), reg, idxVal);
        }

        auto termElement =
            builder.create<hw::ArrayGetOp>(term.getLoc(), term, idxVal);
        createTree(builder, index, termElement, value);
        // This value was used to check the equivalence of elements so useless
        // anymore.
        termElement.erase();
      }
      return;
    }
    builder.create<sv::PAssignOp>(term.getLoc(), reg, next);
  }
}

FirRegLower::RegLowerInfo FirRegLower::lower(FirRegOp reg) {
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

  // For array registers, we annotate ram_style attributes if
  // `addVivadoRAMAddressConflictSynthesisBugWorkaround` is enabled so that we
  // can workaround incorrect optimizations of vivado. See "RAM address conflict
  // and Vivado synthesis bug" issue in the vivado forum for the more detail.
  if (addVivadoRAMAddressConflictSynthesisBugWorkaround &&
      hw::type_isa<hw::ArrayType, hw::UnpackedArrayType>(reg.getType()))
    circt::sv::setSVAttributes(
        svReg.reg,
        sv::SVAttributeAttr::get(builder.getContext(), "ram_style",
                                 R"("distributed")", /*emitAsComment=*/false));

  if (auto innerSymAttr = reg.getInnerSymAttr())
    svReg.reg.setInnerSymAttr(innerSymAttr);

  auto regVal = builder.create<sv::ReadInOutOp>(loc, svReg.reg);

  if (reg.hasReset()) {
    addToAlwaysBlock(
        module.getBodyBlock(), sv::EventControl::AtPosEdge, reg.getClk(),
        [&](OpBuilder &b) {
          // If this is an AsyncReset, ensure that we emit a self connect to
          // avoid erroneously creating a latch construct.
          if (reg.getIsAsync() && areEquivalentValues(reg, reg.getNext()))
            b.create<sv::PAssignOp>(reg.getLoc(), svReg.reg, reg);
          else
            createTree(b, svReg.reg, reg, reg.getNext());
        },
        reg.getIsAsync() ? ResetType::AsyncReset : ResetType::SyncReset,
        sv::EventControl::AtPosEdge, reg.getReset(),
        [&](OpBuilder &builder) {
          builder.create<sv::PAssignOp>(loc, svReg.reg, reg.getResetValue());
        });
    if (reg.getIsAsync()) {
      svReg.asyncResetSignal = reg.getReset();
      svReg.asyncResetValue = reg.getResetValue();
    }
  } else {
    addToAlwaysBlock(
        module.getBodyBlock(), sv::EventControl::AtPosEdge, reg.getClk(),
        [&](OpBuilder &b) { createTree(b, svReg.reg, reg, reg.getNext()); });
  }

  reg.replaceAllUsesWith(regVal.getResult());
  reg.erase();

  return svReg;
}

// Initialize registers by assigning each element recursively instead of
// initializing entire registers. This is necessary as a workaround for
// verilator which allocates many local variables for concat op.
void FirRegLower::initializeRegisterElements(Location loc, OpBuilder &builder,
                                             Value reg, Value randomSource,
                                             unsigned &pos) {
  auto type = reg.getType().cast<sv::InOutType>().getElementType();
  if (auto intTy = hw::type_dyn_cast<IntegerType>(type)) {
    // Use randomSource[pos-1:pos-width] as a random value.
    pos -= intTy.getWidth();
    auto elem = builder.createOrFold<comb::ExtractOp>(loc, randomSource, pos,
                                                      intTy.getWidth());
    builder.create<sv::BPAssignOp>(loc, reg, elem);
  } else if (auto array = hw::type_dyn_cast<hw::ArrayType>(type)) {
    for (unsigned i = 0, e = array.getSize(); i < e; ++i) {
      auto index = getOrCreateConstant(loc, APInt(llvm::Log2_64_Ceil(e), i));
      initializeRegisterElements(
          loc, builder, builder.create<sv::ArrayIndexInOutOp>(loc, reg, index),
          randomSource, pos);
    }
  } else if (auto structType = hw::type_dyn_cast<hw::StructType>(type)) {
    for (auto e : structType.getElements())
      initializeRegisterElements(
          loc, builder,
          builder.create<sv::StructFieldInOutOp>(loc, reg, e.name),
          randomSource, pos);
  } else {
    assert(false && "unsupported type");
  }
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
  unsigned pos = reg.width;
  // Initialize register elements.
  initializeRegisterElements(loc, builder, reg.reg, concat, pos);
}

void FirRegLower::addToAlwaysBlock(Block *block, sv::EventControl clockEdge,
                                   Value clock,
                                   std::function<void(OpBuilder &)> body,
                                   ::ResetType resetStyle,
                                   sv::EventControl resetEdge, Value reset,
                                   std::function<void(OpBuilder &)> resetBody) {
  auto loc = clock.getLoc();
  auto builder = ImplicitLocOpBuilder::atBlockTerminator(loc, block);

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
  patterns.add<CompRegLower<CompRegOp>>(&ctxt);
  patterns.add<CompRegLower<CompRegClockEnabledOp>>(&ctxt);

  if (failed(applyPartialConversion(top, target, std::move(patterns))))
    signalPassFailure();
}

void SeqFIRRTLToSVPass::runOnOperation() {
  hw::HWModuleOp module = getOperation();
  FirRegLower firRegLower(module, disableRegRandomization,
                          addVivadoRAMAddressConflictSynthesisBugWorkaround);
  firRegLower.lower();
  numSubaccessRestored += firRegLower.numSubaccessRestored;
}

std::unique_ptr<Pass> circt::seq::createSeqLowerToSVPass() {
  return std::make_unique<SeqToSVPass>();
}

std::unique_ptr<Pass> circt::seq::createSeqFIRRTLLowerToSVPass(
    const LowerSeqFIRRTLToSVOptions &options) {
  return std::make_unique<SeqFIRRTLToSVPass>(options);
}
