//===- Sig2RegPass.cpp - Implement the Sig2Reg Pass -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement Pass to promote LLHD signals to SSA values.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-sig2reg"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_SIG2REG
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;

namespace {

/// Represents an offset of an interval relative to a root interval. All values
/// describe number of bits, not elements.
struct Offset {
  Offset(uint64_t min, uint64_t max, ArrayRef<Value> dynamic)
      : min(min), max(max), dynamic(dynamic) {}

  Offset(uint64_t idx) : min(idx), max(idx) {}

  // The lower bound of the offset known statically.
  uint64_t min = 0;
  // The upper bound of the offset known statically.
  uint64_t max = -1;
  // A list of SSA values used to compute the final offset.
  SmallVector<Value> dynamic;

  /// Returns if we know the exact offset statically.
  bool isStatic() const { return min == max; }
};

/// Represents an alias interval within a root interval that is written to or
/// read from. All values refer to number of bits, not elements.
struct Interval {
  Interval(const Offset &low, uint64_t bitwidth, Value value,
           llhd::TimeAttr delay = llhd::TimeAttr())
      : low(low), bitwidth(bitwidth), value(value), delay(delay) {}

  // The offset of the interval relative to the root interval (i.e. all the bits
  // of the original signal).
  Offset low;
  // The width of the interval.
  uint64_t bitwidth;
  // The value written to this interval or the OpResult of a read.
  Value value;
  // The delay with which the value is written.
  llhd::TimeAttr delay;
};

class SigPromoter {
public:
  SigPromoter(llhd::SignalOp sigOp) : sigOp(sigOp) {}

  // Start at the signal operation and traverse all alias operations to compute
  // all the intervals and sort them by ascending offset.
  LogicalResult computeIntervals() {
    SmallVector<std::pair<Operation *, Offset>> stack;

    for (auto *user : sigOp->getUsers())
      stack.emplace_back(user, Offset(0));

    while (!stack.empty()) {
      auto currAndOffset = stack.pop_back_val();
      auto *curr = currAndOffset.first;
      auto offset = currAndOffset.second;

      if (curr->getBlock() != sigOp->getBlock()) {
        LLVM_DEBUG(llvm::dbgs() << "  - User in other block, skipping...\n\n");
        return failure();
      }

      auto result =
          TypeSwitch<Operation *, LogicalResult>(curr)
              .Case<llhd::PrbOp>([&](llhd::PrbOp probeOp) {
                auto bw = hw::getBitWidth(probeOp.getResult().getType());
                if (bw <= 0)
                  return failure();

                readIntervals.emplace_back(offset, bw, probeOp.getResult());
                return success();
              })
              .Case<llhd::DrvOp>([&](llhd::DrvOp driveOp) {
                if (driveOp.getEnable()) {
                  LLVM_DEBUG(llvm::dbgs()
                             << "  - Conditional driver, skipping...\n\n");
                  return failure();
                }

                auto timeOp =
                    driveOp.getTime().getDefiningOp<llhd::ConstantTimeOp>();
                if (!timeOp) {
                  LLVM_DEBUG(llvm::dbgs()
                             << "  - Unknown drive delay, skipping...\n\n");
                  return failure();
                }

                auto bw = hw::getBitWidth(driveOp.getValue().getType());
                if (bw <= 0)
                  return failure();

                intervals.emplace_back(offset, bw, driveOp.getValue(),
                                       timeOp.getValueAttr());
                return success();
              })
              .Case<llhd::SigExtractOp>([&](llhd::SigExtractOp extractOp) {
                if (auto constOp =
                        extractOp.getLowBit().getDefiningOp<hw::ConstantOp>();
                    constOp && offset.isStatic()) {
                  for (auto *user : extractOp->getUsers())
                    stack.emplace_back(
                        user,
                        Offset(constOp.getValue().getZExtValue() + offset.min));

                  return success();
                }

                auto bw = hw::getBitWidth(
                    cast<hw::InOutType>(extractOp.getInput().getType())
                        .getElementType());
                if (bw <= 0)
                  return failure();

                SmallVector<Value> indices(offset.dynamic);
                indices.push_back(extractOp.getLowBit());

                for (auto *user : extractOp->getUsers())
                  stack.emplace_back(
                      user, Offset(offset.min, offset.max + bw - 1, indices));

                return success();
              })
              .Default([](auto *op) {
                LLVM_DEBUG(llvm::dbgs() << "  - User that is not a probe or "
                                           "drive, skipping...\n    "
                                        << *op << "\n\n");
                return failure();
              });

      if (failed(result))
        return failure();

      toDelete.push_back(curr);
    }

    llvm::sort(intervals, [](const Interval &a, const Interval &b) {
      return a.low.min < b.low.min;
    });

    LLVM_DEBUG({
      llvm::dbgs() << "  - Detected intervals:\n";
      dumpIntervals(llvm::dbgs(), 4);
    });

    return success();
  }

#ifndef NDEBUG

  /// Print the list of intervals in a readable format for debugging.
  void dumpIntervals(llvm::raw_ostream &os, unsigned indent = 0) {
    os << llvm::indent(indent) << "[\n";
    for (const auto &interval : intervals) {
      os << llvm::indent(indent + 2) << "<from [" << interval.low.min << ", "
         << interval.low.max << "]\n";
      os << llvm::indent(indent + 3) << "width " << interval.bitwidth << "\n";

      for (auto idx : interval.low.dynamic)
        os << llvm::indent(indent + 3) << idx << "\n";

      os << llvm::indent(indent + 3) << "value: " << interval.value << "\n";
      os << llvm::indent(indent + 3) << "delay: " << interval.delay << "\n";
      os << llvm::indent(indent + 2) << ">,\n";
    }
    os << llvm::indent(indent) << "]\n";
  }

#endif

  /// Check if we can promote the entire signal according to the current
  /// limitations of the pass.
  bool isPromotable() {
    for (unsigned i = 0; i < intervals.size(); ++i) {
      if (i >= intervals.size() - 1)
        break;

      if (intervals[i].low.max + intervals[i].bitwidth - 1 >
          intervals[i + 1].low.min) {
        LLVM_DEBUG({
          llvm::dbgs() << "  - Potentially overlapping drives, skipping...\n\n";
        });
        return false;
      }
    }

    return true;
  }

  /// Promote the signal. This builds the necessary operations, replaces the
  /// values, and removes the signal and signal value handling operations.
  void promote() {
    auto bw = hw::getBitWidth(sigOp.getInit().getType());
    assert(bw > 0 && "bw must be known and non-zero");

    OpBuilder builder(sigOp);
    Value val = sigOp.getInit();
    Location loc = sigOp->getLoc();
    auto type = builder.getIntegerType(bw);
    val = builder.createOrFold<hw::BitcastOp>(loc, type, val);

    // Handle the writes by starting with the signal init value and injecting
    // the written values at the right offsets.
    for (auto interval : intervals) {
      Value invMask = builder.create<hw::ConstantOp>(
          loc, APInt::getAllOnes(interval.bitwidth));

      if (uint64_t(bw) > interval.bitwidth) {
        Value pad = builder.create<hw::ConstantOp>(
            loc, APInt::getZero(bw - interval.bitwidth));
        invMask = builder.createOrFold<comb::ConcatOp>(loc, pad, invMask);
      }

      Value amt = buildDynamicIndex(builder, loc, interval.low.min,
                                    interval.low.dynamic, bw);
      invMask = builder.createOrFold<comb::ShlOp>(loc, invMask, amt);
      Value allOnes =
          builder.create<hw::ConstantOp>(loc, APInt::getAllOnes(bw));
      Value mask = builder.createOrFold<comb::XorOp>(loc, invMask, allOnes);
      val = builder.createOrFold<comb::AndOp>(loc, val, mask);

      Value assignVal = builder.createOrFold<hw::BitcastOp>(
          loc, builder.getIntegerType(interval.bitwidth), interval.value);

      if (uint64_t(bw) > interval.bitwidth) {
        Value pad = builder.create<hw::ConstantOp>(
            loc, APInt::getZero(bw - interval.bitwidth));
        assignVal = builder.createOrFold<comb::ConcatOp>(loc, pad, assignVal);
      }

      assignVal = builder.createOrFold<comb::ShlOp>(loc, assignVal, amt);
      if (!isImmediate(interval.delay))
        assignVal =
            builder.createOrFold<llhd::DelayOp>(loc, assignVal, interval.delay);
      val = builder.createOrFold<comb::OrOp>(loc, assignVal, val);
    }

    // Handle the reads by extracting right number of bits at the right offset.
    for (auto interval : readIntervals) {
      if (interval.low.isStatic()) {
        Value read = builder.createOrFold<comb::ExtractOp>(
            loc, builder.getIntegerType(interval.bitwidth), val,
            interval.low.min);
        read = builder.createOrFold<hw::BitcastOp>(
            loc, interval.value.getType(), read);
        interval.value.replaceAllUsesWith(read);
        continue;
      }

      Value read = buildDynamicIndex(builder, loc, interval.low.min,
                                     interval.low.dynamic, bw);
      read = builder.createOrFold<comb::ShrUOp>(loc, val, read);
      read = builder.createOrFold<comb::ExtractOp>(
          loc, builder.getIntegerType(interval.bitwidth), read, 0);
      read = builder.createOrFold<hw::BitcastOp>(loc, interval.value.getType(),
                                                 read);
      interval.value.replaceAllUsesWith(read);
    }

    // Delete all operations operating on signal values.
    for (auto *op : llvm::reverse(toDelete))
      op->erase();

    sigOp->erase();
  }

private:
  /// Given a static offset and a list of dynamic offset values, materialize an
  /// SSA value that adds all these offsets together and is an integer with the
  /// given 'width'.
  Value buildDynamicIndex(OpBuilder &builder, Location loc,
                          uint64_t constOffset, ArrayRef<Value> indices,
                          uint64_t width) {
    Value index = builder.create<hw::ConstantOp>(
        loc, builder.getIntegerType(width), constOffset);

    for (auto idx : indices) {
      auto bw = hw::getBitWidth(idx.getType());
      Value pad =
          builder.create<hw::ConstantOp>(loc, APInt::getZero(width - bw));
      idx = builder.createOrFold<comb::ConcatOp>(loc, pad, idx);
      index = builder.createOrFold<comb::AddOp>(loc, index, idx);
    }

    return index;
  }

  bool isImmediate(llhd::TimeAttr attr) const {
    return attr.getTime() == 0 && attr.getDelta() == 0 &&
           attr.getEpsilon() == 1;
  }

  // The signal to be promoted.
  llhd::SignalOp sigOp;
  // Intervals written to.
  SmallVector<Interval> intervals;
  // Intervals read from.
  SmallVector<Interval> readIntervals;
  // Operations to delete after promotion is done.
  SmallVector<Operation *> toDelete;
};

struct Sig2RegPass : public circt::llhd::impl::Sig2RegBase<Sig2RegPass> {
  void runOnOperation() override;
};
} // namespace

void Sig2RegPass::runOnOperation() {
  hw::HWModuleOp moduleOp = getOperation();

  LLVM_DEBUG(llvm::dbgs() << "=== Sig2Reg in module " << moduleOp.getSymName()
                          << "\n\n");

  for (auto sigOp :
       llvm::make_early_inc_range(moduleOp.getOps<llhd::SignalOp>())) {
    LLVM_DEBUG(llvm::dbgs() << "  - Attempting to promote signal "
                            << sigOp.getName() << "\n");
    SigPromoter promoter(sigOp);
    if (failed(promoter.computeIntervals()) || !promoter.isPromotable())
      continue;

    promoter.promote();
    LLVM_DEBUG(llvm::dbgs() << "  - Successfully promoted!\n\n");
  }

  LLVM_DEBUG({
    if (moduleOp.getOps<llhd::SignalOp>().empty())
      llvm::dbgs() << "  Successfully promoted all signals in module!\n";
  });

  LLVM_DEBUG(llvm::dbgs() << "\n");
}
