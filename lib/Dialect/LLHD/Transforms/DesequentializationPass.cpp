//===- DesequentializationPass.cpp - Convert processes to registers -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements a pass to lower sequential processes to registers. This is not
// always possible. In that case, it just leaves the process unconverted.
//
//===----------------------------------------------------------------------===//

#include "TemporalRegions.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Comb/CombVisitors.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/FVInt.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-desequentialization"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_DESEQUENTIALIZATION
#include "circt/Dialect/LLHD/Transforms/Passes.h.inc"
} // namespace llhd
} // namespace circt

using namespace circt;
using namespace mlir;

namespace {
/// Visits boolean comb operations from a 'root' value to a list of given
/// 'primitives' and computes the result of this expression for each possible
/// input combination.
class CombInterpreter : public comb::CombinationalVisitor<CombInterpreter> {
public:
  /// Takes a list of variables to be considered primitive. The 'truthTable'
  /// list must have the same number of elements and each 'APInt' represents the
  /// possible configurations those primitive variables can be in. This visitor
  /// then computes an 'APInt' representing the result of the 'truthTable' in a
  /// vectorized fashion.
  APInt compute(ArrayRef<Value> primitives, ArrayRef<APInt> truthTable,
                uint64_t width, Value root) {
    assert(primitives.size() == truthTable.size() && "must have same size");

    for (auto [p, t] : llvm::zip(primitives, truthTable))
      results[p] = t;

    this->width = width;

    DenseSet<Operation *> visited;
    SmallVector<Value> worklist;
    worklist.push_back(root);

    while (!worklist.empty()) {
      auto curr = worklist.back();
      if (results.contains(curr)) {
        worklist.pop_back();
        continue;
      }

      auto *defOp = curr.getDefiningOp();
      if (!defOp) {
        worklist.pop_back();
        continue;
      }

      if (!visited.contains(defOp)) {
        visited.insert(defOp);

        bool addedToWorklist =
            TypeSwitch<Operation *, bool>(defOp)
                .Case<comb::AndOp, comb::OrOp, comb::XorOp>([&](auto op) {
                  worklist.append(llvm::to_vector(op->getOperands()));
                  return true;
                })
                .Case([&](comb::ICmpOp op) {
                  if ((op.getPredicate() == circt::comb::ICmpPredicate::eq ||
                       op.getPredicate() == circt::comb::ICmpPredicate::ne) &&
                      op.getLhs().getType().isSignlessInteger(1)) {
                    worklist.append(llvm::to_vector(op->getOperands()));
                    return true;
                  }
                  return false;
                })
                .Case<hw::ConstantOp>([&](hw::ConstantOp op) {
                  results[op.getResult()] = op.getValue().getBoolValue()
                                                ? APInt::getAllOnes(width)
                                                : APInt(width, 0);
                  worklist.pop_back();
                  return true;
                });
        if (addedToWorklist)
          continue;
      }

      dispatchCombinationalVisitor(defOp);
      worklist.pop_back();
    }

    return results[root];
  }

  void visitComb(comb::AndOp op) {
    auto res = APInt::getAllOnes(width);
    for (auto operand : op->getOperands())
      res &= results[operand];
    results[op.getResult()] = res;
  }

  void visitComb(comb::OrOp op) {
    auto res = APInt(width, 0);
    for (auto operand : op->getOperands())
      res |= results[operand];
    results[op.getResult()] = res;
  }

  void visitComb(comb::XorOp op) {
    auto res = results[op->getOperands()[0]];
    for (auto operand : op->getOperands().drop_front())
      res ^= results[operand];
    results[op.getResult()] = res;
  }

  void visitComb(comb::ICmpOp op) {
    auto res = results[op.getLhs()];
    res ^= results[op.getRhs()];
    if (op.getPredicate() == comb::ICmpPredicate::eq)
      res ^= APInt::getAllOnes(width);
    results[op.getResult()] = res;
  }

  // Define the following methods to make the compiler happy. They will never be
  // called.
  void visitComb(comb::AddOp op) { visitInvalidComb(op); }
  void visitComb(comb::SubOp op) { visitInvalidComb(op); }
  void visitComb(comb::MulOp op) { visitInvalidComb(op); }
  void visitComb(comb::DivUOp op) { visitInvalidComb(op); }
  void visitComb(comb::DivSOp op) { visitInvalidComb(op); }
  void visitComb(comb::ModUOp op) { visitInvalidComb(op); }
  void visitComb(comb::ModSOp op) { visitInvalidComb(op); }
  void visitComb(comb::ShlOp op) { visitInvalidComb(op); }
  void visitComb(comb::ShrUOp op) { visitInvalidComb(op); }
  void visitComb(comb::ShrSOp op) { visitInvalidComb(op); }
  void visitComb(comb::ParityOp op) { visitInvalidComb(op); }
  void visitComb(comb::ConcatOp op) { visitInvalidComb(op); }
  void visitComb(comb::ReplicateOp op) { visitInvalidComb(op); }
  void visitComb(comb::ExtractOp op) { visitInvalidComb(op); }
  void visitComb(comb::MuxOp op) { visitInvalidComb(op); }

private:
  DenseMap<Value, APInt> results;
  unsigned width;
};

/// Represents a single register trigger. This is basically one row in the DNF
/// truth table. However, several ones can typically combined into one.
struct Trigger {
  enum class Kind {
    PosEdge,
    NegEdge,
    Edge,
  };

  static StringRef stringify(const Trigger::Kind &kind) {
    switch (kind) {
    case Trigger::Kind::PosEdge:
      return "posedge";
    case Trigger::Kind::NegEdge:
      return "negedge";
    case Trigger::Kind::Edge:
      return "edge";
    }
    llvm::llvm_unreachable_internal("all cases considered above");
  }

  /// Clock and async reset values.
  SmallVector<Value> clocks;

  /// Determines whether this trigger is a clock and at which edge or an async
  /// reset.
  SmallVector<Kind> kinds;

  /// Null Value when AsyncReset or no enable present.
  Value enable;
};
} // namespace

template <typename T>
static T &operator<<(T &os, const Trigger::Kind &kind) {
  return os << Trigger::stringify(kind);
}

namespace {
/// Analyses a boolean expression and converts it to a list of register
/// triggers.
class DnfAnalyzer {
public:
  DnfAnalyzer(Value value, function_ref<bool(Value)> sampledInPast)
      : root(value) {
    assert(value.getType().isSignlessInteger(1) &&
           "only 1-bit signless integers supported");

    DenseSet<Value> alreadyAdded;

    SmallVector<Value> worklist;
    worklist.push_back(value);

    while (!worklist.empty()) {
      Value curr = worklist.pop_back_val();
      auto *defOp = curr.getDefiningOp();
      if (!defOp) {
        if (!alreadyAdded.contains(curr)) {
          primitives.push_back(curr);
          primitiveSampledInPast.push_back(sampledInPast(curr));
          alreadyAdded.insert(curr);
        }
        continue;
      }

      TypeSwitch<Operation *>(defOp)
          .Case<comb::AndOp, comb::OrOp, comb::XorOp>([&](auto op) {
            worklist.append(llvm::to_vector(op->getOperands()));
          })
          .Case([&](comb::ICmpOp op) {
            if ((op.getPredicate() == circt::comb::ICmpPredicate::eq ||
                 op.getPredicate() == circt::comb::ICmpPredicate::ne) &&
                op.getLhs().getType().isSignlessInteger(1)) {
              worklist.append(llvm::to_vector(op->getOperands()));
            } else {
              if (!alreadyAdded.contains(curr)) {
                primitives.push_back(curr);
                primitiveSampledInPast.push_back(sampledInPast(curr));
                alreadyAdded.insert(curr);
              }
            }
          })
          .Case<hw::ConstantOp>([](auto op) { /* ignore */ })
          .Default([&](auto op) {
            if (!alreadyAdded.contains(curr)) {
              primitives.push_back(curr);
              primitiveSampledInPast.push_back(sampledInPast(curr));
              alreadyAdded.insert(curr);
            }
          });
    }

    LLVM_DEBUG({
      for (auto val : primitives)
        llvm::dbgs() << "  - Primitive variable: " << val << "\n";
    });

    this->isClock = SmallVector<bool>(primitives.size(), false);
    this->dontCare = SmallVector<APInt>(primitives.size(),
                                        APInt(1ULL << primitives.size(), 0));
  }

  /// Note that clocks can be dual edge triggered, but this is not directly
  /// returned as a single Trigger but instead two Triggers with the same clock
  /// value, one negedge and one posedge kind. The enable should also match in
  /// that case.
  LogicalResult
  computeTriggers(OpBuilder &builder, Location loc,
                  function_ref<bool(Value, Value)> sampledFromSameSignal,
                  SmallVectorImpl<Trigger> &triggers, unsigned maxPrimitives) {
    if (primitives.size() > maxPrimitives) {
      LLVM_DEBUG({ llvm::dbgs() << "  Too many primitives, skipping...\n"; });
      return failure();
    }

    // Populate the truth table and the result APInt.
    computeTruthTable();

    // Detect primitive variable pairs that form a clock and mark them as such.
    // If a variable sampled in the past cannot be matched with a sample in the
    // present to form a clock, it will return failure.
    if (failed(computeClockValuePairs(sampledFromSameSignal)))
      return failure();

    // Perform boolean expression simplification.
    simplifyTruthTable();

    LLVM_DEBUG({
      llvm::dbgs() << "  - Truth table:\n";

      for (auto [t, d] : llvm::zip(truthTable, dontCare))
        llvm::dbgs() << "    " << FVInt(std::move(t), std::move(d)) << "\n";

      SmallVector<char> str;
      result.toString(str, 2, false);

      llvm::dbgs() << "    ";
      for (unsigned i = 0; i < result.getBitWidth() - str.size(); ++i)
        llvm::dbgs() << '0';

      llvm::dbgs() << str << "\n";
    });

    // Compute the enable expressions for each trigger. Make sure that the SSA
    // value for a specific configuration is reused, allowing for easier
    // canonicalization and merging of triggers later on.
    materializeTriggerEnables(builder, loc);

    // Iterate over the truth table and extract the triggers.
    extractTriggerList(triggers);

    // Canonicalize and merge triggers in the trigger list.
    canonicalizeTriggerList(triggers, builder, loc);

    return success();
  }

private:
  FVInt computeEnableKey(unsigned tableRow) {
    FVInt key = FVInt::getAllX(primitives.size());
    for (unsigned k = 0; k < primitives.size(); ++k) {
      if (dontCare[k][tableRow])
        continue;

      if (primitiveSampledInPast[k])
        continue;

      // TODO: allow the present value of the clock if it is not trivially
      // satisfied by the trigger
      if (isClock[k])
        continue;

      key.setBit(k, truthTable[k][tableRow]);
    }

    return key;
  }

  void extractTriggerList(SmallVectorImpl<Trigger> &triggers) {
    for (uint64_t i = 0, e = 1ULL << primitives.size(); i < e; ++i) {
      if (!result[i])
        continue;

      auto key = computeEnableKey(i);
      Trigger trigger;
      for (auto clk : clockPairs) {
        if (dontCare[clk.second][i] && dontCare[clk.first][i])
          continue;

        trigger.clocks.push_back(primitives[clk.second]);
        trigger.kinds.push_back(truthTable[clk.second][i]
                                    ? Trigger::Kind::PosEdge
                                    : Trigger::Kind::NegEdge);
      }
      trigger.enable = enableMap[key];

      if (!trigger.clocks.empty())
        triggers.push_back(trigger);
    }
  }

  void materializeTriggerEnables(OpBuilder &builder, Location loc) {
    Value trueVal =
        builder.create<hw::ConstantOp>(loc, builder.getBoolAttr(true));
    for (uint64_t i = 0, e = 1ULL << primitives.size(); i < e; ++i) {
      if (!result[i])
        continue;

      auto key = computeEnableKey(i);

      if (!enableMap.contains(key)) {
        SmallVector<Value> conjuncts;
        for (unsigned k = 0; k < primitives.size(); ++k) {
          if (dontCare[k][i])
            continue;

          if (primitiveSampledInPast[k])
            continue;

          // TODO: allow the present value of the clock if it is not trivially
          // satisfied by the trigger
          if (isClock[k])
            continue;

          if (truthTable[k][i]) {
            conjuncts.push_back(primitives[k]);
            continue;
          }
          conjuncts.push_back(
              builder.create<comb::XorOp>(loc, primitives[k], trueVal));
        }
        if (!conjuncts.empty())
          enableMap[key] =
              builder.createOrFold<comb::AndOp>(loc, conjuncts, false);
      }
    }
  }

  LogicalResult computeClockValuePairs(
      function_ref<bool(Value, Value)> sampledFromSameSignal) {
    for (unsigned k = 0; k < primitives.size(); ++k) {
      if (isClock[k])
        continue;

      for (unsigned l = k + 1; l < primitives.size(); ++l) {
        if (sampledFromSameSignal(primitives[k], primitives[l]) &&
            (primitiveSampledInPast[k] != primitiveSampledInPast[l])) {
          if (primitiveSampledInPast[k])
            clockPairs.emplace_back(k, l);
          else
            clockPairs.emplace_back(l, k);
          isClock[k] = true;
          isClock[l] = true;
        }
      }
      if (primitiveSampledInPast[k] && !isClock[k])
        return failure();
    }

    return success();
  }

  void simplifyTruthTable() {
    uint64_t numEntries = 1 << primitives.size();

    // Perform boolean expression simplifification (see Karnaugh maps).
    // NOTE: This is a very simple algorithm that may become a bottleneck.
    // Fortunately, there exist better algorithms we could implement if that
    // becomes necessay.
    for (uint64_t i = 0; i < numEntries; ++i) {
      if (!result[i])
        continue;

      for (uint64_t k = i + 1; k < numEntries; ++k) {
        if (!result[i])
          continue;

        unsigned differenceCount = 0;
        for (unsigned l = 0; l < primitives.size(); ++l) {
          if (truthTable[l][i] != truthTable[l][k])
            ++differenceCount;
          if (differenceCount > 1)
            break;
        }

        if (differenceCount == 1) {
          for (unsigned l = 0; l < primitives.size(); ++l) {
            dontCare[l].setBit(k);
            if (truthTable[l][i] != truthTable[l][k])
              dontCare[l].setBit(i);
          }
        }
      }
    }
  }

  void computeTruthTable() {
    uint64_t numEntries = 1 << primitives.size();
    for (auto _ [[maybe_unused]] : primitives)
      truthTable.push_back(APInt(numEntries, 0));

    for (uint64_t i = 0; i < numEntries; ++i)
      for (unsigned k = 0; k < primitives.size(); ++k)
        truthTable[k].setBitVal(i, APInt(64, i)[k]);

    result =
        CombInterpreter().compute(primitives, truthTable, numEntries, root);
  }

  void canonicalizeTriggerList(SmallVectorImpl<Trigger> &triggers,
                               OpBuilder &builder, Location loc) {
    for (auto *iter1 = triggers.begin(); iter1 != triggers.end(); ++iter1) {
      for (auto *iter2 = iter1 + 1; iter2 != triggers.end(); ++iter2) {
        if (iter1->clocks == iter2->clocks && iter1->kinds == iter2->kinds) {
          iter1->enable =
              builder.create<comb::OrOp>(loc, iter1->enable, iter2->enable);
          triggers.erase(iter2--);
        }
      }
    }

    // TODO: merge negedge and posedge triggers on the same clock with the same
    // enables to an 'edge' trigger.
  }

  Value root;
  SmallVector<Value> primitives;
  SmallVector<bool> isClock;
  SmallVector<bool> primitiveSampledInPast;
  SmallVector<APInt> truthTable;
  SmallVector<APInt> dontCare;
  SmallVector<std::pair<unsigned, unsigned>> clockPairs;
  DenseMap<FVInt, Value> enableMap;
  APInt result;
};

struct DesequentializationPass
    : public llhd::impl::DesequentializationBase<DesequentializationPass> {
  DesequentializationPass()
      : llhd::impl::DesequentializationBase<DesequentializationPass>() {}
  DesequentializationPass(const llhd::DesequentializationOptions &options)
      : llhd::impl::DesequentializationBase<DesequentializationPass>(options) {
    maxPrimitives.setValue(options.maxPrimitives);
  }
  void runOnOperation() override;
  void runOnProcess(llhd::ProcessOp procOp) const;
  LogicalResult
  isSupportedSequentialProcess(llhd::ProcessOp procOp,
                               const llhd::TemporalRegionAnalysis &trAnalysis,
                               SmallVectorImpl<Value> &observed) const;
};
} // namespace

LogicalResult DesequentializationPass::isSupportedSequentialProcess(
    llhd::ProcessOp procOp, const llhd::TemporalRegionAnalysis &trAnalysis,
    SmallVectorImpl<Value> &observed) const {
  unsigned numTRs = trAnalysis.getNumTemporalRegions();

  // We only consider the case with three basic blocks and two TRs, because
  // combinatorial circuits have fewer blocks and don't need
  // desequentialization and more are not supported for now
  // NOTE: 3 basic blocks because of the entry block and one for each TR
  if (numTRs == 1) {
    LLVM_DEBUG({
      llvm::dbgs() << "  Combinational process -> no need to desequentialize\n";
    });
    return failure();
  }

  if (numTRs > 2 || procOp.getBody().getBlocks().size() != 3) {
    LLVM_DEBUG(
        { llvm::dbgs() << "  Complex sequential process -> not supported\n"; });
    return failure();
  }

  bool seenWait = false;
  WalkResult result = procOp.walk([&](llhd::WaitOp op) -> WalkResult {
    LLVM_DEBUG({ llvm::dbgs() << "  Analyzing Wait Operation:\n"; });
    for (auto obs : op.getObserved()) {
      observed.push_back(obs);
      LLVM_DEBUG({ llvm::dbgs() << "  - Observes: " << obs << "\n"; });
    }
    LLVM_DEBUG({ llvm::dbgs() << "\n"; });

    if (seenWait)
      return failure();

    // Check that the block containing the wait is the only exiting block of
    // that TR
    if (!trAnalysis.hasSingleExitBlock(
            trAnalysis.getBlockTR(op.getOperation()->getBlock())))
      return failure();

    seenWait = true;
    return WalkResult::advance();
  });

  if (result.wasInterrupted() || !seenWait) {
    LLVM_DEBUG(
        { llvm::dbgs() << "  Complex sequential process -> not supported\n"; });
    return failure();
  }

  LLVM_DEBUG(
      { llvm::dbgs() << "  Sequential process, attempt lowering...\n"; });

  return success();
}

void DesequentializationPass::runOnProcess(llhd::ProcessOp procOp) const {
  LLVM_DEBUG({
    std::string line(74, '-');
    llvm::dbgs() << "\n===" << line << "===\n";
    llvm::dbgs() << "=== Process\n";
    llvm::dbgs() << "===" << line << "===\n";
  });

  llhd::TemporalRegionAnalysis trAnalysis(procOp);

  // If we don't support it, just skip it.
  SmallVector<Value> observed;
  if (failed(isSupportedSequentialProcess(procOp, trAnalysis, observed)))
    return;

  OpBuilder builder(procOp);
  WalkResult result = procOp.walk([&](llhd::DrvOp op) {
    LLVM_DEBUG({ llvm::dbgs() << "\n  Lowering Drive Operation\n"; });

    if (!op.getEnable()) {
      LLVM_DEBUG({ llvm::dbgs() << "  - No enable condition -> skip\n"; });
      return WalkResult::advance();
    }

    Location loc = op.getLoc();
    builder.setInsertionPoint(op);
    int presentTR = trAnalysis.getBlockTR(op.getOperation()->getBlock());

    auto sampledInPast = [&](Value value) -> bool {
      if (isa<BlockArgument>(value))
        return false;

      if (!procOp->isAncestor(value.getDefiningOp()))
        return false;

      return trAnalysis.getBlockTR(value.getDefiningOp()->getBlock()) !=
             presentTR;
    };

    LLVM_DEBUG({ llvm::dbgs() << "  - Analyzing enable condition...\n"; });

    SmallVector<Trigger> triggers;
    auto sampledFromSameSignal = [](Value val1, Value val2) -> bool {
      if (auto prb1 = val1.getDefiningOp<llhd::PrbOp>())
        if (auto prb2 = val2.getDefiningOp<llhd::PrbOp>())
          return prb1.getSignal() == prb2.getSignal();

      // TODO: consider signals not represented by hw.inout (and thus don't have
      // an llhd.prb op to look at)
      return false;
    };

    if (failed(DnfAnalyzer(op.getEnable(), sampledInPast)
                   .computeTriggers(builder, loc, sampledFromSameSignal,
                                    triggers, maxPrimitives))) {
      LLVM_DEBUG({
        llvm::dbgs() << "  Unable to compute trigger list for drive condition, "
                        "skipping...\n";
      });
      return WalkResult::interrupt();
    }

    LLVM_DEBUG({
      if (triggers.empty())
        llvm::dbgs() << "  - no triggers found!\n";
    });

    LLVM_DEBUG({
      for (auto trigger : triggers) {
        llvm::dbgs() << "  - Trigger\n";
        for (auto [clk, kind] : llvm::zip(trigger.clocks, trigger.kinds))
          llvm::dbgs() << "    - " << kind << " "
                       << "clock: " << clk << "\n";

        if (trigger.enable)
          llvm::dbgs() << "      with enable: " << trigger.enable << "\n";
      }
    });

    // TODO: add support
    if (triggers.size() > 2 || triggers.empty())
      return WalkResult::interrupt();

    // TODO: add support
    if (triggers[0].clocks.size() != 1 || triggers[0].clocks.size() != 1)
      return WalkResult::interrupt();

    // TODO: add support
    if (triggers[0].kinds[0] == Trigger::Kind::Edge)
      return WalkResult::interrupt();

    if (!llvm::any_of(observed, [&](Value val) {
          return sampledFromSameSignal(val, triggers[0].clocks[0]) &&
                 val.getParentRegion() != procOp.getBody();
        }))
      return WalkResult::interrupt();

    Value clock = builder.create<seq::ToClockOp>(loc, triggers[0].clocks[0]);
    Value reset, resetValue;

    if (triggers[0].kinds[0] == Trigger::Kind::NegEdge)
      clock = builder.create<seq::ClockInverterOp>(loc, clock);

    if (triggers[0].enable)
      clock = builder.create<seq::ClockGateOp>(loc, clock, triggers[0].enable);

    if (triggers.size() == 2) {
      // TODO: add support
      if (triggers[1].clocks.size() != 1 || triggers[1].kinds.size() != 1)
        return WalkResult::interrupt();

      // TODO: add support
      if (triggers[1].kinds[0] == Trigger::Kind::Edge)
        return WalkResult::interrupt();

      // TODO: add support
      if (triggers[1].enable)
        return WalkResult::interrupt();

      if (!llvm::any_of(observed, [&](Value val) {
            return sampledFromSameSignal(val, triggers[1].clocks[0]) &&
                   val.getParentRegion() != procOp.getBody();
          }))
        return WalkResult::interrupt();

      reset = triggers[1].clocks[0];
      resetValue = op.getValue();

      if (triggers[1].kinds[0] == Trigger::Kind::NegEdge) {
        Value trueVal =
            builder.create<hw::ConstantOp>(loc, builder.getBoolAttr(true));
        reset = builder.create<comb::XorOp>(loc, reset, trueVal);
      }
    }

    // FIXME: this adds async resets as sync resets and might also add the reset
    // as clock and clock as reset.
    Value regOut = builder.create<seq::CompRegOp>(loc, op.getValue(), clock,
                                                  reset, resetValue);

    op.getEnableMutable().clear();
    op.getValueMutable().assign(regOut);

    LLVM_DEBUG(
        { llvm::dbgs() << "  Lowered Drive Operation successfully!\n\n"; });

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return;

  IRRewriter rewriter(builder);
  auto &entryBlock = procOp.getBody().getBlocks().front();

  // Delete the terminator of all blocks in the process.
  for (Block &block : procOp.getBody().getBlocks()) {
    block.getTerminator()->erase();

    if (!block.isEntryBlock())
      entryBlock.getOperations().splice(entryBlock.end(),
                                        block.getOperations());
  }

  rewriter.inlineBlockBefore(&entryBlock, procOp);
  procOp.erase();

  LLVM_DEBUG({ llvm::dbgs() << "Lowered process successfully!\n"; });
}

void DesequentializationPass::runOnOperation() {
  hw::HWModuleOp moduleOp = getOperation();
  for (auto procOp :
       llvm::make_early_inc_range(moduleOp.getOps<llhd::ProcessOp>()))
    runOnProcess(procOp);
}
