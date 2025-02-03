//===- FirRegLowering.cpp - FirReg lowering utilities ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FirRegLowering.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

using namespace circt;
using namespace hw;
using namespace seq;
using llvm::MapVector;

#define DEBUG_TYPE "lower-seq-firreg"

std::function<bool(const Operation *op)> OpUserInfo::opAllowsReachability =
    [](const Operation *op) -> bool {
  return (isa<comb::MuxOp, ArrayGetOp, ArrayCreateOp>(op));
};

bool ReachableMuxes::isMuxReachableFrom(seq::FirRegOp regOp,
                                        comb::MuxOp muxOp) {
  return llvm::any_of(regOp.getResult().getUsers(), [&](Operation *user) {
    if (!OpUserInfo::opAllowsReachability(user))
      return false;
    buildReachabilityFrom(user);
    return reachableMuxes[user].contains(muxOp);
  });
}

void ReachableMuxes::buildReachabilityFrom(Operation *startNode) {
  // This is a backward dataflow analysis.
  // First build a graph rooted at the `startNode`. Every user of an operation
  // that does not block the reachability is a child node. Then, the ops that
  // are reachable from a node is computed as the union of the Reachability of
  // all its child nodes.
  // The dataflow can be expressed as, for all child in the Children(node)
  // Reachability(node) = node + Union{Reachability(child)}
  if (visited.contains(startNode))
    return;

  // The stack to record enough information for an iterative post-order
  // traversal.
  llvm::SmallVector<OpUserInfo, 16> stk;

  stk.emplace_back(startNode);

  while (!stk.empty()) {
    auto &info = stk.back();
    Operation *currentNode = info.op;

    // Node is being visited for the first time.
    if (info.getAndSetUnvisited())
      visited.insert(currentNode);

    if (info.userIter != info.userEnd) {
      Operation *child = *info.userIter;
      ++info.userIter;
      if (!visited.contains(child))
        stk.emplace_back(child);

    } else { // All children of the node have been visited
      // Any op is reachable from itself.
      reachableMuxes[currentNode].insert(currentNode);

      for (auto *childOp : llvm::make_filter_range(
               info.op->getUsers(), OpUserInfo::opAllowsReachability)) {
        reachableMuxes[currentNode].insert(childOp);
        // Propagate the reachability backwards from m to currentNode.
        auto iter = reachableMuxes.find(childOp);
        assert(iter != reachableMuxes.end());

        // Add all the mux that was reachable from childOp, to currentNode.
        reachableMuxes[currentNode].insert(iter->getSecond().begin(),
                                           iter->getSecond().end());
      }
      stk.pop_back();
    }
  }
}

void FirRegLowering::addToIfBlock(OpBuilder &builder, Value cond,
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

FirRegLowering::FirRegLowering(TypeConverter &typeConverter,
                               hw::HWModuleOp module,
                               bool disableRegRandomization,
                               bool emitSeparateAlwaysBlocks)
    : typeConverter(typeConverter), module(module),
      disableRegRandomization(disableRegRandomization),
      emitSeparateAlwaysBlocks(emitSeparateAlwaysBlocks) {

  reachableMuxes = std::make_unique<ReachableMuxes>(module);
}

void FirRegLowering::lower() {
  // Find all registers to lower in the module.
  auto regs = module.getOps<seq::FirRegOp>();
  if (regs.empty())
    return;

  // Lower the regs to SV regs. Group them by initializer and reset kind.
  SmallVector<RegLowerInfo> randomInit, presetInit;
  llvm::MapVector<Value, SmallVector<RegLowerInfo>> asyncResets;
  for (auto reg : llvm::make_early_inc_range(regs)) {
    auto svReg = lower(reg);
    if (svReg.preset)
      presetInit.push_back(svReg);
    else if (!disableRegRandomization)
      randomInit.push_back(svReg);

    if (svReg.asyncResetSignal)
      asyncResets[svReg.asyncResetSignal].emplace_back(svReg);
  }

  // Compute total width of random space.  Place non-chisel registers at the end
  // of the space.  The Random space is unique to the initial block, due to
  // verilog thread rules, so we can drop trailing random calls if they are
  // unused.
  uint64_t maxBit = 0;
  for (auto reg : randomInit)
    if (reg.randStart >= 0)
      maxBit = std::max(maxBit, (uint64_t)reg.randStart + reg.width);

  for (auto &reg : randomInit) {
    if (reg.randStart == -1) {
      reg.randStart = maxBit;
      maxBit += reg.width;
    }
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
  if (randomInit.empty() && presetInit.empty() && asyncResets.empty())
    return;

  needsRandom = true;

  auto loc = module.getLoc();
  MLIRContext *context = module.getContext();
  auto randInitRef = sv::MacroIdentAttr::get(context, "RANDOMIZE_REG_INIT");

  auto builder =
      ImplicitLocOpBuilder::atBlockTerminator(loc, module.getBodyBlock());

  builder.create<sv::IfDefOp>("ENABLE_INITIAL_REG_", [&] {
    builder.create<sv::OrderedOutputOp>([&] {
      builder.create<sv::IfDefOp>("FIRRTL_BEFORE_INITIAL", [&] {
        builder.create<sv::VerbatimOp>("`FIRRTL_BEFORE_INITIAL");
      });

      builder.create<sv::InitialOp>([&] {
        if (!randomInit.empty()) {
          builder.create<sv::IfDefProceduralOp>("INIT_RANDOM_PROLOG_", [&] {
            builder.create<sv::VerbatimOp>("`INIT_RANDOM_PROLOG_");
          });
          builder.create<sv::IfDefProceduralOp>(randInitRef, [&] {
            // Create randomization vector
            SmallVector<Value> randValues;
            auto numRandomCalls = (maxBit + 31) / 32;
            auto logic = builder.create<sv::LogicOp>(
                loc,
                hw::UnpackedArrayType::get(builder.getIntegerType(32),
                                           numRandomCalls),
                "_RANDOM");
            // Indvar's width must be equal to `ceil(log2(numRandomCalls +
            // 1))` to avoid overflow.
            auto inducionVariableWidth = llvm::Log2_64_Ceil(numRandomCalls + 1);
            auto arrayIndexWith = llvm::Log2_64_Ceil(numRandomCalls);
            auto lb =
                getOrCreateConstant(loc, APInt::getZero(inducionVariableWidth));
            auto ub = getOrCreateConstant(
                loc, APInt(inducionVariableWidth, numRandomCalls));
            auto step =
                getOrCreateConstant(loc, APInt(inducionVariableWidth, 1));
            auto forLoop = builder.create<sv::ForOp>(
                loc, lb, ub, step, "i", [&](BlockArgument iter) {
                  auto rhs = builder.create<sv::MacroRefExprSEOp>(
                      loc, builder.getIntegerType(32), "RANDOM");
                  Value iterValue = iter;
                  if (!iter.getType().isInteger(arrayIndexWith))
                    iterValue = builder.create<comb::ExtractOp>(
                        loc, iterValue, 0, arrayIndexWith);
                  auto lhs = builder.create<sv::ArrayIndexInOutOp>(loc, logic,
                                                                   iterValue);
                  builder.create<sv::BPAssignOp>(loc, lhs, rhs);
                });
            builder.setInsertionPointAfter(forLoop);
            for (uint64_t x = 0; x < numRandomCalls; ++x) {
              auto lhs = builder.create<sv::ArrayIndexInOutOp>(
                  loc, logic,
                  getOrCreateConstant(loc, APInt(arrayIndexWith, x)));
              randValues.push_back(lhs.getResult());
            }

            // Create initialisers for all registers.
            for (auto &svReg : randomInit)
              initialize(builder, svReg, randValues);
          });
        }

        if (!presetInit.empty()) {
          for (auto &svReg : presetInit) {
            auto loc = svReg.reg.getLoc();
            auto elemTy = svReg.reg.getType().getElementType();
            auto cst = getOrCreateConstant(loc, svReg.preset.getValue());

            Value rhs;
            if (cst.getType() == elemTy)
              rhs = cst;
            else
              rhs = builder.create<hw::BitcastOp>(loc, elemTy, cst);

            builder.create<sv::BPAssignOp>(loc, svReg.reg, rhs);
          }
        }

        if (!asyncResets.empty()) {
          // If the register is async reset, we need to insert extra
          // initialization in post-randomization so that we can set the
          // reset value to register if the reset signal is enabled.
          for (auto &reset : asyncResets) {
            //  if (reset) begin
            //    ..
            //  end
            builder.create<sv::IfOp>(reset.first, [&] {
              for (auto &reg : reset.second)
                builder.create<sv::BPAssignOp>(reg.reg.getLoc(), reg.reg,
                                               reg.asyncResetValue);
            });
          }
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
  // If the value is not AndOp, use it as a condition.
  if (!andOp) {
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
FirRegLowering::tryRestoringSubaccess(OpBuilder &builder, Value reg, Value term,
                                      hw::ArrayCreateOp nextRegValue) {
  Value trueVal;
  SmallVector<Value> muxConditions;
  // Compat fix for GCC12's libstdc++, cannot use
  // llvm::enumerate(llvm::reverse(OperandRange)).  See #4900.
  SmallVector<Value> reverseOpValues(llvm::reverse(nextRegValue.getOperands()));
  if (!llvm::all_of(llvm::enumerate(reverseOpValues), [&](auto idxAndValue) {
        // Check that `nextRegValue[i]` is `cond_i ? val : reg[i]`.
        auto [i, value] = idxAndValue;
        auto mux = value.template getDefiningOp<comb::MuxOp>();
        // Ensure that mux has binary flag.
        if (!mux)
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
    if (!indexCompare ||
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
        reg.getLoc(), builder.getI1Type(), commonConditions.takeVector());
  return std::make_tuple(commonConditionValue, indexValue, trueVal);
}

void FirRegLowering::createTree(OpBuilder &builder, Value reg, Value term,
                                Value next) {
  // Get the fanout from this register before we build the tree. While we are
  // creating the tree of if/else statements from muxes, we only want to turn
  // muxes that are on the register's fanout into if/else statements. This is
  // required to get the correct enable inference. But other muxes in the tree
  // should be left as ternary operators. This is desirable because we don't
  // want to create if/else structure for logic unrelated to the register's
  // enable.
  auto firReg = term.getDefiningOp<seq::FirRegOp>();

  SmallVector<std::tuple<Block *, Value, Value, Value>> worklist;
  auto addToWorklist = [&](Value reg, Value term, Value next) {
    worklist.push_back({builder.getBlock(), reg, term, next});
  };

  auto getArrayIndex = [&](Value reg, Value idx) {
    // Create an array index op just after `reg`.
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfterValue(reg);
    return builder.create<sv::ArrayIndexInOutOp>(reg.getLoc(), reg, idx);
  };

  SmallVector<Value, 8> opsToDelete;
  addToWorklist(reg, term, next);
  while (!worklist.empty()) {
    OpBuilder::InsertionGuard guard(builder);
    Block *block;
    Value reg, term, next;
    std::tie(block, reg, term, next) = worklist.pop_back_val();
    builder.setInsertionPointToEnd(block);
    if (areEquivalentValues(term, next))
      continue;

    // If this is a two-state mux within the fanout from the register, we use
    // if/else structure for proper enable inference.
    auto mux = next.getDefiningOp<comb::MuxOp>();
    if (mux && 
        reachableMuxes->isMuxReachableFrom(firReg, mux)) {
      addToIfBlock(
          builder, mux.getCond(),
          [&]() { addToWorklist(reg, term, mux.getTrueValue()); },
          [&]() { addToWorklist(reg, term, mux.getFalseValue()); });
      continue;
    }
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
              Value nextReg = getArrayIndex(reg, index);
              // Create a value to use for equivalence checking in the
              // recursive calls. Add the value to `opsToDelete` so that it can
              // be deleted afterwards.
              auto termElement =
                  builder.create<hw::ArrayGetOp>(term.getLoc(), term, index);
              opsToDelete.push_back(termElement);
              addToWorklist(nextReg, termElement, trueValue);
            },
            []() {});
        ++numSubaccessRestored;
        continue;
      }
      // Compat fix for GCC12's libstdc++, cannot use
      // llvm::enumerate(llvm::reverse(OperandRange)).  See #4900.
      // SmallVector<Value> reverseOpValues(llvm::reverse(array.getOperands()));
      for (auto [idx, value] : llvm::enumerate(array.getOperands())) {
        idx = array.getOperands().size() - idx - 1;
        // Create an index constant.
        auto idxVal = getOrCreateConstant(
            array.getLoc(),
            APInt(std::max(1u, llvm::Log2_64_Ceil(array.getOperands().size())),
                  idx));

        auto &index = arrayIndexCache[{reg, idx}];
        if (!index)
          index = getArrayIndex(reg, idxVal);

        // Create a value to use for equivalence checking in the
        // recursive calls. Add the value to `opsToDelete` so that it can
        // be deleted afterwards.
        auto termElement =
            builder.create<hw::ArrayGetOp>(term.getLoc(), term, idxVal);
        opsToDelete.push_back(termElement);
        addToWorklist(index, termElement, value);
      }
      continue;
    }

    builder.create<sv::PAssignOp>(term.getLoc(), reg, next);
  }

  while (!opsToDelete.empty()) {
    auto value = opsToDelete.pop_back_val();
    assert(value.use_empty());
    value.getDefiningOp()->erase();
  }
}

FirRegLowering::RegLowerInfo FirRegLowering::lower(FirRegOp reg) {
  Location loc = reg.getLoc();
  Type regTy = typeConverter.convertType(reg.getType());

  ImplicitLocOpBuilder builder(reg.getLoc(), reg);
  RegLowerInfo svReg{nullptr, reg.getPresetAttr(), nullptr, nullptr, -1, 0};
  svReg.reg = builder.create<sv::RegOp>(loc, regTy, reg.getNameAttr());
  svReg.width = hw::getBitWidth(regTy);

  if (auto attr = reg->getAttrOfType<IntegerAttr>("firrtl.random_init_start"))
    svReg.randStart = attr.getUInt();

  // Don't move these over
  reg->removeAttr("firrtl.random_init_start");

  // Move Attributes
  svReg.reg->setDialectAttrs(reg->getDialectAttrs());

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
        reg.getIsAsync() ? sv::ResetType::AsyncReset : sv::ResetType::SyncReset,
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
// NOLINTBEGIN(misc-no-recursion)
void FirRegLowering::initializeRegisterElements(Location loc,
                                                OpBuilder &builder, Value reg,
                                                Value randomSource,
                                                unsigned &pos) {
  auto type = cast<sv::InOutType>(reg.getType()).getElementType();
  if (auto intTy = hw::type_dyn_cast<IntegerType>(type)) {
    // Use randomSource[pos-1:pos-width] as a random value.
    pos -= intTy.getWidth();
    auto elem = builder.createOrFold<comb::ExtractOp>(loc, randomSource, pos,
                                                      intTy.getWidth());
    builder.create<sv::BPAssignOp>(loc, reg, elem);
  } else if (auto array = hw::type_dyn_cast<hw::ArrayType>(type)) {
    for (unsigned i = 0, e = array.getNumElements(); i < e; ++i) {
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
// NOLINTEND(misc-no-recursion)

void FirRegLowering::initialize(OpBuilder &builder, RegLowerInfo reg,
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

void FirRegLowering::addToAlwaysBlock(
    Block *block, sv::EventControl clockEdge, Value clock,
    const std::function<void(OpBuilder &)> &body, sv::ResetType resetStyle,
    sv::EventControl resetEdge, Value reset,
    const std::function<void(OpBuilder &)> &resetBody) {
  auto loc = clock.getLoc();
  auto builder = ImplicitLocOpBuilder::atBlockTerminator(loc, block);
  AlwaysKeyType key{builder.getBlock(), clockEdge, clock,
                    resetStyle,         resetEdge, reset};

  sv::AlwaysOp alwaysOp;
  sv::IfOp insideIfOp;
  if (!emitSeparateAlwaysBlocks) {
    std::tie(alwaysOp, insideIfOp) = alwaysBlocks[key];
  }

  if (!alwaysOp) {
    if (reset) {
      assert(resetStyle != sv::ResetType::NoReset);
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
      if (resetStyle == sv::ResetType::AsyncReset) {
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

  if (!emitSeparateAlwaysBlocks) {
    alwaysBlocks[key] = {alwaysOp, insideIfOp};
  }
}
