//===- FirRegLowering.cpp - FirReg lowering utilities ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FirRegLowering.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Support/Utils.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

#include <deque>

using namespace circt;
using namespace hw;
using namespace seq;
using llvm::MapVector;

#define DEBUG_TYPE "lower-seq-firreg"

static Value buildXMRTo(OpBuilder &builder, HierPathOp path, Location loc,
                        Type type) {
  auto name = path.getSymNameAttr();
  auto ref = mlir::FlatSymbolRefAttr::get(name);
  return sv::XMRRefOp::create(builder, loc, type, ref);
}

/// Immediately before the terminator, if present. Otherwise, the block's end.
static Block::iterator getBlockEnd(Block *block) {
  if (block->mightHaveTerminator())
    return Block::iterator(block->getTerminator());
  return block->end();
}

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
        sv::IfOp::create(builder, cond.getLoc(), cond, trueSide, falseSide);
    ifCache.insert({{builder.getBlock(), cond}, newIfOp});
  } else {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(op.getThenBlock());
    trueSide();
    builder.setInsertionPointToEnd(op.getElseBlock());
    falseSide();
  }
}

/// Attach an inner-sym to field-id 0 of the given register, or use an existing
/// inner-sym, if present.
static StringAttr getInnerSymFor(InnerSymbolNamespace &innerSymNS,
                                 seq::FirRegOp reg) {
  auto attr = reg.getInnerSymAttr();

  // If we have an inner sym attribute already, and if there exists a symbol for
  // field-id 0, then just return that.
  if (attr)
    if (auto sym = attr.getSymIfExists(0))
      return sym;

  // Otherwise, we have to create a new inner sym.
  auto *context = reg->getContext();

  auto hint = reg.getName();

  // Create our new property for field 0.
  auto sym = StringAttr::get(context, innerSymNS.newName(hint));
  auto property = hw::InnerSymPropertiesAttr::get(sym);

  // Build the new list of inner sym properties. Since properties are sorted by
  // field ID, our new property is first.
  SmallVector<hw::InnerSymPropertiesAttr> properties = {property};
  if (attr)
    llvm::append_range(properties, attr.getProps());

  // Build the new InnerSymAttr and attach it to the op.
  attr = hw::InnerSymAttr::get(context, properties);
  reg.setInnerSymAttr(attr);

  // Return the name of the new inner sym.
  return sym;
}

static InnerRefAttr getInnerRefTo(StringAttr mod, InnerSymbolNamespace &isns,
                                  seq::FirRegOp reg) {
  auto tgt = getInnerSymFor(isns, reg);
  return hw::InnerRefAttr::get(mod, tgt);
}

namespace {
/// A pair of a register, and an inner-ref attribute.
struct BuriedFirReg {
  FirRegOp reg;
  InnerRefAttr ref;
};
} // namespace

/// Locate the registers under the given HW module, which are not at the
/// top-level of the module body. These registers will be initialized through an
/// NLA. Put an inner symbol on each, and return a list of the buried registers
/// and their inner-symbols.
static std::vector<BuriedFirReg> getBuriedRegs(HWModuleOp module) {
  auto name = SymbolTable::getSymbolName(module);
  InnerSymbolNamespace isns(module);
  std::vector<BuriedFirReg> result;
  for (auto &op : *module.getBodyBlock()) {
    for (auto &region : op.getRegions()) {
      region.walk([&](FirRegOp reg) {
        auto ref = getInnerRefTo(name, isns, reg);
        result.push_back({reg, ref});
      });
    }
  }
  return result;
}

/// Locate all registers which are not at the top-level of their parent HW
/// module. These registers will be initialized through an NLA. Put an inner
/// symbol on each, and return a list of the buried registers and their
/// inner-symbols.
static std::vector<BuriedFirReg> getAllBuriedRegs(ModuleOp top) {
  auto *context = top.getContext();
  std::vector<BuriedFirReg> init;
  auto ms = top.getOps<HWModuleOp>();
  const std::vector<HWModuleOp> modules(ms.begin(), ms.end());
  const auto reduce =
      [](std::vector<BuriedFirReg> acc,
         std::vector<BuriedFirReg> &&xs) -> std::vector<BuriedFirReg> {
    acc.insert(acc.end(), xs.begin(), xs.end());
    return acc;
  };
  return transformReduce(context, modules, init, reduce, getBuriedRegs);
}

/// Construct a hierarchical path op that targets the given register.
static hw::HierPathOp getHierPathTo(OpBuilder &builder, Namespace &ns,
                                    BuriedFirReg entry) {
  auto modName = entry.ref.getModule().getValue();
  auto symName = entry.ref.getName().getValue();
  auto name = ns.newName(Twine(modName) + "_" + symName);

  // Insert the HierPathOp immediately before the parent HWModuleOp, for style.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(entry.reg->getParentOfType<HWModuleOp>());

  auto path = builder.getArrayAttr({entry.ref});
  return hw::HierPathOp::create(builder, entry.reg.getLoc(), name, path);
}

FirRegLowering::PathTable FirRegLowering::createPaths(mlir::ModuleOp top) {
  auto builder = OpBuilder::atBlockBegin(top.getBody());
  PathTable result;
  Namespace ns;
  ns.add(top);
  for (auto entry : getAllBuriedRegs(top))
    result[entry.reg] = getHierPathTo(builder, ns, entry);
  return result;
}

FirRegLowering::FirRegLowering(TypeConverter &typeConverter,
                               hw::HWModuleOp module,
                               const PathTable &pathTable,
                               bool disableRegRandomization,
                               bool emitSeparateAlwaysBlocks)
    : pathTable(pathTable), typeConverter(typeConverter), module(module),
      disableRegRandomization(disableRegRandomization),
      emitSeparateAlwaysBlocks(emitSeparateAlwaysBlocks) {
  reachableMuxes = std::make_unique<ReachableMuxes>(module);
}

void FirRegLowering::lower() {
  lowerInBlock(module.getBodyBlock());
  createInitialBlock();
  module->removeAttr("firrtl.random_init_width");
}

// NOLINTNEXTLINE(misc-no-recursion)
void FirRegLowering::lowerUnderIfDef(sv::IfDefOp ifDefOp) {
  auto cond = ifDefOp.getCond();

  conditions.emplace_back(RegCondition::IfDefThen, cond);
  lowerInBlock(ifDefOp.getThenBlock());
  conditions.pop_back();

  if (ifDefOp.hasElse()) {
    conditions.emplace_back(RegCondition::IfDefElse, cond);
    lowerInBlock(ifDefOp.getElseBlock());
    conditions.pop_back();
  }
}

// NOLINTNEXTLINE(misc-no-recursion)
void FirRegLowering::lowerInBlock(Block *block) {
  for (auto &op : llvm::make_early_inc_range(*block)) {
    if (auto ifDefOp = dyn_cast<sv::IfDefOp>(op)) {
      lowerUnderIfDef(ifDefOp);
      continue;
    }
    if (auto regOp = dyn_cast<seq::FirRegOp>(op)) {
      lowerReg(regOp);
      continue;
    }
    for (auto &region : op.getRegions())
      for (auto &block : region.getBlocks())
        lowerInBlock(&block);
  }
}

SmallVector<Value> FirRegLowering::createRandomizationVector(OpBuilder &builder,
                                                             Location loc) {
  // Compute total width of random space.  Place non-chisel registers at the end
  // of the space.  The Random space is unique to the initial block, due to
  // verilog thread rules, so we can drop trailing random calls if they are
  // unused.
  uint64_t maxBit = 0;
  for (auto reg : randomInitRegs)
    if (reg.randStart >= 0)
      maxBit = std::max(maxBit, (uint64_t)reg.randStart + reg.width);

  for (auto &reg : randomInitRegs) {
    if (reg.randStart == -1) {
      reg.randStart = maxBit;
      maxBit += reg.width;
    }
  }

  // Create randomization vector
  SmallVector<Value> randValues;
  auto numRandomCalls = (maxBit + 31) / 32;
  auto logic = sv::LogicOp::create(
      builder, loc,
      hw::UnpackedArrayType::get(builder.getIntegerType(32), numRandomCalls),
      "_RANDOM");
  // Indvar's width must be equal to `ceil(log2(numRandomCalls +
  // 1))` to avoid overflow.
  auto inducionVariableWidth = llvm::Log2_64_Ceil(numRandomCalls + 1);
  auto arrayIndexWith = llvm::Log2_64_Ceil(numRandomCalls);
  auto lb = getOrCreateConstant(loc, APInt::getZero(inducionVariableWidth));
  auto ub =
      getOrCreateConstant(loc, APInt(inducionVariableWidth, numRandomCalls));
  auto step = getOrCreateConstant(loc, APInt(inducionVariableWidth, 1));
  auto forLoop = sv::ForOp::create(
      builder, loc, lb, ub, step, "i", [&](BlockArgument iter) {
        auto rhs = sv::MacroRefExprSEOp::create(
            builder, loc, builder.getIntegerType(32), "RANDOM");
        Value iterValue = iter;
        if (!iter.getType().isInteger(arrayIndexWith))
          iterValue = comb::ExtractOp::create(builder, loc, iterValue, 0,
                                              arrayIndexWith);
        auto lhs =
            sv::ArrayIndexInOutOp::create(builder, loc, logic, iterValue);
        sv::BPAssignOp::create(builder, loc, lhs, rhs);
      });
  builder.setInsertionPointAfter(forLoop);
  for (uint64_t x = 0; x < numRandomCalls; ++x) {
    auto lhs = sv::ArrayIndexInOutOp::create(
        builder, loc, logic,
        getOrCreateConstant(loc, APInt(arrayIndexWith, x)));
    randValues.push_back(lhs.getResult());
  }

  return randValues;
}

void FirRegLowering::createRandomInitialization(ImplicitLocOpBuilder &builder) {
  auto randInitRef =
      sv::MacroIdentAttr::get(builder.getContext(), "RANDOMIZE_REG_INIT");

  if (!randomInitRegs.empty()) {
    sv::IfDefProceduralOp::create(builder, "INIT_RANDOM_PROLOG_", [&] {
      sv::VerbatimOp::create(builder, "`INIT_RANDOM_PROLOG_");
    });

    sv::IfDefProceduralOp::create(builder, randInitRef, [&] {
      auto randValues = createRandomizationVector(builder, builder.getLoc());
      for (auto &svReg : randomInitRegs)
        initialize(builder, svReg, randValues);
    });
  }
}

void FirRegLowering::createPresetInitialization(ImplicitLocOpBuilder &builder) {
  for (auto &svReg : presetInitRegs) {
    OpBuilder::InsertionGuard guard(builder);

    auto loc = svReg.reg.getLoc();
    auto elemTy = svReg.reg.getType().getElementType();
    auto cst = getOrCreateConstant(loc, svReg.preset.getValue());

    Value rhs;
    if (cst.getType() == elemTy)
      rhs = cst;
    else
      rhs = hw::BitcastOp::create(builder, loc, elemTy, cst);

    buildRegConditions(builder, svReg.reg);
    Value target = svReg.reg;
    if (svReg.path)
      target = buildXMRTo(builder, svReg.path, svReg.reg.getLoc(),
                          svReg.reg.getType());

    sv::BPAssignOp::create(builder, loc, target, rhs);
  }
}

// If a register is async reset, we need to insert extra initialization in
// post-randomization so that we can set the reset value to register if the
// reset signal is enabled.
void FirRegLowering::createAsyncResetInitialization(
    ImplicitLocOpBuilder &builder) {
  for (auto &reset : asyncResets) {
    OpBuilder::InsertionGuard guard(builder);

    //  if (reset) begin
    //    ..
    //  end
    sv::IfOp::create(builder, reset.first, [&]() {
      for (auto &reg : reset.second) {
        OpBuilder::InsertionGuard guard(builder);
        buildRegConditions(builder, reg.reg);
        Value target = reg.reg;
        if (reg.path)
          target = buildXMRTo(builder, reg.path, reg.reg.getLoc(),
                              reg.reg.getType());
        sv::BPAssignOp::create(builder, reg.reg.getLoc(), target,
                               reg.asyncResetValue);
      }
    });
  }
}

void FirRegLowering::createInitialBlock() {
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
  if (randomInitRegs.empty() && presetInitRegs.empty() && asyncResets.empty())
    return;

  needsRandom = true;

  auto loc = module.getLoc();
  auto builder =
      ImplicitLocOpBuilder::atBlockTerminator(loc, module.getBodyBlock());

  sv::IfDefOp::create(builder, "ENABLE_INITIAL_REG_", [&] {
    sv::OrderedOutputOp::create(builder, [&] {
      sv::IfDefOp::create(builder, "FIRRTL_BEFORE_INITIAL", [&] {
        sv::VerbatimOp::create(builder, "`FIRRTL_BEFORE_INITIAL");
      });

      sv::InitialOp::create(builder, [&] {
        createRandomInitialization(builder);
        createPresetInitialization(builder);
        createAsyncResetInitialization(builder);
      });

      sv::IfDefOp::create(builder, "FIRRTL_AFTER_INITIAL", [&] {
        sv::VerbatimOp::create(builder, "`FIRRTL_AFTER_INITIAL");
      });
    });
  });
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

void FirRegLowering::createTree(OpBuilder &builder, Value reg, Value term,
                                Value next) {
  // If-then-else tree limit.
  constexpr size_t limit = 1024;

  // Count of emitted if-then-else ops.
  size_t counter = 0;

  // Get the fanout from this register before we build the tree. While we are
  // creating the tree of if/else statements from muxes, we only want to turn
  // muxes that are on the register's fanout into if/else statements. This is
  // required to get the correct enable inference. But other muxes in the tree
  // should be left as ternary operators. This is desirable because we don't
  // want to create if/else structure for logic unrelated to the register's
  // enable.
  auto firReg = term.getDefiningOp<seq::FirRegOp>();

  std::deque<std::tuple<Block *, Value, Value, Value>> worklist;
  auto addToWorklist = [&](Value reg, Value term, Value next) {
    worklist.emplace_back(builder.getBlock(), reg, term, next);
  };

  auto getArrayIndex = [&](Value reg, Value idx) {
    // Create an array index op just after `reg`.
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfterValue(reg);
    return sv::ArrayIndexInOutOp::create(builder, reg.getLoc(), reg, idx);
  };

  SmallVector<Value, 8> opsToDelete;
  addToWorklist(reg, term, next);
  while (!worklist.empty()) {
    OpBuilder::InsertionGuard guard(builder);
    Block *block;
    Value reg, term, next;
    std::tie(block, reg, term, next) = worklist.front();
    worklist.pop_front();

    builder.setInsertionPointToEnd(block);
    if (areEquivalentValues(term, next))
      continue;

    // If this is a two-state mux within the fanout from the register, we use
    // if/else structure for proper enable inference.
    auto mux = next.getDefiningOp<comb::MuxOp>();
    if (mux && mux.getTwoState() &&
        reachableMuxes->isMuxReachableFrom(firReg, mux)) {
      if (counter >= limit) {
        sv::PAssignOp::create(builder, term.getLoc(), reg, next);
        continue;
      }
      addToIfBlock(
          builder, mux.getCond(),
          [&]() { addToWorklist(reg, term, mux.getTrueValue()); },
          [&]() { addToWorklist(reg, term, mux.getFalseValue()); });
      ++counter;
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
                  hw::ArrayGetOp::create(builder, term.getLoc(), term, index);
              opsToDelete.push_back(termElement);
              addToWorklist(nextReg, termElement, trueValue);
            },
            []() {});
        ++numSubaccessRestored;
        continue;
      }
      // Compat fix for GCC12's libstdc++, cannot use
      // llvm::enumerate(llvm::reverse(OperandRange)).  See #4900.
      // SmallVector<Value>
      // reverseOpValues(llvm::reverse(array.getOperands()));
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
            hw::ArrayGetOp::create(builder, term.getLoc(), term, idxVal);
        opsToDelete.push_back(termElement);
        addToWorklist(index, termElement, value);
      }
      continue;
    }

    sv::PAssignOp::create(builder, term.getLoc(), reg, next);
  }

  while (!opsToDelete.empty()) {
    auto value = opsToDelete.pop_back_val();
    assert(value.use_empty());
    value.getDefiningOp()->erase();
  }
}

void FirRegLowering::lowerReg(FirRegOp reg) {
  Location loc = reg.getLoc();
  Type regTy = typeConverter.convertType(reg.getType());

  HierPathOp path;
  auto lookup = pathTable.find(reg);
  if (lookup != pathTable.end())
    path = lookup->second;

  ImplicitLocOpBuilder builder(reg.getLoc(), reg);
  RegLowerInfo svReg{nullptr, path, reg.getPresetAttr(), nullptr, nullptr,
                     -1,      0};
  svReg.reg = sv::RegOp::create(builder, loc, regTy, reg.getNameAttr());
  auto width = hw::getBitWidth(regTy);
  assert(width && "register type must have a known bit width");
  svReg.width = *width;

  if (auto attr = reg->getAttrOfType<IntegerAttr>("firrtl.random_init_start"))
    svReg.randStart = attr.getUInt();

  // Don't move these over
  reg->removeAttr("firrtl.random_init_start");

  // Move Attributes
  svReg.reg->setDialectAttrs(reg->getDialectAttrs());

  if (auto innerSymAttr = reg.getInnerSymAttr())
    svReg.reg.setInnerSymAttr(innerSymAttr);

  auto regVal = sv::ReadInOutOp::create(builder, loc, svReg.reg);

  if (reg.hasReset()) {
    addToAlwaysBlock(
        reg->getBlock(), sv::EventControl::AtPosEdge, reg.getClk(),
        [&](OpBuilder &b) {
          // If this is an AsyncReset, ensure that we emit a self connect to
          // avoid erroneously creating a latch construct.
          if (reg.getIsAsync() && areEquivalentValues(reg, reg.getNext()))
            sv::PAssignOp::create(b, reg.getLoc(), svReg.reg, reg);
          else
            createTree(b, svReg.reg, reg, reg.getNext());
        },
        reg.getIsAsync() ? sv::ResetType::AsyncReset : sv::ResetType::SyncReset,
        sv::EventControl::AtPosEdge, reg.getReset(),
        [&](OpBuilder &builder) {
          sv::PAssignOp::create(builder, loc, svReg.reg, reg.getResetValue());
        });
    if (reg.getIsAsync()) {
      svReg.asyncResetSignal = reg.getReset();
      svReg.asyncResetValue = reg.getResetValue();
    }
  } else {
    addToAlwaysBlock(
        reg->getBlock(), sv::EventControl::AtPosEdge, reg.getClk(),
        [&](OpBuilder &b) { createTree(b, svReg.reg, reg, reg.getNext()); });
  }

  // Record information required later on to build the initialization code for
  // this register. All initialization is grouped together in a single initial
  // block at the back of the module.
  if (svReg.preset)
    presetInitRegs.push_back(svReg);
  else if (!disableRegRandomization)
    randomInitRegs.push_back(svReg);

  if (svReg.asyncResetSignal)
    asyncResets[svReg.asyncResetSignal].emplace_back(svReg);

  // Remember the ifdef conditions surrounding this register, if present. We
  // will need to place this register's initialization code under the same
  // ifdef conditions.
  if (!conditions.empty())
    regConditionTable.emplace_or_assign(svReg.reg, conditions);

  reg.replaceAllUsesWith(regVal.getResult());
  reg.erase();
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
    sv::BPAssignOp::create(builder, loc, reg, elem);
  } else if (auto array = hw::type_dyn_cast<hw::ArrayType>(type)) {
    for (unsigned i = 0, e = array.getNumElements(); i < e; ++i) {
      auto index = getOrCreateConstant(loc, APInt(llvm::Log2_64_Ceil(e), i));
      initializeRegisterElements(
          loc, builder, sv::ArrayIndexInOutOp::create(builder, loc, reg, index),
          randomSource, pos);
    }
  } else if (auto structType = hw::type_dyn_cast<hw::StructType>(type)) {
    for (auto e : structType.getElements())
      initializeRegisterElements(
          loc, builder,
          sv::StructFieldInOutOp::create(builder, loc, reg, e.name),
          randomSource, pos);
  } else {
    assert(false && "unsupported type");
  }
}
// NOLINTEND(misc-no-recursion)

void FirRegLowering::buildRegConditions(OpBuilder &b, sv::RegOp reg) {
  // If there are no conditions, just return the current insertion point.
  auto lookup = regConditionTable.find(reg);
  if (lookup == regConditionTable.end())
    return;

  // Recreate the conditions under which the register was declared.
  auto &conditions = lookup->second;
  for (auto &condition : conditions) {
    auto kind = condition.getKind();
    if (kind == RegCondition::IfDefThen) {
      auto ifDef = sv::IfDefProceduralOp::create(b, reg.getLoc(),
                                                 condition.getMacro(), []() {});
      b.setInsertionPointToEnd(ifDef.getThenBlock());
      continue;
    }
    if (kind == RegCondition::IfDefElse) {
      auto ifDef = sv::IfDefProceduralOp::create(
          b, reg.getLoc(), condition.getMacro(), []() {}, []() {});

      b.setInsertionPointToEnd(ifDef.getElseBlock());
      continue;
    }
    llvm_unreachable("unknown reg condition type");
  }
}

void FirRegLowering::initialize(OpBuilder &builder, RegLowerInfo reg,
                                ArrayRef<Value> rands) {
  auto loc = reg.reg.getLoc();
  SmallVector<Value> nibbles;
  if (reg.width == 0)
    return;

  OpBuilder::InsertionGuard guard(builder);

  // If the register was defined under ifdefs, we have to guard the
  // initialization code under the same ifdefs. The builder's insertion point
  // will be left inside the guards.
  buildRegConditions(builder, reg.reg);

  // If the register is not located in the toplevel body of the module, we must
  // refer to the register by (local) XMR, since the register will not dominate
  // the initialization block.
  Value target = reg.reg;
  if (reg.path)
    target = buildXMRTo(builder, reg.path, reg.reg.getLoc(), reg.reg.getType());

  uint64_t width = reg.width;
  uint64_t offset = reg.randStart;
  while (width) {
    auto index = offset / 32;
    auto start = offset % 32;
    auto nwidth = std::min(32 - start, width);
    auto elemVal = sv::ReadInOutOp::create(builder, loc, rands[index]);
    auto elem =
        builder.createOrFold<comb::ExtractOp>(loc, elemVal, start, nwidth);
    nibbles.push_back(elem);
    offset += nwidth;
    width -= nwidth;
  }
  auto concat = builder.createOrFold<comb::ConcatOp>(loc, nibbles);
  unsigned pos = reg.width;
  // Initialize register elements.
  initializeRegisterElements(loc, builder, target, concat, pos);
}

void FirRegLowering::addToAlwaysBlock(
    Block *block, sv::EventControl clockEdge, Value clock,
    const std::function<void(OpBuilder &)> &body, sv::ResetType resetStyle,
    sv::EventControl resetEdge, Value reset,
    const std::function<void(OpBuilder &)> &resetBody) {
  auto loc = clock.getLoc();
  ImplicitLocOpBuilder builder(loc, block, getBlockEnd(block));
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
        insideIfOp = sv::IfOp::create(
            builder, reset, []() {}, []() {});
      };
      if (resetStyle == sv::ResetType::AsyncReset) {
        sv::EventControl events[] = {clockEdge, resetEdge};
        Value clocks[] = {clock, reset};

        alwaysOp = sv::AlwaysOp::create(builder, events, clocks, [&]() {
          if (resetEdge == sv::EventControl::AtNegEdge)
            llvm_unreachable("negative edge for reset is not expected");
          createIfOp();
        });
      } else {
        alwaysOp = sv::AlwaysOp::create(builder, clockEdge, clock, createIfOp);
      }
    } else {
      assert(!resetBody);
      alwaysOp = sv::AlwaysOp::create(builder, clockEdge, clock);
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
