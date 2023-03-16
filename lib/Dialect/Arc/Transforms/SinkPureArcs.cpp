//===- SinkPureArcs.cpp
//-----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Sinks arc.state operations with latency 0 and without resets or enables into
// the next arc.state op that cannot be sunk.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/Namespace.h"
#include "circt/Support/SymCache.h"
#include "circt/Support/ValueMapper.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/PriorityQueue.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Timer.h"
#include <cstdint>
#include <iterator>

#define DEBUG_TYPE "sink-pure-arcs"

using namespace circt;
using namespace arc;

namespace {
struct SinkPureArcsPass : public SinkPureArcsBase<SinkPureArcsPass> {
  SinkPureArcsPass() = default;
  SinkPureArcsPass(const SinkPureArcsPass &pass) : SinkPureArcsPass() {}

  void runOnOperation() override;
  bool canSink(StateOp stateOp);

  Statistic numStateOpsSunk{this, "num-sunk",
                            "Number of state operations that were sunk."};
  Statistic numStateOpsDuplicated{this, "num-duplicated",
                                  "Total number of operations cloned."};

  llvm::Timer sinking;
  llvm::Timer cleanup;
  llvm::Timer traversing;
};
} // namespace

bool SinkPureArcsPass::canSink(StateOp stateOp) {
  return stateOp.getLatency() == 0 && !stateOp.getEnable() &&
         !stateOp.getReset();
}

namespace {
// struct SinkRewritePattern : public OpRewritePattern<StateOp> {
//   SinkRewritePattern(MLIRContext *context, Namespace &names) :
//   OpRewritePattern<StateOp>(context), names(names) {}

//   LogicalResult matchAndRewrite(StateOp op,
//                                 PatternRewriter &rewriter) const override {
//     // if (op.getLatency() == 0)
//     //   return failure();

//       bool abc = true;
//       for (auto [i, input] : llvm::enumerate(op.getInputs())) {
//         if (auto predState = input.getDefiningOp<StateOp>(); predState &&
//         predState.getLatency() == 0) {
//           abc = false;
//         }
//       }
//       if (abc)
//       return failure();

//       SmallVector<Type> inputTypes;
//       SmallVector<Type> inputTypes2;
//       auto *bodyBlock = new Block;
//       auto newArgs = bodyBlock->addArguments(op.getInputs().getTypes(),
//       SmallVector<Location>(op.getInputs().size(), op.getLoc())); OpBuilder
//       bodyBuilder(op->getContext());
//       bodyBuilder.setInsertionPointToStart(bodyBlock);
//       auto selfCall = bodyBuilder.create<CallOp>(op.getLoc(),
//       op->getResultTypes(), op.getArc(), SmallVector<Value>{newArgs.begin(),
//       newArgs.end()}); bodyBuilder.create<OutputOp>(op.getLoc(),
//       selfCall->getResults());
//       bodyBuilder.setInsertionPointToStart(bodyBlock);

//       BitVector toDelete(op.getInputs().size());
//       SmallVector<Value> newInputs2;
//       for (auto [i, input] : llvm::enumerate(op.getInputs())) {
//         if (auto predState = input.getDefiningOp<StateOp>(); predState &&
//         predState.getLatency() == 0) {
//           SmallVector<Type> predInputTypes(predState.getInputs().getTypes());
//           inputTypes.append(predInputTypes);
//           auto newArgs = bodyBlock->addArguments(predInputTypes,
//           SmallVector<Location>(predInputTypes.size(), op.getLoc())); auto
//           call = bodyBuilder.create<CallOp>(op.getLoc(),
//           predState->getResultTypes(), predState.getArc(),
//           SmallVector<Value>{newArgs.begin(), newArgs.end()});
//           // TODO: find and assign the correct result, not just 0
//           SmallVector<Value> newInputs(selfCall.getInputs());
//           auto found = std::find(predState.getResults().begin(),
//           predState.getResults().end(), input); assert(found !=
//           predState.getResults().end() && "This should not trigger"); auto
//           idx = std::distance(predState.getResults().begin(), found);
//           newInputs[i] = call->getResult(idx);
//           selfCall.getInputsMutable().assign(newInputs);
//           toDelete.set(i);
//         } else {
//           inputTypes2.push_back(input.getType());
//           newInputs2.push_back(input);
//         }
//       }
//       inputTypes2.append(inputTypes);
//       toDelete.resize(bodyBlock->getNumArguments());
//       bodyBlock->eraseArguments(toDelete);
//       rewriter.setInsertionPointToStart(&op->getParentOfType<ModuleOp>().getRegion().front());
//       auto defOp = rewriter.create<DefineOp>(op.getLoc(),
//       names.newName(op.getArc()), rewriter.getFunctionType(inputTypes2,
//       op.getResultTypes())); defOp->getRegion(0).push_back(bodyBlock);

//         op.setArc(defOp.getSymName());
//         for (auto [i, input] : llvm::enumerate(op.getInputs())) {
//           if (auto predState = input.getDefiningOp<StateOp>(); predState &&
//           predState.getLatency() == 0) {
//             newInputs2.append(llvm::to_vector(predState.getInputs()));
//             // auto deleteIter = std::find(newInputs.begin(),
//             newInputs.end(), predState.getResult(0));
//             // if (deleteIter)
//             // newInputs.erase(deleteIter);
//           }
//         }
//         op.getInputsMutable().assign(newInputs2);

//     return success();
//   }

//   Namespace &names;
// };
// static DenseMap<Value, uint64_t> valueMap;

struct PackArgsInStructPattern : public OpRewritePattern<DefineOp> {
  using OpRewritePattern<DefineOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DefineOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumArguments() <= 1)
      return failure();

    SmallVector<hw::StructType::FieldInfo> fields;
    for (auto arg : op.getArguments()) {
      auto fieldName = StringAttr::get(getContext(), Twine(arg.getArgNumber()));
      fields.push_back({fieldName, arg.getType()});
    }
    auto structTy = hw::StructType::get(getContext(), fields);
    rewriter.setInsertionPointToStart(&op.getBodyBlock());
    auto newArg = op.getBodyBlock().insertArgument(op.getArguments().begin(),
                                                   structTy, op.getLoc());
    for (auto [i, arg] : llvm::enumerate(op.getArguments())) {
      if (i == 0)
        continue;
      auto repl = rewriter.create<hw::StructExtractOp>(op.getLoc(), newArg,
                                                       std::to_string(i - 1));
      rewriter.replaceAllUsesWith(arg, repl);
    }
    BitVector toDelete(op.getNumArguments() + 1, true);
    toDelete.reset(0);
    op.getBodyBlock().eraseArguments(toDelete);
    op.setFunctionType(
        rewriter.getFunctionType(structTy, op.getFunctionType().getResults()));
    return success();
  }
};

struct PackStateInputsInStructPattern : public OpRewritePattern<StateOp> {
  using OpRewritePattern<StateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(StateOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() <= 1)
      return failure();

    SmallVector<hw::StructType::FieldInfo> fields;
    for (auto [i, arg] : llvm::enumerate(op.getInputs())) {
      auto fieldName = StringAttr::get(getContext(), Twine(i));
      fields.push_back({fieldName, arg.getType()});
    }
    auto structTy = hw::StructType::get(getContext(), fields);

    Value strukt = rewriter.create<hw::StructCreateOp>(op.getLoc(), structTy,
                                                       op.getInputs());
    op.getInputsMutable().assign(strukt);

    return success();
  }
};

struct ArgDedupPattern : public OpRewritePattern<DefineOp> {
  ArgDedupPattern(
      MLIRContext *context,
      DenseMap<StringAttr, llvm::SetVector<Operation *>> &arcUserMap)
      : OpRewritePattern<DefineOp>(context), arcUserMap(arcUserMap) {}

  LogicalResult matchAndRewrite(DefineOp op,
                                PatternRewriter &rewriter) const override {
    if (!arcUserMap.count(op.getSymNameAttr()) ||
        arcUserMap[op.getSymNameAttr()].empty()) {
      rewriter.eraseOp(op);
      arcUserMap.erase(op.getSymNameAttr());
      return success();
    }

    auto stateOp = dyn_cast<StateOp>(arcUserMap[op.getSymNameAttr()][0]);
    if (!stateOp)
      return failure();

    DenseSet<Value> inputs;
    DenseMap<Value, unsigned> firstOccurrence;
    llvm::MapVector<unsigned, Value> valueMap;
    for (auto [i, input] : llvm::enumerate(stateOp.getInputs())) {
      if (!firstOccurrence.count(input))
        firstOccurrence[input] = i;
      if (auto prev = inputs.insert(input); !prev.second) {
        valueMap[firstOccurrence[*prev.first]] = *prev.first;
        valueMap[i] = input;
      }
    }

    for (auto *user : arcUserMap[op.getSymNameAttr()]) {
      if (isa<CallOp>(user))
        return failure();
      assert(user);
      if (auto stateOp = dyn_cast<StateOp>(user)) {
        valueMap.remove_if([&](auto entry) {
          return stateOp.getInputs()[entry.first] != entry.second;
        });
        // for (auto [idx, input] : llvm::make_early_inc_range(valueMap)) {
        //   if (stateOp.getInputs()[idx] != input)
        //     valueMap.erase(idx);
        // }
      }
    }

    // It might have been removed above, so we need to recompute it.
    firstOccurrence.clear();

    BitVector toDelete(op.getNumArguments());
    for (auto [i, val] : valueMap) {
      if (!firstOccurrence[val]) {
        firstOccurrence[val] = i;
        continue;
      }

      op.getArgument(i).replaceAllUsesWith(
          op.getArgument(firstOccurrence[val]));
      toDelete.set(i);
    }
    if (toDelete.none())
      return failure();

    op.eraseArguments(toDelete);

    BitVector notDeleted(toDelete);
    notDeleted.flip();
    for (auto *user : arcUserMap[op.getSymNameAttr()]) {
      if (isa<StateOp>(user)) {
        auto stateOp = cast<StateOp>(user);
        SmallVector<Value> newInputs;
        for (auto i : notDeleted.set_bits()) {
          newInputs.push_back(stateOp.getInputs()[i]);
        }
        stateOp.getInputsMutable().assign(newInputs);
        continue;
      }
      user->eraseOperands(toDelete);
    }

    return success();
  }

  DenseMap<StringAttr, llvm::SetVector<Operation *>> &arcUserMap;
};

struct SinkRewritePattern : public OpRewritePattern<StateOp> {
  SinkRewritePattern(
      MLIRContext *context, Namespace &names,
      DenseMap<std::pair<StringAttr, ArrayAttr>, DefineOp> &replacementMap,
      DenseMap<StringAttr, llvm::SetVector<Operation *>> &arcUserMap,
      SmallVector<Operation *> &newArcs)
      : OpRewritePattern<StateOp>(context), names(names),
        replacementMap(replacementMap), arcUserMap(arcUserMap),
        newArcs(newArcs) {}

  DefineOp buildDefineOp(StateOp op, mlir::PatternRewriter &rewriter) const {
    SmallVector<Type> inputTypes;
    SmallVector<Value> newInputs;
    auto *bodyBlock = new Block;
    OpBuilder bodyBuilder(op->getContext());
    bodyBuilder.setInsertionPointToStart(bodyBlock);

    for (auto [i, input] : llvm::enumerate(op.getInputs())) {
      if (auto predState = input.getDefiningOp<StateOp>();
          predState && predState.getLatency() == 0) {
        SmallVector<Type> predInputTypes(predState.getInputs().getTypes());
        auto newArgs = bodyBlock->addArguments(
            predInputTypes,
            SmallVector<Location>(predInputTypes.size(), op.getLoc()));
        auto call = bodyBuilder.create<CallOp>(
            op.getLoc(), predState->getResultTypes(), predState.getArc(),
            SmallVector<Value>{newArgs.begin(), newArgs.end()});
        arcUserMap[call.getArcAttr().getAttr()].insert(call);
        auto idx = cast<OpResult>(input).getResultNumber();
        newInputs.push_back(call->getResult(idx));
        continue;
      }
      auto newArg = bodyBlock->addArgument(input.getType(), op.getLoc());
      newInputs.push_back(newArg);
    }

    auto selfCall = bodyBuilder.create<CallOp>(
        op.getLoc(), op->getResultTypes(), op.getArc(),
        newInputs); // SmallVector<Value>{newArgs.begin(), newArgs.end()});
    arcUserMap[selfCall.getArcAttr().getAttr()].insert(selfCall);
    bodyBuilder.create<OutputOp>(op.getLoc(), selfCall->getResults());
    rewriter.setInsertionPointToStart(
        &op->getParentOfType<ModuleOp>().getRegion().front());
    auto defOp = rewriter.create<DefineOp>(
        op.getLoc(), names.newName(op.getArc()),
        rewriter.getFunctionType(bodyBlock->getArgumentTypes(),
                                 op.getResultTypes()));
    newArcs.push_back(defOp);
    arcUserMap[defOp.getSymNameAttr()] = llvm::SetVector<Operation *>();
    defOp->getRegion(0).push_back(bodyBlock);
    return defOp;
  }

  LogicalResult matchAndRewrite(StateOp op,
                                PatternRewriter &rewriter) const override {
    bool abc = true;
    SmallVector<Attribute> inp;
    ValueRange inputs = op.getInputs();
    bool isStructArgs = false;
    if (inputs.size() == 1 && inputs[0].getType().isa<hw::StructType>()) {
      if (auto structCreate = dyn_cast_or_null<hw::StructCreateOp>(inputs[0].getDefiningOp())) {
        inputs = structCreate.getInput();
        isStructArgs = true;
      }
    }
    for (auto [i, input] : llvm::enumerate(inputs)) {
      if (auto predState = input.getDefiningOp<StateOp>();
          predState && predState.getLatency() == 0) {
        inp.push_back(predState.getArcAttr().getAttr());
        abc = false;
      } else {
        inp.push_back(TypeAttr::get(input.getType()));
      }
    }
    if (abc)
      return failure();

    auto attr = rewriter.getArrayAttr(inp);

    auto key = std::make_pair(op.getArcAttr().getAttr(), attr);
    auto &defOp = replacementMap[key];
    if (!defOp)
      defOp = buildDefineOp(op, rewriter);

    SmallVector<Value> newInputs;
    llvm::SetVector<Operation *> predStates;
    for (auto [i, input] : llvm::enumerate(inputs)) {
      if (auto predState = input.getDefiningOp<StateOp>();
          predState && predState.getLatency() == 0) {
        newInputs.append(llvm::to_vector(predState.getInputs()));
        predStates.insert(predState);
        continue;
      }
      newInputs.push_back(input);
    }
    op.getInputsMutable().assign(newInputs);
    arcUserMap[op.getArcAttr().getAttr()].remove(op);
    op.setArc(defOp.getSymName());
    arcUserMap[defOp.getSymNameAttr()].insert(op);

    for (auto *op : predStates) {
      if (op->use_empty()) {
        op->dropAllReferences();
        op->erase();
      }
    }

    return success();
  }

  Namespace &names;
  DenseMap<std::pair<StringAttr, ArrayAttr>, DefineOp> &replacementMap;
  DenseMap<StringAttr, llvm::SetVector<Operation *>> &arcUserMap;
  SmallVector<Operation *> &newArcs;
};

} // namespace

void SinkPureArcsPass::runOnOperation() {
  sinking.clear();
  cleanup.clear();
  traversing.clear();

  mlir::GreedyRewriteConfig config;
  config.strictMode = mlir::GreedyRewriteStrictness::ExistingOps;
  config.maxIterations = 1;
  config.enableRegionSimplification = false;

  auto module = getOperation();
  SymbolCache cache;
  cache.addDefinitions(module);
  Namespace names;
  names.add(cache);

  SmallVector<Operation *> newArcs;
  SmallVector<Operation *> prePassOps(getOperation().getOps<DefineOp>());
  llvm::SetVector<Operation *> stateOps;
  llvm::SetVector<Operation *> visited;

  // top-down
  getOperation()->walk([&](StateOp op) {
    prePassOps.push_back(op);
    bool hasStatePred = false;
    for (auto [i, input] : llvm::enumerate(op.getInputs())) {
      if (auto predState = input.getDefiningOp<StateOp>();
          predState && predState.getLatency() == 0)
        hasStatePred = true;
    }
    if (!hasStatePred) {
      stateOps.insert(op);
    }
  });

  mlir::RewritePatternSet prePassPatterns(&getContext());
  prePassPatterns.add<PackArgsInStructPattern, PackStateInputsInStructPattern>(
      &getContext());
  if (failed(mlir::applyOpPatternsAndFold(prePassOps,
                                          std::move(prePassPatterns), config)))
    return signalPassFailure();

  // bottom-up
  // getOperation()->walk([&](StateOp op) {
  //   if (op.getLatency() > 0)
  //     stateOps.insert(op);
  // });

  DenseMap<std::pair<StringAttr, ArrayAttr>, DefineOp> replacementMap;

  DenseMap<StringAttr, llvm::SetVector<Operation *>> arcUserMap;
  mlir::SymbolTableCollection symbolTable;
  mlir::SymbolUserMap userMap2(symbolTable, getOperation());
  auto defines = SmallVector<Operation *>(getOperation().getOps<DefineOp>());
  for (auto *def : defines) {
    llvm::SetVector<Operation *> defUses;
    defUses.insert(userMap2.getUsers(def).begin(),
                   userMap2.getUsers(def).end());
    arcUserMap[cast<DefineOp>(def).getSymNameAttr()] = defUses;
  }

  RewritePatternSet sinkPatterns(&getContext());
  sinkPatterns.add<SinkRewritePattern>(&getContext(), names, replacementMap,
                                       arcUserMap, newArcs);
  mlir::FrozenRewritePatternSet sinkPatternsFrozen(std::move(sinkPatterns));

  RewritePatternSet cleanupPatterns(&getContext());
  cleanupPatterns.add<ArgDedupPattern>(&getContext(), arcUserMap);
  mlir::FrozenRewritePatternSet cleanupPatternsFrozen(
      std::move(cleanupPatterns));

  // top-down
  llvm::SetVector<Operation *> next;
  // unsigned maxIter = 2;
  while (!stateOps.empty()) {
    traversing.startTimer();
    next.clear();

    for (auto *op : stateOps) {
      visited.insert(op);
    }

    for (auto *op : stateOps) {
      if (!op)
        continue;
      if (!isa<StateOp>(op))
        continue;
      auto stateOp = cast<StateOp>(op);
      if (stateOp.getLatency() == 0) {
        for (auto *user : op->getUsers()) {
          if (isa<StateOp>(user)) { // && !visited.contains(user)) {
            bool hasStatePred = false;
            for (auto [i, input] :
                 llvm::enumerate(cast<StateOp>(user).getInputs())) {
              if (auto predState = input.getDefiningOp<StateOp>();
                  predState && predState.getLatency() == 0 &&
                  !visited.contains(predState))
                hasStatePred = true;
            }
            if (!hasStatePred) {
              next.insert(user);
            }
          }
        }
      }
    }

    traversing.stopTimer();
    sinking.startTimer();
    if (failed(mlir::applyOpPatternsAndFold(stateOps.getArrayRef(),
                                            sinkPatternsFrozen, config)))
      return signalPassFailure();
    sinking.stopTimer();

    cleanup.startTimer();
    if (failed(mlir::applyOpPatternsAndFold(newArcs, cleanupPatternsFrozen,
                                            config)))
      return signalPassFailure();
    cleanup.stopTimer();

    newArcs.clear();
    stateOps.clear();

    traversing.startTimer();
    for (auto *op : next) {
      if (op)
        stateOps.insert(op);
    }
    traversing.stopTimer();
  }

  // bottom-up
  // while(!stateOps.empty()) {
  //   for (auto *stateOp : llvm::make_early_inc_range(stateOps)) {
  //     if (!isa<StateOp>(stateOp))
  //       continue;
  //     if (llvm::all_of(cast<StateOp>(stateOp).getInputs(), [](auto input) {
  //       if (auto pred = input.template getDefiningOp<StateOp>(); !pred ||
  //       pred.getLatency() > 0)
  //         return true;
  //       return false;
  //     })) {
  //       stateOps.remove(stateOp);
  //     }
  //   }

  //   if (failed(mlir::applyOpPatternsAndFold(stateOps.getArrayRef(),
  //   sinkPatternsFrozen, config)))
  //     return signalPassFailure();

  //   if
  //   (failed(mlir::applyOpPatternsAndFold(SmallVector<Operation*>(getOperation().getOps<DefineOp>()),
  //   cleanupPatternsFrozen, config)))
  //     return signalPassFailure();
  // }

  // for (auto hwmodule : getOperation().getOps<hw::HWModuleOp>()) {
  //   if (failed(mlir::applyPatternsAndFoldGreedily(hwmodule, {})))
  //     return signalPassFailure();
  // }

  llvm::StringMap<llvm::TimeRecord> records;
  records.insert({"sinking", sinking.getTotalTime()});
  records.insert({"traversing", traversing.getTotalTime()});
  records.insert({"cleanup", cleanup.getTotalTime()});
  llvm::TimerGroup group("SinkPureArcs", "sink pure arcs", records);
  group.printAll(llvm::errs());
}

std::unique_ptr<Pass> arc::createSinkPureArcsPass() {
  return std::make_unique<SinkPureArcsPass>();
}
