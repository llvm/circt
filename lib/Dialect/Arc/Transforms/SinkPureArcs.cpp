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
#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/Namespace.h"
#include "circt/Support/SymCache.h"
#include "circt/Support/ValueMapper.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
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
  void runOnModule(hw::HWModuleOp module);
  bool canSink(Operation *op);
  LogicalResult runOnNonPureCallOp(OpBuilder &arcBuilder, CallOpMutableInterface callOp);
  DefineOp outlineIntoArc(OpBuilder &arcBuilder, CallOpMutableInterface callOp, SmallVectorImpl<Value> &newInputs, bool wrapInStruct);

  Namespace names;
  SymbolCache cache;

  Statistic numStateOpsSunk{this, "num-sunk",
                            "Number of state operations that were sunk."};
  Statistic numStateOpsDuplicated{this, "num-duplicated",
                                  "Total number of operations cloned."};
};
} // namespace

bool SinkPureArcsPass::canSink(Operation *op) {
  if (!op)
    return false;
  if (auto callOp = dyn_cast<mlir::CallOpInterface>(op)) {
    if (auto defOp =
            dyn_cast<DefineOp>(cache.getDefinition(callOp.getCallableForCallee()
                                                       .get<SymbolRefAttr>()
                                                       .getLeafReference()))) {
      return defOp.getBodyBlock().getOperations().size() <= 3;
    }
  }
  if (isa<hw::StructCreateOp>(op))
    return false;
  return mlir::isPure(op);
}

DefineOp SinkPureArcsPass::outlineIntoArc(OpBuilder &arcBuilder, CallOpMutableInterface callOp, SmallVectorImpl<Value> &newInputs, bool wrapInStruct) {
  auto calleeName = callOp.getCallableForCallee().get<SymbolRefAttr>().getLeafReference();
  auto currArc = dyn_cast<DefineOp>(cache.getDefinition(calleeName));
  if (!currArc)
    return nullptr;
  // Create a new arc solely for this call.
  auto newArc = arcBuilder.create<arc::DefineOp>(callOp.getLoc(), names.newName(calleeName.getValue()), currArc.getFunctionType());
  cache.addSymbol(newArc);
  auto &block = newArc.getBody().emplaceBlock();

  SmallVector<Operation*> worklist;
  DenseSet<Operation *> visited;
  mlir::IRMapping mapping;

  for (auto operand : callOp.getArgOperands()) {
      auto *toSink = operand.getDefiningOp();
      bool sinkable = canSink(toSink);
      if (!sinkable && !mapping.contains(operand)) {
        auto arg = block.addArgument(operand.getType(), operand.getLoc());
        mapping.map(operand, arg);
        newInputs.push_back(operand);
        continue;
      }
        
      if (sinkable)
        worklist.push_back(toSink);
  }

  OpBuilder builder = OpBuilder::atBlockBegin(&block);
  while (!worklist.empty()) {
    auto *curr = worklist.back();
    if (visited.contains(curr)) {
      worklist.pop_back();
      continue;
    }
    bool added = false;
    for (auto operand : curr->getOperands()) {
      auto *toSink = operand.getDefiningOp();
      bool sinkable = canSink(toSink);
      if (!sinkable && !mapping.contains(operand)) {
        auto arg = block.addArgument(operand.getType(), operand.getLoc());
        mapping.map(operand, arg);
        newInputs.push_back(operand);
        added = true;
        continue;
      }
        
      if (sinkable && !visited.contains(toSink)) {
        worklist.push_back(toSink);
        added = true;
        break;
      }
    }
    if (added)
      continue;
    
    builder.clone(*curr, mapping);
    visited.insert(curr);
    worklist.pop_back();
  }

  // Add block terminator
  SmallVector<Value> outputs;
  for (auto val : callOp.getArgOperands())
    outputs.push_back(mapping.lookup(val));
  auto nestedCall = builder.create<CallOp>(callOp.getLoc(), currArc.getResultTypes(), FlatSymbolRefAttr::get(currArc.getNameAttr()), outputs);
  builder.create<OutputOp>(callOp.getLoc(), nestedCall->getResults());

  // Convert arguments to struct
  if (wrapInStruct) {
    SmallVector<hw::detail::FieldInfo> fields;
    for (auto [i, argTy] : llvm::enumerate(newArc.getBodyBlock().getArgumentTypes()))
      fields.push_back({StringAttr::get(&getContext(), std::to_string(i)), argTy});
    auto structType = hw::StructType::get(&getContext(), fields);
    SmallVector<BlockArgument> oldArguments(newArc.getBodyBlock().getArguments());
    auto structArg = newArc.getBodyBlock().addArgument(structType, newArc.getLoc());
    builder.setInsertionPointToStart(&newArc.getBodyBlock());
    ValueRange structFields = builder.create<hw::StructExplodeOp>(newArc.getLoc(), structArg)->getResults();
    for (auto [field, arg] : llvm::zip(structFields, oldArguments)) {
      arg.replaceAllUsesWith(field);
      newArc.getBodyBlock().eraseArgument(arg.getArgNumber());
    }
  }

  newArc.setFunctionType(builder.getFunctionType(newArc.getBodyBlock().getArgumentTypes(), newArc.getResultTypes()));
  return newArc;
}

LogicalResult SinkPureArcsPass::runOnNonPureCallOp(OpBuilder &arcBuilder, CallOpMutableInterface callOp) {
  SetVector<Operation*> worklist;
  DenseSet<Operation *> visited;

  worklist.insert(callOp);

  while (!worklist.empty()) {
    auto *curr = worklist.back();
    bool added = false;
    for (auto operand : curr->getOperands()) {
      auto *toSink = operand.getDefiningOp();
      bool sinkable = canSink(toSink);
      if (!sinkable || visited.contains(toSink))
        continue;
      if (worklist.contains(toSink)) {
        auto diag = toSink->emitError("combinational cycle detected");
        diag.attachNote(toSink->getLoc()) << *toSink;
        for (auto *op : llvm::reverse(worklist)) {
          diag.attachNote(op->getLoc()) << *op;
          if (op == toSink)
            break;
        }
        signalPassFailure();
        return failure();
      }
      worklist.insert(toSink);
      added = true;
      break;
    }
    if (added)
      continue;

    // Make outlining decision
    bool outline = false;
    for (auto operand : curr->getOperands()) {
      auto *toSink = operand.getDefiningOp();
      if (canSink(toSink))
        outline = true;
    }
    if (auto root = dyn_cast<CallOp>(curr); root && outline && std::distance(curr->getUsers().begin(), curr->getUsers().end())) {
      SmallVector<Value> inputs;
      auto defOp = outlineIntoArc(arcBuilder, root, inputs, true);
      OpBuilder tmpBuilder(curr);
      SmallVector<hw::detail::FieldInfo> fields;
      for (auto [i, input] : llvm::enumerate(inputs))
        fields.push_back({StringAttr::get(&getContext(), std::to_string(i)), input.getType()});
      auto structType = hw::StructType::get(&getContext(), fields);
      Value structInput = tmpBuilder.create<hw::StructCreateOp>(curr->getLoc(), structType, inputs);
      auto replacementCall = tmpBuilder.create<CallOp>(curr->getLoc(), defOp.getResultTypes(), FlatSymbolRefAttr::get(defOp.getNameAttr()), structInput);
      root->replaceAllUsesWith(replacementCall->getResults());
      visited.insert(replacementCall);
      visited.insert(structInput.getDefiningOp());
    }

    visited.insert(curr);
    worklist.pop_back();
  }

  SmallVector<Value> inputs;
  auto defOp = outlineIntoArc(arcBuilder, callOp, inputs, false);

  callOp.setCalleeFromCallable(SymbolRefAttr::get(defOp.getNameAttr()));
  callOp.getArgOperandsMutable().assign(inputs);

  return success();
}

void SinkPureArcsPass::runOnModule(hw::HWModuleOp module) {
  OpBuilder arcBuilder(module);
  for (auto &op : *module.getBodyBlock()) {
    if (auto stateOp = dyn_cast<StateOp>(&op); stateOp && stateOp.getLatency() > 0)
      if (failed(runOnNonPureCallOp(arcBuilder, stateOp)))
        return;
    if (isa<MemoryWritePortOp>(&op))
      if (failed(runOnNonPureCallOp(arcBuilder, cast<CallOpMutableInterface>(&op))))
        return;
  }
}

void SinkPureArcsPass::runOnOperation() {
  cache.addDefinitions(getOperation());
  names.add(cache);

  for (auto hwModule : getOperation().getOps<hw::HWModuleOp>())
    runOnModule(hwModule);
}

std::unique_ptr<Pass> arc::createSinkPureArcsPass() {
  return std::make_unique<SinkPureArcsPass>();
}
