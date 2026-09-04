//===- HandshakeUtils.cpp - handshake pass helper functions -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions for various helper functions used in handshake
// passes.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Handshake/HandshakeUtils.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace handshake;
using namespace mlir;

/// Iterates over the handshake::FuncOp's in the program to build an instance
/// graph. In doing so, we detect whether there are any cycles in this graph, as
/// well as infer a top module for the design by performing a topological sort
/// of the instance graph. The result of this sort is placed in sortedFuncs.
LogicalResult circt::handshake::resolveInstanceGraph(
    ModuleOp moduleOp, InstanceGraph &instanceGraph, std::string &topLevel,
    SmallVectorImpl<std::string> &sortedFuncs) {
  // Create use graph
  auto walkFuncOps = [&](handshake::FuncOp funcOp) {
    auto &funcUses = instanceGraph[funcOp.getName().str()];
    funcOp.walk([&](handshake::InstanceOp instanceOp) {
      funcUses.insert(instanceOp.getModule().str());
    });
  };
  moduleOp.walk(walkFuncOps);

  // find top-level (and cycles) using a topological sort. Initialize all
  // instances as candidate top level modules; these will be pruned whenever
  // they are referenced by another module.
  std::set<std::string> visited, marked, candidateTopLevel;
  SmallVector<std::string> cycleTrace;
  bool cyclic = false;
  llvm::transform(instanceGraph,
                  std::inserter(candidateTopLevel, candidateTopLevel.begin()),
                  [](auto it) { return it.first; });
  std::function<void(const std::string &, SmallVector<std::string>)> cycleUtil =
      [&](const std::string &node, SmallVector<std::string> trace) {
        if (cyclic || visited.count(node))
          return;
        trace.push_back(node);
        if (marked.count(node)) {
          cyclic = true;
          cycleTrace = trace;
          return;
        }
        marked.insert(node);
        for (auto use : instanceGraph[node]) {
          candidateTopLevel.erase(use);
          cycleUtil(use, trace);
        }
        marked.erase(node);
        visited.insert(node);
        sortedFuncs.insert(sortedFuncs.begin(), node);
      };
  for (auto it : instanceGraph) {
    if (visited.count(it.first) == 0)
      cycleUtil(it.first, {});
    if (cyclic)
      break;
  }

  if (cyclic) {
    auto err = moduleOp.emitOpError();
    err << "cannot deduce top level function - cycle "
           "detected in instance graph (";
    llvm::interleave(
        cycleTrace, err, [&](auto node) { err << node; }, "->");
    err << ").";
    return err;
  }
  assert(!candidateTopLevel.empty() &&
         "if non-cyclic, there should be at least 1 candidate top level");

  if (candidateTopLevel.size() > 1) {
    auto err = moduleOp.emitOpError();
    err << "multiple candidate top-level modules detected (";
    llvm::interleaveComma(candidateTopLevel, err,
                          [&](auto topLevel) { err << topLevel; });
    err << "). Please remove one of these from the source program.";
    return err;
  }
  topLevel = *candidateTopLevel.begin();
  return success();
}

LogicalResult
circt::handshake::verifyAllValuesHasOneUse(handshake::FuncOp funcOp) {
  if (funcOp.isExternal())
    return success();

  auto checkUseFunc = [&](Operation *op, Value v, StringRef desc,
                          unsigned idx) -> LogicalResult {
    auto numUses = std::distance(v.getUses().begin(), v.getUses().end());
    if (numUses == 0)
      return op->emitOpError() << desc << " " << idx << " has no uses.";
    if (numUses > 1)
      return op->emitOpError() << desc << " " << idx << " has multiple uses.";
    return success();
  };

  for (auto &subOp : funcOp.getOps()) {
    for (auto res : llvm::enumerate(subOp.getResults())) {
      if (failed(checkUseFunc(&subOp, res.value(), "result", res.index())))
        return failure();
    }
  }

  Block &entryBlock = funcOp.front();
  for (auto barg : enumerate(entryBlock.getArguments())) {
    if (failed(checkUseFunc(funcOp.getOperation(), barg.value(), "argument",
                            barg.index())))
      return failure();
  }
  return success();
}

// NOLINTNEXTLINE(misc-no-recursion)
static Type tupleToStruct(TupleType tuple) {
  auto *ctx = tuple.getContext();
  mlir::SmallVector<hw::StructType::FieldInfo, 8> hwfields;
  for (auto [i, innerType] : llvm::enumerate(tuple)) {
    Type convertedInnerType = innerType;
    if (auto tupleInnerType = dyn_cast<TupleType>(innerType))
      convertedInnerType = tupleToStruct(tupleInnerType);
    hwfields.push_back({StringAttr::get(ctx, "field" + std::to_string(i)),
                        convertedInnerType});
  }

  return hw::StructType::get(ctx, hwfields);
}

// Converts 't' into a valid HW type. This is strictly used for converting
// 'index' types into a fixed-width type.
Type circt::handshake::toValidType(Type t) {
  return TypeSwitch<Type, Type>(t)
      .Case<IndexType>(
          [&](IndexType it) { return IntegerType::get(it.getContext(), 64); })
      .Case<TupleType>([&](TupleType tt) {
        llvm::SmallVector<Type> types;
        for (auto innerType : tt)
          types.push_back(toValidType(innerType));
        return tupleToStruct(
            mlir::TupleType::get(types[0].getContext(), types));
      })
      .Case<hw::StructType>([&](auto st) {
        llvm::SmallVector<hw::StructType::FieldInfo> structFields(
            st.getElements());
        for (auto &field : structFields)
          field.type = toValidType(field.type);
        return hw::StructType::get(st.getContext(), structFields);
      })
      .Case<NoneType>(
          [&](NoneType nt) { return IntegerType::get(nt.getContext(), 0); })
      .Default([&](Type t) { return t; });
}

namespace {

/// A class to be used with getPortInfoForOp. Provides an opaque interface for
/// generating the port names of an operation; handshake operations generate
/// names by the Handshake NamedIOInterface;  and other operations, such as
/// arith ops, are assigned default names.
class HandshakePortNameGenerator {
public:
  explicit HandshakePortNameGenerator(Operation *op)
      : builder(op->getContext()) {
    auto namedOpInterface = dyn_cast<handshake::NamedIOInterface>(op);
    if (namedOpInterface)
      inferFromNamedOpInterface(namedOpInterface);
    else if (auto funcOp = dyn_cast<handshake::FuncOp>(op))
      inferFromFuncOp(funcOp);
    else
      inferDefault(op);
  }

  StringAttr inputName(unsigned idx) { return inputs[idx]; }
  StringAttr outputName(unsigned idx) { return outputs[idx]; }

private:
  using IdxToStrF = const std::function<std::string(unsigned)> &;
  void infer(Operation *op, IdxToStrF &inF, IdxToStrF &outF) {
    llvm::transform(
        llvm::enumerate(op->getOperandTypes()), std::back_inserter(inputs),
        [&](auto it) { return builder.getStringAttr(inF(it.index())); });
    llvm::transform(
        llvm::enumerate(op->getResultTypes()), std::back_inserter(outputs),
        [&](auto it) { return builder.getStringAttr(outF(it.index())); });
  }

  void inferDefault(Operation *op) {
    infer(
        op, [](unsigned idx) { return "in" + std::to_string(idx); },
        [](unsigned idx) { return "out" + std::to_string(idx); });
  }

  void inferFromNamedOpInterface(handshake::NamedIOInterface op) {
    infer(
        op, [&](unsigned idx) { return op.getOperandName(idx); },
        [&](unsigned idx) { return op.getResultName(idx); });
  }

  void inferFromFuncOp(handshake::FuncOp op) {
    auto inF = [&](unsigned idx) { return op.getArgName(idx).str(); };
    auto outF = [&](unsigned idx) { return op.getResName(idx).str(); };
    llvm::transform(
        llvm::enumerate(op.getArgumentTypes()), std::back_inserter(inputs),
        [&](auto it) { return builder.getStringAttr(inF(it.index())); });
    llvm::transform(
        llvm::enumerate(op.getResultTypes()), std::back_inserter(outputs),
        [&](auto it) { return builder.getStringAttr(outF(it.index())); });
  }

  Builder builder;
  llvm::SmallVector<StringAttr> inputs;
  llvm::SmallVector<StringAttr> outputs;
};
} // namespace

static void replaceFirstUse(Operation *op, Value oldVal, Value newVal) {
  for (int i = 0, e = op->getNumOperands(); i < e; ++i)
    if (op->getOperand(i) == oldVal) {
      op->setOperand(i, newVal);
      break;
    }
}

void circt::handshake::insertFork(Value result, bool isLazy,
                                  OpBuilder &rewriter) {
  // Get successor operations
  std::vector<Operation *> opsToProcess;
  for (auto &u : result.getUses())
    opsToProcess.push_back(u.getOwner());

  // Insert fork after op
  rewriter.setInsertionPointAfterValue(result);
  auto forkSize = opsToProcess.size();
  Operation *newOp;
  if (isLazy)
    newOp = LazyForkOp::create(rewriter, result.getLoc(), result, forkSize);
  else
    newOp = ForkOp::create(rewriter, result.getLoc(), result, forkSize);

  // Modify operands of successor
  // opsToProcess may have multiple instances of same operand
  // Replace uses one by one to assign different fork outputs to them
  for (int i = 0, e = forkSize; i < e; ++i)
    replaceFirstUse(opsToProcess[i], result, newOp->getResult(i));
}

// NOLINTNEXTLINE(misc-no-recursion)
esi::ChannelType circt::handshake::esiWrapper(Type t) {
  return TypeSwitch<Type, esi::ChannelType>(t)
      .Case<esi::ChannelType>([](auto t) { return t; })
      .Case<TupleType>(
          [&](TupleType tt) { return esiWrapper(tupleToStruct(tt)); })
      .Case<NoneType>([](NoneType nt) {
        // todo: change when handshake switches to i0
        return esiWrapper(IntegerType::get(nt.getContext(), 0));
      })
      .Default([](auto t) {
        return esi::ChannelType::get(t.getContext(), toValidType(t));
      });
}

hw::ModulePortInfo circt::handshake::getPortInfoForOpTypes(Operation *op,
                                                           TypeRange inputs,
                                                           TypeRange outputs) {
  SmallVector<hw::PortInfo> pinputs, poutputs;

  HandshakePortNameGenerator portNames(op);
  auto *ctx = op->getContext();

  Type i1Type = IntegerType::get(ctx, 1);
  Type clkType = seq::ClockType::get(ctx);

  // Add all inputs of funcOp.
  unsigned inIdx = 0;
  for (auto arg : llvm::enumerate(inputs)) {
    pinputs.push_back(
        {{portNames.inputName(arg.index()), esiWrapper(arg.value()),
          hw::ModulePort::Direction::Input},
         arg.index(),
         {}});
    inIdx++;
  }

  // Add all outputs of funcOp.
  for (auto res : llvm::enumerate(outputs)) {
    poutputs.push_back(
        {{portNames.outputName(res.index()), esiWrapper(res.value()),
          hw::ModulePort::Direction::Output},
         res.index(),
         {}});
  }

  // Add clock and reset signals.
  if (op->hasTrait<mlir::OpTrait::HasClock>()) {
    pinputs.push_back({{StringAttr::get(ctx, "clock"), clkType,
                        hw::ModulePort::Direction::Input},
                       inIdx++,
                       {}});
    pinputs.push_back({{StringAttr::get(ctx, "reset"), i1Type,
                        hw::ModulePort::Direction::Input},
                       inIdx,
                       {}});
  }

  return hw::ModulePortInfo{pinputs, poutputs};
}
