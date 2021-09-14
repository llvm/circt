//===- ExpandWhens.cpp - Expand WhenOps into muxed operations ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ExpandWhens pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/FieldRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"

using namespace circt;
using namespace firrtl;

/// This is a determistic mapping of a FieldRef to the last operation which set
/// a value to it.
using ScopeMap = llvm::MapVector<FieldRef, Operation *>;

/// Move all operations from a source block in to a destination block. Leaves
/// the source block empty.
static void mergeBlock(Block &destination, Block::iterator insertPoint,
                       Block &source) {
  destination.getOperations().splice(insertPoint, source.getOperations());
}

//===----------------------------------------------------------------------===//
// Last Connect Resolver
//===----------------------------------------------------------------------===//

namespace {

/// This visitor visits process a block resolving last connect semantics
/// and expanding WhenOps.
template <typename ConcreteT>
class LastConnectResolver : public FIRRTLVisitor<ConcreteT> {
protected:
  bool changed = false;

  /// Map of destinations and the operation which is driving a value to it in
  /// the current scope. This is used for resolving last connect semantics, and
  /// for retrieving the responsible connect operation.
  ScopeMap &scope;

public:
  LastConnectResolver(ScopeMap &scope) : scope(scope) {}

  using FIRRTLVisitor<ConcreteT>::visitExpr;
  using FIRRTLVisitor<ConcreteT>::visitDecl;
  using FIRRTLVisitor<ConcreteT>::visitStmt;

  /// Records a connection to a destination. This will delete a previous
  /// connection to a destination if there was one. Returns true if an old
  /// connect was erased.
  bool setLastConnect(FieldRef dest, Operation *connection) {
    // Try to insert, if it doesn't insert, replace the previous value.
    auto itAndInserted = scope.insert({dest, connection});
    if (!std::get<1>(itAndInserted)) {
      auto iterator = std::get<0>(itAndInserted);
      auto changed = false;
      // Delete the old connection if it exists. Null connections are inserted
      // on declarations.
      if (auto *oldConnect = iterator->second) {
        oldConnect->erase();
        changed = true;
      }
      iterator->second = connection;
      return changed;
    }
    return false;
  }

  /// Get the destination value from a connection.  This supports any operation
  /// which is capable of driving a value.
  static Value getDestinationValue(Operation *op) {
    return cast<ConnectOp>(op).dest();
  }

  /// Get the source value from a connection. This supports any operation which
  /// is capable of driving a value.
  static Value getConnectedValue(Operation *op) {
    return cast<ConnectOp>(op).src();
  }

  /// For every leaf field in the sink, record that it exists and should be
  /// initialized.
  void declareSinks(Value value, Flow flow) {
    auto type = value.getType();
    unsigned id = 0;

    // Recurse through a bundle and declare each leaf sink node.
    std::function<void(Type, Flow)> declare = [&](Type type, Flow flow) {
      // If this is a bundle type, recurse to each of the fields.
      if (auto bundleType = type.dyn_cast<BundleType>()) {
        for (auto &element : bundleType.getElements()) {
          id++;
          if (element.isFlip)
            declare(element.type, swapFlow(flow));
          else
            declare(element.type, flow);
        }
        return;
      }

      // If this is an analog type, it does not need to be tracked.
      if (auto analogType = type.dyn_cast<AnalogType>())
        return;

      // If it is a leaf node with Flow::Sink or Flow::Duplex, it must be
      // initialized.
      if (flow != Flow::Source)
        scope[{value, id}] = nullptr;
    };
    declare(type, flow);
  }

  /// If a value has an outer flip, convert the value to passive.
  Value convertToPassive(OpBuilder &builder, Location loc, Value input) {
    auto inType = input.getType().cast<FIRRTLType>();
    return builder
        .create<mlir::UnrealizedConversionCastOp>(
            loc, inType.getPassiveType(), input)
        .getResult(0);
  }

  /// Take two connection operations and merge them in to a new connect under a
  /// condition.  Destination of both connects should be `dest`.
  ConnectOp flattenConditionalConnections(OpBuilder &b, Location loc,
                                          Value dest, Value cond,
                                          Operation *whenTrueConn,
                                          Operation *whenFalseConn) {
    auto whenTrue = getConnectedValue(whenTrueConn);
    auto whenFalse = getConnectedValue(whenFalseConn);
    auto newValue = b.createOrFold<MuxPrimOp>(loc, cond, whenTrue, whenFalse);
    auto newConnect = b.create<ConnectOp>(loc, dest, newValue);
    whenTrueConn->erase();
    whenFalseConn->erase();
    return newConnect;
  }

  void visitDecl(WireOp op) { declareSinks(op.result(), Flow::Duplex); }

  void visitDecl(RegOp op) {
    // Registers are initialized to themselves.
    // TODO: register of bundle type are not supported.
    auto connect = OpBuilder(op->getBlock(), ++Block::iterator(op))
                       .create<ConnectOp>(op.getLoc(), op, op);
    scope[getFieldRefFromValue(op.result())] = connect;
  }

  void visitDecl(RegResetOp op) {
    // Registers are initialized to themselves.
    // TODO: register of bundle type are not supported.
    assert(!op.result().getType().isa<BundleType>() &&
           "registers can't be bundle type");
    auto connect = OpBuilder(op->getBlock(), ++Block::iterator(op))
                       .create<ConnectOp>(op.getLoc(), op, op);
    scope[getFieldRefFromValue(op.result())] = connect;
  }

  void visitDecl(InstanceOp op) {
    // Track any instance inputs which need to be connected to for init
    // coverage.
    auto ref = op.getReferencedModule();
    for (auto result : llvm::enumerate(op.results()))
      if (ref.getPortDirection(result.index()) == Direction::Out)
        declareSinks(result.value(), Flow::Source);
      else
        declareSinks(result.value(), Flow::Sink);
  }

  void visitDecl(MemOp op) {
    // Track any memory inputs which require connections.
    for (auto result : op.results())
      declareSinks(result, Flow::Sink);
  }

  void visitStmt(PartialConnectOp op) {
    llvm_unreachable("PartialConnectOps should have been removed.");
  }

  void visitStmt(ConnectOp op) {
    setLastConnect(getFieldRefFromValue(op.dest()), op);
  }

  /// Combine the connect statements from each side of the block. There are 5
  /// cases to consider. If all are set, last connect semantics dictate that it
  /// is actually the third case.
  ///
  /// Prev | Then | Else | Outcome
  /// -----|------|------|-------
  ///      |  set |      | then
  ///      |      |  set | else
  ///      |  set |  set | mux(p, then, else)
  ///  set |  set |      | mux(p, then, prev)
  ///  set |      |  set | mux(p, prev, else)
  ///
  /// If the value was declared in the block, then it does not need to have been
  /// assigned a previous value.  If the value was declared before the block,
  /// then there is an incomplete initialization error.
  void mergeScopes(ScopeMap &thenScope, ScopeMap &elseScope,
                   Value thenCondition) {

    // Process all connects in the `then` block.
    for (auto &destAndConnect : thenScope) {
      auto dest = std::get<0>(destAndConnect);
      auto thenConnect = std::get<1>(destAndConnect);

      // `dest` is set in `then` only.
      auto itAndInserted = scope.insert({dest, thenConnect});
      if (std::get<1>(itAndInserted))
        continue;
      auto outerIt = std::get<0>(itAndInserted);

      // `dest` is set in `then` and `else`.
      auto elseIt = elseScope.find(dest);
      if (elseIt != elseScope.end()) {
        auto &elseConnect = std::get<1>(*elseIt);
        // Create a new connect with `mux(p, then, else)`.
        OpBuilder connectBuilder(elseConnect);
        auto newConnect = flattenConditionalConnections(
            connectBuilder, elseConnect->getLoc(),
            getDestinationValue(thenConnect), thenCondition, thenConnect,
            elseConnect);
        setLastConnect(dest, newConnect);
        // Do not process connect in the else scope.
        elseScope.erase(dest);
        continue;
      }

      // `dest` is null in the outer scope.
      auto &outerConnect = std::get<1>(*outerIt);
      if (!outerConnect) {
        thenConnect->erase();
        continue;
      }

      // `dest` is set in the outer scope.
      // Create a new connect with mux(p, then, outer)
      OpBuilder connectBuilder(thenConnect);
      auto newConnect = flattenConditionalConnections(
          connectBuilder, thenConnect->getLoc(),
          getDestinationValue(thenConnect), thenCondition, thenConnect,
          outerConnect);
      outerIt->second = newConnect;
    }

    // Process all connects in the `else` block.
    for (auto &destAndConnect : elseScope) {
      auto dest = std::get<0>(destAndConnect);
      auto elseConnect = std::get<1>(destAndConnect);

      // `dest` is set in `then` only.
      auto itAndInserted = scope.insert({dest, elseConnect});
      if (std::get<1>(itAndInserted))
        continue;

      // `dest` is null in the outer scope.
      auto outerIt = std::get<0>(itAndInserted);
      auto &outerConnect = std::get<1>(*outerIt);
      if (!outerConnect) {
        elseConnect->erase();
        continue;
      }

      // `dest` is set in the outer scope.
      // Create a new connect with mux(p, outer, else).
      OpBuilder connectBuilder(elseConnect);
      auto newConnect = flattenConditionalConnections(
          connectBuilder, elseConnect->getLoc(),
          getDestinationValue(outerConnect), thenCondition, outerConnect,
          elseConnect);
      outerIt->second = newConnect;
    }
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// WhenOpVisitor
//===----------------------------------------------------------------------===//

/// This extends the ModuleVisitor with additional funcationality that is only
/// required in side a WhenOp.  This visitor handles all Simulation constructs.
namespace {
class WhenOpVisitor : public LastConnectResolver<WhenOpVisitor> {

public:
  WhenOpVisitor(ScopeMap &scope, Value condition)
      : LastConnectResolver<WhenOpVisitor>(scope), condition(condition) {}

  using LastConnectResolver<WhenOpVisitor>::visitExpr;
  using LastConnectResolver<WhenOpVisitor>::visitDecl;
  using LastConnectResolver<WhenOpVisitor>::visitStmt;

  /// Process a block, recording each declaration, and expanding all whens.
  void process(Block &block);

  /// Simulation Constructs.
  void visitStmt(AssertOp op);
  void visitStmt(AssumeOp op);
  void visitStmt(CoverOp op);
  void visitStmt(ModuleOp op);
  void visitStmt(PrintFOp op);
  void visitStmt(StopOp op);
  void visitStmt(WhenOp op);

private:
  /// And a 1-bit value with the current condition.  If we are in the outer
  /// scope, i.e. not in a WhenOp region, then there is no condition.
  Value andWithCondition(Operation *op, Value value) {
    // 'and' the value with the current condition.
    return OpBuilder(op).createOrFold<AndPrimOp>(
        condition.getLoc(), condition.getType(), condition, value);
  }

private:
  /// The current wrapping condition. If null, we are in the outer scope.
  Value condition;
};
} // namespace

void WhenOpVisitor::process(Block &block) {

  for (auto &op : llvm::make_early_inc_range(block)) {
    dispatchVisitor(&op);
  }
}

void WhenOpVisitor::visitStmt(PrintFOp op) {
  op.condMutable().assign(andWithCondition(op, op.cond()));
}

void WhenOpVisitor::visitStmt(StopOp op) {
  op.condMutable().assign(andWithCondition(op, op.cond()));
}

void WhenOpVisitor::visitStmt(AssertOp op) {
  op.enableMutable().assign(andWithCondition(op, op.enable()));
}

void WhenOpVisitor::visitStmt(AssumeOp op) {
  op.enableMutable().assign(andWithCondition(op, op.enable()));
}

void WhenOpVisitor::visitStmt(CoverOp op) {
  op.enableMutable().assign(andWithCondition(op, op.enable()));
}

void WhenOpVisitor::visitStmt(WhenOp whenOp) {
  OpBuilder b(whenOp);
  Block *parentBlock = whenOp->getBlock();
  auto condition = whenOp.condition();

  // Process both sides of the the WhenOp, fixing up all simulation
  // contructs, and resolving last connect semantics in each block. This process
  // returns the set of connects in each side of the when op.

  // Process the `then` block.
  ScopeMap thenScope;
  auto thenCondition = andWithCondition(whenOp, condition);
  auto &thenBlock = whenOp.getThenBlock();
  WhenOpVisitor(thenScope, thenCondition).process(thenBlock);
  mergeBlock(*parentBlock, Block::iterator(whenOp), thenBlock);

  // Process the `else` block.
  ScopeMap elseScope;
  if (whenOp.hasElseRegion()) {
    auto notOp = b.createOrFold<NotPrimOp>(whenOp.getLoc(), condition.getType(),
                                           condition);
    Value elseCondition = andWithCondition(whenOp, notOp);
    auto &elseBlock = whenOp.getElseBlock();
    WhenOpVisitor(elseScope, elseCondition).process(elseBlock);
    mergeBlock(*parentBlock, Block::iterator(whenOp), elseBlock);
  }

  mergeScopes(thenScope, elseScope, condition);

  // Delete the now empty WhenOp.
  whenOp.erase();
}

//===----------------------------------------------------------------------===//
// ModuleOpVisitor
//===----------------------------------------------------------------------===//

namespace {
class ModuleVisitor : public LastConnectResolver<ModuleVisitor> {
public:
  ModuleVisitor() : LastConnectResolver<ModuleVisitor>(outerScope) {}

  // Unshadow the overloads.
  using LastConnectResolver<ModuleVisitor>::visitExpr;
  using LastConnectResolver<ModuleVisitor>::visitDecl;
  using LastConnectResolver<ModuleVisitor>::visitStmt;

  void visitStmt(WhenOp whenOp);
  void visitStmt(ConnectOp connectOp);

  /// Run expand whens on the Module.  This will emit an error for each
  /// incomplete initialization found. If an initialiazation error was detected,
  /// this will return failure and leave the IR in an inconsistent state.  If
  /// the pass was a success, returns true if nothing changed.
  mlir::FailureOr<bool> run(FModuleOp op);

private:
  /// The outermost scope of the module body.
  ScopeMap outerScope;

  /// Tracks if anything in the IR has changed.
  bool anythingChanged = false;
};
} // namespace

mlir::FailureOr<bool> ModuleVisitor::run(FModuleOp module) {
  // Track any results (flipped arguments) of the module for init coverage.
  for (auto it : llvm::enumerate(module.getArguments())) {
    auto flow = module.getPortDirection(it.index()) == Direction::In
                    ? Flow::Source
                    : Flow::Sink;
    declareSinks(it.value(), flow);
  }

  // Process the body of the module.
  for (auto &op : llvm::make_early_inc_range(*module.getBodyBlock())) {
    dispatchVisitor(&op);
  }

  // Check for any incomplete initialization.
  for (auto destAndConnect : outerScope) {
    // If there is valid connection to this destination, everything is good.
    auto *connect = std::get<1>(destAndConnect);
    if (connect)
      continue;
    // Get the op which defines the sink, and emit an error.
    auto dest = std::get<0>(destAndConnect);
    dest.getDefiningOp()->emitError("sink \"" + getFieldName(dest) +
                                    "\" not fully initialized");
    return failure();
  }
  return mlir::FailureOr<bool>(anythingChanged);
}

void ModuleVisitor::visitStmt(ConnectOp op) {
  anythingChanged |= setLastConnect(getFieldRefFromValue(op.dest()), op);
}

void ModuleVisitor::visitStmt(WhenOp whenOp) {
  Block *parentBlock = whenOp->getBlock();
  auto condition = whenOp.condition();

  // Process both sides of the the WhenOp, fixing up all simulation contructs,
  // and resolving last connect semantics in each block. This process returns
  // the set of connects in each side of the when op.

  // Process the `then` block.
  ScopeMap thenScope;
  auto &thenBlock = whenOp.getThenBlock();
  WhenOpVisitor(thenScope, condition).process(thenBlock);
  mergeBlock(*parentBlock, Block::iterator(whenOp), thenBlock);

  // Process the `else` block.
  ScopeMap elseScope;
  if (whenOp.hasElseRegion()) {
    OpBuilder b(whenOp);
    auto notCondition = b.createOrFold<NotPrimOp>(
        whenOp.getLoc(), condition.getType(), condition);
    auto &elseBlock = whenOp.getElseBlock();
    WhenOpVisitor(elseScope, notCondition).process(elseBlock);
    mergeBlock(*parentBlock, Block::iterator(whenOp), elseBlock);
  }

  mergeScopes(thenScope, elseScope, condition);

  // If we are deleting a WhenOp something definitely changed.
  anythingChanged = true;

  // Delete the now empty WhenOp.
  whenOp.erase();
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

class ExpandWhensPass : public ExpandWhensBase<ExpandWhensPass> {
  void runOnOperation() override;
};

void ExpandWhensPass::runOnOperation() {
  // Pass returns failure if something went wrong, or a bool indicating whether
  // something changed.
  auto failureOrChanged = ModuleVisitor().run(getOperation());
  if (failed(failureOrChanged)) {
    signalPassFailure();
  } else if (!*failureOrChanged) {
    markAllAnalysesPreserved();
  }
}

std::unique_ptr<mlir::Pass> circt::firrtl::createExpandWhensPass() {
  return std::make_unique<ExpandWhensPass>();
}
