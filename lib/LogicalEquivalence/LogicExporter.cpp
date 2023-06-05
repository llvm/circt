//===- LogicExporter.cpp - class to extrapolate CIRCT IR logic --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file defines the logic-exporting class for the `circt-lec` tool.
///
//===----------------------------------------------------------------------===//

#include "circt/LogicalEquivalence/LogicExporter.h"
#include "circt/LogicalEquivalence/Circuit.h"
#include "circt/LogicalEquivalence/Solver.h"
#include "circt/LogicalEquivalence/Utility.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "lec-exporter"

namespace {
/// Helper function to provide a common debug formatting for
/// an operation's list of operands.
template <class ConcreteOp>
static void debugOperands(ConcreteOp op) {
  for (const mlir::OpOperand &operand : op->getOpOperands()) {
    mlir::Value value = operand.get();
    lec::dbgs() << "Operand:\n";
    lec::Scope indent;
    lec::printValue(value);
  }
}

/// Helper function to provide a common debug formatting for
/// an operation's result.
static void debugOpResult(mlir::Value result) {
  lec::dbgs() << "Result:\n";
  lec::Scope indent;
  lec::printValue(result);
}

/// Helper function to provide a common debug formatting for
/// an operation's list of results.
template <class ConcreteOp>
static void debugOpResults(mlir::OpTrait::VariadicResults<ConcreteOp> *op) {
  lec::dbgs() << "Results:\n";
  for (mlir::OpResult result : op->getResults()) {
    lec::Scope indent;
    lec::dbgs() << "#" << result.getResultNumber() << " ";
    debugOpResult(result);
  }
}

/// Helper function to provide a common debug formatting for
/// an operation's list of attributes.
static void debugAttributes(llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  lec::dbgs() << "Attributes:\n";
  lec::Scope indent;
  for (mlir::NamedAttribute attr : attributes) {
    lec::dbgs() << attr.getName().getValue() << ": " << attr.getValue() << "\n";
  }
}
} // anonymous namespace

/// Initializes the exporter by visiting the builtin module.
mlir::LogicalResult LogicExporter::run(mlir::ModuleOp &builtinModule) {
  mlir::LogicalResult outcome =
      Visitor::visitBuiltin(builtinModule, circuit, moduleName);
  return outcome;
}

//===----------------------------------------------------------------------===//
// Visitor implementation
//===----------------------------------------------------------------------===//
// StmtVisitor implementation
//===----------------------------------------------------------------------===//

mlir::LogicalResult
LogicExporter::Visitor::visitStmt(circt::hw::InstanceOp &op,
                                  Solver::Circuit *circuit) {
  LLVM_DEBUG(lec::dbgs() << "Visiting hw.instance\n");
  lec::Scope indent;
  LLVM_DEBUG(lec::dbgs() << op->getName() << "\n");
  LLVM_DEBUG(debugAttributes(op->getAttrs()));
  LLVM_DEBUG(debugOperands(op));
  LLVM_DEBUG(debugOpResults(&op));
  llvm::StringRef instanceName = op.getInstanceName();
  LLVM_DEBUG(lec::dbgs() << "Instance name: " << instanceName << "\n");
  llvm::StringRef targetModule = op.getModuleName();
  LLVM_DEBUG(lec::dbgs() << "Target module name: " << targetModule << "\n");
  llvm::Optional<llvm::StringRef> innerSym = op.getInnerSym();
  LLVM_DEBUG(lec::dbgs() << "Inner symbol: " << innerSym << "\n");

  auto hwModule = llvm::dyn_cast_if_present<circt::hw::HWModuleOp>(
      op.getReferencedModule());
  if (hwModule) {
    circuit->addInstance(instanceName.str(), hwModule, op->getOperands(),
                         op->getResults());
    return mlir::success();
  }
  op.emitError("expected referenced module `" + targetModule + "` not found");
  return mlir::failure();
}

mlir::LogicalResult
LogicExporter::Visitor::visitStmt(circt::hw::OutputOp &op,
                                  Solver::Circuit *circuit) {
  LLVM_DEBUG(lec::dbgs() << "Visiting hw.output\n");
  lec::Scope indent;
  LLVM_DEBUG(debugOperands(op));
  for (auto operand : op.getOperands())
    circuit->addOutput(operand);
  return mlir::success();
}

/// Collects unhandled `hw` statement operations.
mlir::LogicalResult
LogicExporter::Visitor::visitStmt(mlir::Operation *op,
                                  Solver::Circuit *circuit) {
  return visitUnhandledOp(op);
}

/// Handles invalid `hw` statement operations.
mlir::LogicalResult
LogicExporter::Visitor::visitInvalidStmt(mlir::Operation *op,
                                         Solver::Circuit *circuit) {
  // op is not valid for StmtVisitor.
  // Attempt dispatching it to TypeOpVisitor next.
  return dispatchTypeOpVisitor(op, circuit);
}

//===----------------------------------------------------------------------===//
// TypeOpVisitor implementation
//===----------------------------------------------------------------------===//

mlir::LogicalResult
LogicExporter::Visitor::visitTypeOp(circt::hw::ConstantOp &op,
                                    Solver::Circuit *circuit) {
  LLVM_DEBUG(lec::dbgs() << "Visiting hw.constant\n");
  lec::Scope indent;
  mlir::Value result = op.getResult();
  LLVM_DEBUG(lec::printValue(result));
  mlir::APInt value = op.getValue();
  LLVM_DEBUG(lec::printAPInt(value));
  circuit->addConstant(result, value);
  return mlir::success();
}

/// Collects unhandled `hw` type operations.
mlir::LogicalResult
LogicExporter::Visitor::visitTypeOp(mlir::Operation *op,
                                    Solver::Circuit *circuit) {
  return visitUnhandledOp(op);
}

/// Handles invalid `hw` type operations.
mlir::LogicalResult
LogicExporter::Visitor::visitInvalidTypeOp(mlir::Operation *op,
                                           Solver::Circuit *circuit) {
  // op is neither valid for StmtVisitor nor TypeOpVisitor.
  // Attempt dispatching it to CombinationalVisitor next.
  return dispatchCombinationalVisitor(op, circuit);
}

//===----------------------------------------------------------------------===//
// CombinationalVisitor implementation
//===----------------------------------------------------------------------===//

// This macro is used to reject the visited operation when n-state logic is
// not supported.
#define REJECT_N_STATE_LOGIC()                                                 \
  if (!twoState) {                                                             \
    op.emitError("`bin` attribute unset, but n-state logic is not supported"); \
    return mlir::failure();                                                    \
  }

// This macro implements the visiting function for a `comb` operation accepting
// a variadic number of operands.
template <typename OpTy, typename FnTy>
static mlir::LogicalResult visitVariadicCombOp(Solver::Circuit *circuit,
                                               OpTy op, FnTy fn) {
  LLVM_DEBUG(lec::dbgs() << "Visiting " << op->getName() << "\n");
  lec::Scope indent;
  LLVM_DEBUG(debugOperands(op));
  bool twoState = op.getTwoState();
  REJECT_N_STATE_LOGIC();
  mlir::Value result = op.getResult();
  LLVM_DEBUG(debugOpResult(result));
  (circuit->*fn)(result, op.getOperands());
  return mlir::success();
}

// This macro implements the visiting function for a `comb` operation accepting
// two operands.
template <typename OpTy, typename FnTy>
static mlir::LogicalResult visitBinaryCombOp(Solver::Circuit *circuit, OpTy op,
                                             FnTy fn) {
  LLVM_DEBUG(lec::dbgs() << "Visiting " << op->getName() << "\n");
  lec::Scope indent;
  LLVM_DEBUG(debugOperands(op));
  bool twoState = op.getTwoState();
  REJECT_N_STATE_LOGIC();
  auto lhs = op.getLhs();
  auto rhs = op.getRhs();
  auto result = op.getResult();
  LLVM_DEBUG(debugOpResult(result));
  (circuit->*fn)(result, lhs, rhs);
  return mlir::success();
}

// This macro implements the visiting function for a `comb` operation accepting
// one operand.
template <typename OpTy, typename FnTy>
static mlir::LogicalResult visitUnaryCombOp(Solver::Circuit *circuit, OpTy op,
                                            FnTy fn) {
  LLVM_DEBUG(lec::dbgs() << "Visiting " << op->getName() << "\n");
  lec::Scope indent;
  LLVM_DEBUG(debugOperands(op));
  bool twoState = op.getTwoState();
  REJECT_N_STATE_LOGIC();
  auto input = op.getInput();
  auto result = op.getResult();
  LLVM_DEBUG(debugOpResult(result));
  (circuit->*fn)(result, input);
  return mlir::success();
}

mlir::LogicalResult
LogicExporter::Visitor::visitComb(circt::comb::AddOp &op,
                                  Solver::Circuit *circuit) {
  return visitVariadicCombOp(circuit, op, &Solver::Circuit::performAdd);
}

mlir::LogicalResult
LogicExporter::Visitor::visitComb(circt::comb::AndOp &op,
                                  Solver::Circuit *circuit) {
  return visitVariadicCombOp(circuit, op, &Solver::Circuit::performAnd);
}

mlir::LogicalResult
LogicExporter::Visitor::visitComb(circt::comb::ConcatOp &op,
                                  Solver::Circuit *circuit) {
  LLVM_DEBUG(lec::dbgs() << "Visiting comb.concat\n");
  lec::Scope indent;
  LLVM_DEBUG(debugOperands(op));
  mlir::Value result = op.getResult();
  LLVM_DEBUG(debugOpResult(result));
  circuit->performConcat(result, op.getOperands());
  return mlir::success();
}

mlir::LogicalResult
LogicExporter::Visitor::visitComb(circt::comb::DivSOp &op,
                                  Solver::Circuit *circuit) {
  return visitBinaryCombOp(circuit, op, &Solver::Circuit::performDivS);
}

mlir::LogicalResult
LogicExporter::Visitor::visitComb(circt::comb::DivUOp &op,
                                  Solver::Circuit *circuit) {
  return visitBinaryCombOp(circuit, op, &Solver::Circuit::performDivU);
}

mlir::LogicalResult
LogicExporter::Visitor::visitComb(circt::comb::ExtractOp &op,
                                  Solver::Circuit *circuit) {
  LLVM_DEBUG(lec::dbgs() << "Visiting comb.extract\n");
  lec::Scope indent;
  LLVM_DEBUG(debugOperands(op));
  auto input = op.getInput();
  uint32_t lowBit = op.getLowBit();
  LLVM_DEBUG(lec::dbgs() << "lowBit: " << lowBit << "\n");
  auto result = op.getResult();
  LLVM_DEBUG(debugOpResult(result));
  circuit->performExtract(result, input, lowBit);
  return mlir::success();
}

mlir::LogicalResult
LogicExporter::Visitor::visitComb(circt::comb::ICmpOp &op,
                                  Solver::Circuit *circuit) {
  LLVM_DEBUG(lec::dbgs() << "Visiting comb.icmp\n");
  lec::Scope indent;
  LLVM_DEBUG(debugOperands(op));
  bool twoState = op.getTwoState();
  REJECT_N_STATE_LOGIC();
  circt::comb::ICmpPredicate predicate = op.getPredicate();
  auto lhs = op.getLhs();
  auto rhs = op.getRhs();
  auto result = op.getResult();
  LLVM_DEBUG(debugOpResult(result));
  mlir::LogicalResult comparisonResult =
      circuit->performICmp(result, predicate, lhs, rhs);
  return comparisonResult;
}

mlir::LogicalResult
LogicExporter::Visitor::visitComb(circt::comb::ModSOp &op,
                                  Solver::Circuit *circuit) {
  return visitBinaryCombOp(circuit, op, &Solver::Circuit::performModS);
}

mlir::LogicalResult
LogicExporter::Visitor::visitComb(circt::comb::ModUOp &op,
                                  Solver::Circuit *circuit) {
  return visitBinaryCombOp(circuit, op, &Solver::Circuit::performModU);
}

mlir::LogicalResult
LogicExporter::Visitor::visitComb(circt::comb::MulOp &op,
                                  Solver::Circuit *circuit) {
  return visitVariadicCombOp(circuit, op, &Solver::Circuit::performMul);
}

mlir::LogicalResult
LogicExporter::Visitor::visitComb(circt::comb::MuxOp &op,
                                  Solver::Circuit *circuit) {
  LLVM_DEBUG(lec::dbgs() << "Visiting comb.mux\n");
  lec::Scope indent;
  LLVM_DEBUG(debugOperands(op));
  bool twoState = op.getTwoState();
  REJECT_N_STATE_LOGIC();
  auto cond = op.getCond();
  auto trueValue = op.getTrueValue();
  auto falseValue = op.getFalseValue();
  auto result = op.getResult();
  LLVM_DEBUG(debugOpResult(result));
  circuit->performMux(result, cond, trueValue, falseValue);
  return mlir::success();
}

mlir::LogicalResult
LogicExporter::Visitor::visitComb(circt::comb::OrOp &op,
                                  Solver::Circuit *circuit) {
  return visitVariadicCombOp(circuit, op, &Solver::Circuit::performOr);
}

mlir::LogicalResult
LogicExporter::Visitor::visitComb(circt::comb::ParityOp &op,
                                  Solver::Circuit *circuit) {
  return visitUnaryCombOp(circuit, op, &Solver::Circuit::performParity);
}

mlir::LogicalResult
LogicExporter::Visitor::visitComb(circt::comb::ReplicateOp &op,
                                  Solver::Circuit *circuit) {
  LLVM_DEBUG(lec::dbgs() << "Visiting comb.replicate\n");
  lec::Scope indent;
  LLVM_DEBUG(debugOperands(op));
  auto input = op.getInput();
  auto result = op.getResult();
  LLVM_DEBUG(debugOpResult(result));
  circuit->performReplicate(result, input);
  return mlir::success();
}

mlir::LogicalResult
LogicExporter::Visitor::visitComb(circt::comb::ShlOp &op,
                                  Solver::Circuit *circuit) {
  return visitBinaryCombOp(circuit, op, &Solver::Circuit::performShl);
}

mlir::LogicalResult
LogicExporter::Visitor::visitComb(circt::comb::ShrSOp &op,
                                  Solver::Circuit *circuit) {
  return visitBinaryCombOp(circuit, op, &Solver::Circuit::performShrS);
}

mlir::LogicalResult
LogicExporter::Visitor::visitComb(circt::comb::ShrUOp &op,
                                  Solver::Circuit *circuit) {
  return visitBinaryCombOp(circuit, op, &Solver::Circuit::performShrU);
}

mlir::LogicalResult
LogicExporter::Visitor::visitComb(circt::comb::SubOp &op,
                                  Solver::Circuit *circuit) {
  return visitVariadicCombOp(circuit, op, &Solver::Circuit::performSub);
}

mlir::LogicalResult
LogicExporter::Visitor::visitComb(circt::comb::XorOp &op,
                                  Solver::Circuit *circuit) {
  return visitVariadicCombOp(circuit, op, &Solver::Circuit::performXor);
}

//===----------------------------------------------------------------------===//
// Additional Visitor implementations
//===----------------------------------------------------------------------===//

/// Handles `builtin.module` logic exporting.
mlir::LogicalResult
LogicExporter::Visitor::visitBuiltin(mlir::ModuleOp &op,
                                     Solver::Circuit *circuit,
                                     llvm::StringRef targetModule) {
  LLVM_DEBUG(lec::dbgs() << "Visiting `builtin.module`\n");
  lec::Scope indent;
  // Currently only `hw.module` handling is implemented.
  for (auto hwModule : op.getOps<circt::hw::HWModuleOp>()) {
    llvm::StringRef moduleName = hwModule.getName();
    LLVM_DEBUG(lec::dbgs() << "found `hw.module@" << moduleName << "`\n");

    // When no module name is specified the first module encountered is
    // selected.
    if (targetModule.empty() || moduleName == targetModule) {
      lec::Scope indent;
      LLVM_DEBUG(lec::dbgs() << "proceeding with this module\n");
      return visitHW(hwModule, circuit);
    }
  }
  op.emitError("expected `" + targetModule + "` module not found");
  return mlir::failure();
}

/// Handles `hw.module` logic exporting.
mlir::LogicalResult LogicExporter::Visitor::visitHW(circt::hw::HWModuleOp &op,
                                                    Solver::Circuit *circuit) {
  LLVM_DEBUG(lec::dbgs() << "Visiting `hw.module@" << op.getName() << "`\n");
  lec::Scope indent;
  LLVM_DEBUG(debugAttributes(op->getAttrs()));
  LLVM_DEBUG(lec::dbgs() << "Arguments:\n");
  for (mlir::BlockArgument argument : op.getArguments()) {
    lec::Scope indent;
    LLVM_DEBUG(lec::dbgs() << "Argument\n");
    {
      lec::Scope indent;
      LLVM_DEBUG(lec::printValue(argument));
    }
    circuit->addInput(argument);
  }

  // Traverse the module's IR, dispatching the appropriate visiting function.
  Visitor visitor;
  for (mlir::Operation &op : op.getOps()) {
    mlir::LogicalResult outcome = visitor.dispatch(&op, circuit);
    if (outcome.failed())
      return outcome;
  }

  return mlir::success();
}

/// Reports a failure whenever an unhandled operation is visited.
mlir::LogicalResult
LogicExporter::Visitor::visitUnhandledOp(mlir::Operation *op) {
  return mlir::failure();
}

/// Dispatches an operation to the appropriate visit function.
mlir::LogicalResult LogicExporter::Visitor::dispatch(mlir::Operation *op,
                                                     Solver::Circuit *circuit) {
  // Attempt dispatching the operation to the StmtVisitor; if it is an invalid
  // `hw` statement operation, the StmtVisitor will dispatch it to another
  // visitor, and so on in a chain until it gets dispatched to the appropriate
  // visitor.
  return dispatchStmtVisitor(op, circuit);
}
