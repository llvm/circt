//===- LogicExporter.cpp - Pass to extrapolate CIRCT IR logic ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file defines the logic-exporting pass for the `circt-lec` tool.
///
//===----------------------------------------------------------------------===//

#include "LogicExporter.h"
#include "Circuit.h"
#include "Solver.h"
#include "Utility.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "pass"

namespace {
/// Helper function to provide a common debug formatting for
/// an operation's list of operands.
template <class ConcreteOp>
static inline void debugOperands(ConcreteOp op) {
  for (const mlir::OpOperand &operand : op->getOpOperands()) {
    mlir::Value value = operand.get();
    lec::dbgs << "Operand:\n";
    INDENT();
    lec::printValue(value);
  }
}

/// Helper function to provide a common debug formatting for
/// an operation's result.
static inline void debugOpResult(const mlir::Value &result) {
  lec::dbgs << "Result:\n";
  INDENT();
  lec::printValue(result);
}

/// Helper function to provide a common debug formatting for
/// an operation's list of results.
template <class ConcreteOp>
static inline void
debugOpResults(mlir::OpTrait::VariadicResults<ConcreteOp> *op) {
  lec::dbgs << "Results:\n";
  for (const mlir::OpResult result : op->getResults()) {
    INDENT();
    lec::dbgs << "#" << result.getResultNumber() << " ";
    debugOpResult(result);
  }
}

/// Helper function to provide a common debug formatting for
/// an operation's list of attributes.
static inline void
debugAttributes(llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  lec::dbgs << "Attributes:\n";
  INDENT();
  for (mlir::NamedAttribute attr : attributes) {
    lec::dbgs << attr.getName().getValue() << ": " << attr.getValue() << "\n";
  }
}
} // anonymous namespace

/// Initializes the pass by visiting the builtin module.
void LogicExporter::runOnOperation() {
  mlir::Operation *op = mlir::Pass::getOperation();
  if (auto builtinModule = llvm::dyn_cast<mlir::ModuleOp>(op)) {
    mlir::LogicalResult outcome =
        Visitor::visitBuiltin(builtinModule, circuit, moduleName);
    if (mlir::failed(outcome))
      mlir::Pass::signalPassFailure();
  } else {
    op->emitError("expected `builtin.module`");
    mlir::Pass::signalPassFailure();
  }
}

/// Visits the given `builtin.module` in search of a specified `hw.module`
/// and returns it.
circt::hw::HWModuleOp
LogicExporter::fetchModuleOp(mlir::ModuleOp builtinModule,
                             llvm::StringRef targetModule) {
  for (const mlir::Operation &op : builtinModule.getOps()) {
    if (auto hwModule = llvm::dyn_cast<circt::hw::HWModuleOp>(op)) {
      llvm::StringRef moduleName = hwModule.getName();
      LLVM_DEBUG(lec::dbgs << "found `hw.module@" << moduleName << "`\n");

      if (moduleName == targetModule)
        return hwModule;
    }
  }
  builtinModule.emitError("expected `" + targetModule + "` module not found");
  // Suppress the compiler's warning.
  return circt::hw::HWModuleOp();
}

//===----------------------------------------------------------------------===//
// Visitor implementation
//===----------------------------------------------------------------------===//
// StmtVisitor implementation
//===----------------------------------------------------------------------===//

mlir::LogicalResult
LogicExporter::Visitor::visitStmt(circt::hw::InstanceOp &op,
                                  Solver::Circuit *circuit) {
  LLVM_DEBUG(lec::dbgs << "Visiting hw.instance\n");
  INDENT();
  LLVM_DEBUG(lec::dbgs << op->getName() << "\n");
  LLVM_DEBUG(debugAttributes(op->getAttrs()));
  LLVM_DEBUG(debugOperands(op));
  LLVM_DEBUG(debugOpResults(&op));
  llvm::StringRef instanceName = op.instanceName();
  LLVM_DEBUG(lec::dbgs << "Instance name: " << instanceName << "\n");
  llvm::StringRef targetModule = op.getModuleName();
  LLVM_DEBUG(lec::dbgs << "Target module name: " << targetModule << "\n");
  llvm::Optional<llvm::StringRef> innerSym = op.getInnerSym();
  LLVM_DEBUG(lec::dbgs << "Inner symbol: " << innerSym << "\n");

  mlir::ModuleOp builtinModule = op->getParentOfType<mlir::ModuleOp>();
  circt::hw::HWModuleOp hwModule =
      LogicExporter::fetchModuleOp(builtinModule, targetModule);
  circuit->addInstance(instanceName.str(), hwModule, op->getOperands(),
                       op->getResults());
  return mlir::success();
}

mlir::LogicalResult
LogicExporter::Visitor::visitStmt(circt::hw::OutputOp &op,
                                  Solver::Circuit *circuit) {
  LLVM_DEBUG(lec::dbgs << "Visiting hw.output\n");
  INDENT();
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
  LLVM_DEBUG(lec::dbgs << "Visiting hw.constant\n");
  INDENT();
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

// This macro implements the visiting function for a `comb` operation accepting
// a variadic number of operands.
#define visitVariadicCombOp(OP_NAME, MLIR_NAME, TYPE)                          \
  mlir::LogicalResult LogicExporter::Visitor::visitComb(                       \
      TYPE op, Solver::Circuit *circuit) {                                     \
    LLVM_DEBUG(lec::dbgs << "Visiting " #MLIR_NAME "\n");                      \
    INDENT();                                                                  \
    LLVM_DEBUG(debugOperands(op));                                             \
    mlir::Value result = op.getResult();                                       \
    LLVM_DEBUG(debugOpResult(result));                                         \
    circuit->perform##OP_NAME(result, op.getOperands());                       \
    return mlir::success();                                                    \
  }

// This macro implements the visiting function for a `comb` operation accepting
// two operands.
#define visitBinaryCombOp(OP_NAME, MLIR_NAME, TYPE)                            \
  mlir::LogicalResult LogicExporter::Visitor::visitComb(                       \
      TYPE op, Solver::Circuit *circuit) {                                     \
    LLVM_DEBUG(lec::dbgs << "Visiting " #MLIR_NAME "\n");                      \
    INDENT();                                                                  \
    LLVM_DEBUG(debugOperands(op));                                             \
    mlir::Value lhs = op.getLhs();                                             \
    mlir::Value rhs = op.getRhs();                                             \
    mlir::Value result = op.getResult();                                       \
    LLVM_DEBUG(debugOpResult(result));                                         \
    circuit->perform##OP_NAME(result, lhs, rhs);                               \
    return mlir::success();                                                    \
  }

// This macro implements the visiting function for a `comb` operation accepting
// one operand.
#define visitUnaryCombOp(OP_NAME, MLIR_NAME, TYPE)                             \
  mlir::LogicalResult LogicExporter::Visitor::visitComb(                       \
      TYPE op, Solver::Circuit *circuit) {                                     \
    LLVM_DEBUG(lec::dbgs << "Visiting " #MLIR_NAME "\n");                      \
    INDENT();                                                                  \
    LLVM_DEBUG(debugOperands(op));                                             \
    mlir::Value input = op.getInput();                                         \
    mlir::Value result = op.getResult();                                       \
    LLVM_DEBUG(debugOpResult(result));                                         \
    circuit->perform##OP_NAME(result, input);                                  \
    return mlir::success();                                                    \
  }

visitVariadicCombOp(Add, comb.add, circt::comb::AddOp &);

visitVariadicCombOp(And, comb.and, circt::comb::AndOp &);

visitVariadicCombOp(Concat, comb.concat, circt::comb::ConcatOp &);

visitBinaryCombOp(DivS, comb.divs, circt::comb::DivSOp &);

visitBinaryCombOp(DivU, comb.divu, circt::comb::DivUOp &);

mlir::LogicalResult
LogicExporter::Visitor::visitComb(circt::comb::ExtractOp &op,
                                  Solver::Circuit *circuit) {
  LLVM_DEBUG(lec::dbgs << "Visiting comb.extract\n");
  INDENT();
  LLVM_DEBUG(debugOperands(op));
  mlir::Value input = op.getInput();
  uint32_t lowBit = op.getLowBit();
  LLVM_DEBUG(lec::dbgs << "lowBit: " << lowBit << "\n");
  mlir::Value result = op.getResult();
  LLVM_DEBUG(debugOpResult(result));
  circuit->performExtract(result, input, lowBit);
  return mlir::success();
}

mlir::LogicalResult
LogicExporter::Visitor::visitComb(circt::comb::ICmpOp &op,
                                  Solver::Circuit *circuit) {
  LLVM_DEBUG(lec::dbgs << "Visiting comb.icmp\n");
  INDENT();
  LLVM_DEBUG(debugOperands(op));
  circt::comb::ICmpPredicate predicate = op.getPredicate();
  mlir::Value lhs = op.getLhs();
  mlir::Value rhs = op.getRhs();
  mlir::Value result = op.getResult();
  LLVM_DEBUG(debugOpResult(result));
  circuit->performICmp(result, predicate, lhs, rhs);
  return mlir::success();
}

visitBinaryCombOp(ModS, comb.mods, circt::comb::ModSOp &);

visitBinaryCombOp(ModU, comb.modu, circt::comb::ModUOp &);

visitVariadicCombOp(Mul, comb.mul, circt::comb::MulOp &);

mlir::LogicalResult
LogicExporter::Visitor::visitComb(circt::comb::MuxOp &op,
                                  Solver::Circuit *circuit) {
  LLVM_DEBUG(lec::dbgs << "Visiting comb.mux\n");
  INDENT();
  LLVM_DEBUG(debugOperands(op));
  mlir::Value cond = op.getCond();
  mlir::Value trueValue = op.getTrueValue();
  mlir::Value falseValue = op.getFalseValue();
  mlir::Value result = op.getResult();
  LLVM_DEBUG(debugOpResult(result));
  circuit->performMux(result, cond, trueValue, falseValue);
  return mlir::success();
}

visitVariadicCombOp(Or, comb.or, circt::comb::OrOp &);

visitUnaryCombOp(Parity, comb.parity, circt::comb::ParityOp &);

visitUnaryCombOp(Replicate, comb.replicate, circt::comb::ReplicateOp &);

visitBinaryCombOp(Shl, comb.shl, circt::comb::ShlOp &);

visitBinaryCombOp(ShrS, comb.shrs, circt::comb::ShrSOp &);

visitBinaryCombOp(ShrU, comb.shru, circt::comb::ShrUOp &);

visitVariadicCombOp(Sub, comb.sub, circt::comb::SubOp &);

visitVariadicCombOp(Xor, comb.xor, circt::comb::XorOp &);

//===----------------------------------------------------------------------===//
// Additional Visitor implementations
//===----------------------------------------------------------------------===//

/// Handles `builtin.module` logic exporting.
mlir::LogicalResult
LogicExporter::Visitor::visitBuiltin(mlir::ModuleOp &op,
                                     Solver::Circuit *circuit,
                                     llvm::StringRef targetModule) {
  LLVM_DEBUG(lec::dbgs << "Visiting `builtin.module`\n");
  INDENT();
  for (mlir::Operation &op : op.getOps()) {
    if (auto hwModule = llvm::dyn_cast<circt::hw::HWModuleOp>(op)) {
      llvm::StringRef moduleName = hwModule.getName();
      LLVM_DEBUG(lec::dbgs << "found `hw.module@" << moduleName << "`\n");

      // When no module name is specified the first module encountered is
      // selected.
      if (targetModule.empty() || moduleName == targetModule) {
        INDENT();
        LLVM_DEBUG(lec::dbgs << "proceeding with this module\n");
        return visitHW(hwModule, circuit);
      }
    } else
      op.emitWarning("only `hw.module` checking is implemented");
    // return mlir::failure();
  }
  op.emitError("expected `" + targetModule + "` module not found");
  return mlir::failure();
}

/// Handles `hw.module` logic exporting.
mlir::LogicalResult LogicExporter::Visitor::visitHW(circt::hw::HWModuleOp &op,
                                                    Solver::Circuit *circuit) {
  LLVM_DEBUG(lec::dbgs << "Visiting `hw.module@" << op.getName() << "`\n");
  INDENT();
  LLVM_DEBUG(debugAttributes(op->getAttrs()));
  LLVM_DEBUG(lec::dbgs << "Arguments:\n");
  for (mlir::BlockArgument argument : op.getArguments()) {
    INDENT();
    LLVM_DEBUG(lec::dbgs << "Argument\n");
    {
      INDENT();
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

#undef DEBUG_TYPE
