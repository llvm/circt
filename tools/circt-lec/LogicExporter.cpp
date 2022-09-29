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
        Builtin::visitModule(builtinModule, circuit, moduleName);
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
                             circt::StringRef targetModule) {
  for (const circt::Operation &op : builtinModule.getOps()) {
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
// `Builtin` dialect implementation
//===----------------------------------------------------------------------===//

mlir::LogicalResult
LogicExporter::Builtin::visitModule(mlir::ModuleOp &op,
                                    Solver::Circuit *circuit,
                                    circt::StringRef targetModule) {
  LLVM_DEBUG(lec::dbgs << "Visiting `builtin.module`\n");
  INDENT();
  for (circt::Operation &op : op.getOps()) {
    if (auto hwModule = llvm::dyn_cast<circt::hw::HWModuleOp>(op)) {
      llvm::StringRef moduleName = hwModule.getName();
      LLVM_DEBUG(lec::dbgs << "found `hw.module@" << moduleName << "`\n");

      // When no module name is specified the first module encountered is
      // selected.
      if (targetModule.empty() || moduleName == targetModule) {
        INDENT();
        LLVM_DEBUG(lec::dbgs << "proceeding with this module\n");
        return LogicExporter::HW::visitModule(hwModule, circuit);
      }
    } else
      op.emitWarning("only `hw.module` checking is implemented");
    // return mlir::failure();
  }
  op.emitError("expected `" + targetModule + "` module not found");
  return mlir::failure();
}

//===----------------------------------------------------------------------===//
// `hw` dialect implementation
//===----------------------------------------------------------------------===//

mlir::LogicalResult LogicExporter::HW::visitConstant(circt::hw::ConstantOp &op,
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

mlir::LogicalResult LogicExporter::HW::visitModule(circt::hw::HWModuleOp &op,
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
  for (circt::Operation &op : op.getOps()) {
    mlir::LogicalResult outcome = LogicExporter::visitOperation(&op, circuit);
    if (outcome.failed())
      return outcome;
  }

  return mlir::success();
}

mlir::LogicalResult LogicExporter::HW::visitInstance(circt::hw::InstanceOp &op,
                                                     Solver::Circuit *circuit) {
  LLVM_DEBUG(lec::dbgs << "Visiting hw.instance\n");
  INDENT();
  LLVM_DEBUG(lec::dbgs << op->getName() << "\n");
  LLVM_DEBUG(debugAttributes(op->getAttrs()));
  LLVM_DEBUG(debugOperands(op));
  LLVM_DEBUG(debugOpResults(&op));
  circt::StringRef instanceName = op.instanceName();
  LLVM_DEBUG(lec::dbgs << "Instance name: " << instanceName << "\n");
  circt::StringRef targetModule = op.getModuleName();
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

mlir::LogicalResult LogicExporter::HW::visitOutput(circt::hw::OutputOp &op,
                                                   Solver::Circuit *circuit) {
  LLVM_DEBUG(lec::dbgs << "Visiting hw.output\n");
  INDENT();
  LLVM_DEBUG(debugOperands(op));
  for (auto operand : op.getOperands())
    circuit->addOutput(operand);
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// `comb` dialect implementation
//===----------------------------------------------------------------------===//

// This macro implements the visiting function for a `comb` operation accepting
// a variadic number of operands.
#define visitVariadicCombOp(OP_NAME, MLIR_NAME, TYPE)                          \
  mlir::LogicalResult LogicExporter::Comb::visit##OP_NAME(                     \
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
  mlir::LogicalResult LogicExporter::Comb::visit##OP_NAME(                     \
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
  mlir::LogicalResult LogicExporter::Comb::visit##OP_NAME(                     \
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
LogicExporter::Comb::visitExtract(circt::comb::ExtractOp &op,
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

mlir::LogicalResult LogicExporter::Comb::visitICmp(circt::comb::ICmpOp &op,
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

mlir::LogicalResult LogicExporter::Comb::visitMux(circt::comb::MuxOp &op,
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
// Operation visitor
//===----------------------------------------------------------------------===//

/// Dispatches an operation to the appropriate visit function.
mlir::LogicalResult LogicExporter::visitOperation(mlir::Operation *op,
                                                  Solver::Circuit *circuit) {
  mlir::LogicalResult outcome =
      llvm::TypeSwitch<mlir::Operation *, mlir::LogicalResult>(op)
          .Case<circt::hw::ConstantOp>([&](circt::hw::ConstantOp &op) {
            return LogicExporter::HW::visitConstant(op, circuit);
          })
          .Case<circt::hw::InstanceOp>([&](circt::hw::InstanceOp &op) {
            return LogicExporter::HW::visitInstance(op, circuit);
          })
          .Case<circt::hw::OutputOp>([&](circt::hw::OutputOp &op) {
            return LogicExporter::HW::visitOutput(op, circuit);
          })
          .Case<circt::comb::AddOp>([&](circt::comb::AddOp &op) {
            return LogicExporter::Comb::visitAdd(op, circuit);
          })
          .Case<circt::comb::AndOp>([&](circt::comb::AndOp &op) {
            return LogicExporter::Comb::visitAnd(op, circuit);
          })
          .Case<circt::comb::ConcatOp>([&](circt::comb::ConcatOp &op) {
            return LogicExporter::Comb::visitConcat(op, circuit);
          })
          .Case<circt::comb::DivSOp>([&](circt::comb::DivSOp &op) {
            return LogicExporter::Comb::visitDivS(op, circuit);
          })
          .Case<circt::comb::DivUOp>([&](circt::comb::DivUOp &op) {
            return LogicExporter::Comb::visitDivU(op, circuit);
          })
          .Case<circt::comb::ExtractOp>([&](circt::comb::ExtractOp &op) {
            return LogicExporter::Comb::visitExtract(op, circuit);
          })
          .Case<circt::comb::ICmpOp>([&](circt::comb::ICmpOp &op) {
            return LogicExporter::Comb::visitICmp(op, circuit);
          })
          .Case<circt::comb::ModSOp>([&](circt::comb::ModSOp &op) {
            return LogicExporter::Comb::visitModS(op, circuit);
          })
          .Case<circt::comb::ModUOp>([&](circt::comb::ModUOp &op) {
            return LogicExporter::Comb::visitModU(op, circuit);
          })
          .Case<circt::comb::MulOp>([&](circt::comb::MulOp &op) {
            return LogicExporter::Comb::visitMul(op, circuit);
          })
          .Case<circt::comb::MuxOp>([&](circt::comb::MuxOp &op) {
            return LogicExporter::Comb::visitMux(op, circuit);
          })
          .Case<circt::comb::OrOp>([&](circt::comb::OrOp &op) {
            return LogicExporter::Comb::visitOr(op, circuit);
          })
          .Case<circt::comb::ParityOp>([&](circt::comb::ParityOp &op) {
            return LogicExporter::Comb::visitParity(op, circuit);
          })
          .Case<circt::comb::ReplicateOp>([&](circt::comb::ReplicateOp &op) {
            return LogicExporter::Comb::visitReplicate(op, circuit);
          })
          .Case<circt::comb::ShlOp>([&](circt::comb::ShlOp &op) {
            return LogicExporter::Comb::visitShl(op, circuit);
          })
          .Case<circt::comb::ShrSOp>([&](circt::comb::ShrSOp &op) {
            return LogicExporter::Comb::visitShrS(op, circuit);
          })
          .Case<circt::comb::ShrUOp>([&](circt::comb::ShrUOp &op) {
            return LogicExporter::Comb::visitShrU(op, circuit);
          })
          .Case<circt::comb::SubOp>([&](circt::comb::SubOp &op) {
            return LogicExporter::Comb::visitSub(op, circuit);
          })
          .Case<circt::comb::XorOp>([&](circt::comb::XorOp &op) {
            return LogicExporter::Comb::visitXor(op, circuit);
          })
          .Default([](mlir::Operation *op) {
            op->emitOpError("is not implemented");
            return mlir::failure();
          });
  return outcome;
}

#undef DEBUG_TYPE
