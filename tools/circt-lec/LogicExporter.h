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

#ifndef LEC_LOGICEXPORTER_H
#define LEC_LOGICEXPORTER_H

#include "Solver.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include <string>

/// A pass traversing MLIR IR to extrapolate the logic of a given circuit.
///
/// This class implements a MLIR pass which searches the IR for the specified
/// `hw.module` describing a circuit. It will then traverse its operations and
/// collect the underlying logical constraints within an abstract circuit
/// representation.
struct LogicExporter :
public mlir::PassWrapper<LogicExporter, mlir::OperationPass<>> {
public:
  LogicExporter(circt::StringRef moduleName, Solver::Circuit *circuit) :
    moduleName(moduleName), circuit(circuit) {};

  /// Initializes the pass by visiting the builtin module.
  void runOnOperation() override;

private:
  /// Visits the given `builtin.module` in search of a specified `hw.module`
  /// and returns it.
  static circt::hw::HWModuleOp fetchModuleOp(
      mlir::ModuleOp builtinModule, circt::StringRef targetModule);

  /// A base class for representing MLIR dialects.
  class Dialect {};

  /// This class collects logic-exporting functions for the `builtin` dialect.
  struct Builtin : public LogicExporter::Dialect {
    static mlir::LogicalResult visitModule(mlir::ModuleOp &op,
        Solver::Circuit *circuit, circt::StringRef targetModule);
  };

  /// This class collects logic-exporting functions for the `hw` dialect.
  struct HW : public LogicExporter::Dialect {
    static mlir::LogicalResult visitConstant(circt::hw::ConstantOp &op,
        Solver::Circuit *circuit);
    static mlir::LogicalResult visitModule(circt::hw::HWModuleOp &op,
        Solver::Circuit *circuit);
    static mlir::LogicalResult visitInstance(circt::hw::InstanceOp &op,
        Solver::Circuit *circuit);
    static mlir::LogicalResult visitOutput(circt::hw::OutputOp &op,
        Solver::Circuit *circuit);
  };

  /// This class collects logic-exporting functions for the `comb` dialect.
  struct Comb : public LogicExporter::Dialect {
    static mlir::LogicalResult visitAdd(circt::comb::AddOp &op,
        Solver::Circuit *circuit);
    static mlir::LogicalResult visitAnd(circt::comb::AndOp &op,
        Solver::Circuit *circuit);
    static mlir::LogicalResult visitConcat(circt::comb::ConcatOp &op,
        Solver::Circuit *circuit);
    static mlir::LogicalResult visitDivS(circt::comb::DivSOp &op,
        Solver::Circuit *circuit);
    static mlir::LogicalResult visitDivU(circt::comb::DivUOp &op,
        Solver::Circuit *circuit);
    static mlir::LogicalResult visitExtract(circt::comb::ExtractOp &op, 
        Solver::Circuit *circuit);
    static mlir::LogicalResult visitICmp(circt::comb::ICmpOp &op,
        Solver::Circuit *circuit);
    static mlir::LogicalResult visitModS(circt::comb::ModSOp &op,
        Solver::Circuit *circuit);
    static mlir::LogicalResult visitModU(circt::comb::ModUOp &op,
        Solver::Circuit *circuit);
    static mlir::LogicalResult visitMul(circt::comb::MulOp &op,
        Solver::Circuit *circuit);
    static mlir::LogicalResult visitMux(circt::comb::MuxOp &op,
        Solver::Circuit *circuit);
    static mlir::LogicalResult visitOr(circt::comb::OrOp &op,
        Solver::Circuit *circuit);
    static mlir::LogicalResult visitParity(circt::comb::ParityOp &op,
        Solver::Circuit *circuit);
    static mlir::LogicalResult visitReplicate(circt::comb::ReplicateOp &op,
        Solver::Circuit *circuit);
    static mlir::LogicalResult visitShl(circt::comb::ShlOp &op,
        Solver::Circuit *circuit);
    static mlir::LogicalResult visitShrS(circt::comb::ShrSOp &op,
        Solver::Circuit *circuit);
    static mlir::LogicalResult visitShrU(circt::comb::ShrUOp &op,
        Solver::Circuit *circuit);
    static mlir::LogicalResult visitSub(circt::comb::SubOp &op,
        Solver::Circuit *circuit);
    static mlir::LogicalResult visitXor(circt::comb::XorOp &op,
        Solver::Circuit *circuit);
  };

  /// Dispatches an operation to the appropriate visit function.
  static mlir::LogicalResult visitOperation(mlir::Operation *op,
      Solver::Circuit *circuit);

  // for Solver::Circuit::addInstance to access LogicExporter::HW::visitModule
  friend Solver::Circuit;

  /// The specified module name to look for when traversing the input file.
  std::string moduleName;
  /// The circuit representation to hold the logical constraints extracted
  /// from the IR.
  Solver::Circuit *circuit;
};

#endif // LEC_LOGICEXPORTER_H
