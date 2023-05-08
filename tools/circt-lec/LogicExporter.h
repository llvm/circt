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

// NOLINTNEXTLINE
#ifndef TOOLS_CIRCT_LEC_LOGICEXPORTER_H
#define TOOLS_CIRCT_LEC_LOGICEXPORTER_H

#include "Solver.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Comb/CombVisitors.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWVisitors.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include <string>

/// A class traversing MLIR IR to extrapolate the logic of a given circuit.
///
/// This class implements a MLIR exporter which searches the IR for the
/// specified `hw.module` describing a circuit. It will then traverse its
/// operations and collect the underlying logical constraints within an
/// abstract circuit representation.
class LogicExporter {
public:
  LogicExporter(llvm::StringRef moduleName, Solver::Circuit *circuit)
      : moduleName(moduleName), circuit(circuit) {}

  /// Initializes the exporting by visiting the builtin module.
  mlir::LogicalResult run(mlir::ModuleOp &module);

private:
  /// This class provides logic-exporting functions for the implemented
  /// operations, along with a dispatcher to visit the correct handler.
  struct Visitor
      : public circt::hw::StmtVisitor<Visitor, mlir::LogicalResult,
                                      Solver::Circuit *>,
        public circt::hw::TypeOpVisitor<Visitor, mlir::LogicalResult,
                                        Solver::Circuit *>,
        public circt::comb::CombinationalVisitor<Visitor, mlir::LogicalResult,
                                                 Solver::Circuit *> {
    // StmtVisitor definitions
    // Handle implemented `hw` statement operations.
    static mlir::LogicalResult visitStmt(circt::hw::InstanceOp &op,
                                         Solver::Circuit *circuit);
    static mlir::LogicalResult visitStmt(circt::hw::OutputOp &op,
                                         Solver::Circuit *circuit);

    /// Collects unhandled `hw` statement operations.
    static mlir::LogicalResult visitStmt(mlir::Operation *op,
                                         Solver::Circuit *circuit);

    /// Handles invalid `hw` statement operations.
    mlir::LogicalResult visitInvalidStmt(mlir::Operation *op,
                                         Solver::Circuit *circuit);

    // TypeOpVisitor definitions
    // Handle implemented `hw` type operations.
    static mlir::LogicalResult visitTypeOp(circt::hw::ConstantOp &op,
                                           Solver::Circuit *circuit);

    /// Collects unhandled `hw` type operations.
    static mlir::LogicalResult visitTypeOp(mlir::Operation *op,
                                           Solver::Circuit *circuit);

    /// Handles invalid `hw` type operations.
    mlir::LogicalResult visitInvalidTypeOp(mlir::Operation *op,
                                           Solver::Circuit *circuit);

    // CombinationalVisitor definitions
    // Handle implemented `comb` operations.
    static mlir::LogicalResult visitComb(circt::comb::AddOp &op,
                                         Solver::Circuit *circuit);
    static mlir::LogicalResult visitComb(circt::comb::AndOp &op,
                                         Solver::Circuit *circuit);
    static mlir::LogicalResult visitComb(circt::comb::ConcatOp &op,
                                         Solver::Circuit *circuit);
    static mlir::LogicalResult visitComb(circt::comb::DivSOp &op,
                                         Solver::Circuit *circuit);
    static mlir::LogicalResult visitComb(circt::comb::DivUOp &op,
                                         Solver::Circuit *circuit);
    static mlir::LogicalResult visitComb(circt::comb::ExtractOp &op,
                                         Solver::Circuit *circuit);
    static mlir::LogicalResult visitComb(circt::comb::ICmpOp &op,
                                         Solver::Circuit *circuit);
    static mlir::LogicalResult visitComb(circt::comb::ModSOp &op,
                                         Solver::Circuit *circuit);
    static mlir::LogicalResult visitComb(circt::comb::ModUOp &op,
                                         Solver::Circuit *circuit);
    static mlir::LogicalResult visitComb(circt::comb::MulOp &op,
                                         Solver::Circuit *circuit);
    static mlir::LogicalResult visitComb(circt::comb::MuxOp &op,
                                         Solver::Circuit *circuit);
    static mlir::LogicalResult visitComb(circt::comb::OrOp &op,
                                         Solver::Circuit *circuit);
    static mlir::LogicalResult visitComb(circt::comb::ParityOp &op,
                                         Solver::Circuit *circuit);
    static mlir::LogicalResult visitComb(circt::comb::ReplicateOp &op,
                                         Solver::Circuit *circuit);
    static mlir::LogicalResult visitComb(circt::comb::ShlOp &op,
                                         Solver::Circuit *circuit);
    static mlir::LogicalResult visitComb(circt::comb::ShrSOp &op,
                                         Solver::Circuit *circuit);
    static mlir::LogicalResult visitComb(circt::comb::ShrUOp &op,
                                         Solver::Circuit *circuit);
    static mlir::LogicalResult visitComb(circt::comb::SubOp &op,
                                         Solver::Circuit *circuit);
    static mlir::LogicalResult visitComb(circt::comb::XorOp &op,
                                         Solver::Circuit *circuit);

    // Additional definitions
    /// Handles `builtin.module` logic exporting.
    static mlir::LogicalResult visitBuiltin(mlir::ModuleOp &op,
                                            Solver::Circuit *circuit,
                                            llvm::StringRef targetModule);

    /// Handles `hw.module` logic exporting.
    static mlir::LogicalResult visitHW(circt::hw::HWModuleOp &op,
                                       Solver::Circuit *circuit);

    /// Reports a failure whenever an unhandled operation is visited.
    static mlir::LogicalResult visitUnhandledOp(mlir::Operation *op);

    /// Dispatches an operation to the appropriate visit function.
    mlir::LogicalResult dispatch(mlir::Operation *op, Solver::Circuit *circuit);
  };

  // For Solver::Circuit::addInstance to access Visitor::visitHW.
  friend Solver::Circuit;

  /// The specified module name to look for when traversing the input file.
  std::string moduleName;
  /// The circuit representation to hold the logical constraints extracted
  /// from the IR.
  Solver::Circuit *circuit;
};

#endif // TOOLS_CIRCT_LEC_LOGICEXPORTER_H
