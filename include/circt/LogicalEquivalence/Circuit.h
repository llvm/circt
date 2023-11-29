//===-- Circuit.h - intermediate representation for circuits ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file defines an intermediate representation for circuits acting as
/// an abstraction for constraints defined over an SMT's solver context.
///
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE
#ifndef TOOLS_CIRCT_LEC_CIRCUIT_H
#define TOOLS_CIRCT_LEC_CIRCUIT_H

#include "Solver.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include <string>
#include <z3++.h>

namespace circt {

/// The representation of a circuit within a logical engine.
///
/// This class defines a circuit as an abstraction of its underlying
/// logical constraints. Its various methods act in a builder pattern fashion,
/// declaring new constraints over a Z3 context.
class Solver::Circuit {
public:
  Circuit(llvm::Twine name, Solver &solver) : name(name.str()), solver(solver) {
    assignments = 0;
    populateCombTransformTable();
  };
  /// Populate the table of combinational transforms
  void populateCombTransformTable();
  /// Add an input to the circuit; internally a new value gets allocated.
  void addInput(mlir::Value);
  /// Add an output to the circuit.
  void addOutput(mlir::Value);
  /// Add a new clock to the list of clocks.
  void addClk(mlir::Value);
  /// Recover the inputs.
  llvm::ArrayRef<z3::expr> getInputs();
  /// Recover the outputs.
  llvm::ArrayRef<z3::expr> getOutputs();

  /// Execute a clock cycle and check that the properties hold throughout
  bool checkCycle(int count);

  // `hw` dialect operations.
  void addConstant(mlir::Value result, const mlir::APInt &value);
  void addInstance(llvm::StringRef instanceName, circt::hw::HWModuleOp op,
                   mlir::OperandRange arguments, mlir::ResultRange results);

  // `comb` dialect operations.
  void performAdd(mlir::Value result, mlir::OperandRange operands);
  void performAnd(mlir::Value result, mlir::OperandRange operands);
  void performConcat(mlir::Value result, mlir::OperandRange operands);
  void performDivS(mlir::Value result, mlir::Value lhs, mlir::Value rhs);
  void performDivU(mlir::Value result, mlir::Value lhs, mlir::Value rhs);
  void performExtract(mlir::Value result, mlir::Value input, uint32_t lowBit);
  void performICmp(mlir::Value result, circt::comb::ICmpPredicate predicate,
                   mlir::Value lhs, mlir::Value rhs);
  void performModS(mlir::Value result, mlir::Value lhs, mlir::Value rhs);
  void performModU(mlir::Value result, mlir::Value lhs, mlir::Value rhs);
  void performMul(mlir::Value result, mlir::OperandRange operands);
  void performMux(mlir::Value result, mlir::Value cond, mlir::Value trueValue,
                  mlir::Value falseValue);
  void performOr(mlir::Value result, mlir::OperandRange operands);
  void performParity(mlir::Value result, mlir::Value input);
  void performReplicate(mlir::Value result, mlir::Value input);
  void performShl(mlir::Value result, mlir::Value lhs, mlir::Value rhs);
  void performShrS(mlir::Value result, mlir::Value lhs, mlir::Value rhs);
  void performShrU(mlir::Value result, mlir::Value lhs, mlir::Value rhs);
  void performSub(mlir::Value result, mlir::OperandRange operands);
  void performXor(mlir::Value result, mlir::OperandRange operands);

  // `seq` dialect operations.
  void performCompReg(mlir::Value input, mlir::Value clk, mlir::Value data,
                      mlir::Value reset, mlir::Value resetValue);
  void performFirReg(mlir::Value next, mlir::Value clk, mlir::Value data,
                     mlir::Value reset, mlir::Value resetValue);
  void performFromClock(mlir::Value result, mlir::Value input);

  // `verif` dialect operations.
  void performAssert(mlir::Value property);
  void performAssume(mlir::Value property);
  void performCover(mlir::Value property);

private:
  /// Struct to represent computational registers
  struct CompRegStruct {
    mlir::Value input;
    mlir::Value clk;
    mlir::Value data;
    mlir::Value reset;
    mlir::Value resetValue;
  };
  /// Struct to represent FIRTLL registers
  struct FirRegStruct {
    mlir::Value next;
    mlir::Value clk;
    mlir::Value data;
    mlir::Value reset;
    mlir::Value resetValue;
    bool isAsync;
  };
  /// A type for the different operand sets a wire might need to store
  using WireVariant = std::variant<
      std::tuple<mlir::Value, llvm::StringLiteral>,
      std::tuple<mlir::Value, mlir::Value, llvm::StringLiteral>,
      std::tuple<mlir::Value, mlir::Value, mlir::Value, llvm::StringLiteral>,
      /*Variadic Ops*/
      std::tuple<mlir::OperandRange, llvm::StringLiteral>,
      /*ICmpOp:*/
      std::tuple<circt::comb::ICmpPredicate, mlir::Value, mlir::Value,
                 llvm::StringLiteral>,
      /*ParityOp & ReplicateOp:*/
      std::tuple<mlir::Value, unsigned int, llvm::StringLiteral>,
      /*ExtractOp:*/
      std::tuple<mlir::Value, uint32_t, int, llvm::StringLiteral>>;

  /// Helper function for performing a variadic operation: it executes a
  /// lambda over a range of operands.
  z3::expr
  variadicOperation(std::tuple<mlir::OperandRange, llvm::StringLiteral> opInfo);
  /// Returns the expression allocated for the input value in the logical
  /// backend if one has been allocated - otherwise allocates and returns a
  /// new expression
  z3::expr fetchOrAllocateExpr(mlir::Value value);
  /// Allocates a constant value in the logical backend and returns its
  /// representing expression.
  void allocateConstant(mlir::Value opResult, const mlir::APInt &opValue);
  /// Constrains the result of a MLIR operation to be equal a given logical
  /// express, simulating an assignment.
  void constrainResult(mlir::Value &result, WireVariant opInfo);

  /// Convert from bitvector to bool sort.
  z3::expr bvToBool(const z3::expr &condition);
  /// Convert from a boolean sort to the corresponding 1-width bitvector.
  z3::expr boolToBv(const z3::expr &condition);

  /// Apply variadic operation and update the state given a variadic comb
  /// transform
  void applyCombVariadicOperation(
      mlir::Value,
      std::pair<mlir::OperandRange,
                std::function<z3::expr(const z3::expr &, const z3::expr &)>>);

  /// Push solver constraints assigning registers and inputs to their current
  /// state
  void loadStateConstraints();
  /// Execute a clock posedge (i.e. update registers and combinatorial logic)
  void runClockPosedge();
  /// Execute a clock negedge (i.e. update combinatorial logic)
  void runClockNegedge();
  /// Assign a new set of symbolic values to all inputs
  void updateInputs(int count, bool posedge);
  /// Check that the properties hold for the current state
  bool checkState();
  /// Update combinatorial logic states (to propagate new inputs/reg outputs)
  void applyCombUpdates();

  z3::expr generateConstraint(WireVariant opInfo);

  /// The name of the circuit; it corresponds to its scope within the parsed
  /// IR.
  std::string name;
  /// A counter for how many assignments have occurred; it's used to uniquely
  /// name new values as they have to be represented within the logical
  /// engine's context.
  unsigned assignments;
  /// The solver environment the circuit belongs to.
  Solver &solver;
  /// The list for the circuit's inputs.
  llvm::SmallVector<z3::expr> inputs;
  /// The list for the circuit's outputs.
  llvm::SmallVector<z3::expr> outputs;

  // Duplicates of these lists are, for now, created holding the corresponding
  // MLIR value. It may eventually be nicer to have a dedicated ID that can be
  // mapped to different Z3 constructs
  /// The list for the circuit's inputs.
  llvm::SmallVector<mlir::Value> inputsByVal;
  /// The list for the circuit's outputs.
  llvm::SmallVector<mlir::Value> outputsByVal;

  /// The list for the circuit's registers.
  llvm::SmallVector<std::variant<CompRegStruct, FirRegStruct>> regs;

  /// The list for the circuit's wires.
  llvm::SmallVector<std::pair<mlir::Value, WireVariant>> wires;
  /// The list for the circuit's clocks.
  // Note: currently circt-mc supports only single clocks, but this is a
  // vector to avoid later reworking.
  llvm::SmallVector<mlir::Value> clks;
  /// A map from IR values to their corresponding logical representation.
  llvm::DenseMap<mlir::Value, z3::expr> exprTable;
  /// A map from IR values to their corresponding state.
  llvm::DenseMap<mlir::Value, z3::expr> stateTable;
  /// A type to represent the different representations of combinational
  /// transforms

  using TransformVariant =
      std::variant<std::function<z3::expr(const z3::expr &)>,
                   std::function<z3::expr(const z3::expr &, const z3::expr &)>,
                   std::function<z3::expr(const z3::expr &, const z3::expr &,
                                          const z3::expr &)>,
                   /*ICmpOp:*/
                   std::function<z3::expr(circt::comb::ICmpPredicate,
                                          const z3::expr &, const z3::expr &)>,
                   /*ParityOp & ReplicateOp:*/
                   std::function<z3::expr(const z3::expr &, unsigned int)>,
                   /*ExtractOp:*/
                   std::function<z3::expr(const z3::expr &, uint32_t, int)>>;
  /// A map from wire values to their corresponding transformations.
  llvm::DenseMap<llvm::StringRef, TransformVariant> combTransformTable;
  /// A map from IR values to their corresponding name.
  llvm::DenseMap<mlir::Value, std::string> nameTable;
};

} // namespace circt

#endif // TOOLS_CIRCT_LEC_CIRCUIT_H
