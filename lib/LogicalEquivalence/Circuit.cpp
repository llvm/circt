//===-- Circuit.cpp - intermediate representation for circuits --*- C++ -*-===//
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

#include "circt/LogicalEquivalence/Circuit.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/LogicalEquivalence/LogicExporter.h"
#include "circt/LogicalEquivalence/Solver.h"
#include "circt/LogicalEquivalence/Utility.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#define DEBUG_TYPE "lec-circuit"

using namespace mlir;
using namespace circt;

/// Populate the table of combinational transforms
void Solver::Circuit::populateCombTransformTable() {
  this->combTransformTable.insert(std::pair(
      comb::AddOp::getOperationName(),
      (std::function<z3::expr(const z3::expr &, const z3::expr &)>)[](
          auto op1, auto op2) { return op1 + op2; }));
  this->combTransformTable.insert(std::pair(
      comb::AndOp::getOperationName(),
      (std::function<z3::expr(const z3::expr &, const z3::expr &)>)[](
          auto op1, auto op2) { return z3::operator&(op1, op2); }));
  this->combTransformTable.insert(std::pair(
      comb::ConcatOp::getOperationName(),
      (std::function<z3::expr(const z3::expr &, const z3::expr &)>)[](
          auto op1, auto op2) { return z3::concat(op1, op2); }));
  this->combTransformTable.insert(std::pair(
      comb::DivSOp::getOperationName(),
      (std::function<z3::expr(const z3::expr &, const z3::expr &)>)[](
          auto op1, auto op2) { return z3::operator/(op1, op2); }));
  this->combTransformTable.insert(std::pair(
      comb::DivUOp::getOperationName(),
      (std::function<z3::expr(const z3::expr &, const z3::expr &)>)[](
          auto op1, auto op2) { return z3::udiv(op1, op2); }));
  this->combTransformTable.insert(std::pair(
      comb::ModSOp::getOperationName(),
      (std::function<z3::expr(const z3::expr &, const z3::expr &)>)[](
          auto op1, auto op2) { return z3::smod(op1, op2); }));
  this->combTransformTable.insert(std::pair(
      comb::ModUOp::getOperationName(),
      (std::function<z3::expr(const z3::expr &, const z3::expr &)>)[](
          auto op1, auto op2) { return z3::urem(op1, op2); }));
  this->combTransformTable.insert(std::pair(
      comb::MulOp::getOperationName(),
      (std::function<z3::expr(const z3::expr &, const z3::expr &)>)[](
          auto op1, auto op2) { return op1 * op2; }));
  this->combTransformTable.insert(std::pair(
      comb::OrOp::getOperationName(),
      (std::function<z3::expr(const z3::expr &, const z3::expr &)>)[](
          auto op1, auto op2) { return op1 | op2; }));
  this->combTransformTable.insert(std::pair(
      comb::ShlOp::getOperationName(),
      (std::function<z3::expr(const z3::expr &, const z3::expr &)>)[](
          auto op1, auto op2) { return z3::shl(op1, op2); }));
  this->combTransformTable.insert(std::pair(
      comb::ShrSOp::getOperationName(),
      (std::function<z3::expr(const z3::expr &, const z3::expr &)>)[](
          auto op1, auto op2) { return z3::ashr(op1, op2); }));
  this->combTransformTable.insert(std::pair(
      comb::SubOp::getOperationName(),
      (std::function<z3::expr(const z3::expr &, const z3::expr &)>)[](
          auto op1, auto op2) { return op1 - op2; }));
  this->combTransformTable.insert(std::pair(
      comb::XorOp::getOperationName(),
      (std::function<z3::expr(const z3::expr &, const z3::expr &)>)[](
          auto op1, auto op2) { return op1 ^ op2; }));
  this->combTransformTable.insert(std::pair(
      comb::ExtractOp::getOperationName(),
      (std::function<z3::expr(const z3::expr &, uint32_t, int)>)[](
          const z3::expr &op1, uint32_t lowBit, int width) {
        return op1.extract(lowBit + width - 1, lowBit);
      }));
  this->combTransformTable.insert(std::pair(
      comb::ICmpOp::getOperationName(),
      (std::function<z3::expr(circt::comb::ICmpPredicate, const z3::expr &,
                              const z3::expr &)>)[this](
          circt::comb::ICmpPredicate predicate, auto lhsExpr, auto rhsExpr) {
        z3::expr result(solver.context);
        switch (predicate) {
        case circt::comb::ICmpPredicate::eq:
          result = lhsExpr == rhsExpr;
          break;
        case circt::comb::ICmpPredicate::ne:
          result = lhsExpr != rhsExpr;
          break;
        case circt::comb::ICmpPredicate::slt:
          result = (z3::slt(lhsExpr, rhsExpr));
          break;
        case circt::comb::ICmpPredicate::sle:
          result = (z3::sle(lhsExpr, rhsExpr));
          break;
        case circt::comb::ICmpPredicate::sgt:
          result = (z3::sgt(lhsExpr, rhsExpr));
          break;
        case circt::comb::ICmpPredicate::sge:
          result = (z3::sge(lhsExpr, rhsExpr));
          break;
        case circt::comb::ICmpPredicate::ult:
          result = (z3::ult(lhsExpr, rhsExpr));
          break;
        case circt::comb::ICmpPredicate::ule:
          result = (z3::ule(lhsExpr, rhsExpr));
          break;
        case circt::comb::ICmpPredicate::ugt:
          result = (z3::ugt(lhsExpr, rhsExpr));
          break;
        case circt::comb::ICmpPredicate::uge:
          result = (z3::uge(lhsExpr, rhsExpr));
          break;
        // Multi-valued logic comparisons are not supported.
        case circt::comb::ICmpPredicate::ceq:
        case circt::comb::ICmpPredicate::weq:
        case circt::comb::ICmpPredicate::cne:
        case circt::comb::ICmpPredicate::wne:
          assert(false && "Multi-valued logic comparisons are not supported.");
        };
        return boolToBv(result);
      }));
  this->combTransformTable.insert(std::pair(
      comb::MuxOp::getOperationName(),
      (std::function<z3::expr(const z3::expr &, const z3::expr &,
                              const z3::expr &)>)[this](
          auto condExpr, auto tvalue, auto fvalue) {
        return z3::ite(bvToBool(condExpr), tvalue, fvalue);
      }));
  this->combTransformTable.insert(std::pair(
      comb::ParityOp::getOperationName(),
      (std::function<z3::expr(const z3::expr &, unsigned int)>)[](
          auto op1, unsigned int width) {
        // input has 1 or more bits
        z3::expr parity = op1.extract(0, 0);
        // calculate parity with every other bit
        for (unsigned int i = 1; i < width; i++) {
          parity = parity ^ op1.extract(i, i);
        }
        return parity;
      }));
  this->combTransformTable.insert(std::pair(
      comb::ReplicateOp::getOperationName(),
      (std::function<z3::expr(const z3::expr &, unsigned int)>)[](
          auto op1, unsigned int times) {
        z3::expr replicate = op1;
        for (unsigned int i = 1; i < times; i++) {
          replicate = z3::concat(replicate, op1);
        }
        return replicate;
      }));
  this->combTransformTable.insert(std::pair(
      seq::FromClockOp::getOperationName(), [](auto op1) { return op1; }));
};

/// Add an input to the circuit; internally a new value gets allocated.
void Solver::Circuit::addInput(Value value) {
  LLVM_DEBUG(lec::dbgs() << name << " addInput\n");
  lec::Scope indent;
  z3::expr input = fetchOrAllocateExpr(value);
  inputs.insert(inputs.end(), input);
  inputsByVal.insert(inputsByVal.end(), value);
}

/// Add an output to the circuit.
void Solver::Circuit::addOutput(Value value) {
  LLVM_DEBUG(lec::dbgs() << name << " addOutput\n");
  // Referenced value already assigned, fetching from expression table.
  z3::expr output = fetchOrAllocateExpr(value);
  outputs.insert(outputs.end(), output);
  outputsByVal.insert(outputsByVal.end(), value);
}

/// Add a clock to the list of clocks.
void Solver::Circuit::addClk(mlir::Value value) {
  if (clks.size() == 1) {
    assert(clks[0] == value && "More than one clock detected - currently "
                               "circt-mc only supports one clock in designs.");
  } else {
    assert(clks.empty() && "Too many clocks added to circuit model.");
    // Check that value is in inputs (i.e. is an external signal and won't be
    // affected by design components)
    auto *inputSearch =
        std::find(inputsByVal.begin(), inputsByVal.end(), value);
    assert(inputSearch != inputsByVal.end() &&
           "Clock is not an input signal - circt-mc currently only supports "
           "external clocks.");
    clks.push_back(value);
  }
}

/// Recover the inputs.
llvm::ArrayRef<z3::expr> Solver::Circuit::getInputs() { return inputs; }

/// Recover the outputs.
llvm::ArrayRef<z3::expr> Solver::Circuit::getOutputs() { return outputs; }

//===----------------------------------------------------------------------===//
// `hw` dialect operations
//===----------------------------------------------------------------------===//

void Solver::Circuit::addConstant(Value opResult, const APInt &opValue) {
  LLVM_DEBUG(lec::dbgs() << name << " addConstant\n");
  lec::Scope indent;
  allocateConstant(opResult, opValue);
}

void Solver::Circuit::addInstance(llvm::StringRef instanceName,
                                  circt::hw::HWModuleOp op,
                                  OperandRange arguments, ResultRange results) {
  LLVM_DEBUG(lec::dbgs() << name << " addInstance\n");
  lec::Scope indent;
  LLVM_DEBUG(lec::dbgs() << "instance name: " << instanceName << "\n");
  LLVM_DEBUG(lec::dbgs() << "module name: " << op->getName() << "\n");
  // There is no preventing multiple instances holding the same name.
  // As an hack, a suffix is used to differentiate them.
  std::string suffix = "_" + std::to_string(assignments);
  Circuit instance(name + "@" + instanceName + suffix, solver);
  // Export logic to the instance's circuit by visiting the IR of the
  // instanced module.
  auto res = LogicExporter(op.getModuleName(), &instance).run(op);
  (void)res; // Suppress Warning
  assert(res.succeeded() && "Instance visit failed");

  // Constrain the inputs and outputs of the instanced circuit to,
  // respectively, the arguments and results of the instance operation.
  {
    LLVM_DEBUG(lec::dbgs() << "instance inputs:\n");
    lec::Scope indent;
    auto *input = instance.inputs.begin();
    for (Value argument : arguments) {
      LLVM_DEBUG(lec::dbgs() << "input\n");
      z3::expr argExpr = fetchOrAllocateExpr(argument);
      solver.solver.add(argExpr == *input++);
    }
  }
  {
    LLVM_DEBUG(lec::dbgs() << "instance results:\n");
    lec::Scope indent;
    auto *output = instance.outputs.begin();
    for (circt::OpResult result : results) {
      z3::expr resultExpr = fetchOrAllocateExpr(result);
      solver.solver.add(resultExpr == *output++);
    }
  }
}

//===----------------------------------------------------------------------===//
// `comb` dialect operations
//===----------------------------------------------------------------------===//

void Solver::Circuit::performAdd(Value result, OperandRange operands) {
  LLVM_DEBUG(lec::dbgs() << name << " perform Add\n");
  lec::Scope indent;
  WireVariant opInfo = std::tuple(operands, comb::AddOp::getOperationName());
  constrainResult(result, opInfo);
}

void Solver::Circuit::performAnd(Value result, OperandRange operands) {
  LLVM_DEBUG(lec::dbgs() << name << " perform And\n");
  lec::Scope indent;
  WireVariant opInfo = std::tuple(operands, comb::AndOp::getOperationName());
  constrainResult(result, opInfo);
}

void Solver::Circuit::performConcat(Value result, OperandRange operands) {
  LLVM_DEBUG(lec::dbgs() << name << " perform Concat\n");
  lec::Scope indent;
  WireVariant opInfo = std::tuple(operands, comb::ConcatOp::getOperationName());
  constrainResult(result, opInfo);
}

void Solver::Circuit::performDivS(Value result, Value lhs, Value rhs) {
  LLVM_DEBUG(lec::dbgs() << name << " perform DivS\n");
  lec::Scope indent;
  LLVM_DEBUG(lec::dbgs() << "lhs:\n");
  z3::expr lhsExpr = fetchOrAllocateExpr(lhs);
  LLVM_DEBUG(lec::dbgs() << "rhs:\n");
  z3::expr rhsExpr = fetchOrAllocateExpr(rhs);
  WireVariant opInfo = std::tuple(lhs, rhs, comb::DivSOp::getOperationName());
  constrainResult(result, opInfo);
}

void Solver::Circuit::performDivU(Value result, Value lhs, Value rhs) {
  LLVM_DEBUG(lec::dbgs() << name << " perform DivU\n");
  lec::Scope indent;
  LLVM_DEBUG(lec::dbgs() << "lhs:\n");
  z3::expr lhsExpr = fetchOrAllocateExpr(lhs);
  LLVM_DEBUG(lec::dbgs() << "rhs:\n");
  z3::expr rhsExpr = fetchOrAllocateExpr(rhs);
  WireVariant opInfo = std::tuple(lhs, rhs, comb::DivUOp::getOperationName());
  constrainResult(result, opInfo);
}

void Solver::Circuit::performExtract(Value result, Value input,
                                     uint32_t lowBit) {
  LLVM_DEBUG(lec::dbgs() << name << " performExtract\n");
  lec::Scope indent;
  LLVM_DEBUG(lec::dbgs() << "input:\n");
  z3::expr inputExpr = fetchOrAllocateExpr(input);
  unsigned width = result.getType().getIntOrFloatBitWidth();
  LLVM_DEBUG(lec::dbgs() << "width: " << width << "\n");
  WireVariant opInfo =
      std::tuple(input, lowBit, width, comb::ExtractOp::getOperationName());
  constrainResult(result, opInfo);
}

void Solver::Circuit::performICmp(Value result,
                                  circt::comb::ICmpPredicate predicate,
                                  Value lhs, Value rhs) {
  LLVM_DEBUG(lec::dbgs() << name << " performICmp\n");
  assert(predicate != circt::comb::ICmpPredicate::ceq &&
         predicate != circt::comb::ICmpPredicate::weq &&
         predicate != circt::comb::ICmpPredicate::cne &&
         predicate != circt::comb::ICmpPredicate::wne &&
         "Multi-valued logic comparisons are not supported.");
  lec::Scope indent;
  LLVM_DEBUG(lec::dbgs() << "lhs:\n");
  z3::expr lhsExpr = fetchOrAllocateExpr(lhs);
  LLVM_DEBUG(lec::dbgs() << "rhs:\n");
  z3::expr rhsExpr = fetchOrAllocateExpr(rhs);
  WireVariant opInfo =
      std::tuple(predicate, lhs, rhs, comb::ICmpOp::getOperationName());
  constrainResult(result, opInfo);
}

void Solver::Circuit::performModS(Value result, Value lhs, Value rhs) {
  LLVM_DEBUG(lec::dbgs() << name << " perform ModS\n");
  lec::Scope indent;
  LLVM_DEBUG(lec::dbgs() << "lhs:\n");
  z3::expr lhsExpr = fetchOrAllocateExpr(lhs);
  LLVM_DEBUG(lec::dbgs() << "rhs:\n");
  z3::expr rhsExpr = fetchOrAllocateExpr(rhs);
  WireVariant opInfo = std::tuple(lhs, rhs, comb::ModSOp::getOperationName());
  constrainResult(result, opInfo);
}

void Solver::Circuit::performModU(Value result, Value lhs, Value rhs) {
  LLVM_DEBUG(lec::dbgs() << name << " perform ModU\n");
  lec::Scope indent;
  LLVM_DEBUG(lec::dbgs() << "lhs:\n");
  z3::expr lhsExpr = fetchOrAllocateExpr(lhs);
  LLVM_DEBUG(lec::dbgs() << "rhs:\n");
  z3::expr rhsExpr = fetchOrAllocateExpr(rhs);
  WireVariant opInfo = std::tuple(lhs, rhs, comb::ModUOp::getOperationName());
  constrainResult(result, opInfo);
}

void Solver::Circuit::performMul(Value result, OperandRange operands) {
  LLVM_DEBUG(lec::dbgs() << name << " perform Mul\n");
  lec::Scope indent;
  WireVariant opInfo = std::tuple(operands, comb::MulOp::getOperationName());
  constrainResult(result, opInfo);
}

void Solver::Circuit::performMux(Value result, Value cond, Value trueValue,
                                 Value falseValue) {
  LLVM_DEBUG(lec::dbgs() << name << " performMux\n");
  lec::Scope indent;
  LLVM_DEBUG(lec::dbgs() << "cond:\n");
  z3::expr condExpr = fetchOrAllocateExpr(cond);
  LLVM_DEBUG(lec::dbgs() << "trueValue:\n");
  z3::expr tvalue = fetchOrAllocateExpr(trueValue);
  LLVM_DEBUG(lec::dbgs() << "falseValue:\n");
  z3::expr fvalue = fetchOrAllocateExpr(falseValue);
  // Conversion due to z3::ite requiring a bool rather than a bitvector.
  WireVariant opInfo =
      std::tuple(cond, trueValue, falseValue, comb::MuxOp::getOperationName());
  constrainResult(result, opInfo);
}

void Solver::Circuit::performOr(Value result, OperandRange operands) {
  LLVM_DEBUG(lec::dbgs() << name << " perform Or\n");
  lec::Scope indent;
  WireVariant opInfo = std::tuple(operands, comb::OrOp::getOperationName());
  constrainResult(result, opInfo);
}

void Solver::Circuit::performParity(Value result, Value input) {
  LLVM_DEBUG(lec::dbgs() << name << " performParity\n");
  lec::Scope indent;
  LLVM_DEBUG(lec::dbgs() << "input:\n");
  z3::expr inputExpr = fetchOrAllocateExpr(input);

  unsigned width = inputExpr.get_sort().bv_size();

  WireVariant opInfo =
      std::tuple(input, width, comb::ParityOp::getOperationName());
  constrainResult(result, opInfo);
}

void Solver::Circuit::performReplicate(Value result, Value input) {
  LLVM_DEBUG(lec::dbgs() << name << " performReplicate\n");
  lec::Scope indent;
  LLVM_DEBUG(lec::dbgs() << "input:\n");
  z3::expr inputExpr = fetchOrAllocateExpr(input);

  unsigned int final = result.getType().getIntOrFloatBitWidth();
  unsigned int initial = input.getType().getIntOrFloatBitWidth();
  unsigned int times = final / initial;
  LLVM_DEBUG(lec::dbgs() << "replies: " << times << "\n");

  z3::expr replicate = inputExpr;
  for (unsigned int i = 1; i < times; i++) {
    replicate = z3::concat(replicate, inputExpr);
  }

  WireVariant opInfo =
      std::tuple(input, times, comb::ReplicateOp::getOperationName());
  constrainResult(result, opInfo);
}

void Solver::Circuit::performShl(Value result, Value lhs, Value rhs) {
  LLVM_DEBUG(lec::dbgs() << name << " perform Shl\n");
  lec::Scope indent;
  LLVM_DEBUG(lec::dbgs() << "lhs:\n");
  z3::expr lhsExpr = fetchOrAllocateExpr(lhs);
  LLVM_DEBUG(lec::dbgs() << "rhs:\n");
  z3::expr rhsExpr = fetchOrAllocateExpr(rhs);
  z3::expr op = z3::shl(lhsExpr, rhsExpr);
  WireVariant opInfo = std::tuple(lhs, rhs, comb::ShlOp::getOperationName());
  constrainResult(result, opInfo);
}

// Arithmetic shift right.
void Solver::Circuit::performShrS(Value result, Value lhs, Value rhs) {
  LLVM_DEBUG(lec::dbgs() << name << " perform ShrS\n");
  lec::Scope indent;
  LLVM_DEBUG(lec::dbgs() << "lhs:\n");
  z3::expr lhsExpr = fetchOrAllocateExpr(lhs);
  LLVM_DEBUG(lec::dbgs() << "rhs:\n");
  z3::expr rhsExpr = fetchOrAllocateExpr(rhs);
  WireVariant opInfo = std::tuple(lhs, rhs, comb::ShrSOp::getOperationName());
  constrainResult(result, opInfo);
}

// Logical shift right.
void Solver::Circuit::performShrU(Value result, Value lhs, Value rhs) {
  LLVM_DEBUG(lec::dbgs() << name << " perform ShrU\n");
  lec::Scope indent;
  LLVM_DEBUG(lec::dbgs() << "lhs:\n");
  z3::expr lhsExpr = fetchOrAllocateExpr(lhs);
  LLVM_DEBUG(lec::dbgs() << "rhs:\n");
  z3::expr rhsExpr = fetchOrAllocateExpr(rhs);
  WireVariant opInfo = std::tuple(lhs, rhs, comb::ShrUOp::getOperationName());
  constrainResult(result, opInfo);
}

void Solver::Circuit::performSub(Value result, OperandRange operands) {
  LLVM_DEBUG(lec::dbgs() << name << " perform Sub\n");
  lec::Scope indent;
  WireVariant opInfo = std::tuple(operands, comb::SubOp::getOperationName());
  constrainResult(result, opInfo);
}

void Solver::Circuit::performXor(Value result, OperandRange operands) {
  LLVM_DEBUG(lec::dbgs() << name << " perform Xor\n");
  lec::Scope indent;
  WireVariant opInfo = std::tuple(operands, comb::XorOp::getOperationName());
  constrainResult(result, opInfo);
}

/// Helper function for performing a variadic operation: it executes a lambda
/// over a range of operands.
z3::expr Solver::Circuit::variadicOperation(
    std::tuple<mlir::OperandRange, llvm::StringLiteral> opInfo) {
  mlir::OperandRange operands = std::get<0>(opInfo);
  llvm::StringLiteral operationName = std::get<1>(opInfo);
  auto functionPair = combTransformTable.find(operationName);
  assert(functionPair != combTransformTable.end() &&
         "No function to represent given operation");
  auto operation =
      std::get<std::function<z3::expr(const z3::expr &, const z3::expr &)>>(
          functionPair->second);
  // Allocate operands if unallocated
  LLVM_DEBUG(lec::dbgs() << "variadic operation\n");
  lec::Scope indent;
  // Vacuous base case.
  auto it = operands.begin();
  Value operand = *it;
  z3::expr varOp = fetchOrAllocateExpr(operand);
  {
    LLVM_DEBUG(lec::dbgs() << "first operand:\n");
    lec::Scope indent;
    LLVM_DEBUG(lec::printValue(operand));
  }
  ++it;
  // Inductive step.
  while (it != operands.end()) {
    operand = *it;
    varOp = operation(varOp, fetchOrAllocateExpr(operand));
    {
      LLVM_DEBUG(lec::dbgs() << "next operand:\n");
      lec::Scope indent;
      LLVM_DEBUG(lec::printValue(operand));
    }
    ++it;
  }

  return varOp;
}

/// Allocates an IR value in the logical backend and returns its representing
/// expression.
z3::expr Solver::Circuit::fetchOrAllocateExpr(Value value) {
  z3::expr expr(solver.context);
  auto exprPair = exprTable.find(value);
  if (exprPair != exprTable.end()) {
    LLVM_DEBUG(lec::dbgs() << "value already allocated:\n");
    lec::Scope indent;
    expr = exprPair->second;
    LLVM_DEBUG(lec::printExpr(expr));
    LLVM_DEBUG(lec::printValue(value));
  } else {
    std::string valueName = name + "%" + std::to_string(assignments++);
    LLVM_DEBUG(lec::dbgs() << "allocating value:\n");
    lec::Scope indent;
    auto nameInsertion = nameTable.insert(std::pair(value, valueName));
    assert(nameInsertion.second && "Name not inserted in state table");
    Type type = value.getType();
    auto isClockType = hw::type_isa<seq::ClockType>(type);
    assert((type.isSignlessInteger() || isClockType) && "Unsupported type");
    unsigned int width = isClockType ? 1 : type.getIntOrFloatBitWidth();
    // Technically allowed for the `hw` dialect but
    // disallowed for `comb` operations; should check separately.
    assert(width > 0 && "0-width integers are not supported"); // NOLINT
    expr = solver.context.bv_const(valueName.c_str(), width);
    LLVM_DEBUG(lec::printExpr(expr));
    LLVM_DEBUG(lec::printValue(value));
    auto exprInsertion = exprTable.insert(std::pair(value, expr));
    (void)exprInsertion; // Suppress Warning
    assert(exprInsertion.second && "Value not inserted in expression table");
    // Populate state table
    std::string stateName = valueName + std::string("_init");
    z3::expr stateExpr = solver.context.bv_const(stateName.c_str(), width);
    auto stateInsertion = stateTable.insert(std::pair(value, stateExpr));
    (void)stateInsertion; // Suppress Warning
    assert(stateInsertion.second && "Value not inserted in state table");
    Builder builder(solver.mlirCtx);
    StringAttr symbol = builder.getStringAttr(valueName);
    auto symInsertion = solver.symbolTable.insert(std::pair(symbol, value));
    (void)symInsertion; // Suppress Warning
    assert(symInsertion.second && "Value not inserted in symbol table");
    mlir::StringAttr stateSymbol = builder.getStringAttr(stateName);
    auto symStateInsertion =
        solver.symbolTable.insert(std::pair(stateSymbol, value));
    (void)symStateInsertion; // Suppress Warning
    assert(symStateInsertion.second && "State not inserted in symbol table");
  }
  return expr;
}

/// Allocates a constant value in the logical backend and returns its
/// representing expression.
void Solver::Circuit::allocateConstant(Value result, const APInt &value) {
  // `The constant operation produces a constant value
  //  of standard integer type without a sign`
  const z3::expr constant =
      solver.context.bv_val(value.getZExtValue(), value.getBitWidth());
  //  Check whether the constant has been pre-allocated
  auto allocatedPair = exprTable.find(result);
  if (allocatedPair == exprTable.end()) {
    // If not, then allocate
    auto insertion = exprTable.insert(std::pair(result, constant));
    assert(insertion.second && "Constant not inserted in expression table");
    (void)insertion; // Suppress Warning
    auto stateInsertion = stateTable.insert(std::pair(result, constant));
    (void)stateInsertion; // Suppress Warning
    assert(stateInsertion.second && "Value not inserted in state table");
    LLVM_DEBUG(lec::printExpr(constant));
    LLVM_DEBUG(lec::printValue(result));
  } else {
    // If it has, then we force equivalence to the constant (we cannot just
    // overwrite in the table as when it was allocated, a constraint was
    // already formed using the symbolic form).
    solver.solver.add(allocatedPair->second == constant);
    LLVM_DEBUG(lec::dbgs() << "constraining symbolic value to constant:\n");
    lec::Scope indent;
    LLVM_DEBUG(lec::printExpr(constant));
    LLVM_DEBUG(lec::printValue(result));
  }
}

/// Constrains the result of a MLIR operation to be equal a given logical
/// express, simulating an assignment.
void Solver::Circuit::constrainResult(Value &result, WireVariant opInfo) {
  LLVM_DEBUG(lec::dbgs() << "constraining result:\n");
  lec::Scope indent;
  z3::expr expr = generateConstraint(opInfo);
  {
    LLVM_DEBUG(lec::dbgs() << "result expression:\n");
    lec::Scope indent;
    LLVM_DEBUG(lec::printExpr(expr));
  }
  z3::expr resExpr = fetchOrAllocateExpr(result);
  z3::expr constraint = resExpr == expr;
  {
    LLVM_DEBUG(lec::dbgs() << "adding constraint:\n");
    lec::Scope indent;
    LLVM_DEBUG(lec::dbgs() << constraint.to_string() << "\n");
  }
  solver.solver.add(constraint);
  wires.push_back(std::pair(result, opInfo));
}

z3::expr Solver::Circuit::generateConstraint(WireVariant opInfo) {
  auto table = exprTable;
  z3::expr result(solver.context);
  if (auto *info =
          std::get_if<std::tuple<mlir::Value, llvm::StringLiteral>>(&opInfo)) {
    mlir::Value input = std::get<0>(*info);
    llvm::StringLiteral operationName = std::get<1>(*info);
    auto updateFuncPair = combTransformTable.find(operationName);
    assert(updateFuncPair != combTransformTable.end() &&
           "Combinational value to update has no update function");
    result = std::get<std::function<z3::expr(const z3::expr &)>>(
        updateFuncPair->second)(fetchOrAllocateExpr(input));
  } else if (auto *info = std::get_if<
                 std::tuple<mlir::Value, mlir::Value, llvm::StringLiteral>>(
                 &opInfo)) {
    mlir::Value input0 = std::get<0>(*info);
    mlir::Value input1 = std::get<1>(*info);
    llvm::StringLiteral operationName = std::get<2>(*info);
    auto updateFuncPair = combTransformTable.find(operationName);
    assert(updateFuncPair != combTransformTable.end() &&
           "Combinational value to update has no update function");
    result =
        std::get<std::function<z3::expr(const z3::expr &, const z3::expr &)>>(
            updateFuncPair->second)(fetchOrAllocateExpr(input0),
                                    fetchOrAllocateExpr(input1));
  } else if (auto *info =
                 std::get_if<std::tuple<mlir::Value, mlir::Value, mlir::Value,
                                        llvm::StringLiteral>>(&opInfo)) {
    mlir::Value input0 = std::get<0>(*info);
    mlir::Value input1 = std::get<1>(*info);
    mlir::Value input2 = std::get<2>(*info);
    llvm::StringLiteral operationName = std::get<3>(*info);
    auto updateFuncPair = combTransformTable.find(operationName);
    assert(updateFuncPair != combTransformTable.end() &&
           "Combinational value to update has no update function");
    result = std::get<std::function<z3::expr(const z3::expr &, const z3::expr &,
                                             const z3::expr &)>>(
        updateFuncPair->second)(fetchOrAllocateExpr(input0),
                                fetchOrAllocateExpr(input1),
                                fetchOrAllocateExpr(input2));
  } else if (auto *info = std::get_if<
                 std::tuple<mlir::OperandRange, llvm::StringLiteral>>(
                 &opInfo)) {
    result = variadicOperation(*info);
  } else if (auto *info =
                 std::get_if<std::tuple<circt::comb::ICmpPredicate, mlir::Value,
                                        mlir::Value, llvm::StringLiteral>>(
                     &opInfo)) {
    circt::comb::ICmpPredicate predicate = std::get<0>(*info);
    mlir::Value input0 = std::get<1>(*info);
    mlir::Value input1 = std::get<2>(*info);
    llvm::StringLiteral operationName = std::get<3>(*info);
    auto updateFuncPair = combTransformTable.find(operationName);
    assert(updateFuncPair != combTransformTable.end() &&
           "Combinational value to update has no update function");
    result = std::get<std::function<z3::expr(
        circt::comb::ICmpPredicate, const z3::expr &, const z3::expr &)>>(
        updateFuncPair->second)(predicate, fetchOrAllocateExpr(input0),
                                fetchOrAllocateExpr(input1));
  } else if (auto *info = std::get_if<
                 std::tuple<mlir::Value, unsigned int, llvm::StringLiteral>>(
                 &opInfo)) {
    mlir::Value input = std::get<0>(*info);
    int num = std::get<1>(*info);
    llvm::StringLiteral operationName = std::get<2>(*info);
    auto updateFuncPair = combTransformTable.find(operationName);
    assert(updateFuncPair != combTransformTable.end() &&
           "Combinational value to update has no update function");
    result = std::get<std::function<z3::expr(const z3::expr &, unsigned int)>>(
        updateFuncPair->second)(fetchOrAllocateExpr(input), num);
  } else if (auto *info = std::get_if<
                 std::tuple<mlir::Value, uint32_t, int, llvm::StringLiteral>>(
                 &opInfo)) {
    mlir::Value input = std::get<0>(*info);
    uint32_t lowBit = std::get<1>(*info);
    int width = std::get<2>(*info);
    llvm::StringLiteral operationName = std::get<3>(*info);
    auto updateFuncPair = combTransformTable.find(operationName);
    assert(updateFuncPair != combTransformTable.end() &&
           "Combinational value to update has no update function");
    result = std::get<std::function<z3::expr(const z3::expr &, uint32_t, int)>>(
        updateFuncPair->second)(fetchOrAllocateExpr(input), lowBit, width);
  }
  return result;
}

/// Convert from bitvector to bool sort.
z3::expr Solver::Circuit::bvToBool(const z3::expr &condition) {
  // bitvector is true if it's different from 0
  return condition != 0;
}

/// Convert from a boolean sort to the corresponding 1-width bitvector.
z3::expr Solver::Circuit::boolToBv(const z3::expr &condition) {
  return z3::ite(condition, solver.context.bv_val(1, 1),
                 solver.context.bv_val(0, 1));
}

/// Push solver constraints assigning registers and inputs to their current
/// state
void Solver::Circuit::loadStateConstraints() {
  for (auto input : inputsByVal) {
    auto symbolPair = exprTable.find(input);
    assert(symbolPair != exprTable.end() &&
           "Z3 expression not found for input value");
    auto statePair = stateTable.find(input);
    assert(statePair != stateTable.end() &&
           "Z3 state not found for input value");
    solver.solver.add(symbolPair->second == statePair->second);
  }
  for (auto reg : regs) {
    mlir::Value regData;
    if (auto *compReg = std::get_if<CompRegStruct>(&reg)) {
      regData = compReg->data;
    } else if (auto *firReg = std::get_if<FirRegStruct>(&reg)) {
      regData = firReg->data;
    }
    auto symbolPair = exprTable.find(regData);
    assert(symbolPair != exprTable.end() &&
           "Z3 expression not found for register output");
    auto statePair = stateTable.find(regData);
    assert(statePair != stateTable.end() &&
           "Z3 state not found for register output");

    solver.solver.add(symbolPair->second == statePair->second);
  }
  // Combinatorial values are handled by the constraints we already have, so
  // we do not need their state
}

/// Execute a clock posedge (i.e. update registers and combinatorial logic)
void Solver::Circuit::runClockPosedge() {
  for (auto clk : clks) {
    // Currently we explicitly handle only one clock, so we can just update
    // every clock in clks (of which there are 0 or 1)
    stateTable.find(clk)->second = solver.context.bv_val(1, 1);
  }
  for (auto reg : regs) {
    // Fetch values from reg structs
    mlir::Value input;
    mlir::Value data;
    mlir::Value reset;
    mlir::Value resetValue;
    if (auto *compReg = std::get_if<CompRegStruct>(&reg)) {
      input = compReg->input;
      data = compReg->data;
      reset = compReg->reset;
      resetValue = compReg->resetValue;
    } else if (auto *firReg = std::get_if<FirRegStruct>(&reg)) {
      input = firReg->next;
      data = firReg->data;
      reset = firReg->reset;
      resetValue = firReg->resetValue;
    }
    // Currently, there is no difference in CompReg and FirReg handling, as
    // async resets aren't supported
    z3::expr inputState = stateTable.find(input)->second;
    // Make sure that a reset value is present
    if (reset) {
      z3::expr resetState = stateTable.find(reset)->second;
      z3::expr resetValueState = stateTable.find(resetValue)->second;
      z3::expr newState =
          z3::ite(bvToBool(resetState), resetValueState, inputState);
      stateTable.find(data)->second = newState;
    } else {
      // Otherwise, simply update output state to be the same as input state
      stateTable.find(data)->second = inputState;
    }
  }
  // Update combinational updates so register outputs can propagate
  applyCombUpdates();
}

/// Execute a clock negedge (i.e. update combinatorial logic)
void Solver::Circuit::runClockNegedge() {
  for (auto clk : clks) {
    // Currently we explicitly handle only one clock, so we can just update
    // every clock in clks (of which there are 0 or 1)
    stateTable.find(clk)->second = solver.context.bv_val(0, 1);
  }
  // Update combinational updates so changes in inputs can propagate
  applyCombUpdates();
}

/// Assign a new set of symbolic values to all inputs
void Solver::Circuit::updateInputs(int count, bool posedge) {
  mlir::Builder builder(solver.mlirCtx);
  for (auto input : inputsByVal) {
    // We update clocks literally, so skip this for clocks
    if (std::find(clks.begin(), clks.end(), input) != clks.end()) {
      continue;
    }
    llvm::DenseMap<mlir::Value, z3::expr>::iterator currentStatePair =
        stateTable.find(input);
    if (currentStatePair != stateTable.end()) {
      int width = input.getType().getIntOrFloatBitWidth();
      std::string valueName = nameTable.find(input)->second;
      std::string edgeString(posedge ? "_pos" : "_neg");
      std::string symbolName =
          (valueName + "_" + std::to_string(count) + edgeString).c_str();
      currentStatePair->second =
          solver.context.bv_const(symbolName.c_str(), width);
      mlir::StringAttr symbol = builder.getStringAttr(symbolName);
      auto symInsertion = solver.symbolTable.insert(std::pair(symbol, input));
      assert(symInsertion.second && "Value not inserted in symbol table");
    }
  }
}

/// Check that the properties hold for the current state
bool Solver::Circuit::checkState() {
  solver.solver.push();
  loadStateConstraints();
  auto result = solver.solver.check();
  solver.solver.pop();
  switch (result) {
  case z3::sat:
    solver.printModel();
    return false;
    break;
  case z3::unsat:
    return true;
    break;
  default:
    // TODO: maybe add handler for other return vals?
    return false;
  }
}

/// Execute a clock cycle and check that the properties hold throughout
bool Solver::Circuit::checkCycle(int count) {
  updateInputs(count, true);
  runClockPosedge();
  if (!checkState()) {
    return false;
  }
  updateInputs(count, false);
  runClockNegedge();
  return checkState();
}

/// Update combinatorial logic states (to propagate new inputs/reg outputs)
void Solver::Circuit::applyCombUpdates() {
  for (auto wire : wires) {
    auto resultState = wire.first;
    auto opInfo = wire.second;
    if (auto *info = std::get_if<std::tuple<mlir::Value, llvm::StringLiteral>>(
            &opInfo)) {
      mlir::Value input = std::get<0>(*info);
      llvm::StringLiteral operationName = std::get<1>(*info);
      auto updateFuncPair = combTransformTable.find(operationName);
      assert(updateFuncPair != combTransformTable.end() &&
             "Combinational value to update has no update function");
      stateTable.find(resultState)->second =
          std::get<std::function<z3::expr(const z3::expr &)>>(
              updateFuncPair->second)(fetchOrAllocateExpr(input));
    } else if (auto *info = std::get_if<
                   std::tuple<mlir::Value, mlir::Value, llvm::StringLiteral>>(
                   &opInfo)) {
      mlir::Value input0 = std::get<0>(*info);
      mlir::Value input1 = std::get<1>(*info);
      llvm::StringLiteral operationName = std::get<2>(*info);
      auto updateFuncPair = combTransformTable.find(operationName);
      assert(updateFuncPair != combTransformTable.end() &&
             "Combinational value to update has no update function");
      stateTable.find(resultState)->second =
          std::get<std::function<z3::expr(const z3::expr &, const z3::expr &)>>(
              updateFuncPair->second)(fetchOrAllocateExpr(input0),
                                      fetchOrAllocateExpr(input1));
    } else if (auto *info =
                   std::get_if<std::tuple<mlir::Value, mlir::Value, mlir::Value,
                                          llvm::StringLiteral>>(&opInfo)) {
      mlir::Value input0 = std::get<0>(*info);
      mlir::Value input1 = std::get<1>(*info);
      mlir::Value input2 = std::get<2>(*info);
      llvm::StringLiteral operationName = std::get<3>(*info);
      auto updateFuncPair = combTransformTable.find(operationName);
      assert(updateFuncPair != combTransformTable.end() &&
             "Combinational value to update has no update function");
      stateTable.find(resultState)->second = std::get<std::function<z3::expr(
          const z3::expr &, const z3::expr &, const z3::expr &)>>(
          updateFuncPair->second)(fetchOrAllocateExpr(input0),
                                  fetchOrAllocateExpr(input1),
                                  fetchOrAllocateExpr(input2));
    } else if (auto *info = std::get_if<
                   std::tuple<mlir::OperandRange, llvm::StringLiteral>>(
                   &opInfo)) {
      stateTable.find(resultState)->second = variadicOperation(*info);
    } else if (auto *info = std::get_if<
                   std::tuple<circt::comb::ICmpPredicate, mlir::Value,
                              mlir::Value, llvm::StringLiteral>>(&opInfo)) {
      circt::comb::ICmpPredicate predicate = std::get<0>(*info);
      mlir::Value input0 = std::get<1>(*info);
      mlir::Value input1 = std::get<2>(*info);
      llvm::StringLiteral operationName = std::get<3>(*info);
      auto updateFuncPair = combTransformTable.find(operationName);
      assert(updateFuncPair != combTransformTable.end() &&
             "Combinational value to update has no update function");
      stateTable.find(resultState)->second = std::get<std::function<z3::expr(
          circt::comb::ICmpPredicate, const z3::expr &, const z3::expr &)>>(
          updateFuncPair->second)(predicate, fetchOrAllocateExpr(input0),
                                  fetchOrAllocateExpr(input1));
    } else if (auto *info = std::get_if<
                   std::tuple<mlir::Value, unsigned int, llvm::StringLiteral>>(
                   &opInfo)) {
      mlir::Value input = std::get<0>(*info);
      int num = std::get<1>(*info);
      llvm::StringLiteral operationName = std::get<2>(*info);
      auto updateFuncPair = combTransformTable.find(operationName);
      assert(updateFuncPair != combTransformTable.end() &&
             "Combinational value to update has no update function");
      stateTable.find(resultState)->second =
          std::get<std::function<z3::expr(const z3::expr &, unsigned int)>>(
              updateFuncPair->second)(fetchOrAllocateExpr(input), num);
    } else if (auto *info = std::get_if<
                   std::tuple<mlir::Value, uint32_t, int, llvm::StringLiteral>>(
                   &opInfo)) {
      mlir::Value input = std::get<0>(*info);
      uint32_t lowBit = std::get<1>(*info);
      int width = std::get<2>(*info);
      llvm::StringLiteral operationName = std::get<3>(*info);
      auto updateFuncPair = combTransformTable.find(operationName);
      assert(updateFuncPair != combTransformTable.end() &&
             "Combinational value to update has no update function");
      stateTable.find(resultState)->second =
          std::get<std::function<z3::expr(const z3::expr &, uint32_t, int)>>(
              updateFuncPair->second)(fetchOrAllocateExpr(input), lowBit,
                                      width);
    }
  }
}

/// Helper function for applying a variadic update operation: it executes a
/// lambda over a range of operands and updates the state.
void Solver::Circuit::applyCombVariadicOperation(
    mlir::Value result,
    const std::pair<mlir::OperandRange,
                    std::function<z3::expr(const z3::expr &, const z3::expr &)>>
        operationPair) {
  LLVM_DEBUG(lec::dbgs() << "comb variadic operation\n");
  lec::Scope indent;
  mlir::OperandRange operands = operationPair.first;
  std::function<z3::expr(const z3::expr &, const z3::expr &)> operation =
      operationPair.second;
  // Vacuous base case.
  auto it = operands.begin();
  mlir::Value operand = *it;
  z3::expr varOp = exprTable.find(operand)->second;
  {
    LLVM_DEBUG(lec::dbgs() << "first operand:\n");
    lec::Scope indent;
    LLVM_DEBUG(lec::printValue(operand));
  }
  ++it;
  // Inductive step.
  while (it != operands.end()) {
    operand = *it;
    varOp = operation(varOp, exprTable.find(operand)->second);
    {
      LLVM_DEBUG(lec::dbgs() << "next operand:\n");
      lec::Scope indent;
      LLVM_DEBUG(lec::printValue(operand));
    }
    ++it;
  };
  stateTable.find(result)->second = varOp;
}

//===----------------------------------------------------------------------===//
// `seq` dialect operations
//===----------------------------------------------------------------------===//
void Solver::Circuit::performCompReg(mlir::Value input, mlir::Value clk,
                                     mlir::Value data, mlir::Value reset,
                                     mlir::Value resetValue) {
  z3::expr regData = fetchOrAllocateExpr(data);
  CompRegStruct reg;
  reg.input = input;
  reg.clk = clk;
  reg.data = data;
  reg.reset = reset;
  reg.resetValue = resetValue;
  regs.push_back(reg);
  addClk(clk);
}

void Solver::Circuit::performFirReg(mlir::Value next, mlir::Value clk,
                                    mlir::Value data, mlir::Value reset,
                                    mlir::Value resetValue) {
  z3::expr regData = fetchOrAllocateExpr(data);
  FirRegStruct reg;
  reg.next = next;
  reg.clk = clk;
  reg.data = data;
  reg.reset = reset;
  reg.resetValue = resetValue;
  regs.push_back(reg);
  addClk(clk);
}

void Solver::Circuit::performFromClock(mlir::Value result, mlir::Value input) {
  z3::expr resultState = fetchOrAllocateExpr(result);
  z3::expr inputState = fetchOrAllocateExpr(input);
  // Constrain the result directly to the input's value
  WireVariant opInfo = std::tuple(input, seq::FromClockOp::getOperationName());
  constrainResult(result, opInfo);
  // combTransformTable.insert(std::pair(
  //     result, std::pair(std::make_tuple(input), [](auto op1) { return op1;
  //     })));
}

//===----------------------------------------------------------------------===//
// `seq` dialect operations
//===----------------------------------------------------------------------===//
void Solver::Circuit::performAssert(mlir::Value property) {
  z3::expr propExpr = exprTable.find(property)->second;
  solver.solver.add(!bvToBool(propExpr));
}

void Solver::Circuit::performAssume(mlir::Value property) {
  z3::expr propExpr = exprTable.find(property)->second;
  solver.solver.add(bvToBool(propExpr));
}

#undef DEBUG_TYPE
