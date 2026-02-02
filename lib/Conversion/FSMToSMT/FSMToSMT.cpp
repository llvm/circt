//===- FSMToSMT.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "circt/Conversion/FSMToSMT.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/FSM/FSMDialect.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/LLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Dialect/SMT/IR/SMTTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

namespace circt {
#define GEN_PASS_DEF_CONVERTFSMTOSMT
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace fsm;

//===----------------------------------------------------------------------===//
// Convert FSM to SMT pass
//===----------------------------------------------------------------------===//

namespace {

struct LoweringConfig {
  // If true, include a `time` parameter in the relation, to be incremented at
  // every transition.
  bool withTime = false;
  // Width of `time` parameter (if present)
  unsigned timeWidth = 5;
};

class MachineOpConverter {
public:
  MachineOpConverter(OpBuilder &builder, MachineOp machineOp,
                     const LoweringConfig &cfg)
      : machineOp(machineOp), b(builder), cfg(cfg) {}
  LogicalResult dispatch();

private:
  struct PendingAssertion {
    int stateId;
    Region *outputRegion;
  };

  struct Transition {
    int from;
    int to;
    bool hasGuard = false, hasAction = false, hasOutput = false;
    Region *guard = nullptr, *action = nullptr, *output = nullptr;
  };

  Transition parseTransition(fsm::TransitionOp t, int from,
                             SmallVector<std::string> &states, Location &loc) {
    std::string nextState = t.getNextState().str();
    Transition tr = {from, insertStates(states, nextState)};
    if (t.hasGuard()) {
      tr.hasGuard = true;
      tr.guard = &t.getGuard();
    }
    if (t.hasAction()) {
      tr.hasAction = true;
      tr.action = &t.getAction();
    }
    return tr;
  }

  static int insertStates(SmallVector<std::string> &states,
                          llvm::StringRef st) {
    for (auto [id, s] : llvm::enumerate(states))
      if (s == st)
        return id;
    states.push_back(st.str()); // materialize once, stored in vector
    return states.size() - 1;
  }

  Region *
  getOutputRegion(const SmallVector<std::pair<Region *, int>> &outputOfStateId,
                  int stateId) {
    for (auto oid : outputOfStateId)
      if (stateId == oid.second)
        return oid.first;
    llvm_unreachable("State could not be found.");
  }

  SmallVector<std::pair<mlir::Value, mlir::Value>>
  mapSmtToFsm(Location loc, OpBuilder b, SmallVector<Value> smtValues,
              int numArgs, int numOut, SmallVector<Value> fsmArgs,
              SmallVector<Value> fsmVars) {
    // map each SMT-quantified variable to a corresponding FSM variable or
    // argument
    SmallVector<std::pair<mlir::Value, mlir::Value>> fsmToCast;
    for (auto [idx, fq] : llvm::enumerate(smtValues)) {
      if (int(idx) < numArgs) { // arguments
        auto convCast = UnrealizedConversionCastOp::create(
            b, loc, fsmArgs[idx].getType(), fq);
        fsmToCast.push_back({fsmArgs[idx], convCast->getResult(0)});
      } else if (numArgs + numOut <= int(idx)) { // variables
        if (cfg.withTime && idx == smtValues.size() - 1)
          break;
        auto convCast = UnrealizedConversionCastOp::create(
            b, loc, fsmVars[idx - numArgs - numOut].getType(), fq);
        fsmToCast.push_back(
            {fsmVars[idx - numArgs - numOut], convCast->getResult(0)});
      }
    }
    return fsmToCast;
  }

  IRMapping
  createIRMapping(const SmallVector<std::pair<Value, Value>> &fsmToCast,
                  const IRMapping &constMapper) {
    IRMapping mapping;
    for (auto couple : fsmToCast) {
      mapping.map(couple.first, couple.second);
    }
    for (auto &pair : constMapper.getValueMap()) {
      mapping.map(pair.first, pair.second);
    }
    return mapping;
  }

  Value bv1toSmtBool(OpBuilder &b, Location loc, Value i1Value) {
    auto castVal = UnrealizedConversionCastOp::create(
        b, loc, b.getType<smt::BitVectorType>(1), i1Value);
    return smt::EqOp::create(b, loc, castVal->getResult(0),
                             smt::BVConstantOp::create(b, loc, 1, 1));
  }

  MachineOp machineOp;
  OpBuilder &b;
  LoweringConfig cfg;
};

LogicalResult MachineOpConverter::dispatch() {
  b.setInsertionPoint(machineOp);
  auto loc = machineOp.getLoc();
  auto machineArgs = machineOp.getArguments();

  mlir::SmallVector<mlir::Value> fsmVars;
  mlir::SmallVector<mlir::Value> fsmArgs;
  mlir::SmallVector<mlir::Type> quantifiableTypes;

  TypeRange typeRange;
  ValueRange valueRange;

  auto solver = smt::SolverOp::create(b, loc, typeRange, valueRange);
  solver.getBodyRegion().emplaceBlock();
  b.setInsertionPointToStart(solver.getBody());

  // Collect arguments and their types
  for (auto a : machineArgs) {
    fsmArgs.push_back(a);
    quantifiableTypes.push_back(
        b.getType<smt::BitVectorType>(a.getType().getIntOrFloatBitWidth()));
  }
  size_t numArgs = fsmArgs.size();

  // Collect output types
  if (!machineOp.getResultTypes().empty()) {
    for (auto t : machineOp.getResultTypes()) {
      quantifiableTypes.push_back(
          b.getType<smt::BitVectorType>(t.getIntOrFloatBitWidth()));
    }
  }

  size_t numOut = quantifiableTypes.size() - numArgs;

  // Collect FSM variables, their types, and initial values
  SmallVector<llvm::APInt> varInitValues;
  for (auto t : machineOp.front().getOps<fsm::VariableOp>()) {
    auto intAttr = dyn_cast<IntegerAttr>(t.getInitValueAttr());
    varInitValues.push_back(intAttr.getValue());
    quantifiableTypes.push_back(
        b.getType<smt::BitVectorType>(t.getType().getIntOrFloatBitWidth()));
    fsmVars.push_back(t.getResult());
  }

  // Map constant operations to their clones in the new solver region
  IRMapping constMapper;
  for (auto constOp : machineOp.front().getOps<hw::ConstantOp>()) {
    b.clone(*constOp, constMapper);
  }

  // Do not allow any operations other than constants outside of FSM regions
  for (auto &op : machineOp.front().getOperations()) {
    if (!isa<fsm::FSMDialect>(op.getDialect()) &&
        !isa<hw::HWDialect>(op.getDialect())) {
      op.emitError(
          "Operations other than constants are not supported outside FSM "
          "output, guard, and action regions.");
      return failure();
    }
  }

  // Add a time variable if flag is enabled
  if (cfg.withTime) {
    quantifiableTypes.push_back(b.getType<smt::BitVectorType>(cfg.timeWidth));
  }

  size_t numVars = varInitValues.size();

  // Store all the transitions state0 -> state1 in the FSM
  SmallVector<MachineOpConverter::Transition> transitions;
  // Store all the functions `F_state(outs, vars, [time]) -> Bool` describing
  // the activation of each state
  SmallVector<Value> stateFunctions;
  // Store the name of each state
  SmallVector<std::string> states;
  // Store the output region of each state
  SmallVector<std::pair<Region *, int>> outputOfStateId;

  // Get FSM initial state and store it in the states vector
  std::string initialState = machineOp.getInitialState().str();
  insertStates(states, initialState);

  // Only outputs and variables belong in the state function's domain, since the
  // arguments are universally quantified.
  SmallVector<Type> stateFunDomain;
  for (auto i = numArgs; i < quantifiableTypes.size(); i++)
    stateFunDomain.push_back(quantifiableTypes[i]);

  // For each state, declare one SMT function: `F_state(outs, vars, [time]) ->
  // Bool`, returning `true`  when `state` is activated.
  for (auto stateOp : machineOp.front().getOps<fsm::StateOp>()) {
    mlir::StringAttr funName =
        b.getStringAttr(("F_" + stateOp.getName().str()));
    auto rangeTy = b.getType<smt::BoolType>();
    auto funTy = b.getType<smt::SMTFuncType>(stateFunDomain, rangeTy);
    auto acFun = smt::DeclareFunOp::create(b, loc, funTy, funName);
    stateFunctions.push_back(acFun);
    auto fromState = insertStates(states, stateOp.getName().str());
    outputOfStateId.push_back({&stateOp.getOutput(), fromState});
  }

  // For each transition `state0 &&& guard01 -> state1`, construct an
  // implication `F_state_0(outs, vars, [time]) &&& guard01(outs, vars) ->
  // F_state_1(outs, vars, [time])`, simulating the activation of a transition
  // and of the state it reaches.
  for (auto stateOp : machineOp.front().getOps<fsm::StateOp>()) {
    std::string stateName = stateOp.getName().str();
    auto fromState = insertStates(states, stateName);
    if (!stateOp.getTransitions().empty()) {
      for (auto tr :
           stateOp.getTransitions().front().getOps<fsm::TransitionOp>()) {
        auto t = parseTransition(tr, fromState, states, loc);
        if (!stateOp.getOutput().empty()) {
          t.hasOutput = true;
          t.output = getOutputRegion(outputOfStateId, t.to);
        } else {
          t.hasOutput = false;
        }
        transitions.push_back(t);
      }
    }
  }

  SmallVector<PendingAssertion> assertions;

  // The initial assertion sets variables to their initial values and sets `time
  // == 0` (if present), computing the output values of the initial state
  // accordingly.
  auto initialAssertion = smt::ForallOp::create(
      b, loc, quantifiableTypes,
      [&](OpBuilder &b, Location loc,
          const SmallVector<Value> &forallQuantified) -> Value {
        // map each SMT-quantified variable to a corresponding FSM variable or
        // argument
        SmallVector<std::pair<mlir::Value, mlir::Value>> fsmToCast =
            mapSmtToFsm(loc, b, forallQuantified, numArgs, numOut, fsmArgs,
                        fsmVars);
        auto *initOutputReg = getOutputRegion(outputOfStateId, 0);
        SmallVector<Value> castOutValues;
        if (initOutputReg->empty()) {
          for (auto i = numArgs; i < numArgs + numOut; i++)
            castOutValues.push_back(forallQuantified[i]);
        } else {
          
          // replace initial values in fsmToCast to create the IR mapping 
          
          for (auto [id, couple] : llvm::enumerate(fsmToCast)) {
            if (numArgs <= id && id < numArgs + numVars) 
              fsmToCast[id] = {couple.first,
                               hw::ConstantOp::create(b, loc, varInitValues[id - numArgs])};
          }

          IRMapping mapping = createIRMapping(fsmToCast, constMapper);

          SmallVector<mlir::Value> combOutputValues;
          
          // Clone all the operations in the output region except `OutputOp` and
          // `AssertOp`, replacing FSM variables and arguments with the results
          // of unrealized conversion casts and replacing constants with their
          // new clones
          for (auto &op : initOutputReg->front()) {
            auto *newOp = b.clone(op, mapping);
            if (!isa<verif::AssertOp>(newOp)) {
              // Retrieve all the operands of the output operation
              if (isa<fsm::OutputOp>(newOp)) {
                for (auto out : newOp->getOperands())
                  combOutputValues.push_back(out);
                newOp->erase();
              }
            } else {
              // Store the assertion operations, with a copy of the region
              assertions.push_back({0, initOutputReg});
              newOp->erase();
            }
          }

          // Cast the (comb) results obtained from the output region to SMT
          // types, to pass them as arguments of the state function

          for (auto [idx, out] : llvm::enumerate(combOutputValues)) {
            auto convCast = UnrealizedConversionCastOp::create(
                b, loc, forallQuantified[numArgs + idx].getType(), out);
            castOutValues.push_back(convCast->getResult(0));
          }
        }

        // Assign variables and output values their initial value
        SmallVector<mlir::Value> initialCondition;
        for (auto [idx, q] : llvm::enumerate(forallQuantified)) {
          if (cfg.withTime) {
            if (numArgs + numOut <= idx &&
                idx < forallQuantified.size() -
                          1) { // FSM variables are assigned their initial value
                               // as an SMT constant
              initialCondition.push_back(smt::BVConstantOp::create(
                  b, loc, varInitValues[idx - numArgs - numOut]));
            } else if (numArgs <= idx &&
                       idx < forallQuantified.size() -
                                 1) { // FSM outputs are assigned the value
                                      // computed in the output region
              initialCondition.push_back(castOutValues[idx - numArgs]);
            } else if (idx == forallQuantified.size() -
                                  1) // If present, time is set to 0
              initialCondition.push_back(
                  smt::BVConstantOp::create(b, loc, 0, cfg.timeWidth));
          } else {
            if (numArgs + numOut <= idx) { // FSM variables are assigned their
                                           // initial value as an SMT constant
              initialCondition.push_back(smt::BVConstantOp::create(
                  b, loc, varInitValues[idx - numArgs - numOut]));
            } else if (numArgs <= idx) { // FSM outputs are assigned the value
                                         // computed in the output region
              initialCondition.push_back(castOutValues[idx - numArgs]);
            }
          }
        }

        return smt::ApplyFuncOp::create(b, loc, stateFunctions[0],
                                        initialCondition);
      });

  // Assert initial condition
  smt::AssertOp::create(b, loc, initialAssertion);

  // Double quantified arguments' types to consider their value at both the
  // initial and the arriving state
  SmallVector<Type> transitionQuantified;
  for (auto [id, ty] : llvm::enumerate(quantifiableTypes)) {
    if (id < numArgs) {
      transitionQuantified.push_back(ty);
      transitionQuantified.push_back(ty);
    } else {
      transitionQuantified.push_back(ty);
    }
  }

  // For each transition
  // `F_state0(vars, outs, [time]) &&& guard (args, vars) ->
  // F_state1(updatedVars, updatedOuts, [time + 1])`, build a corresponding
  // implication.

  for (auto [transId, transition] : llvm::enumerate(transitions)) {

    // return a SmallVector of updated SMT values as arguments of the arriving
    // state function
    auto action =
        [&](SmallVector<Value> &actionArgsOutsVarsVals) -> SmallVector<Value> {
      // Map each SMT-quantified variable to a corresponding FSM variable or
      // argument
      SmallVector<std::pair<mlir::Value, mlir::Value>> fsmToCast = mapSmtToFsm(
          loc, b, actionArgsOutsVarsVals, numArgs, numOut, fsmArgs, fsmVars);

      // Retrieve action region
      SmallVector<Value> castUpdatedVars;

      // Initialize to the previous value
      for (size_t i = 0; i < numVars; i++) {
        castUpdatedVars.push_back(actionArgsOutsVarsVals[numArgs + numOut + i]);
      }

      if (transition.hasAction) {

        auto *actionReg = transition.action;

        IRMapping mapping = createIRMapping(fsmToCast, constMapper);

        SmallVector<std::pair<mlir::Value, mlir::Value>> combActionValues;

        // Clone all the operations in the action region except for `UpdateOp`
        // and `AssertOp`, replacing FSM variables and arguments with the
        // results of unrealized conversion casts and replacing constants with
        // their new clone
        for (auto &op : actionReg->front()) {
          auto *newOp = b.clone(op, mapping);
          if (!isa<verif::AssertOp>(newOp)) {
            // Retrieve the updated values and their operands
            if (isa<fsm::UpdateOp>(newOp)) {
              auto varToUpdate = newOp->getOperand(0);
              auto updatedValue = newOp->getOperand(1);

              for (auto [id, var] : llvm::enumerate(fsmToCast)) {
                if (var.second == varToUpdate) {

                  // Cast the updated value to the appropriate SMT type
                  auto convCast = UnrealizedConversionCastOp::create(
                      b, loc, actionArgsOutsVarsVals[numOut + id].getType(),
                      updatedValue);

                  castUpdatedVars[id - numArgs] = convCast->getResult(0);
                }
              }
              newOp->erase();
            }
          } else {
            // Ignore assertions in action regions
            mlir::emitWarning(loc, "Assertions in action regions are ignored.");
            newOp->erase();
          }
        }
      }
      return castUpdatedVars;
    };

    // Return an SMT value for the transition's guard
    auto guard = [&](SmallVector<Value> &actionArgsOutsVarsVals) -> Value {
      // Map each SMT-quantified variable to a corresponding FSM variable or
      // argument
      SmallVector<std::pair<mlir::Value, mlir::Value>> fsmToCast = mapSmtToFsm(
          loc, b, actionArgsOutsVarsVals, numArgs, numOut, fsmArgs, fsmVars);

      IRMapping mapping = createIRMapping(fsmToCast, constMapper);

      Value guardVal;

      // Clone all the operations in the guard region except for `ReturnOp` and
      // `AssertOp,
      for (auto &op : transition.guard->front()) {
        auto *newOp = b.clone(op, mapping);
        if (!isa<verif::AssertOp>(newOp)) {
          // Retrieve the guard value
          if (isa<fsm::ReturnOp>(newOp)) {
            // Cast the guard value to an SMT boolean type
            auto castVal = mlir::UnrealizedConversionCastOp::create(
                b, loc, b.getType<smt::BitVectorType>(1), newOp->getOperand(0));

            guardVal = bv1toSmtBool(b, loc, castVal.getResult(0));
            newOp->erase();
          }
        } else {
          // Ignore assertions in guard regions

          mlir::emitWarning(loc, "Assertions in guard regions are ignored.");
          newOp->erase();
        }
      }
      return guardVal;
    };

    // Return a SmallVector of SMT values output at the arriving state
    auto output =
        [&](SmallVector<Value> &outputArgsOutsVarsVals) -> SmallVector<Value> {
      // map each SMT-quantified variable to a corresponding FSM variable or
      // argument
      SmallVector<std::pair<mlir::Value, mlir::Value>> fsmToCast = mapSmtToFsm(
          loc, b, outputArgsOutsVarsVals, numArgs, numOut, fsmArgs, fsmVars);
      SmallVector<Value> castOutputVars;
      auto *outputReg = transition.output;
      IRMapping mapping = createIRMapping(fsmToCast, constMapper);

      // Clone each operation in the output region except for `OutputOp`,
      // replacing FSM variables and arguments with the results of unrealized
      // conversion casts and replacing constants with their new clone
      for (auto &op : outputReg->front()) {
        auto *newOp = b.clone(op, mapping);
        if (!isa<verif::AssertOp>(newOp)) {

          // Retrieve all the operands of the output operation
          if (isa<fsm::OutputOp>(newOp)) {
            for (auto [id, operand] : llvm::enumerate(newOp->getOperands())) {

              // Cast the output value to the appropriate SMT type
              auto convCast = UnrealizedConversionCastOp::create(
                  b, loc, outputArgsOutsVarsVals[numArgs + id].getType(),
                  operand);
              castOutputVars.push_back(convCast->getResult(0));
            }
            newOp->erase();
          }
        } else {
          // Store the assertion operations, with a copy of the region
          assertions.push_back({transition.to, outputReg});
          newOp->erase();
        }
      }

      return castOutputVars;
    };

    auto forall = smt::ForallOp::create(
        b, loc, transitionQuantified,
        [&](OpBuilder &b, Location loc,
            ValueRange doubledQuantifiedVars) -> Value {
          SmallVector<Value> startingArgsOutsVars;
          SmallVector<Value> startingFunArgs;
          SmallVector<Value> arrivingArgsOutsVars;
          SmallVector<Value> arrivingFunArgs;

          for (size_t i = 0; i < 2 * numArgs; i++) {
            if (i % 2 == 1)
              arrivingArgsOutsVars.push_back(doubledQuantifiedVars[i]);
            else
              startingArgsOutsVars.push_back(doubledQuantifiedVars[i]);
          }
          for (auto i = 2 * numArgs; i < doubledQuantifiedVars.size(); ++i) {
            startingArgsOutsVars.push_back(doubledQuantifiedVars[i]);
            arrivingArgsOutsVars.push_back(doubledQuantifiedVars[i]);
            startingFunArgs.push_back(doubledQuantifiedVars[i]);
          }

          // Apply the starting-state function to the starting variables and
          // outputs
          auto lhs = smt::ApplyFuncOp::create(
              b, loc, stateFunctions[transition.from], startingFunArgs);

          // Update the variables according to the action region
          auto updatedCastVals = action(startingArgsOutsVars);
          for (size_t i = 0; i < numVars; i++) {
            arrivingArgsOutsVars[numArgs + numOut + i] = updatedCastVals[i];
          }

          // Depending on the updated variables, compute the output values at
          // the arriving state
          if (transition.hasOutput) {
            auto outputCastVals = output(arrivingArgsOutsVars);

            // Add the output values to the arriving-state function inputs
            for (auto o : outputCastVals) {
              arrivingFunArgs.push_back(o);
            }
          }

          // Add the updated variable values to the arriving-state function
          // inputs
          for (auto u : updatedCastVals)
            arrivingFunArgs.push_back(u);

          // Increment time variable if necessary
          if (cfg.withTime) {
            auto timeVal = doubledQuantifiedVars.back();
            auto oneConst = smt::BVConstantOp::create(b, loc, 1, cfg.timeWidth);
            auto incrementedTime = smt::BVAddOp::create(
                b, loc, timeVal.getType(), timeVal, oneConst);
            arrivingFunArgs.push_back(incrementedTime);
          }

          auto rhs = smt::ApplyFuncOp::create(
              b, loc, stateFunctions[transition.to], arrivingFunArgs);

          // If there is a guard, compute its value with the variable and
          // argument values at the starting state
          if (transition.hasGuard) {
            auto guardVal = guard(startingArgsOutsVars);
            auto guardedlhs = smt::AndOp::create(b, loc, lhs, guardVal);
            return smt::ImpliesOp::create(b, loc, guardedlhs, rhs);
          }
          return smt::ImpliesOp::create(b, loc, lhs, rhs);
        });

    smt::AssertOp::create(b, loc, forall);
  }

  // Produce an implication for each `verif.assert` operation found in output
  // regions
  for (auto pa : assertions) {

    auto forall = smt::ForallOp::create(
        b, loc, quantifiableTypes,
        [&](OpBuilder &b, Location loc, ValueRange forallQuantified) -> Value {
          // map each SMT-quantified variable to a corresponding FSM variable or
          // argument
          SmallVector<std::pair<mlir::Value, mlir::Value>> fsmToCast =
              mapSmtToFsm(loc, b, forallQuantified, numArgs, numOut, fsmArgs,
                          fsmVars);

          IRMapping mapping = createIRMapping(fsmToCast, constMapper);

          Value returnVal = smt::BoolConstantOp::create(b, loc, true);

          // Clone each operation in the output region except for FSM ones,
          // replacing FSM variables and arguments with the results of
          // unrealized conversion casts and replacing constants with their new
          // clone
          for (auto &op : pa.outputRegion->front()) {
            if (!isa<fsm::OutputOp>(op) && !isa<fsm::UpdateOp>(op) &&
                !isa<fsm::ReturnOp>(op)) {
              auto *newOp = b.clone(op, mapping);

              // Retrieve the assertion values
              if (isa<verif::AssertOp>(newOp)) {
                auto assertedVal = newOp->getOperand(0);
                auto castVal = mlir::UnrealizedConversionCastOp::create(
                    b, loc, b.getType<smt::BitVectorType>(1), assertedVal);

                //  Convert to SMT boolean type
                auto toBool = bv1toSmtBool(b, loc, castVal.getResult(0));
                auto inState = smt::ApplyFuncOp::create(
                    b, loc, stateFunctions[pa.stateId],
                    forallQuantified.drop_front(numArgs));

                // Produce an implication `F_state(outs, vars, [time]) ->
                // assertedVal`
                returnVal = smt::ImpliesOp::create(b, loc, inState, toBool);
                newOp->erase();
              }
            }
          }
          return returnVal;
        });

    smt::AssertOp::create(b, loc, forall);
  }

  smt::YieldOp::create(b, loc, typeRange, valueRange);

  machineOp.erase();
  return success();
}

struct FSMToSMTPass : public circt::impl::ConvertFSMToSMTBase<FSMToSMTPass> {
  void runOnOperation() override;
};

void FSMToSMTPass::runOnOperation() {
  auto module = getOperation();
  OpBuilder b(module);

  auto machineOps = to_vector(module.getOps<fsm::MachineOp>());
  if (machineOps.empty()) {
    return;
  }

  // Read options from the generated base
  LoweringConfig cfg;
  cfg.withTime = withTime; // default false
  cfg.timeWidth = timeWidth; // default 5

  for (auto machine : llvm::make_early_inc_range(module.getOps<MachineOp>())) {
    MachineOpConverter converter(b, machine, cfg);
    if (failed(converter.dispatch())) {
      signalPassFailure();
      return;
    }
  }
}

} // namespace

std::unique_ptr<mlir::Pass> circt::createConvertFSMToSMTPass() {
  return std::make_unique<FSMToSMTPass>();
}
