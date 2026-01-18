//===- FSMToSMT.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "circt/Conversion/FSMToSMT.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/LLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/SMT/IR/SMTDialect.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Dialect/SMT/IR/SMTTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

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
    Value predicateFsm;
  };

  struct Transition {
    int from;
    int to;
    bool hasGuard = false, hasAction = false, hasOutput = false;
    Region *guard = nullptr, *action = nullptr, *output = nullptr;
  };

  static Value bvToBool(OpBuilder &b, Location loc, Value v) {
    if (auto bvTy = llvm::dyn_cast<smt::BitVectorType>(v.getType())) {
      if (bvTy.getWidth() == 1) {
        auto one = smt::BVConstantOp::create(b, loc, 1, 1);
        return smt::EqOp::create(b, loc, v, one);
      }
    }
    if (llvm::isa<smt::BoolType>(v.getType()))
      return v;
    v.getDefiningOp()->emitError()
        << "bvToBool expected !smt.bv<1> or !smt.bool, got " << v;
    assert(false && "bvToBool type mismatch");
    return v;
  }

  static Value boolToBv(OpBuilder &b, Location loc, Value pred) {

    if (llvm::isa<smt::BoolType>(pred.getType())) {
      auto one = smt::BVConstantOp::create(b, loc, 1, 1);
      auto zero = smt::BVConstantOp::create(b, loc, 0, 1);
      return smt::IteOp::create(b, loc, b.getType<smt::BitVectorType>(1), pred,
                                one, zero);
    }
    if (auto bvTy = llvm::dyn_cast<smt::BitVectorType>(pred.getType()))
      if (bvTy.getWidth() == 1)
        return pred;
    pred.getDefiningOp()->emitError()
        << "boolToBv expected !smt.bool or !smt.bv<1>, got " << pred;
    assert(false && "boolToBv type mismatch");
    return pred;
  }

  static unsigned getPackedBitWidth(Type t) {
    if (auto intTy = llvm::dyn_cast<IntegerType>(t))
      return intTy.getIntOrFloatBitWidth();

    if (auto bvTy = llvm::dyn_cast<smt::BitVectorType>(t))
      return bvTy.getWidth();

    llvm::errs() << "Unsupported type for bitwidth computation in FSMToSMT: "
                 << t << "\n";
    assert(false && "Unsupported type in getPackedBitWidth");
    return 0;
  }

  smt::BVCmpPredicate getSmtBVPred(comb::ICmpPredicate p) {
    switch (p) {
    case comb::ICmpPredicate::slt:
      return smt::BVCmpPredicate::slt;
    case comb::ICmpPredicate::sle:
      return smt::BVCmpPredicate::sle;
    case comb::ICmpPredicate::sgt:
      return smt::BVCmpPredicate::sgt;
    case comb::ICmpPredicate::sge:
      return smt::BVCmpPredicate::sge;
    case comb::ICmpPredicate::ult:
      return smt::BVCmpPredicate::ult;
    case comb::ICmpPredicate::ule:
      return smt::BVCmpPredicate::ule;
    case comb::ICmpPredicate::ugt:
      return smt::BVCmpPredicate::ugt;
    case comb::ICmpPredicate::uge:
      return smt::BVCmpPredicate::uge;
    }
    assert(false && "unsupported comparison predicate");
  }

  Value numericToBool(Value v, Location loc) {
    if (llvm::isa<smt::BoolType>(v.getType()))
      return v;
    if (auto bvTy = llvm::dyn_cast<smt::BitVectorType>(v.getType())) {
      if (bvTy.getWidth() == 1) {
        auto one = smt::BVConstantOp::create(b, loc, 1, 1);
        return smt::EqOp::create(b, loc, v, one);
      }
      v.getDefiningOp()->emitError()
          << "expected !smt.bv<1> for numericToBool, got " << v;
      assert(false && "non-1-width BV to bool");
    }
    // int mode: v != 0
    return smt::EqOp::create(b, loc, v,
                             smt::BVConstantOp::create(b, loc, 1, 2));
  }

  Value toBv(Value v, Location loc) {
    if (auto bvTy = llvm::dyn_cast<smt::BitVectorType>(v.getType()))
      return v;
    if (llvm::isa<smt::BoolType>(v.getType()))
      return boolToBv(b, loc, v);
    assert(false && "Non-SMT value passed to toBV");
  }

  Value getCombValue(Operation &op, Location &loc, SmallVector<Value> args) {

    // widths for BV ops only (for result typing)
    SmallVector<int> widths;
    for (auto arg : args) {
      if (auto bvTy = llvm::dyn_cast<smt::BitVectorType>(arg.getType())) {
        widths.push_back(bvTy.getWidth());
      } else if (llvm::isa<smt::BoolType>(arg.getType())) {
        widths.push_back(1);
      } else {
        op.emitError() << "getCombValue received a non-SMT value: " << arg;
        assert(false && "Non-SMT value passed to getCombValue");
      }
    }

    // comb.add
    if (auto addOp = dyn_cast<comb::AddOp>(op)) {
      return smt::BVAddOp::create(
          b, loc, b.getType<smt::BitVectorType>(widths[0]), args);
    }

    // comb.sub
    if (auto subOp = dyn_cast<comb::SubOp>(op)) {
      Value rhs = toBv(args[1], loc);
      auto rhsTy = llvm::cast<smt::BitVectorType>(rhs.getType());
      Value neg = smt::BVNegOp::create(b, loc, rhsTy, rhs);
      return smt::BVAddOp::create(
          b, loc, b.getType<smt::BitVectorType>(widths[0]), {args[0], neg});
    }

    // comb.mul
    if (auto mulOp = dyn_cast<comb::MulOp>(op)) {
      return smt::BVMulOp::create(
          b, loc, b.getType<smt::BitVectorType>(widths[0]), args);
    }

    // comb.and
    if (auto andOp = dyn_cast<comb::AndOp>(op)) {
      if (args.size() == 1)
        return args[0];
      Value result = args[0];
      for (size_t i = 1; i < args.size(); ++i)
        result =
            smt::BVAndOp::create(b, loc, result.getType(), result, args[i]);
      return result;
    }

    // comb.or
    if (auto orOp = dyn_cast<comb::OrOp>(op)) {
      Value result = args[0];
      for (size_t i = 1; i < args.size(); ++i)
        result = smt::BVOrOp::create(b, loc, result.getType(), result, args[i]);
      return result;
    }

    // comb.xor
    if (auto xorOp = dyn_cast<comb::XorOp>(op)) {
      Value result = toBv(args[0], loc);
      for (size_t i = 1; i < args.size(); ++i)
        result = smt::BVXOrOp::create(b, loc, result.getType(), result,
                                      toBv(args[i], loc));
      return result;
    }

    // comb.mux
    if (auto mux = dyn_cast<comb::MuxOp>(op)) {
      assert(args.size() == 3 && "MuxOp should have 3 arguments");
      Value condBool = numericToBool(args[0], loc);
      // Return type is the type of data arms.
      Type resTy = args[1].getType();
      return smt::IteOp::create(b, loc, resTy, condBool, args[1], args[2]);
    }

    // comb.concat
    if (auto concatOp = dyn_cast<comb::ConcatOp>(op)) {
      Value acc = toBv(args[0], loc);
      int accW = dyn_cast<smt::BitVectorType>(acc.getType()).getWidth();
      for (size_t i = 1; i < args.size(); ++i) {
        Value next = toBv(args[i], loc);
        int nextW = dyn_cast<smt::BitVectorType>(next.getType()).getWidth();
        auto resTy = b.getType<smt::BitVectorType>(accW + nextW);
        acc = smt::ConcatOp::create(b, loc, resTy, acc, next);
        accW += nextW;
      }
      return acc;
    }

    // comb.extract
    if (auto extOp = dyn_cast<comb::ExtractOp>(op)) {
      unsigned low = extOp.getLowBit();
      unsigned width = extOp.getType().getIntOrFloatBitWidth();
      auto resTy = b.getType<smt::BitVectorType>(width);
      return smt::ExtractOp::create(b, loc, resTy, low,
                                    toBv(args.front(), loc));
    }

    // comb.replicate
    if (auto repOp = dyn_cast<comb::ReplicateOp>(op)) {
      unsigned count = repOp.getMultiple();
      Value in = toBv(args[0], loc);
      return smt::RepeatOp::create(b, loc, count, in);
    }

    // comb.shru
    if (comb::ShrUOp shruOp = dyn_cast<comb::ShrUOp>(op)) {
      SmallVector<Value> bvArgs;
      for (auto a : args)
        bvArgs.push_back(toBv(a, loc));
      auto bvOp = smt::BVLShrOp::create(b, loc, bvArgs);
      return smt::BVLShrOp::create(b, loc, bvOp.getType(), bvArgs);
    }

    // comb.icmp
    if (auto icmp = dyn_cast<comb::ICmpOp>(op)) {
      auto pred = icmp.getPredicate();
      // eq/ne first
      if (pred == comb::ICmpPredicate::eq) {
        // Generic eq works for both Int/BV; result Bool.
        Value eq = smt::EqOp::create(b, loc, args);
        return boolToBv(b, icmp.getLoc(), eq);
      }
      if (pred == comb::ICmpPredicate::ne) {
        auto dis = smt::DistinctOp::create(b, loc, args);
        return boolToBv(b, icmp.getLoc(), dis);
      }
      // ordered comparisons
      auto p = getSmtBVPred(pred);
      return smt::BVCmpOp::create(b, loc, p, toBv(args[0], loc),
                                  toBv(args[1], loc));
    }

    llvm::errs() << "\n\nunsupported comb op: " << op;
    assert(false && "unsupported comb operation");
    return Value();
  }

  Value getSmtValue(Value v,
                    const SmallVector<std::pair<Value, Value>> &fsmArgVals,
                    Location &loc) {
    // if an SMT value is already mapped, return it
    for (auto fav : fsmArgVals)
      if (v == fav.first)
        return fav.second;
    // if it's already an SMT value, retrieve it from the corresponding comb
    // operation
    if (v.getDefiningOp()->getName().getDialect()->getNamespace() == "comb") {
      SmallVector<Value> combArgs;
      for (auto arg : v.getDefiningOp()->getOperands()) {
        auto lowered = getSmtValue(arg, fsmArgVals, loc);
        combArgs.push_back(lowered);
      }
      return getCombValue(*v.getDefiningOp(), loc, combArgs);
    }
    // if it's a constant, lower it to an SMT constant
    if (auto cst = dyn_cast<hw::ConstantOp>(v.getDefiningOp())) {
      auto ap = cst.getValue();
      return smt::BVConstantOp::create(b, loc, ap);
    }
    llvm::errs() << "\n\nunsupported getSmtValue op: " << v;
    assert(false && "unsupported getSmtValue operation");
  }

  SmallVector<Value>
  parseOutputRegion(Region *outputRegion,
                    SmallVector<PendingAssertion> &assertions,
                    const SmallVector<std::pair<Value, Value>> &valToSmt,
                    Location &loc, int stateId) {
    SmallVector<Value> outputSmtValues;
    // Parse the output region and retrieve the output values
    for (auto &op : outputRegion->getOps()) {
      // Whenever we find an OutputOp, backtrack the output values.
      if (auto outOp = dyn_cast<fsm::OutputOp>(op)) {
        // Retrieve the SMT values mapped to each operand in OutputOp
        for (auto outputOperand : outOp->getOperands()) {
          auto v = getSmtValue(outputOperand, valToSmt, loc);
          outputSmtValues.push_back(v);
        }
      }
      if (auto a = dyn_cast<verif::AssertOp>(op)) {
        // Store original FSM value; defer lowering.
        assertions.push_back({stateId, a.getOperand(0)});
      }
    }

    return outputSmtValues;
  }

  Transition parseTransition(fsm::TransitionOp t, int from,
                             SmallVector<std::string> &states, Location &loc) {
    std::string nextState = t.getNextState().str();
    Transition tr = {.from = from, .to = insertStates(states, nextState)};
    if (!t.getGuard().empty()) {
      tr.hasGuard = true;
      tr.guard = &t.getGuard();
    }
    if (!t.getAction().empty()) {
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
    abort();
  }

  MachineOp machineOp;
  OpBuilder &b;
  LoweringConfig cfg;
};

LogicalResult MachineOpConverter::dispatch() {
  b.setInsertionPoint(machineOp);
  auto loc = machineOp.getLoc();
  auto machineArgs = machineOp.getArguments();

  // Store the FSM arguments, outputs, variables types and values
  SmallVector<Type> argsOutsVarsTypes;
  SmallVector<Value> argsOutsVarsVals;

  TypeRange typeRange;
  ValueRange valueRange;

  auto solver = smt::SolverOp::create(b, loc, typeRange, valueRange);
  solver.getBodyRegion().emplaceBlock();
  b.setInsertionPointToStart(solver.getBody());

  // Collect arguments and their types
  for (auto a : machineArgs) {
    auto width = a.getType().getIntOrFloatBitWidth();
    argsOutsVarsTypes.push_back(b.getType<smt::BitVectorType>(width));
    argsOutsVarsVals.push_back(a);
  }
  size_t numArgs = argsOutsVarsTypes.size();

  // Collect output variables and their types
  if (!machineOp.getResultTypes().empty()) {
    for (auto o : machineOp.getResultTypes()) {
      auto width = o.getIntOrFloatBitWidth();
      argsOutsVarsTypes.push_back(b.getType<smt::BitVectorType>(width));
      // zero-initialized outputs
      auto ov = smt::BVConstantOp::create(b, loc, 0, width);
      argsOutsVarsVals.push_back(ov);
    }
  }
  size_t numOut = argsOutsVarsVals.size() - numArgs;

  // Collect FSM variables, their types, and initial values
  SmallVector<llvm::APInt> varInitValues;
  for (auto variableOp : machineOp.front().getOps<fsm::VariableOp>()) {
    auto width = variableOp.getType().getIntOrFloatBitWidth();
    argsOutsVarsTypes.push_back(b.getType<smt::BitVectorType>(width));
    // retrieve initial value if available, set to 0#w otherwise
    auto intAttr = dyn_cast<IntegerAttr>(variableOp.getInitValueAttr());
    varInitValues.push_back(intAttr.getValue());
    argsOutsVarsVals.push_back(variableOp->getOpResult(0));
  }
  size_t numVars = varInitValues.size();

  // Add an extra `time` parameter, if enabled.
  if (cfg.withTime) {
    argsOutsVarsTypes.push_back(b.getType<smt::BitVectorType>(cfg.timeWidth));
  }

  // Store all the transitions state0 -> state1 in the FSM
  SmallVector<MachineOpConverter::Transition> transitions;
  // Store all the functions F_state(outs, vars, [time]) -> Bool describing the
  // activation of each state
  SmallVector<Value> stateFunctions;
  // Store the name of each state
  SmallVector<std::string> states;
  // Store the output region of each state
  SmallVector<std::pair<Region *, int>> outputOfStateId;

  // Get FSM initial state and store it in the states vector
  std::string initialState = machineOp.getInitialState().str();
  insertStates(states, initialState);

  // We retrieve the types of outputs and variables only, since the arguments
  // are universally quantified outside the state functions. These will be the
  // arguments of each state-activation function F_state.
  SmallVector<Type> outsVarsTypes;
  for (auto i = numArgs; i < argsOutsVarsTypes.size(); ++i) {
    // only outputs and variables
    outsVarsTypes.push_back(argsOutsVarsTypes[i]);
  }

  // For each state, declare one SMT function: F_state(outs, vars, [time]) ->
  // Bool, returning `true`  when `state` is activated.
  for (auto stateOp : machineOp.front().getOps<fsm::StateOp>()) {
    mlir::StringAttr funName =
        b.getStringAttr(("F_" + stateOp.getName().str()));
    auto range = b.getType<smt::BoolType>();
    // Only the variables and output are arguments of the state function, since
    // the FSM arguments are universally quantified outside the state functions.
    auto funTy = b.getType<smt::SMTFuncType>(outsVarsTypes, range);
    auto acFun = smt::DeclareFunOp::create(b, loc, funTy, funName);
    stateFunctions.push_back(acFun);
    auto fromState = insertStates(states, stateOp.getName().str());
    outputOfStateId.push_back({&stateOp.getOutput(), fromState});
  }

  // For each transition `state0 &&& guard01 -> state1`, we construct an
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

  // In the initial assertion we set variables to their initial values, time ==
  // 0 (if present), initialize the arguments to `0` and compute the output
  // values of the initial state accordingly.
  auto initState = smt::ForallOp::create(
      b, loc, argsOutsVarsTypes,
      [&](OpBuilder &b, Location loc,
          const SmallVector<Value> &forallArgsOutsVars) -> Value {
        // Retrieve output region of the initial state
        auto *initOutputReg = getOutputRegion(outputOfStateId, 0);
        // store the initial variables' values
        SmallVector<Value> retrievedInitVarVals;
        // retrieve the initial value for all the quantified variables (exclude
        // time)
        for (size_t i = numOut + numArgs; i < numOut + numArgs + numVars; i++) {
          auto val = varInitValues[i - (numOut + numArgs)];
          auto type = argsOutsVarsTypes[i];
          // depending on the type, create the appropriate SMT constant
          if (auto bvType = llvm::dyn_cast<smt::BitVectorType>(type)) {
            // BitVectorType
            assert(val.getBitWidth() == bvType.getWidth() &&
                   "init width mismatch for variable");
            retrievedInitVarVals.push_back(
                smt::BVConstantOp::create(b, loc, val));
          } else if (llvm::isa<smt::BoolType>(type)) {
            // BoolType
            retrievedInitVarVals.push_back(
                smt::BoolConstantOp::create(b, loc, val != 0));
          } else {
            // IntType
            auto attr =
                b.getIntegerAttr(b.getIntegerType(val.getBitWidth()), val);
            retrievedInitVarVals.push_back(
                smt::IntConstantOp::create(b, loc, attr));
          }
        }

        // store the SMT values corresponding to the output values
        SmallVector<Value> outputSmtValues;
        // push the quantified SMT values for the arguments

        // Evaluate output region at init to compute outputs (if present).
        // The evaluation of the output could depend on the FSM arguments, which
        // are initialized to 0 at the first iteration.
        if (!initOutputReg->empty()) {
          // Map each FSM variable to a corresponding SMT value.
          SmallVector<std::pair<Value, Value>> valToSmt;

          // First initialize the map with the quantified arguments
          for (auto [id, val] : llvm::enumerate(argsOutsVarsVals))
            valToSmt.push_back({val, forallArgsOutsVars[id]});

          // Replace the SMT values in the map for variables
          for (auto [i, a] : llvm::enumerate(argsOutsVarsVals)) {
            if (i >= numOut + numArgs && i < numOut + numArgs + numVars) {
              valToSmt.push_back(
                  {a, retrievedInitVarVals[i - numOut - numArgs]});
            }
          }
          // retrieve the output SMT values and assertions from the output
          // region
          outputSmtValues =
              parseOutputRegion(initOutputReg, assertions, valToSmt, loc, 0);
        }

        // Store the variables and output (and optionally time) at the initial
        // state.
        SmallVector<Value> initStateArgs;

        for (auto i = numArgs; i < forallArgsOutsVars.size(); ++i) {
          // output values are the ones computed from the output region
          if (i < numOut + numArgs) {
            // outputs
            initStateArgs.push_back(outputSmtValues[i - numArgs]);
          } else if (i < numOut + numArgs + numVars) {
            // variables
            initStateArgs.push_back(
                retrievedInitVarVals[i - (numOut + numArgs)]);
          } else {
            initStateArgs.push_back(forallArgsOutsVars[i]); // time
          }
        }

        auto inInit =
            smt::ApplyFuncOp::create(b, loc, stateFunctions[0], initStateArgs);

        if (cfg.withTime) {
          // time is the last forall arg
          Value zeroTime = smt::BVConstantOp::create(b, loc, 0, cfg.timeWidth);
          Value atZero =
              smt::EqOp::create(b, loc, forallArgsOutsVars.back(), zeroTime);
          return smt::ImpliesOp::create(b, loc, atZero, inInit);
        }

        return inInit;
      });

  smt::AssertOp::create(b, loc, initState);

  // For each transition
  // F_state0(vars, outs, [time]) &&& guard (args, vars) ->
  // F_state1(updatedVars, updatedOuts, [time + 1]), build a corresponding
  // implication.

  for (auto [transId, transition] : llvm::enumerate(transitions)) {

    // return a SmallVector of updated SMT values for arguments (unchanged),
    // variables and outputs after evaluating the action and output regions
    auto action =
        [&](SmallVector<Value> actionArgsOutsVarsVals) -> SmallVector<Value> {
      // map each FSM variable to a corresponding SMT value.
      SmallVector<std::pair<Value, Value>> valToSmt;
      // initialize the map with the quantified arguments
      for (auto [id, av] : llvm::enumerate(argsOutsVarsVals))
        valToSmt.push_back({av, actionArgsOutsVarsVals[id]});
      // store the updated SMT values for each variable
      SmallVector<Value> updatedSmtValues;

      // if the transition has an action, evaluate it and update the variables
      // accordingly
      if (transition.hasAction) {
        for (auto [j, pairMap] : llvm::enumerate(valToSmt)) {
          bool found = false;
          for (auto &op : transition.action->getOps()) {
            if (auto updateOp = dyn_cast<fsm::UpdateOp>(op)) {
              // if the return operand of the update operation correponds to the
              // variable being updated pairMap.first retrieve its SMT value
              if (updateOp->getOperand(0) == pairMap.first) {
                auto nv = getSmtValue(updateOp->getOperand(1), valToSmt, loc);
                updatedSmtValues.push_back(nv);
                found = true;
              }
            }
          }
          // if the variable is not updated, keep the quantified value
          if (!found)
            updatedSmtValues.push_back(pairMap.second);
        }
      } else {
        // if there is no action, variables remain unchanged
        for (auto [j, pairMap] : llvm::enumerate(valToSmt))
          updatedSmtValues.push_back(pairMap.second);
      }
      // Time update (if present).
      if (cfg.withTime) {
        Value nextTime;
        auto oneT = smt::BVConstantOp::create(b, loc, 1, cfg.timeWidth);
        auto resTy = b.getType<smt::BitVectorType>(cfg.timeWidth);
        nextTime = smt::BVAddOp::create(
            b, loc, resTy,
            SmallVector<Value>{actionArgsOutsVarsVals.back(), oneT});
        updatedSmtValues.push_back(nextTime);
      }
      return updatedSmtValues;
    };

    // return an SMT Bool value for the guard condition of the transition,
    auto guard = [&](SmallVector<Value> guardArgsOutsVarsVals) -> Value {
      if (transition.hasGuard) {
        // map each FSM variable to a corresponding SMT value.
        SmallVector<std::pair<Value, Value>> valToSmt;
        // initialize the map with the quantified arguments
        for (auto [id, av] : llvm::enumerate(argsOutsVarsVals))
          valToSmt.push_back({av, guardArgsOutsVarsVals[id]});

        for (auto &op : transition.guard->getOps())
          if (auto retOp = dyn_cast<fsm::ReturnOp>(op)) {
            // get the SMT value returned by the guard region
            auto gVal = getSmtValue(retOp->getOperand(0), valToSmt, loc);
            // Convert to Bool if it is a bitvector of width 1
            if (auto bvTy = llvm::dyn_cast<smt::BitVectorType>(gVal.getType()))
              if (bvTy.getWidth() == 1) {
                auto one = smt::BVConstantOp::create(b, loc, 1, 1);
                return smt::EqOp::create(b, loc, gVal, one);
              }
            return gVal;
          }
      }
      // return true if the transition has no guard
      return smt::BoolConstantOp::create(b, loc, true);
    };

    auto output =
        [&](SmallVector<Value> outputArgsOutsVarsVals) -> SmallVector<Value> {
      SmallVector<Value> outputSmtValues;
      if (transition.hasOutput) {
        // map each FSM variable to a corresponding SMT value.
        SmallVector<std::pair<Value, Value>> valToSmt;
        // initialize the map with the quantified arguments
        for (auto [id, val] : llvm::enumerate(argsOutsVarsVals))
          valToSmt.push_back({val, outputArgsOutsVarsVals[id]});
        // retrieve the output SMT values and assertions from the output region
        outputSmtValues = parseOutputRegion(transition.output, assertions,
                                            valToSmt, loc, transition.to);
      }
      return outputSmtValues;
    };

    // For each transition, assert:
    // Forall (argsNew,argsOld,...):
    // F_from(outOld, varsOld, [time]) AND guard(argsOld, outOld, varsOld)
    //    => F_to(outNew, varsNew, [time+1])
    // we need two copies of each args: argsOld, argsNew
    SmallVector<Type> doubleArgsOutsVarsTypes;
    for (auto [id, ty] : llvm::enumerate(argsOutsVarsTypes)) {
      if (id < numArgs) {
        doubleArgsOutsVarsTypes.push_back(ty);
        doubleArgsOutsVarsTypes.push_back(ty);
      } else {
        doubleArgsOutsVarsTypes.push_back(ty);
      }
    }
    auto forall = smt::ForallOp::create(
        b, loc, doubleArgsOutsVarsTypes,
        [&](OpBuilder &b, Location loc,
            ValueRange forallDoubledArgsOutputVarsInputs) -> Value {
          SmallVector<Value> startingArgsOutsVars;
          SmallVector<Value> arrivingArgsOutsVars;

          for (size_t i = 0; i < 2 * numArgs; i++) {
            if (i % 2 == 1)
              startingArgsOutsVars.push_back(
                  forallDoubledArgsOutputVarsInputs[i]);
            else
              arrivingArgsOutsVars.push_back(
                  forallDoubledArgsOutputVarsInputs[i]);
          }

          // Initialize state function arguments (outs, vars, [time])
          SmallVector<Value> stateFuncArgs;
          for (auto i = 2 * numArgs;
               i < forallDoubledArgsOutputVarsInputs.size(); ++i) {
            stateFuncArgs.push_back(forallDoubledArgsOutputVarsInputs[i]);
          }

          // the state function only takes outputs and variables (and optionally
          // time) as arguments
          auto startingStateFun = smt::ApplyFuncOp::create(
              b, loc, stateFunctions[transition.from], stateFuncArgs);

          for (auto funcArg : stateFuncArgs) {
            startingArgsOutsVars.push_back(funcArg);
            arrivingArgsOutsVars.push_back(funcArg);
          }

          SmallVector<Value> actionedStateFuncArgs =
              action(startingArgsOutsVars);
          SmallVector<Value> outputStateFuncArgs = output(arrivingArgsOutsVars);

          SmallVector<Value> actionedStateFuncArgsForFunc;

          // only outs, vars, [time] are arguments of the state function
          for (auto i = numArgs; i < actionedStateFuncArgs.size(); ++i) {
            if (i < numOut + numArgs) {
              // outputs are the ones computed from the output region
              actionedStateFuncArgsForFunc.push_back(
                  outputStateFuncArgs[i - numArgs]);
            } else {
              // variables and time result from the update in the action region
              actionedStateFuncArgsForFunc.push_back(actionedStateFuncArgs[i]);
            }
          }

          auto rhs =
              smt::ApplyFuncOp::create(b, loc, stateFunctions[transition.to],
                                       actionedStateFuncArgsForFunc);

          auto guardVal = guard(startingArgsOutsVars);
          auto lhs = smt::AndOp::create(b, loc, startingStateFun, guardVal);
          return smt::ImpliesOp::create(b, loc, lhs, rhs);
        });

    smt::AssertOp::create(b, loc, forall);
  }

  // Lower each captured verif.assert as a safety clause:
  // Forall x. F_state(x) => predicate(x)
  for (auto &pa : assertions) {
    auto forall = smt::ForallOp::create(
        b, loc, argsOutsVarsTypes,
        [&](OpBuilder &b, Location loc, ValueRange forallInputs) {
          SmallVector<std::pair<Value, Value>> avToSmt;
          for (auto [i, av] : llvm::enumerate(argsOutsVarsVals))
            avToSmt.push_back({av, forallInputs[i]});

          Value predVal = getSmtValue(pa.predicateFsm, avToSmt, loc);
          Value predBool = bvToBool(b, loc, predVal);

          Value inState = smt::ApplyFuncOp::create(
              b, loc, stateFunctions[pa.stateId], forallInputs);

          return smt::ImpliesOp::create(b, loc, inState, predBool);
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
    // markAllAnalysesPreserved();
    return;
  }

  // Read options from the generated base
  LoweringConfig cfg;
  cfg.withTime = withTime; // default false
  cfg.timeWidth = 5;

  for (auto machine : llvm::make_early_inc_range(module.getOps<MachineOp>())) {
    MachineOpConverter converter(b, machine, cfg);
    if (failed(converter.dispatch())) {
      signalPassFailure();
      return;
    }
    module.walk([&](circt::hw::ConstantOp cst) { cst.erase(); });
  }
}

} // namespace

std::unique_ptr<mlir::Pass> circt::createConvertFSMToSMTPass() {
  return std::make_unique<FSMToSMTPass>();
}