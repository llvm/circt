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
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/LLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/SMT/IR/SMTDialect.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Dialect/SMT/IR/SMTTypes.h"
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
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <circt/Dialect/HW/HWTypes.h>
#include <cstdlib>
#include <memory>
#include <string>
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

  // If true, include a `time` parameter in the relation, to be incremented at every transition.
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

  // Helpers (mode-aware).
  Type numTypeForWidth(unsigned w) {
    return b.getType<smt::BitVectorType>(w);
  }

  Value zeroConst(unsigned w, Location loc) {
    return b.create<smt::BVConstantOp>(loc, 0, w);
  }

  Value oneConst(unsigned w, Location loc) {
    return b.create<smt::BVConstantOp>(loc, 1, w);
  }

  // BV-only: (!smt.bv<1> -> !smt.bool). If already Bool, return as-is.
  static Value bv1ToBool(OpBuilder &b, Location loc, Value v) {
    if (auto bvTy = llvm::dyn_cast<smt::BitVectorType>(v.getType())) {
      if (bvTy.getWidth() == 1) {
        auto one = b.create<smt::BVConstantOp>(loc, 1, 1);
        return b.create<smt::EqOp>(loc, v, one);
      }
    }
    if (llvm::isa<smt::BoolType>(v.getType()))
      return v;
    v.getDefiningOp()->emitError()
        << "bv1ToBool expected !smt.bv<1> or !smt.bool, got " << v;
    assert(false && "bv1ToBool type mismatch");
    return v;
  }

  // BV-only: (!smt.bool -> !smt.bv<1>). If already BV<1>, return as-is.
  static Value boolToBV1(OpBuilder &b, Location loc, Value pred) {
    if (llvm::isa<smt::BoolType>(pred.getType())) {
      auto one = b.create<smt::BVConstantOp>(loc, 1, 1);
      auto zero = b.create<smt::BVConstantOp>(loc, 0, 1);
      return b.create<smt::IteOp>(loc, b.getType<smt::BitVectorType>(1), pred,
                                  one, zero);
    }
    if (auto bvTy = llvm::dyn_cast<smt::BitVectorType>(pred.getType()))
      if (bvTy.getWidth() == 1)
        return pred;
    pred.getDefiningOp()->emitError()
        << "boolToBV1 expected !smt.bool or !smt.bv<1>, got " << pred;
    assert(false && "boolToBV1 type mismatch");
    return pred;
  }

  // Mode-aware: !smt.bool -> numeric (BV<1> or Int(0/1))
  Value boolToNumeric(Value v, Location loc) {
    if (!llvm::isa<smt::BoolType>(v.getType()))
      return v;
    return boolToBV1(b, loc, v);
  }

  // Mode-aware: numeric (BV<1> or Int) -> Bool
  Value numericToBool(Value v, Location loc) {
    if (llvm::isa<smt::BoolType>(v.getType()))
      return v;
    if (auto bvTy = llvm::dyn_cast<smt::BitVectorType>(v.getType())) {
      if (bvTy.getWidth() == 1) {
        auto one = b.create<smt::BVConstantOp>(loc, 1, 1);
        return b.create<smt::EqOp>(loc, v, one);
      }
      v.getDefiningOp()->emitError()
          << "expected !smt.bv<1> for numericToBool, got " << v;
      assert(false && "non-1-width BV to bool");
    }
    // int mode: v != 0
    auto z = oneConst(/*w*/ 2, loc);
    return b.create<smt::EqOp>(loc, v, z);
  }
  
  // Type width computation for lowering aggregates.
  static unsigned getPackedBitWidth(Type t) {
    if (auto intTy = llvm::dyn_cast<IntegerType>(t))
      return intTy.getIntOrFloatBitWidth();

    if (auto bvTy = llvm::dyn_cast<smt::BitVectorType>(t))
      return bvTy.getWidth();

    if (auto arrTy = llvm::dyn_cast<hw::ArrayType>(t)) {
      unsigned elemW = getPackedBitWidth(arrTy.getElementType());
      return elemW * arrTy.getNumElements();
    }

    if (auto structTy = llvm::dyn_cast<hw::StructType>(t)) {
      unsigned w = 0;
      for (auto elem : structTy.getElements())
        w += getPackedBitWidth(elem.type);
      return w;
    }

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
  
  // Build SMT for a comb op (mode-aware).
  Value getCombValue(Operation &op, Location &loc,
                     SmallVector<Value> args) {
    auto toBV = [&](Value v) -> Value {
      if (isa<smt::IntType>(v.getType()))
        return b.create<smt::Int2BVOp>(loc, b.getType<smt::BitVectorType>(64), v);
      if (auto bvTy = llvm::dyn_cast<smt::BitVectorType>(v.getType()))
        return v;
      if (llvm::isa<smt::BoolType>(v.getType()))
        return boolToBV1(b, loc, v);
    };


    auto toBool = [&](Value v) -> Value {
      if (llvm::isa<smt::BoolType>(v.getType()))
        return v;
      return numericToBool(v, loc);
    };

    auto bvWidth = [&](Value v) -> int {
      if (auto bvTy = llvm::dyn_cast<smt::BitVectorType>(v.getType()))
        return bvTy.getWidth();
      if (llvm::isa<smt::BoolType>(v.getType()))
        return 1;
      op.emitError() << "expected SMT BV or Bool operand, got " << v;
      assert(false && "unexpected SMT operand type");
      return 1;
    };

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
      return b.create<smt::BVAddOp>(loc, b.getType<smt::BitVectorType>(widths[0]),
                                      args);
    }

    // comb.sub
    if (auto subOp = dyn_cast<comb::SubOp>(op)) {
      Value rhs = toBV(args[1]);
      auto rhsTy = llvm::cast<smt::BitVectorType>(rhs.getType());
      Value neg = b.create<smt::BVNegOp>(loc, rhsTy, rhs);
      return b.create<smt::BVAddOp>(loc,
            b.getType<smt::BitVectorType>(widths[0]), args[0], neg);
    }

    // comb.mul
    if (auto mulOp = dyn_cast<comb::MulOp>(op)) {
      return b.create<smt::BVMulOp>(loc, b.getType<smt::BitVectorType>(widths[0]),
                                      args);
    }

    // comb.and 
    if (auto andOp = dyn_cast<comb::AndOp>(op)) {
      if (args.size() == 1)
        return args[0];
      Value result = toBV(args[0]);
      for (size_t i = 1; i < args.size(); ++i)
        result = b.create<smt::BVAndOp>(loc, result, toBV(args[i]));
      return result;
    }

    // comb.or 
    if (auto orOp = dyn_cast<comb::OrOp>(op)) {
      Value result = toBV(args[0]);
      for (size_t i = 1; i < args.size(); ++i)
        result = b.create<smt::BVOrOp>(loc, result, toBV(args[i]));
      return result;
    }

    // comb.xor 
    if (auto xorOp = dyn_cast<comb::XorOp>(op)) {
      Value result = toBV(args[0]);
      for (size_t i = 1; i < args.size(); ++i)
        result = b.create<smt::BVXOrOp>(loc, result, toBV(args[i]));
      return result;
    }

    // comb.mux
    if (auto mux = dyn_cast<comb::MuxOp>(op)) {
      assert(args.size() == 3 && "MuxOp should have 3 arguments");
      Value condBool = toBool(args[0]);
      // Return type is the type of data arms.
      Type resTy = args[1].getType();
      return b.create<smt::IteOp>(loc, resTy, condBool, args[1], args[2]);
    }
    
    // comb.concat
    if (auto concatOp = dyn_cast<comb::ConcatOp>(op)) {
      Value acc = toBV(args[0]);
      int accW = bvWidth(acc);
      for (size_t i = 1; i < args.size(); ++i) {
        Value next = toBV(args[i]);
        int nextW = bvWidth(next);
        auto resTy = b.getType<smt::BitVectorType>(accW + nextW);
        acc = b.create<smt::ConcatOp>(loc, resTy, acc, next);
        accW += nextW;
      }
      return acc;
    }

    // comb.extract
    if (auto extOp = dyn_cast<comb::ExtractOp>(op)) {
      unsigned low = extOp.getLowBit();
      unsigned width = extOp.getType().getIntOrFloatBitWidth();
      auto resTy = b.getType<smt::BitVectorType>(width);
      return b.create<smt::ExtractOp>(loc, resTy, low, toBV(args.front()));
    }

    // comb.replicate
    if (auto repOp = dyn_cast<comb::ReplicateOp>(op)) {
      unsigned count = repOp.getMultiple();
      Value in = toBV(args[0]);
      return b.create<smt::RepeatOp>(loc, count, in);
    }

    // comb.shru
    if (comb::ShrUOp shruOp = dyn_cast<comb::ShrUOp>(op)) {
      SmallVector<Value> bvArgs; 
      for (auto a : args) bvArgs.push_back(toBV(a));
      auto bvOp = b.create<smt::BVLShrOp>(loc, bvArgs);
      return bvOp;
    }
  
    // comb.icmp
    if (auto icmp = dyn_cast<comb::ICmpOp>(op)) {
      auto pred = icmp.getPredicate();
      // eq/ne first
      if (pred == comb::ICmpPredicate::eq) {
        // Generic eq works for both Int/BV; result Bool.
        Value eq = b.create<smt::EqOp>(loc, args);
        return boolToBV1(b, icmp.getLoc(), eq);
      }
      if (pred == comb::ICmpPredicate::ne) {
        auto dis = b.create<smt::DistinctOp>(loc, args);
        return boolToBV1(b, icmp.getLoc(), dis);
      }
      // ordered comparisons
      auto p = getSmtBVPred(pred);
      return b.create<smt::BVCmpOp>(loc, p, toBV(args[0]), toBV(args[1]));
    }

    llvm::outs() << "\n\nunsupported comb op: " << op;
    assert(false && "unsupported comb operation");
    return Value();
  }

  Value getSmtValue(Value v,
                    const SmallVector<std::pair<Value, Value>> &fsmArgVals,
                    Location &loc) {
    for (auto fav : fsmArgVals)
      if (v == fav.first)
        return fav.second;

    if (v.getDefiningOp()->getName().getDialect()->getNamespace() == "comb") {
      SmallVector<Value> combArgs;
      for (auto arg : v.getDefiningOp()->getOperands()) {
        auto lowered = getSmtValue(arg, fsmArgVals, loc);
        combArgs.push_back(lowered);
      }
      return getCombValue(*v.getDefiningOp(), loc, combArgs);
    }

    if (auto cst = dyn_cast<hw::ConstantOp>(v.getDefiningOp())) {
      // Lower constants based on mode and bitwidth.
      auto ap = cst.getValue();
      return b.create<smt::BVConstantOp>(loc, ap);
    }

    llvm::outs() << "\n\nunsupported getSmtValue op: " << v;
    return v;
  }

  struct Transition {
    int from;
    int to;
    bool hasGuard = false, hasAction = false, hasOutput = false;
    Region *guard = nullptr, *action = nullptr, *output = nullptr;
  };

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

  static int insertStates(SmallVector<std::string> &states, llvm::StringRef st) {
    for (auto [id, s] : llvm::enumerate(states))
      if (s == st)
        return id;
    states.push_back(st.str());   // materialize once, stored in vector
    return states.size() - 1;
  }

  Region *getOutputRegion(
      const SmallVector<std::pair<Region *, int>>& outputOfStateId, int stateId) {
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

  int numArgs = 0;
  int numOut = 0;

  TypeRange typeRange;
  ValueRange valueRange;

  auto solver = b.create<smt::SolverOp>(loc, typeRange, valueRange);
  solver.getBodyRegion().emplaceBlock();
  b.setInsertionPointToStart(solver.getBody());

  // Collect arguments and their types
  for (auto a : machineArgs) {
    unsigned w = getPackedBitWidth(a.getType());
    argsOutsVarsTypes.push_back(numTypeForWidth(w));
    argsOutsVarsVals.push_back(a);
    numArgs++;
  }

  // Collect output variables and their types
  if (!machineOp.getResultTypes().empty()) {
    for (auto o : machineOp.getResultTypes()) {
      unsigned w = getPackedBitWidth(o);
      argsOutsVarsTypes.push_back(numTypeForWidth(w));
      auto ov = zeroConst(w, loc);
      argsOutsVarsVals.push_back(ov);
      numOut++;
    }
  }

  // Collect FSM variables, their types, and initial values
  SmallVector<llvm::APInt> varInitValues;
  for (auto variableOp : machineOp.front().getOps<fsm::VariableOp>()) {
    unsigned w = getPackedBitWidth(variableOp.getType());
    argsOutsVarsTypes.push_back(numTypeForWidth(w));
    auto initVal = variableOp.getInitValueAttr();
    if (auto intAttr = dyn_cast<IntegerAttr>(initVal))
      varInitValues.push_back(intAttr.getValue());
    else
      varInitValues.emplace_back(1, 0); // default false if missing
    argsOutsVarsVals.push_back(variableOp->getOpResult(0));
  }
  size_t numVars = varInitValues.size();

  // Add an extra `time` parameter, if enabled. 
  if (cfg.withTime) {
    argsOutsVarsTypes.push_back(b.getType<smt::BitVectorType>(cfg.timeWidth));
  }

  // Store all the transitions state0 -> state1 in the FSM
  SmallVector<MachineOpConverter::Transition> transitions;
  // Store all the functions F_state(outs, vars, [time]) -> Bool describing the activation of each state
  SmallVector<Value> stateFunctions;
  // Store the name of each state 
  SmallVector<std::string> states;
  // Store the output region of each state
  SmallVector<std::pair<Region *, int>> outputOfStateId;

  // Get FSM initial state and store it in the states vector
  std::string initialState = machineOp.getInitialState().str();
  insertStates(states, initialState);
  
  // We retrieve the types of outputs and variables only, since the arguments
  // are universally quantified outside the state functions. These will be the arguments 
  // of each state-activation function F_state.
  SmallVector<Type> outsVarsTypes;
  for (auto i = numArgs; i < argsOutsVarsTypes.size(); ++i) {
    // only outputs and variables
    if (numArgs <= i)
      outsVarsTypes.push_back(argsOutsVarsTypes[i]);
  }

  // For each state, declare one SMT function: F_state(outs, vars, [time]) -> Bool, 
  // returning `true`  when `state` is activated.
  for (auto stateOp : machineOp.front().getOps<fsm::StateOp>()) {
    mlir::StringAttr funName = b.getStringAttr(("F_" + stateOp.getName().str()));
    auto range = b.getType<smt::BoolType>();
    // Only the variables and output are arguments of the state function, since
    // the FSM arguments are universally quantified outside the state functions. 
    auto funTy = b.getType<smt::SMTFuncType>(outsVarsTypes, range);
    smt::DeclareFunOp acFun = b.create<smt::DeclareFunOp>(loc, funTy, funName);
    stateFunctions.push_back(acFun);
    auto fromState = insertStates(states, stateOp.getName().str());
    outputOfStateId.push_back({&stateOp.getOutput(), fromState});
  }

  // For each transition `state0 &&& guard01 -> state1`, we construct an implication 
  // `F_state_0(outs, vars, [time]) &&& guard01(outs, vars) -> F_state_1(outs, vars, [time])`, 
  // simulating the activation of a transition and of the state it reaches.
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

  // Collect `verif.assert` properties, to be lowered subsequently.
  struct PendingAssertion {
    int stateId;
    Value predicateFsm; 
  };
  SmallVector<PendingAssertion> assertions;

  // In the initial assertion we set variables to their initial values, time == 0 (if present), 
  // initialize the arguments to `0` and compute the output values of the initial state accordingly. 
  auto init_state = b.create<smt::ForallOp>(
      loc, argsOutsVarsTypes,
      [&](OpBuilder &b, Location loc, SmallVector<Value> forallArgs) -> Value {
        SmallVector<Value> initArgs;         
        SmallVector<Value> outputSmtValues;  
        SmallVector<Value> initVarValues;    
        auto initOutputReg = getOutputRegion(outputOfStateId, 0);

        // Build var init constants in SMT order (variables occupy positions
        // [numArgs + numOut, numArgs + numOut + numVars)).
        for (auto [i, a] : llvm::enumerate(forallArgs)) {
          // variable 
          if (int(i) >= numOut + numArgs && int(i) < int(numOut + numArgs + numVars)) {
            size_t varIdx = i - (numOut + numArgs);
            auto ap = varInitValues[varIdx];
            // Determine SMT type for this quantified symbol.
            Type qt = argsOutsVarsTypes[i];
            if (auto bvTy = llvm::dyn_cast<smt::BitVectorType>(qt)) {
              assert(ap.getBitWidth() == bvTy.getWidth() &&
                     "init width mismatch for variable");
              initVarValues.push_back(b.create<smt::BVConstantOp>(loc, ap));
            } else if (llvm::isa<smt::BoolType>(qt)) {
              initVarValues.push_back(b.create<smt::BoolConstantOp>(
                  loc, ap != 0));
            } else {
              // IntType
              auto attr =
                  b.getIntegerAttr(b.getIntegerType(ap.getBitWidth()), ap);
              initVarValues.push_back(b.create<smt::IntConstantOp>(loc, attr));
            }
          }
        }

        // Evaluate output region at init to compute outputs (if present).
        if (!initOutputReg->empty()) {
          SmallVector<std::pair<Value, Value>> avToSmt;
          for (auto [i, a] : llvm::enumerate(argsOutsVarsVals)) {
            if (int(i) >= numOut + numArgs &&
                int(i) < int(numOut + numArgs + numVars)) {
              avToSmt.push_back({a, initVarValues[i - numOut - numArgs]});
            } else {
              avToSmt.push_back({a, forallArgs[i]});
            }
          }

          for (auto &op : initOutputReg->getOps()) {
            if (auto outOp = dyn_cast<fsm::OutputOp>(op)) {
              for (auto outs : outOp->getOperands()) {
                bool found = false;
                for (auto [i, fav] : llvm::enumerate(avToSmt)) {
                  if (outs == fav.first && i < numArgs) {
                    outputSmtValues.push_back(forallArgs[i]);
                    found = true;
                  }
                }
                if (!found) {
                  auto v = getSmtValue(outs, avToSmt, loc);
                  outputSmtValues.push_back(v);
                }
              }
            }
          }
        }

        // `initArgs` only contains variables, outputs and optionally time. 
        for (auto [i, a] : llvm::enumerate(forallArgs)) {
          // we initialize outputs to the values computed from the output region
          if (int(i) >= numArgs && int(i) < numOut + numArgs &&
              int(i) < int(numOut + numArgs + numVars)) {
            initArgs.push_back(outputSmtValues[i - numArgs]); // outs
          } 
          // we initialize variables to their initial value
          else if (int(i) >= numOut + numArgs &&
                     int(i) < int(numOut + numArgs + numVars)) {
            initArgs.push_back(initVarValues[i - numOut - numArgs]); // vars
          }
        }

        auto inInit = b.create<smt::ApplyFuncOp>(loc, stateFunctions[0],
                                                 initArgs);

        if (cfg.withTime) {
          // time is the last forall arg
          Value zeroTime = b.create<smt::BVConstantOp>(loc, 0, cfg.timeWidth);
          Value atZero = b.create<smt::EqOp>(loc, forallArgs.back(), zeroTime);
          return b.create<smt::ImpliesOp>(loc, atZero, inInit);
        }

        return inInit;
      });

  b.create<smt::AssertOp>(loc, init_state);

  // Transition semantics.
  for (auto [id1, t1] : llvm::enumerate(transitions)) {
    auto action = [&](SmallVector<Value> actionArgs)
        -> SmallVector<Value> {
      // actionArgs are the current tuple (args, outs, vars, [time]).
      SmallVector<Value> outputSmtValues;

      if (t1.hasOutput) {
        SmallVector<std::pair<Value, Value>> avToSmt;
        for (auto [id, av] : llvm::enumerate(argsOutsVarsVals))
          avToSmt.push_back({av, actionArgs[id]});
        for (auto &op : t1.output->getOps()) {
          if (auto outOp = dyn_cast<fsm::OutputOp>(op)) {
            for (auto outs : outOp->getOperands()) {
              auto v = getSmtValue(outs, avToSmt, loc);
              outputSmtValues.push_back(v);
            }
          }
          if (auto a = dyn_cast<verif::AssertOp>(op)) {
            // Store original FSM value; defer lowering.
            assertions.push_back({t1.to, a.getOperand(0)});
          }
        }
      }

      SmallVector<std::pair<Value, Value>> avToSmt;
      SmallVector<Value> updatedSmtValues;

      for (auto [id, av] : llvm::enumerate(argsOutsVarsVals))
        avToSmt.push_back({av, actionArgs[id]});

      if (t1.hasAction) {
        // Copy current values; override with updates where present.
        for (auto [j, uv] : llvm::enumerate(avToSmt)) {
          bool found = false;
          for (auto &op : t1.action->getOps()) {
            if (auto upd = dyn_cast<fsm::UpdateOp>(op)) {
              if (upd->getOperand(0) == uv.first) {
                auto nv = getSmtValue(upd->getOperand(1), avToSmt, loc);
                updatedSmtValues.push_back(nv);
                found = true;
              }
            }
          }
          if (!found)
            updatedSmtValues.push_back(uv.second);
        }
      } else {
        for (auto [j, uv] : llvm::enumerate(avToSmt))
          updatedSmtValues.push_back(uv.second);
      }

      // Overwrite outputs from output region synthesis.
      for (auto [i, ov] : llvm::enumerate(outputSmtValues))
        updatedSmtValues[numArgs + i] = ov;

      // Time update (if present).
      if (cfg.withTime) {
        Value nextTime;
        auto oneT = b.create<smt::BVConstantOp>(loc, 1, cfg.timeWidth);
        auto resTy = b.getType<smt::BitVectorType>(cfg.timeWidth);
        nextTime = b.create<smt::BVAddOp>(
            loc, resTy, SmallVector<Value>{actionArgs.back(), oneT});
        updatedSmtValues.push_back(nextTime);
      }

      return updatedSmtValues;
    };

    auto guard1 = [&](SmallVector<Value> guardArgs) -> Value {
      if (t1.hasGuard) {
        SmallVector<std::pair<Value, Value>> avToSmt;
        for (auto [av, a] : llvm::zip(argsOutsVarsVals, guardArgs))
          avToSmt.push_back({av, a});
        for (auto &op : t1.guard->getOps())
          if (auto retOp = dyn_cast<fsm::ReturnOp>(op)) {
            auto gVal = getSmtValue(retOp->getOperand(0), avToSmt, loc);
            // If numeric-1-bit, convert to Bool.
            if (auto bvTy = llvm::dyn_cast<smt::BitVectorType>(gVal.getType()))
              if (bvTy.getWidth() == 1) {
                auto one = b.create<smt::BVConstantOp>(loc, 1, 1);
                return b.create<smt::EqOp>(loc, gVal, one);
              }
            return gVal; // Already Bool
          }
      }
      return b.create<smt::BoolConstantOp>(loc, true);
    };

    // For each transition, assert:
    // Forall (argsNew,argsOld,others): F_from(argsOld, ...) AND guard(argsOld, ...)
    //    => F_to(argsNew, outputs/vars updated, [time+1])
    SmallVector<Type> forallTypes;
    for (auto [id, ty] : llvm::enumerate(argsOutsVarsTypes)) {
      if (id < numArgs) {
        forallTypes.push_back(ty);
        forallTypes.push_back(ty);
      } else {
        forallTypes.push_back(ty);
      }
    }

    auto forall = b.create<smt::ForallOp>(
        loc, forallTypes,
        [&](OpBuilder &b, Location loc, ValueRange outVarsInputs) -> Value {
          SmallVector<Value> startingStateArgs;
          SmallVector<Value> arrivingStateArgs;
          for (auto [idx, fdi] : llvm::enumerate(outVarsInputs)) {
              startingStateArgs.push_back(fdi);
              arrivingStateArgs.push_back(fdi);
          }

          auto inFrom = b.create<smt::ApplyFuncOp>(loc, stateFunctions[t1.from],
                                                   startingStateArgs);
          auto actionedArgs = action(startingStateArgs);
          // for (auto [ida, aa] : llvm::enumerate(actionedArgs))
          //   if (ida < numArgs)
          //     actionedArgs[ida] = arrivingStateArgs[ida];

          auto rhs =
              b.create<smt::ApplyFuncOp>(loc, stateFunctions[t1.to],
                                         actionedArgs);
          auto guard = guard1(startingStateArgs);
          auto lhs = b.create<smt::AndOp>(loc, inFrom, guard);
          return b.create<smt::ImpliesOp>(loc, lhs, rhs);
        });

    b.create<smt::AssertOp>(loc, forall);
  }

  // Lower each captured verif.assert as a safety clause:
  // Forall x. F_state(x) => predicate(x)
  for (auto &pa : assertions) {
    auto forall = b.create<smt::ForallOp>(
        loc, argsOutsVarsTypes,
        [&](OpBuilder &b, Location loc, ValueRange forallInputs) {
          SmallVector<std::pair<Value, Value>> avToSmt;
          for (auto [i, av] : llvm::enumerate(argsOutsVarsVals))
            avToSmt.push_back({av, forallInputs[i]});

          Value predVal = getSmtValue(pa.predicateFsm, avToSmt, loc);
          Value predBool =  bv1ToBool(b, loc, predVal);

          Value inState = b.create<smt::ApplyFuncOp>(
              loc, stateFunctions[pa.stateId], forallInputs);

          return b.create<smt::ImpliesOp>(loc, inState, predBool);
        });
    b.create<smt::AssertOp>(loc, forall);
  }

  b.create<smt::YieldOp>(loc, typeRange, valueRange);
  machineOp.erase();
  return success();
}

struct FSMToSMTPass
    : public circt::impl::ConvertFSMToSMTBase<FSMToSMTPass> {
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

  // Read options from the generated base. Defaults:
  // withTime (false), bitVecOrInt ("bitVec").
  LoweringConfig cfg;
  cfg.withTime = withTime;
  // Normalize option string.
  // if (mode == "int" || mode == "ints" || mode == "integer")
  //   cfg.useBitVec = false;
  // else
  //   cfg.useBitVec = true; // default

  // Optional: set time width; keep 5 as stable default (can be parameterized later).
  cfg.timeWidth = 5;

  for (auto machine : llvm::make_early_inc_range(module.getOps<MachineOp>())) {
    MachineOpConverter converter(b, machine, cfg);
    if (failed(converter.dispatch())) {
      signalPassFailure();
      return;
    }
    // Clean up dangling HW constants left around.
    module.walk([&](circt::hw::ConstantOp cst) { cst.erase(); });
  }
}

} // namespace

std::unique_ptr<mlir::Pass> circt::createConvertFSMToSMTPass() {
  return std::make_unique<FSMToSMTPass>();
}