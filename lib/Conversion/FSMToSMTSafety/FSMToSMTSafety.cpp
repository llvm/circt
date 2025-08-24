//===- FSMToSMT.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "circt/Conversion/FSMToSMTSafety.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SMT/IR/SMTTypes.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <circt/Dialect/HW/HWTypes.h>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

namespace circt {
#define GEN_PASS_DEF_CONVERTFSMTOSMTSAFETY
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace fsm;

//===----------------------------------------------------------------------===//
// Convert FSM to SMT pass
//===----------------------------------------------------------------------===//

namespace {

class MachineOpConverter {
public:
  MachineOpConverter(OpBuilder &builder, MachineOp machineOp)
      : machineOp(machineOp), b(builder) {}
  LogicalResult dispatch();

private:
  MachineOp machineOp;
  OpBuilder &b;
};
} // namespace

struct Transition {
  int from;
  int to;
  bool hasGuard, hasAction, hasOutput;
  Region *guard, *action, *output;
};

int insertStates(llvm::SmallVector<std::string> &states, std::string &st) {
  for (auto [id, s] : llvm::enumerate(states)) {
    if (s == st) {
      return id;
    }
  }
  states.push_back(st);
  return states.size() - 1;
}

void printFsmArgVals(
    const llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> &fsmArgVals) {
  for (auto [v1, v2] : fsmArgVals)
    llvm::outs() << "\n\nv1: " << v1 << ", v2: " << v2;
}

mlir::smt::BVCmpPredicate getSmtPred(circt::comb::ICmpPredicate cmpPredicate) {
  switch (cmpPredicate) {
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
}

mlir::Value getCombValue(Operation &op, Location &loc, OpBuilder &b, llvm::SmallVector<mlir::Value> args){
  llvm::SmallVector<int> widths;
  for (auto arg : args){
    // assert(arg.getType().isA<smt::BitVectorType>)
    llvm::outs() << "\n\narg: " << arg;
    auto a = llvm::dyn_cast<smt::BitVectorType>(arg.getType());
    if(!a){
      llvm::outs() << "a is null";
    }
    llvm::outs()<<"\n\ntype: "<<arg;
    widths.push_back(a.getWidth());
  }
  // we need to modulo all the operations considering the width of the mlir value!
  if (auto addOp = mlir::dyn_cast<comb::AddOp>(op)){
    return b.create<smt::BVAddOp>(loc, b.getType<smt::BitVectorType>(widths[0]), args);
  }
  if (auto andOp = mlir::dyn_cast<comb::AndOp>(op))
    return b.create<smt::BVAndOp>(loc, b.getType<smt::BitVectorType>(widths[0]), args);
  if (auto xorOp = mlir::dyn_cast<comb::XorOp>(op))
    return b.create<smt::BVXOrOp>(loc, b.getType<smt::BitVectorType>(widths[0]), args);
  if (auto orOp = mlir::dyn_cast<comb::OrOp>(op))
    return b.create<smt::BVOrOp>(loc, b.getType<smt::BitVectorType>(widths[0]), args);
  if (comb::MuxOp m = mlir::dyn_cast<comb::MuxOp>(op)){
    
    if(args.size() != 3){
      for(auto arg : args){
        llvm::outs() << "\n\nmux arg: " << arg;
      }
    }
    // assert(args.size() == 3 && "MuxOp should have 3 arguments");
    return b.create<smt::IteOp>(loc, b.getType<smt::BitVectorType>(widths[1]), args);
  }
  if (auto concatOp = mlir::dyn_cast<comb::ConcatOp>(op))
    return b.create<smt::ConcatOp>(loc, b.getType<smt::BitVectorType>(widths[0] + widths[1]), args);
  if (auto mulOp = mlir::dyn_cast<comb::MulOp>(op))
  {
    return b.create<smt::BVMulOp>(loc, b.getType<smt::BitVectorType>(widths[0]), args);
  }
  if (auto icmp = mlir::dyn_cast<comb::ICmpOp>(op)){
    if(icmp.getPredicate() == circt::comb::ICmpPredicate::eq){
      auto eq = b.create<smt::EqOp>(loc, args);
      return eq; //b.getType<smt::BitVectorType>(widths[0], eq)
    }
    if(icmp.getPredicate() == circt::comb::ICmpPredicate::ne){
      return b.create<smt::DistinctOp>(loc, args);
    }
    auto predicate = getSmtPred(icmp.getPredicate());
    return b.create<smt::BVCmpOp>(loc, predicate, args[0], args[1]);
  }
}



mlir::Value getSmtValue(mlir::Value op, const llvm::SmallVector<std::pair<mlir::Value, mlir::Value>>& fsmArgVals, OpBuilder &b, Location &loc){
  // op can be an arg/var of the fsm
  printFsmArgVals(fsmArgVals);

  llvm::outs() << "\n\nlooking for : " << op;


  for (auto fav: fsmArgVals){
    if (op == fav.first){
      return fav.second;
    } 
  }
  if (op.getDefiningOp()->getName().getDialect()->getNamespace() == "comb"){
    // op can be the result of a comb operation 
    auto op1 = getSmtValue(op.getDefiningOp()->getOperand(0), fsmArgVals, b, loc);
    auto op2 = getSmtValue(op.getDefiningOp()->getOperand(1), fsmArgVals, b, loc);
    // llvm::SmallVector<mlir::Value> combArgs = {op1, op2};
    llvm::SmallVector<mlir::Value> combArgs;
    for(auto arg : op.getDefiningOp()->getOperands()){
      auto toRet = getSmtValue(arg, fsmArgVals, b, loc);
      combArgs.push_back(toRet);
    }
    return getCombValue(*op.getDefiningOp(), loc, b, combArgs);
  }
  // op can be a constant
  if (auto constop = mlir::dyn_cast<hw::ConstantOp>(op.getDefiningOp())){
    return b.create<smt::BVConstantOp>(loc, constop.getValue());
  }
  assert(false);
}


Transition parseTransition(fsm::TransitionOp t, int from,
                           llvm::SmallVector<std::string> &states,
                           Location &loc, OpBuilder &b) {
  std::string nextState = t.getNextState().str();
  // llvm::outs()<<"\n\ntransition from "<<states[from]<<" to
  // "<<states[insertStates(states, nextState)]; t->dump();
  Transition tr = {.from = from, .to = insertStates(states, nextState)};
  if (!t.getGuard().empty()) {
    tr.hasGuard = true;
    tr.guard = &t.getGuard();
  }
  if (!t.getAction().empty()) {
    tr.hasAction = true;
    tr.action = &t.getAction();
  }
  // todo: output
  return tr;
}

Region* getOutputRegion(llvm::SmallVector<std::pair<mlir::Region*, int>> outputOfStateId, int stateId){
  for (auto oid: outputOfStateId)
    if (stateId == oid.second)
      return oid.first;
  abort();
}

LogicalResult MachineOpConverter::dispatch() {

  b.setInsertionPoint(machineOp);
  auto loc = machineOp.getLoc();
  auto machineArgs = machineOp.getArguments();

  llvm::SmallVector<mlir::Type> argVarWidths;

  llvm::SmallVector<mlir::Value> argVars;

  int numArgs = 0;
  int numOut = 0;

  mlir::TypeRange typeRange;
  mlir::ValueRange valueRange;

  auto solver = b.create<smt::SolverOp>(loc, typeRange, valueRange);

  solver.getBodyRegion().emplaceBlock();

  b.setInsertionPointToStart(solver.getBody());

  // fsm arguments
  for (auto a : machineArgs){
    // auto cast = llvm::dyn_cast<smt::BitVectorType>(a.getType());
    argVarWidths.push_back(b.getType<smt::BitVectorType>(a.getType().getIntOrFloatBitWidth()));
    argVars.push_back(a);
    numArgs++;
  }

  // fsm outputs
  if (machineOp.getResultTypes().size() > 0){
    for (auto o : machineOp.getResultTypes()){
      auto intVal = b.getType<smt::BitVectorType>(o.getIntOrFloatBitWidth());
      argVarWidths.push_back(intVal);
      auto ov = b.create<smt::BVConstantOp>(loc, 0, o.getIntOrFloatBitWidth());
      argVars.push_back(ov);
      numOut++;
    }
  }

  llvm::SmallVector<int> varInitValues;

  // fsm variables
  for (auto variableOp : machineOp.front().getOps<fsm::VariableOp>()) {
    auto intVal = b.getType<smt::BitVectorType>(variableOp.getType().getIntOrFloatBitWidth());
    auto initVal = variableOp.getInitValueAttr();
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(initVal))
      varInitValues.push_back(intAttr.getInt());
    argVarWidths.push_back(intVal);
    argVars.push_back(variableOp->getOpResult(0));
  }
  llvm::SmallVector<Transition> transitions;
  llvm::SmallVector<mlir::Value> stateFunctions;

  llvm::SmallVector<std::string> states;
  llvm::SmallVector<std::pair<mlir::Region*, int>> outputOfStateId;

  // populate states vector, each state has its unique index that is used to
  // populate transitions, too

  // the first state is a support one we add to ensure that there is one unique
  // initial transition activated as initial condition of the fsm
  std::string initialState = machineOp.getInitialState().str();

  insertStates(states, initialState);

  // time is an 32-bit bitvec
  argVarWidths.push_back(b.getType<smt::BitVectorType>(32));

  // populate state functions and transitions vector
  for (auto stateOp : machineOp.front().getOps<fsm::StateOp>()) {
    std::string stateName = stateOp.getName().str();
    mlir::StringAttr acFunName =
        b.getStringAttr(("F_" + stateOp.getName().str()));
    auto range = b.getType<smt::BoolType>();
    smt::DeclareFunOp acFun = b.create<smt::DeclareFunOp>(
        loc, b.getType<smt::SMTFuncType>(argVarWidths, range), acFunName);
    stateFunctions.push_back(acFun);
    auto fromState = insertStates(states, stateName);
    outputOfStateId.push_back({&stateOp.getOutput(), fromState});
  }


  // populate vector of transitions
  for (auto stateOp : machineOp.front().getOps<fsm::StateOp>()) {
    std::string stateName = stateOp.getName().str();
    auto fromState = insertStates(states, stateName);
    if (!stateOp.getTransitions().empty()) {
      for (auto tr :
           stateOp.getTransitions().front().getOps<fsm::TransitionOp>()) {
        auto t = parseTransition(tr, fromState, states, loc, b);
        if (!stateOp.getOutput().empty()) {
          t.hasOutput = true;
          t.output = getOutputRegion(outputOfStateId, t.to); // now look for it! &stateOp.getOutput();
        } else {
          t.hasOutput = false;
        }
        transitions.push_back(t);
      }
    }
  }

  // initial condition
  // the key is the number of the state in `stateFunctions`
  llvm::SmallVector<std::pair<int, mlir::Value>> assertions;


  auto forall = b.create<smt::ForallOp>(
      loc, argVarWidths,
      [&varInitValues, &stateFunctions, &numOut, &argVars, &numArgs, &outputOfStateId](OpBuilder &b, Location loc,
                        llvm::SmallVector<mlir::Value> forallArgs) -> mlir::Value {
        llvm::SmallVector<mlir::Value> initArgs;
        // nb. args also has the time
        
        llvm::SmallVector<mlir::Value> outputSmtValues; 
        llvm::SmallVector<mlir::Value> initVarValues; 

        auto initOutputReg = getOutputRegion(outputOfStateId, 0); // the index of the initial state is always zero s

        // first we collect the initial data of variables 
        for (auto [i, a] : llvm::enumerate(forallArgs)) {
          if (int(i) >= numOut + numArgs && int(i) < forallArgs.size() - 1) {
            auto cast = llvm::dyn_cast<smt::BitVectorType>(a.getType());
            llvm::outs()<<"\n\nwidth is: "<<cast.getWidth();
            auto initVarVal = b.create<smt::BVConstantOp>(loc, varInitValues[i - numOut - numArgs], cast.getWidth());
            initVarValues.push_back(initVarVal);
          }
        }

        if (!initOutputReg->empty()){
          llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> avToSmt;
          for (auto [i, a] : llvm::enumerate(argVars)){
            if (int(i) >= numOut + numArgs && int(i) < forallArgs.size() - 1) {
              avToSmt.push_back({a, initVarValues[i - numOut - numArgs]});
            } else {
              avToSmt.push_back({a, a});
            }
          }

          for (auto &op : initOutputReg->getOps()) {
            // todo: check that updates requiring inputs for operations work
            if (auto outputOp = mlir::dyn_cast<fsm::OutputOp>(op)) {
              for (auto outs : outputOp->getOperands()) {
                auto found = false;
                for (auto [i, fav]: llvm::enumerate(avToSmt)){
                  if (outs == fav.first && i < numArgs){
                    outputSmtValues.push_back(forallArgs[i]);
                    found = true;
                  }
                } 
                if(!found) {
                  auto toRet = getSmtValue(outs, avToSmt, b, loc);
                  outputSmtValues.push_back(toRet);
                }
              }
            }
          }
        }

        for (auto [i, a] : llvm::enumerate(forallArgs)) {
            if (int(i) >= numArgs && int(i) < numOut + numArgs && int(i) < forallArgs.size() - 1) { // outputs
              initArgs.push_back(outputSmtValues[i - numArgs]); 
            } else if (int(i) >= numOut + numArgs && int(i) < forallArgs.size() - 1) { // variables
              initArgs.push_back(initVarValues[i - numOut - numArgs]);
            } else {
              initArgs.push_back(a);
            }
          }




        // retrieve output region constraint at the initial state

        auto initTime = b.create<smt::BVConstantOp>(loc, 0, 32);
        auto lhs = b.create<smt::EqOp>(loc, forallArgs.back(), initTime);
        auto rhs = b.create<smt::ApplyFuncOp>(loc, stateFunctions[0], initArgs);

        return b.create<smt::ImpliesOp>(loc, lhs, rhs);
      });

  b.create<smt::AssertOp>(loc, forall);

  // create solver region


  for (auto [id1, t1] : llvm::enumerate(transitions)) {
    //   // each implication op is in the same region
    auto action = [&t1, &loc, this, &argVars,
                   &numArgs, &assertions](llvm::SmallVector<mlir::Value> actionArgs)
        -> llvm::SmallVector<mlir::Value> {
      // args includes the time, argvars does not
      // update outputs if possible first
      llvm::SmallVector<mlir::Value> outputSmtValues;

      if (t1.hasOutput) {
        llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> avToSmt;
        for (auto [id, av] : llvm::enumerate(argVars))
          avToSmt.push_back({av, actionArgs[id]});
        for (auto &op : t1.output->getOps()) {
          // todo: check that updates requiring inputs for operations work
          if (auto outputOp = mlir::dyn_cast<fsm::OutputOp>(op)) {
            for (auto outs : outputOp->getOperands()) {
              auto toRet =
                  getSmtValue(outs, avToSmt, b, loc);
              outputSmtValues.push_back(toRet);
            }
          }
          // if we find a verif.assert operation, we store the SMT formula it checks in `assertions`
          if (auto outputOp = mlir::dyn_cast<verif::AssertOp>(op)) {
            auto operand = outputOp->getOperand(0);
            auto assertion = getSmtValue(operand, avToSmt, b, loc);
            assertions.push_back({t1.from, assertion});
          }
        }
      }

      if (t1.hasAction) {
        llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> avToSmt;
        llvm::SmallVector<mlir::Value> updatedSmtValues;
        // argvars has both inputs and time
        for (auto [id, av] : llvm::enumerate(argVars))
          avToSmt.push_back({av, actionArgs[id]});
        for (auto [j, uv] : llvm::enumerate(avToSmt)) {
          // only variables can be updated and time is updated separately
          bool found = false;
          // look for updates in the region
          for (auto &op : t1.action->getOps()) {
            // todo: check that updates requiring inputs for operations work
            if (auto updateOp = mlir::dyn_cast<fsm::UpdateOp>(op)) {
              if (updateOp->getOperand(0) == uv.first) {
                auto updatedVal = getSmtValue(updateOp->getOperand(1), avToSmt,
                                              b, loc);
                updatedSmtValues.push_back(updatedVal);
                found = true;
              }
            }
          }
          if (!found) // the value is not updated in the region
            updatedSmtValues.push_back(uv.second);
        }

        // update time
        auto c1 = b.create<smt::BVConstantOp>(loc, 1, 32);
        llvm::SmallVector<mlir::Value> timeArgs = {actionArgs.back(), c1};
        auto newTime = b.create<smt::BVAddOp>(
            loc, b.getType<smt::BitVectorType>(32), timeArgs);
        updatedSmtValues.push_back(newTime);
        // push output values
        for (auto [i, outputVal] : llvm::enumerate(outputSmtValues)) {
          updatedSmtValues[numArgs + i] = outputVal;
        }
        return updatedSmtValues;
      }
      llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> avToSmt;
      llvm::SmallVector<mlir::Value> updatedSmtValues;
      for (auto [id, av] : llvm::enumerate(argVars))
        avToSmt.push_back({av, actionArgs[id]});
      for (auto [j, uv] : llvm::enumerate(avToSmt)) {
        updatedSmtValues.push_back(uv.second);
      }
      // update time
      // mlir::IntegerAttr intAttr = b.getI32IntegerAttr(1);
      auto oAttr = b.getI32IntegerAttr(1);
        auto c1 = b.create<smt::IntConstantOp>(loc, oAttr);
        llvm::SmallVector<mlir::Value> timeArgs = {actionArgs.back(), c1};
        auto newTime = b.create<smt::BVAddOp>(
            loc, b.getType<smt::BitVectorType>(32), timeArgs);
        updatedSmtValues.push_back(newTime);
      // push output values
      for (auto [i, outputVal] : llvm::enumerate(outputSmtValues)) {
        updatedSmtValues[numArgs + i] = outputVal;
      }
      return updatedSmtValues;
    };

    auto guard1 = [&t1, &loc, this, &argVars](
                      llvm::SmallVector<mlir::Value> guardArgs) -> mlir::Value {
      if (t1.hasGuard) {
        llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> avToSmt;
        for (auto [av, a] : llvm::zip(argVars, guardArgs))
          avToSmt.push_back({av, a});
        for (auto &op : t1.guard->getOps())
          if (auto retOp = mlir::dyn_cast<fsm::ReturnOp>(op)) {
            auto tmp = getSmtValue(retOp->getOperand(0), avToSmt,
                                   b, loc);
            return tmp;
          }
      } else {
        return b.create<smt::BoolConstantOp>(loc, true);
      }
    };


    llvm::SmallVector<mlir::Type> forallargVarWidths;
    for (auto [id, avt] : llvm::enumerate(argVarWidths)){
      if (id < numArgs){
        forallargVarWidths.push_back(avt);
        forallargVarWidths.push_back(avt);
      } else {
        forallargVarWidths.push_back(avt);
      }
    }

    auto forall = b.create<smt::ForallOp>(
        loc, forallargVarWidths,
        [&guard1, &action, &t1, &stateFunctions, &numArgs,
         &numOut](OpBuilder &b, Location loc, ValueRange forallDoubleInputs) {
          // split new and old arguments

          llvm::SmallVector<mlir::Value> startingStateArgs;
          llvm::SmallVector<mlir::Value> arrivingStateArgs;
          for (auto [idx, fdi] : llvm::enumerate(forallDoubleInputs)){
            if (idx < numArgs*2){
              if (idx % 2 == 1){
                startingStateArgs.push_back(fdi);
              }else{ 
                arrivingStateArgs.push_back(fdi);
              }
            } else {
              startingStateArgs.push_back(fdi);
              arrivingStateArgs.push_back(fdi);
            }
          }


          auto t1ac = b.create<smt::ApplyFuncOp>(loc, stateFunctions[t1.from],
                                                 startingStateArgs);
          auto actionedArgs = action(startingStateArgs);
          for(auto [ida, aa] : llvm::enumerate(actionedArgs))
            if (ida < numArgs)
              actionedArgs[ida] = arrivingStateArgs[ida];

          auto rhs = b.create<smt::ApplyFuncOp>(loc, stateFunctions[t1.to],
                                                actionedArgs);
          auto guard = guard1(startingStateArgs);
            
          auto lhs = b.create<smt::AndOp>(loc, t1ac, guard);
          auto ret = b.create<smt::ImpliesOp>(loc, lhs, rhs);
          return ret;
          
        });

    b.create<smt::AssertOp>(loc, forall);
  }

  for (auto assertion : assertions) { 

    auto forall = b.create<smt::ForallOp>(
        loc, argVarWidths,
        [&stateFunctions, &assertion](OpBuilder &b, Location loc, ValueRange forallInputs) {


          auto stateFun = b.create<smt::ApplyFuncOp>(loc, stateFunctions[assertion.first],
                                                 forallInputs);
          // SMT formula representing the negation of the safety property expressed in verif.assert
          auto negatedProperty = b.create<smt::NotOp>(loc, assertion.second);
          auto rhs = b.create<smt::BoolConstantOp>(loc, false);
            
          auto lhs = b.create<smt::AndOp>(loc, stateFun, negatedProperty);
          auto ret = b.create<smt::ImpliesOp>(loc, lhs, rhs);
          return ret;
          
      });

    b.create<smt::AssertOp>(loc, forall);
  }


  // b.getBlock()->dump();

  b.create<smt::YieldOp>(loc, typeRange, valueRange);

  machineOp.erase();

  return success();
}

namespace {
struct FSMToSMTSafetyPass : public circt::impl::ConvertFSMToSMTSafetyBase<FSMToSMTSafetyPass> {
  void runOnOperation() override;
};

void FSMToSMTSafetyPass::runOnOperation() {

  auto module = getOperation();
  auto b = OpBuilder(module);

  // // only continue if at least one fsm exists

  auto machineOps = to_vector(module.getOps<fsm::MachineOp>());
  if (machineOps.empty()) {
    markAllAnalysesPreserved();
    return;
  }

  for (auto machine : llvm::make_early_inc_range(module.getOps<MachineOp>())) {
    MachineOpConverter converter(b, machine);

    if (failed(converter.dispatch())) {
      signalPassFailure();
      return;
    }
  }
}
} // namespace

std::unique_ptr<mlir::Pass> circt::createConvertFSMToSMTSafetyPass() {
  return std::make_unique<FSMToSMTSafetyPass>();
}