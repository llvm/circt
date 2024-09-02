//===- FSMToSMT.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Conversion/FSMToSMT.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SMT/SMTOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Transforms/DialectConversion.h"
#include <circt/Dialect/HW/HWTypes.h>
#include <memory>
#include <string_view>
#include <utility>

namespace circt {
#define GEN_PASS_DEF_CONVERTFSMTOSMT
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace fsm;

//===----------------------------------------------------------------------===//
// Conversion pattern
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// Convert FSM to SMT pass
//===----------------------------------------------------------------------===//

namespace {

class MachineOpConverter{
  public: 
    MachineOpConverter(OpBuilder &builder, MachineOp machineOp)
      : machineOp(machineOp), b(builder){}
    LogicalResult dispatch();
  
  private: 

    llvm::SmallVector<mlir::Value> var;

    FailureOr<Value> convertArgsAndVars(MachineOp machineOp, llvm::SmallVector<mlir::Value> vars);


    // converts StateOp to smt::function_decl
    // FailureOr<Value> convertState(StateOp state, llvm::SmallVector<mlir::Value> vars);
  MachineOp machineOp;
  OpBuilder &b;
};
} //namespace 

struct Transition{
  int from;
  int to;
  bool hasGuard, hasAction, hasOutput;
  smt::DeclareFunOp activeFun;
  Region *guard, *action, *output;
};


int insertStates(llvm::SmallVector<std::string> &states, const std::string& st){
  for(auto [id, s]: llvm::enumerate(states)){
    if(s == st)
      return id;
  }
  states.push_back(st);
  return states.size()-1;
}

circt::smt::IntPredicate getSmtPred(circt::comb::ICmpPredicate cmpPredicate){
  switch (cmpPredicate) {
      case comb::ICmpPredicate::slt:
          return smt::IntPredicate::lt;
      case comb::ICmpPredicate::sle:
          return smt::IntPredicate::le;
      case comb::ICmpPredicate::sgt:
          return smt::IntPredicate::gt;
      case comb::ICmpPredicate::sge:
          return smt::IntPredicate::ge;
      case comb::ICmpPredicate::ult:
          return smt::IntPredicate::lt;
      case comb::ICmpPredicate::ule:
          return smt::IntPredicate::le;
      case comb::ICmpPredicate::ugt:
          return smt::IntPredicate::gt;
      case comb::ICmpPredicate::uge:
          return smt::IntPredicate::ge;
  }
}


mlir::Value getCombValue(Operation &op, Location &loc, OpBuilder &b, llvm::SmallVector<mlir::Value> args){

  if (auto addOp = llvm::dyn_cast<comb::AddOp>(op))
    return b.create<smt::IntAddOp>(loc, b.getType<smt::IntType>(), args);
  if (auto andOp = llvm::dyn_cast<comb::AndOp>(op))
    return b.create<smt::AndOp>(loc, b.getType<smt::BoolType>(), args);
  if (auto xorOp = llvm::dyn_cast<comb::XorOp>(op))
    return b.create<smt::XOrOp>(loc, b.getType<smt::BoolType>(), args);
  if (auto orOp = llvm::dyn_cast<comb::OrOp>(op))
    return b.create<smt::AndOp>(loc, b.getType<smt::BoolType>(), args);
  if (auto mulOp = llvm::dyn_cast<comb::MulOp>(op))
    return b.create<smt::IntMulOp>(loc, b.getType<smt::IntType>(), args);
  if (auto icmp = llvm::dyn_cast<comb::ICmpOp>(op)){
    if(icmp.getPredicate() == circt::comb::ICmpPredicate::eq){
      auto tmp = b.create<smt::EqOp>(loc, args);
      return tmp;
    } else if(icmp.getPredicate() == circt::comb::ICmpPredicate::ne){
      auto tmp = b.create<smt::DistinctOp>(loc, args);
      return tmp;
    } else {
      auto predicate = getSmtPred(icmp.getPredicate());
      auto tmp = b.create<smt::IntCmpOp>(loc, predicate, args[0], args[1]);
      return tmp;
    }
  }
}

mlir::Value getSmtValue(mlir::Value op, llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> fsmArgVals, OpBuilder &b, Location &loc){
  llvm::outs()<<"\n\ngetting value of "<<op;
  // op can be a constant
  if (auto constop = dyn_cast<hw::ConstantOp>(op.getDefiningOp())){
    return b.create<smt::IntConstantOp>(loc, constop.getValueAttr());
  }

  if (op.getDefiningOp()->getName().getDialect()->getNamespace() == "comb"){
    // op can be the result of a comb operation 
    auto op1 = getSmtValue(op.getDefiningOp()->getOperand(0), fsmArgVals, b, loc);
    llvm::outs()<<"\n\ngotten value of "<<op1;
    auto op2 = getSmtValue(op.getDefiningOp()->getOperand(1), fsmArgVals, b, loc);
    llvm::outs()<<"\n\ngotten value of "<<op2;
    llvm::SmallVector<mlir::Value> combArgs = {op1, op2};
    return getCombValue(*op.getDefiningOp(), loc, b, combArgs);
  }
  // op can be an arg/var of the fsm
  for (auto fav: fsmArgVals)
    if (op == fav.first){
      llvm::outs()<<"\n\n\nfoooooounbd \n"<<fav.second<<", "<<fav.first;
      return fav.second;
    } 
}


Transition parseTransition(fsm::TransitionOp t, int from, llvm::SmallVector<std::string> &states, 
    llvm::SmallVector<mlir::Type> argVarTypes,
    Location &loc, OpBuilder &b){
  mlir::StringAttr acFunName = b.getStringAttr(("t"+std::to_string(from)+std::to_string(insertStates(states, t.getNextState().str()))));
  auto range = b.getType<smt::BoolType>();
  smt::DeclareFunOp acFun = b.create<smt::DeclareFunOp>(loc, b.getType<smt::SMTFuncType>(argVarTypes, range), acFunName);
  Transition tr = {.from = from, .to = insertStates(states, t.getNextState().str()), .activeFun = acFun};
  if (!t.getGuard().empty()){
    tr.hasGuard = true;
    tr.guard = &t.getGuard();
  }
  if(!t.getAction().empty()){
    tr.hasAction = true;
    tr.action = &t.getAction();
  }
  // todo: output
  return tr;
}


LogicalResult MachineOpConverter::dispatch(){
  b.setInsertionPoint(machineOp);
  auto loc = machineOp.getLoc();
  auto args = machineOp.getArguments();
  llvm::SmallVector<mlir::Type> argVarTypes;
  llvm::SmallVector<mlir::Value> argVars;
  int numArgs = 0;


  // everything is an Int (even i1) because we want to have everything in the same structure
  // and we can not mix ints and bools in the same vec
  for (auto a : args){
    auto intVal = b.getType<smt::IntType>();
    // push twice because we will need it after to model the "next value" of inputs
    argVarTypes.push_back(intVal);
    argVarTypes.push_back(intVal);
    argVars.push_back(a);
    argVars.push_back(a);
  }
  numArgs = argVars.size();

  for (auto variableOp : machineOp.front().getOps<fsm::VariableOp>()) {
    auto intVal = b.getType<smt::IntType>();
    argVarTypes.push_back(intVal);
    argVars.push_back(variableOp->getOpResult(0));
  }

  llvm::SmallVector<Transition> transitions;

  llvm::SmallVector<std::string> states;

  // populate states vector, each state has its unique index that is used to populate transitions, too
  insertStates(states, "supp");
  
  // the first state is a support one we add to ensure that there is one unique initial transition activated
  // as initial condition of the fsm
  std::string initialState = machineOp.getInitialState().str();

  insertStates(states, initialState);

  // add time

  argVarTypes.push_back(b.getType<smt::IntType>());

  // the "fake" initial state is connect to the real one with a transition with no guards nor action

  mlir::StringAttr acFunName = b.getStringAttr(("t01"));
  auto range = b.getType<smt::BoolType>();
  smt::DeclareFunOp acFun = b.create<smt::DeclareFunOp>(loc, b.getType<smt::SMTFuncType>(argVarTypes, range), acFunName);

  Transition support = {.from = 0, .to = 1, .hasGuard = false, .hasAction = false, .hasOutput = false, .activeFun =acFun};

  transitions.push_back(support);

  // todo: populate outputs

  // populate transitions vector

  for (auto stateOp : machineOp.front().getOps<fsm::StateOp>()) {
    auto fromState = insertStates(states, stateOp.getName().str());
    for (auto tr: stateOp.getTransitions().front().getOps<fsm::TransitionOp>()){
      auto t = parseTransition(tr, fromState, states, argVarTypes, loc, b);
      transitions.push_back(t);
      // push back function
      // transitionActive.push_back(ValueParamT Elt)
    }
  }
  llvm::outs()<<"\n\nbefore everything\n\n";



  // create solver region

  for(auto [id1, t1] : llvm::enumerate(transitions)){
    for(auto [id2, t2] : llvm::enumerate(transitions)){
      if(id1!=id2 && t1.to == t2.from){
        // each implication op is in the same region
        
        auto action = [&t1, &loc, this, &argVars, &numArgs](llvm::SmallVector<mlir::Value> args) -> llvm::SmallVector<mlir::Value> {
          llvm::outs()<<"\nargs size: "<<args.size();
          llvm::outs()<<"\nvarArgs size: "<<argVars.size();

          if (t1.hasAction) {
            llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> avToSmt;
            llvm::SmallVector<mlir::Value> updatedSmtValues;

            for(auto [av, a] : llvm::zip(argVars, args))
              avToSmt.push_back({av, a});
            for (auto [j, uv]: llvm::enumerate(avToSmt)){
              if(int(j) < numArgs*2){ // arguments 
                llvm::outs()<<"\nupdating: "<<uv.second;
                updatedSmtValues.push_back(uv.second);
              } else { // only variables can be updated and time is updated separately
                bool found = false;
                // look for updates in the region
                for(auto &op: t1.action->getOps()){
                  if (auto updateOp = dyn_cast<fsm::UpdateOp>(op)){
                    llvm::outs()<<"\n\nfound update op: "<<updateOp;
                    auto updatedVal = getSmtValue(updateOp->getOperand(1), avToSmt, b, loc);
                    llvm::outs()<<"\n\nupdated with: "<<updatedVal;

                    updatedSmtValues.push_back(updatedVal);
                    found = true;
                  }
                }
                if(!found) // the value is not updated in the region 
                  updatedSmtValues.push_back(uv.second);
              }
            }
            mlir::IntegerAttr intAttr = b.getI32IntegerAttr(1);
            auto c1 = b.create<smt::IntConstantOp>(loc, intAttr);
            llvm::SmallVector<mlir::Value> timeArgs = {args.back(), c1};
            auto newTime = b.create<smt::IntAddOp>(loc, b.getType<smt::IntType>(), timeArgs);
            updatedSmtValues.push_back(newTime);
            llvm::outs()<<"\nupdating time: "<<newTime;
            llvm::outs()<<"\nupDATED VALUES: "<<updatedSmtValues.size()<<"\n";
            for (auto uvs: updatedSmtValues)
              llvm::outs()<<uvs<<"\n";
            return updatedSmtValues;
          } else {
            llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> avToSmt;
            llvm::SmallVector<mlir::Value> updatedSmtValues;

            for(auto [av, a] : llvm::zip(argVars, args))
              avToSmt.push_back({av, a});
            for (auto [j, uv]: llvm::enumerate(avToSmt)){
              updatedSmtValues.push_back(uv.second);
            }
            mlir::IntegerAttr intAttr = b.getI32IntegerAttr(1);
            auto c1 = b.create<smt::IntConstantOp>(loc, intAttr);
            llvm::SmallVector<mlir::Value> timeArgs = {args.back(), c1};
            auto newTime = b.create<smt::IntAddOp>(loc, b.getType<smt::IntType>(), timeArgs);
            updatedSmtValues.push_back(newTime);
            llvm::outs()<<"\n2updating time: "<<newTime;
            llvm::outs()<<"\n2pDATED VALUES: "<<updatedSmtValues.size();
            for (auto uvs: updatedSmtValues)
              llvm::outs()<<uvs;
            return updatedSmtValues;
          }
        };

        auto guard1 = [&t1, &loc, this, &argVars](llvm::SmallVector<mlir::Value> args) -> mlir::Value {
          if (t1.hasGuard){
            llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> avToSmt;
            for(auto [av, a] : llvm::zip(argVars, args))
              avToSmt.push_back({av, a});
            for(auto &op: t1.guard->getOps())
              if (auto retOp = dyn_cast<fsm::ReturnOp>(op))
                return getSmtValue(retOp->getOperand(0), avToSmt, b, loc);
          } else {
            return b.create<smt::BoolConstantOp>(loc, true);
          }
        };

        auto guard2 = [&t2, &loc, this, &argVars](llvm::SmallVector<mlir::Value> args) -> mlir::Value {
          if (t2.hasGuard){
            llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> avToSmt;
            for(auto [av, a] : llvm::zip(argVars, args))
              avToSmt.push_back({av, a});
            for(auto &op: t2.guard->getOps())
              if (auto retOp = dyn_cast<fsm::ReturnOp>(op))
                return getSmtValue(retOp->getOperand(0), avToSmt, b, loc);
          } else {
            return b.create<smt::BoolConstantOp>(loc, true);
          }
        };

        auto forall = b.create<smt::ForallOp>(loc, argVarTypes, [&guard1, &guard2, &action, &t1, &t2](OpBuilder &b, Location loc, ValueRange args) { 
              auto t1ac = b.create<smt::ApplyFuncOp>(loc, t1.activeFun, args);
              auto actionedArgs = action(args);
              llvm::outs()<<"\nactioned args: \n";
              for(auto aa: actionedArgs)
                llvm::outs()<<aa<<"\n";
              auto rhs = b.create<smt::ApplyFuncOp>(loc, t2.activeFun, action(args));
              auto t1AndGuard1 = b.create<smt::AndOp>(loc, t1ac, guard1(args));
              auto lhs = b.create<smt::AndOp>(loc, t1AndGuard1, guard2(args));
              return b.create<smt::ImpliesOp>(loc, lhs, rhs); 
        });

      }
    }
  }


  // add mutual exclusion

  // machineOp.erase();

  return success();
}

namespace {
struct FSMToSMTPass : public circt::impl::ConvertFSMToSMTBase<FSMToSMTPass> {
  void runOnOperation() override;
};


void FSMToSMTPass::runOnOperation() {

  auto module = getOperation();
  auto b = OpBuilder(module);

  // // only continue if at least one fsm exists

  // auto machineOps = to_vector(module.getOps<fsm::MachineOp>());
  // if (machineOps.empty()) {
  //   markAllAnalysesPreserved();
  //   return;
  // }


  for (auto machine : llvm::make_early_inc_range(module.getOps<MachineOp>())) {
    MachineOpConverter converter(b, machine);

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