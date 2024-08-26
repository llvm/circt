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

mlir::Value getCombValue(Operation &op, Location &loc, OpBuilder &b){
  if (auto addOp = llvm::dyn_cast<comb::AddOp>(op)){
    return b.create<smt::AndOp>(loc, addOp->getOperands());
  }
  if (auto andOp = llvm::dyn_cast<comb::AndOp>(op))
    return b.create<smt::AndOp>(loc, andOp->getOperands());
  if (auto xorOp = llvm::dyn_cast<comb::XorOp>(op))
    return b.create<smt::AndOp>(loc, xorOp->getOperands());
  if (auto orOp = llvm::dyn_cast<comb::OrOp>(op))
    return b.create<smt::AndOp>(loc, orOp->getOperand(0), orOp->getOperand(1));
  if (auto mulOp = llvm::dyn_cast<comb::MulOp>(op))
    return b.create<smt::AndOp>(loc, mulOp->getOperand(0), mulOp->getOperand(1));
  if (auto icmp = llvm::dyn_cast<comb::ICmpOp>(op)){
    circt::comb::ICmpPredicate predicate = icmp.getPredicate();
      switch (predicate) {
      case circt::comb::ICmpPredicate::eq:
        return b.create<smt::EqOp>(loc, icmp->getOperand(0), icmp->getOperand(1));
      case circt::comb::ICmpPredicate::ne:
        return b.create<smt::DistinctOp>(loc, icmp->getOperand(0), icmp->getOperand(1));
      }
    }
}

Transition parseTransition(fsm::TransitionOp t, int from, llvm::SmallVector<std::string> &states, 
    const SmallVector<std::pair<mlir::Value, mlir::Value>> variables, llvm::SmallVector<mlir::Type> argVarTypes,
    Location &loc, OpBuilder &b, 
    llvm::SmallVector<mlir::Value> vecVal){
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

llvm::SmallVector<mlir::Value> getAction(Transition &t, OpBuilder &b, llvm::SmallVector<mlir::Value> &argVars, Location &loc){
  llvm::SmallVector<mlir::Value> updatedValues;
  llvm::SmallVector<mlir::Value> tmpValues;
  for (auto v: argVars){
    bool found = false;
    for(auto &op: t.action->getOps()){
      if(!found){
        if (auto updateOp = dyn_cast<fsm::UpdateOp>(op)){
          if(v == updateOp->getOperand(0)){
            // either we've defined this already and is in tmpValues 
            for(auto tv : tmpValues){
              if(tv == updateOp->getOperand(1)){
                // now generate an appropriate smt expression for this 
                updatedValues.push_back(tv);
                found = true;
              }
            }

            // or it's a variable/arg
            if(!found){
              for(auto av : argVars){
                if(av == updateOp->getOperand(1)){
                  // now generate an appropriate smt expression for this 
                  updatedValues.push_back(av);
                  found = true;
                }
              }
            }
            // or it's a constant 
            if(!found){
              if (auto constop = dyn_cast<hw::ConstantOp>(updateOp->getOperand(1).getDefiningOp())) {
                updatedValues.push_back(constop);
              }
            }
          }
        } else {
          tmpValues.push_back(op.getResult(0));
        }
      }
    }
  }
  return updatedValues;

}

mlir::Value getGuard(Transition &t, OpBuilder &b, llvm::SmallVector<mlir::Value> &argVars, Location &loc){
  llvm::SmallVector<mlir::Value> updatedValues;
  llvm::SmallVector<mlir::Value> tmpValues;
    for(auto &op: t.guard->getOps()){
      if (auto retOp = dyn_cast<fsm::ReturnOp>(op)){
        for(auto tv : tmpValues){
          if(tv == retOp->getOperand(0)){
            return tv;
          }
        }
        for(auto av : argVars){
          if(av == retOp->getOperand(0)){
            return av;
          }
        }
        if (auto constop = dyn_cast<hw::ConstantOp>(retOp->getOperand(0).getDefiningOp())) {
          return constop;
        }


      } else {
        tmpValues.push_back(op.getResult(0));
      }
  }

}

LogicalResult MachineOpConverter::dispatch(){
  b.setInsertionPoint(machineOp);
  auto loc = machineOp.getLoc();
  auto args = machineOp.getArguments();
  llvm::SmallVector<mlir::Type> argVarTypes;
  llvm::SmallVector<mlir::Value> argVars;
  llvm::SmallVector<mlir::Value> val;
  llvm::SmallVector<int> initVal;


  SmallVector<std::pair<mlir::Value, mlir::Value>> variables;
  // everything is an Int (even i1) because we want to have everything in the same structure
  // and we can not mix ints and bools in the same vec
  for (auto a : args){
    auto intVal = b.getType<smt::IntType>();
    argVarTypes.push_back(intVal);
    argVars.push_back(a);
  }

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

  // the "fake" initial state is connect to the real one with a transition with no guards nor action

  mlir::StringAttr acFunName = b.getStringAttr(("t01"));
  auto range = b.getType<smt::BoolType>();
  smt::DeclareFunOp acFun = b.create<smt::DeclareFunOp>(loc, b.getType<smt::SMTFuncType>(argVarTypes, range), acFunName);

  Transition support = {.from = 0, .to = 1, .hasGuard = false, .hasAction = false, .hasOutput = false, .activeFun =acFun};

  transitions.push_back(support);

  // todo: populate outputs

  // populate transitions vector

  // add time
  llvm::SmallVector<smt::DeclareFunOp> transitionActive;

  for (auto stateOp : machineOp.front().getOps<fsm::StateOp>()) {
    auto fromState = insertStates(states, stateOp.getName().str());
    for (auto tr: stateOp.getTransitions().front().getOps<fsm::TransitionOp>()){
      auto t = parseTransition(tr, fromState, states, variables, argVarTypes, loc, b, argVars);
      transitions.push_back(t);
      // push back function
      // transitionActive.push_back(ValueParamT Elt)
    }
  }

  // time

  argVarTypes.push_back(b.getType<smt::IntType>());

  // create solver region

  for(auto [id1, t1] : llvm::enumerate(transitions)){
    for(auto [id2, t2] : llvm::enumerate(transitions)){
      if(id1!=id2 && t1.to == t2.from){
        // each implication op is in the same region
        llvm::SmallVector<mlir::Value> updatedVec;
        if (t1.hasAction){
           updatedVec = getAction(t1, b, argVars, loc);
          for(auto uv: updatedVec)
            llvm::outs()<<"\nupdated: "<<uv;
        } else {
          updatedVec = argVars;
        }
        mlir::Value g1;
        if (t1.hasGuard){
          g1 = getGuard(t1, b, argVars, loc);
          // g1 is either a comb operation or an arg or a var or a constant
        } else {
          g1 = b.create<hw::ConstantOp>(loc, b.getType<hw::IntType>(), 1);
        }

        mlir::Value g2;
        if (t1.hasGuard){
          g2 = getGuard(t1, b, argVars, loc);
          // g1 is either a comb operation or an arg or a var or a constant
        } else {
          g2 = b.create<hw::ConstantOp>(loc, b.getType<hw::IntType>(), 1);
        }
        // create a forall region whose args are the args and vars, these will be referenced by the implication


        auto action = [&updatedVec, &loc, this, &argVars](llvm::SmallVector<mlir::Value> tmpArgVars)->llvm::SmallVector<mlir::Value>{
          // problem: make sure that the order is right
          // return corresponding smt thingy
          llvm::SmallVector<mlir::Value> ret;
          for (auto uv: updatedVec) {
            bool found = false;
            if(auto constOp = dyn_cast<hw::ConstantOp>(uv.getDefiningOp())){
              ret.push_back(b.create<smt::IntConstantOp>(loc, constOp.getValueAttr()));
              found = true;
            }
            if(!found){
              for (auto [tav, av]: llvm::zip(tmpArgVars,argVars)){
                if (uv == av){
                  ret.push_back(tav);
                  found = true;
                }
              }
            }
            if(!found){
              ret.push_back(getCombValue(*uv.getDefiningOp(), loc, b));
            }
          }
        };

        auto guard1 = [&g1, &loc, this, &argVars](llvm::SmallVector<mlir::Value> tmpArgVars)-> mlir::Value {
          // problem: make sure that the order is right
          // return corresponding smt thingy
          llvm::SmallVector<mlir::Value> ret;
          if(auto constOp = dyn_cast<hw::ConstantOp>(g1.getDefiningOp())){
            return b.create<smt::IntConstantOp>(loc, constOp.getValueAttr());
          }
            for (auto [tav, av]: llvm::zip(tmpArgVars,argVars)){
              if (g1 == av){
                return tav;
              }
            }
            return (getCombValue(*g1.getDefiningOp(), loc, b));
        };

        auto guard2 = [&g2, &loc, this, &argVars](llvm::SmallVector<mlir::Value> tmpArgVars)-> mlir::Value {
          // problem: make sure that the order is right
          // return corresponding smt thingy
          llvm::SmallVector<mlir::Value> ret;
          if(auto constOp = dyn_cast<hw::ConstantOp>(g2.getDefiningOp())){
            return b.create<smt::IntConstantOp>(loc, constOp.getValueAttr());
          }
            for (auto [tav, av]: llvm::zip(tmpArgVars,argVars)){
              if (g2 == av){
                return tav;
              }
            }
            return (getCombValue(*g2.getDefiningOp(), loc, b));
        };

        auto forall = b.create<smt::ForallOp>(loc, argVarTypes, [&guard1, &guard2, &action, &t1, &t2](OpBuilder &b, Location loc, ValueRange args) { 

          auto actionResult = action(args);
          for(auto ar: actionResult){
            llvm::outs()<< "\nar: ";
            llvm::outs()<< ar;
          }

          auto t1ac = b.create<smt::ApplyFuncOp>(loc, t1.activeFun, args);
          auto rhs = b.create<smt::ApplyFuncOp>(loc, t2.activeFun, action(args));
          auto andLhs = b.create<smt::AndOp>(loc, guard1(args), guard2(args));
          auto lhs = b.create<smt::AndOp>(loc, t1ac, andLhs);
          
          // auto g1 
          // auto g2 
          return b.create<smt::ImpliesOp>(loc, lhs, rhs); 
        });
      }
    }
    
  }

  // add mutual exclusion

  machineOp.erase();


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

  b.setInsertionPointToStart(module.getBody());

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