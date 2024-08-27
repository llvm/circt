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

mlir::Value getCombValue(Operation &op, Location &loc, OpBuilder &b, const llvm::SmallVector<mlir::Value> &args){
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
    circt::comb::ICmpPredicate predicate = icmp.getPredicate();
      switch (predicate) {
      case circt::comb::ICmpPredicate::eq:
        return b.create<smt::EqOp>(loc, b.getType<smt::BoolType>(), args);
      case circt::comb::ICmpPredicate::ne:
        return b.create<smt::DistinctOp>(loc, b.getType<smt::BoolType>(), args);
      }
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

int getIndex(llvm::SmallVector<mlir::Value> argVars, mlir::Value v){
  for (auto [i, a] : llvm::enumerate(argVars))
    if (a == v)
      return i;
  return -1;
}

llvm::SmallVector<mlir::Value> getAction(Transition &t, OpBuilder &b, llvm::SmallVector<mlir::Value> argVars, Location &loc){
  llvm::SmallVector<mlir::Value> updatedValues(argVars);
  for (auto [i, v]: llvm::enumerate(argVars)){
    llvm::SmallVector<mlir::Value> tmpValues;
    bool found = false;
    for(auto &op: t.action->getOps()){
      if(!found){
        if (auto updateOp = dyn_cast<fsm::UpdateOp>(op)){
          if(v == updateOp->getOperand(0)){
            // either we've defined this already and is in tmpValues 
            for(auto tv : tmpValues){
              if(tv == updateOp->getOperand(1)){
                // now generate an appropriate smt expression for this 
                updatedValues[i]=tv;
                found = true;
              }
            }
            // or it's a constant 
            if(!found){
              if (auto constop = dyn_cast<hw::ConstantOp>(updateOp->getOperand(1).getDefiningOp())) {
                updatedValues[i]=constop;
              }
            }
            // or it's a variable/arg
            int id = getIndex(argVars, updateOp->getOperand(1));
            if(!found && id != -1)
              updatedValues[i]=argVars[id];
          }
        } else {
          tmpValues.push_back(op.getResult(0));
        }
      }
    }
  }
  return updatedValues;
}


mlir::Value getSmt(Operation &op, llvm::SmallVector<mlir::Value> &args, llvm::SmallVector<mlir::Value> &argVars, llvm::SmallVector<mlir::Value> &tmpSmtVal, OpBuilder &b, Location &loc){
  // atm can only be a comb operation 
  auto operands = op.getOperands();
  llvm::SmallVector<mlir::Value> smtOperands;
  for(auto opr : operands){
    auto found = false;
      // each operand is either a const, an arg/var, another operation
    if (auto constop = dyn_cast<hw::ConstantOp>(opr.getDefiningOp())){
      smtOperands.push_back(b.create<smt::IntConstantOp>(loc, constop.getValueAttr()));
      found = true;
    }
    if(!found){
      for (auto [i, av]: llvm::enumerate(argVars)){
        if (opr == av){
          smtOperands.push_back(args[i]);
          found = true;
        }
      }
    }
    if(!found){
      smtOperands.push_back(getSmt(*opr.getDefiningOp(), args, argVars ,tmpSmtVal, b, loc));
    }
  }
  return getCombValue(op, loc, b, operands);
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
    argVarTypes.push_back(intVal);
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

  // add time

  argVarTypes.push_back(b.getType<smt::IntType>());

  // create solver region

  for(auto [id1, t1] : llvm::enumerate(transitions)){
    for(auto [id2, t2] : llvm::enumerate(transitions)){
      if(id1!=id2 && t1.to == t2.from){
        // each implication op is in the same region
        llvm::SmallVector<mlir::Value> updatedVec(argVars);
        llvm::SmallVector<mlir::Value> tmpArgVars(argVars);

        if (t1.hasAction){
           updatedVec = getAction(t1, b, argVars, loc);
          for(auto uv: updatedVec)
            llvm::outs()<<"\nupdated: "<<uv;
        } else {
          updatedVec = tmpArgVars;
        }
        

        if (t1.hasGuard){
          auto *reg = t1.guard;
          auto guard1 = [&reg, &loc, this, &argVars](llvm::SmallVector<mlir::Value> args)-> mlir::Value {
            // problem: make sure that the order is right
            // return corresponding smt thingy
            llvm::SmallVector<mlir::Value> tmpSmtVal;
            llvm::SmallVector<mlir::Value> tmpVal;
            for(auto &op: reg->getOps()){
              if (auto retOp = dyn_cast<fsm::ReturnOp>(op)){                           
                for(auto [i, tv] : llvm::enumerate(tmpVal)){
                  if(tv == retOp->getOperand(0)){
                    return tmpSmtVal[i];
                  }
                }
                if (auto constop = dyn_cast<hw::ConstantOp>(retOp->getOperand(0).getDefiningOp())) {
                  // must return smt constant
                  return b.create<smt::IntConstantOp>(loc, constop.getValueAttr());
                }
                int id = getIndex(argVars, retOp->getOperand(0));
                return args[id];
              }
              tmpVal.push_back(op.getResult(0));
              tmpSmtVal.push_back(getSmt(*op.getResult(0).getDefiningOp(), args, argVars, tmpSmtVal, b, loc));
            }
          };
        }


        


        // create a forall region whose args are the args and vars, these will be referenced by the implication


        // auto action = [&updatedVec, &loc, this, &argVars](llvm::SmallVector<mlir::Value> args)->llvm::SmallVector<mlir::Value>{
        //   // problem: make sure that the order is right
        //   // return corresponding smt thingy
        //   llvm::outs()<<"\ninside action";
        //   llvm::SmallVector<mlir::Value> ret(updatedVec);
        //   for (auto [i, uv]: llvm::enumerate(updatedVec)) {
        //     llvm::outs()<<"\nlooking for"<<uv;
        //     bool found = false;
        //     if(auto constOp = dyn_cast<hw::ConstantOp>(uv.getDefiningOp())){
        //       ret[i] = b.create<smt::IntConstantOp>(loc, constOp.getValueAttr());
        //       llvm::outs()<<"\n1found"<<ret[i];
        //       found = true;
        //     }
        //     if(!found){
        //       int id = getIndex(argVars, uv);
        //       if (id!=-1){
        //         ret[i] = args[id];
        //         found = true;
        //         llvm::outs()<<"\n2found"<<ret[i];  

        //       }
        //     }
        //     if(!found){
        //       auto a1 = uv.getDefiningOp()->getOperand(0);
        //       auto a2 = uv.getDefiningOp()->getOperand(1);
        //       // now need to find the smt value corresponding to these arguments: either constants, variables, or previously defined in the region 

        //       ret[i] = getCombValue(*uv.getDefiningOp(), loc, b, args);
        //       llvm::outs()<<"\n3found"<<ret[i];
        //       found = true;
        //     }
        //   }
        //   return ret;
        // };

        // auto forall = b.create<smt::ForallOp>(loc, argVarTypes, [&guard1, &guard2, &action, &t1, &t2](OpBuilder &b, Location loc, ValueRange args) { 
        //   llvm::outs()<<"\ninside forall";

        //   for(auto arg: args)
        //     llvm::outs()<<arg;
          
        //   auto actionResult = action(args);
        //   for(auto ar: actionResult){
        //     llvm::outs()<< "\nar: ";
        //     llvm::outs()<< ar;
        //   }

        //   return b.create<smt::ApplyFuncOp>(loc, t1.activeFun, args);
        //   // auto rhs = b.create<smt::ApplyFuncOp>(loc, t2.activeFun, action(args));
        //   // auto andLhs = b.create<smt::AndOp>(loc, guard1(args), guard2(args));
        //   // auto lhs = b.create<smt::AndOp>(loc, t1ac, andLhs);
          
        //   // auto g1 
        //   // auto g2 
        //   // return b.create<smt::ImpliesOp>(loc, lhs, rhs); 
        // });
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