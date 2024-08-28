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
  llvm::outs()<<"\n\ncomb args: ";
  for(auto a: args)
    llvm::outs()<<a;

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
      case circt::comb::ICmpPredicate::eq: {
        auto tmp = b.create<smt::EqOp>(loc, b.getType<smt::BoolType>(), args);
        llvm::outs()<<"\n\nreturnign comb value: "<<tmp;
        return tmp;
      }
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

 
mlir::Value getSmt(Operation &op, llvm::SmallVector<mlir::Value> &args, llvm::SmallVector<mlir::Value> &argVars, llvm::SmallVector<mlir::Value> &tmpSmtVal, OpBuilder &b, Location &loc){
  // atm can only be a comb operation 
  auto operands = op.getOperands();
  llvm::SmallVector<mlir::Value> smtOperands;
  for(auto opr : operands){
    llvm::outs()<<"\n\ngetting smt value of: "<<opr;
    auto found = false;
      // each operand is either a const, an arg/var, another operation
    if (auto constop = dyn_cast<hw::ConstantOp>(opr.getDefiningOp())){
      auto tmp = b.create<smt::IntConstantOp>(loc, constop.getValueAttr());
      llvm::outs()<<"\n\n became"<<tmp;
      
      smtOperands.push_back(tmp);
      found = true;
    }
    if(!found){
      for (auto [i, av]: llvm::enumerate(argVars)){
        if (opr == av){
          llvm::outs()<<"\n\n became"<<args[i];

          smtOperands.push_back(args[i]);
          found = true;
        }
      }
    }
    if(!found){
      auto tmp = getSmt(*opr.getDefiningOp(), args, argVars ,tmpSmtVal, b, loc);
      llvm::outs()<<"\n\n became"<<tmp;
      smtOperands.push_back(tmp);
    }
  }
  return getCombValue(op, loc, b, smtOperands);
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



  // create solver region

  for(auto [id1, t1] : llvm::enumerate(transitions)){
    for(auto [id2, t2] : llvm::enumerate(transitions)){
      if(id1!=id2 && t1.to == t2.from){
        // each implication op is in the same region
        llvm::SmallVector<mlir::Value> tmpArgVars(argVars);


        if (t1.hasGuard && !t1.hasAction){
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
                    llvm::outs()<<"\nreturning: "<<tmpSmtVal[i];
                    return tmpSmtVal[i];
                  }
                }
                if (auto constop = dyn_cast<hw::ConstantOp>(retOp->getOperand(0).getDefiningOp())) {
                  // must return smt constant
                  auto retVal = b.create<smt::IntConstantOp>(loc, constop.getValueAttr());
                  llvm::outs()<<"\nreturning: "<<retVal;
                  return retVal;
                }
                int id = getIndex(argVars, retOp->getOperand(0));
                llvm::outs()<<"\nreturning: "<<args[id];
                return args[id];
              }
              tmpVal.push_back(op.getResult(0));
              tmpSmtVal.push_back(getSmt(*op.getResult(0).getDefiningOp(), args, argVars, tmpSmtVal, b, loc));
            }
          };

          if (t2.hasGuard){
            auto *reg = t2.guard;
            auto guard2 = [&reg, &loc, this, &argVars](llvm::SmallVector<mlir::Value> args)-> mlir::Value {
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
          
            // create a forall region whose args are the args and vars, these will be referenced by the implication

            b.create<smt::ForallOp>(loc, argVarTypes, [&guard1, &guard2, &t1, &t2](OpBuilder &b, Location loc, ValueRange args) { 

              llvm::outs()<<"\ninside forall\n\n\n";
                            b.getBlock()->dump();

              for(auto arg: args)
                llvm::outs()<<arg;
              
              // auto actionResult = action(args);
              // for(auto ar: actionResult){
              //   llvm::outs()<< "\nar: ";
              //   llvm::outs()<< ar;
              // }

              llvm::outs()<<"\n\nfunction1:\n"<<t1.activeFun;
              llvm::outs()<<"\n\nfunction2:\n"<<t2.activeFun;


              auto t1ac = b.create<smt::ApplyFuncOp>(loc, t1.activeFun, args);
              auto rhs = b.create<smt::ApplyFuncOp>(loc, t2.activeFun, args);
              auto andLhs = b.create<smt::AndOp>(loc, guard1(args), guard2(args));
              auto lhs = b.create<smt::AndOp>(loc, t1ac, andLhs);
              
              return b.create<smt::ImpliesOp>(loc, lhs, rhs); 
            });
          } else {

              b.create<smt::ForallOp>(loc, argVarTypes, [&guard1, &t1, &t2](OpBuilder &b, Location loc, ValueRange args) { 

              llvm::outs()<<"\ninside forall\n\n\n";
                            b.getBlock()->dump();

              for(auto arg: args)
                llvm::outs()<<arg;
              
              // auto actionResult = action(args);
              // for(auto ar: actionResult){
              //   llvm::outs()<< "\nar: ";
              //   llvm::outs()<< ar;
              // }

              llvm::outs()<<"\n\nfunction1:\n"<<t1.activeFun;
              llvm::outs()<<"\n\nfunction2:\n"<<t2.activeFun;


              auto t1ac = b.create<smt::ApplyFuncOp>(loc, t1.activeFun, args);
              auto rhs = b.create<smt::ApplyFuncOp>(loc, t2.activeFun, args);
              auto lhs = b.create<smt::AndOp>(loc, t1ac, guard1(args));
              return b.create<smt::ImpliesOp>(loc, lhs, rhs); 
            });

          }
        }
        else if(t1.hasGuard && t1.hasAction){

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
                    llvm::outs()<<"\nreturning: "<<tmpSmtVal[i];
                    return tmpSmtVal[i];
                  }
                }
                if (auto constop = dyn_cast<hw::ConstantOp>(retOp->getOperand(0).getDefiningOp())) {
                  // must return smt constant
                  auto retVal = b.create<smt::IntConstantOp>(loc, constop.getValueAttr());
                  llvm::outs()<<"\nreturning: "<<retVal;
                  return retVal;
                }
                int id = getIndex(argVars, retOp->getOperand(0));
                llvm::outs()<<"\nreturning: "<<args[id];
                return args[id];
              }
              tmpVal.push_back(op.getResult(0));
              tmpSmtVal.push_back(getSmt(*op.getResult(0).getDefiningOp(), args, argVars, tmpSmtVal, b, loc));
            }
          };

          auto reg2 = t1.action;

          auto action = [&reg2, &loc, this, &argVars, &numArgs](llvm::SmallVector<mlir::Value> args)->llvm::SmallVector<mlir::Value>{
            // problem: make sure that the order is right
            // return corresponding smt thingy

            llvm::SmallVector<mlir::Value> tmpSmtVal;
            llvm::SmallVector<mlir::Value> updatedSmtVal(argVars);



            for (auto [j, uv]: llvm::enumerate(updatedSmtVal)){
              if (int(j)  >= int(numArgs)){
                bool found = false;
                llvm::SmallVector<mlir::Value> tmpVal;
                for(auto &op: reg2->getOps()){
                  if (auto updateOp = dyn_cast<fsm::UpdateOp>(op)){
                    if (updateOp->getOperand(0) == uv){
                      for(auto tv : tmpVal){
                        if(tv == updateOp->getOperand(1)){
                          updatedSmtVal[j] = tv;
                          found = true;
                        } 
                      } 
                      if(!found){
                        if (auto constop = dyn_cast<hw::ConstantOp>(updateOp->getOperand(0).getDefiningOp())) {
                          // must return smt constant
                          updatedSmtVal[j] =  b.create<smt::IntConstantOp>(loc, constop.getValueAttr());
                          found = true;
                        } else {
                          int id = getIndex(argVars, updateOp->getOperand(0));
                          if (id!=-1){
                            updatedSmtVal[j] = args[id];
                            found = true;
                          }
                        }
                      }
                    }
                  } else {
                    tmpVal.push_back(op.getResult(0));
                    tmpSmtVal.push_back(getSmt(*op.getResult(0).getDefiningOp(), args, argVars, tmpSmtVal, b, loc));
                  }
                }
                if(!found)
                  updatedSmtVal[j] = getSmt(*uv.getDefiningOp(), args, argVars, tmpSmtVal, b, loc);
              }
            }
            return tmpSmtVal;
          };


          if (t2.hasGuard){
            auto *reg = t2.guard;
            auto guard2 = [&reg, &loc, this, &argVars](llvm::SmallVector<mlir::Value> args)-> mlir::Value {
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
          
            // create a forall region whose args are the args and vars, these will be referenced by the implication

            b.create<smt::ForallOp>(loc, argVarTypes, [&guard1, &guard2, &action, &t1, &t2](OpBuilder &b, Location loc, ValueRange args) { 

              llvm::outs()<<"\ninside forall\n\n\n";
                            b.getBlock()->dump();

              for(auto arg: args)
                llvm::outs()<<arg;
              
              // auto actionResult = action(args);
              // for(auto ar: actionResult){
              //   llvm::outs()<< "\nar: ";
              //   llvm::outs()<< ar;
              // }

              llvm::outs()<<"\n\nfunction1:\n"<<t1.activeFun;
              llvm::outs()<<"\n\nfunction2:\n"<<t2.activeFun;


              auto t1ac = b.create<smt::ApplyFuncOp>(loc, t1.activeFun, args);
              auto rhs = b.create<smt::ApplyFuncOp>(loc, t2.activeFun, action(args));
              auto andLhs = b.create<smt::AndOp>(loc, guard1(args), guard2(action(args)));
              auto lhs = b.create<smt::AndOp>(loc, t1ac, andLhs);
              
              return b.create<smt::ImpliesOp>(loc, lhs, rhs); 
            });
          } else {

              b.create<smt::ForallOp>(loc, argVarTypes, [&guard1, &t1, &t2, &action](OpBuilder &b, Location loc, ValueRange args) { 

              llvm::outs()<<"\ninside forall\n\n\n";
                            b.getBlock()->dump();

              for(auto arg: args)
                llvm::outs()<<arg;
              
              // auto actionResult = action(args);
              // for(auto ar: actionResult){
              //   llvm::outs()<< "\nar: ";
              //   llvm::outs()<< ar;
              // }

              llvm::outs()<<"\n\nfunction1:\n"<<t1.activeFun;
              llvm::outs()<<"\n\nfunction2:\n"<<t2.activeFun;


              auto t1ac = b.create<smt::ApplyFuncOp>(loc, t1.activeFun, args);
              auto rhs = b.create<smt::ApplyFuncOp>(loc, t2.activeFun, action(args));
              auto lhs = b.create<smt::AndOp>(loc, t1ac, guard1(args));
              return b.create<smt::ImpliesOp>(loc, lhs, rhs); 
            });

          }
          
        } 
        // else if (!t1.hasGuard && t1.hasAction){

        // } else {

        // }
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