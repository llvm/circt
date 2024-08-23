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
#include <functional>
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

mlir::Value getSmt(mlir::Value opr, SmallVector<std::pair<mlir::Value, mlir::Value>> &variables, Location &loc, 
OpBuilder &b){
  for(auto &e: variables){
    if (e.first == opr)
      return e.second;
  }
  if (auto constOp = dyn_cast<hw::ConstantOp>(opr.getDefiningOp())){
    if (constOp.getType().getIntOrFloatBitWidth() > 1)
      return b.create<smt::IntConstantOp>(loc, constOp->getResult(0));
        return b.create<smt::BoolConstantOp>(loc, constOp->getResult(0));
  }
}

int insertStates(llvm::SmallVector<std::string> states, const std::string& st){
  for(auto [id, s]: llvm::enumerate(states)){
    if(s == st)
      return id;
  }
  states.push_back(st);
  return states.size()-1;
}

mlir::Value getCombValue(Operation &op, Location &loc, OpBuilder &b){
  // if (auto addOp = llvm::dyn_cast<comb::AddOp>(op)){
  //   auto newOp = b.create<smt::IntAddOp>(loc);
  //   return newOp;
  // }
  if (auto andOp = llvm::dyn_cast<comb::AndOp>(op))
    return b.create<smt::AndOp>(loc, andOp->getOperands());
  // if (auto xorOp = llvm::dyn_cast<comb::XorOp>(op))
  //   return b.create<smt::XOrOp>(loc, xorOp->getOperand(0), xorOp->getOperand(1));
  // if (auto orOp = llvm::dyn_cast<comb::OrOp>(op))
  //   return b.create<smt::AndOp>(loc, orOp->getOperand(0), orOp->getOperand(1));
  // if (auto mulOp = llvm::dyn_cast<comb::MulOp>(op))
  //   return b.create<smt::AndOp>(loc, mulOp->getOperand(0), mulOp->getOperand(1));
  // if (auto icmp = llvm::dyn_cast<comb::ICmpOp>(op)){
  //   circt::comb::ICmpPredicate predicate = icmp.getPredicate();
  //     switch (predicate) {
  //     case circt::comb::ICmpPredicate::eq:
  //       return b.create<smt::EqOp>(loc, icmp->getOperand(0), icmp->getOperand(1));
  //     case circt::comb::ICmpPredicate::ne:
  //       return b.create<smt::DistinctOp>(loc, icmp->getOperand(0), icmp->getOperand(1));
  //     }
  //   }
  
}


mlir::Value getGuardExp(SmallVector<std::pair<mlir::Value, mlir::Value>> variables, Region &r, Location &loc, OpBuilder &b){
  for (auto &op : r.getOps()){
      if (auto retOp = llvm::dyn_cast<fsm::ReturnOp>(op)){
        for(auto &v: variables){
          if(v.first == retOp->getOperand(0))
            llvm::outs()<<"\nreturning "<<v.first;
            return v.first;
        }
      } else {
        llvm::SmallVector<mlir::Value> vec;
        for (auto opr : op.getOperands()){
          vec.push_back(getSmt(opr, variables, loc, b));
        }
        variables.push_back({getCombValue(op, loc, b), op.getResult(0)});
      }
    }
}

llvm::SmallVector<mlir::Value> getAction(llvm::SmallVector<mlir::Value> &toUpdate, SmallVector<std::pair<mlir::Value, mlir::Value>> variables, Region &r, Location &loc, OpBuilder &b){
  llvm::SmallVector<mlir::Value> updatedVec;
  for(auto v : toUpdate){
    bool found = false;
    for (auto &op: r.getOps()){
      if(!found){
        if (auto updateop = dyn_cast<fsm::UpdateOp>(op)) {
          if (v == updateop.getOperands()[0]) {
            updatedVec.push_back(
                getSmt(updateop.getOperands()[1], variables, loc, b));
            found = true;
          }
        } else {
          llvm::SmallVector<mlir::Value> vec;
          for(auto opr:op.getOperands())
            vec.push_back(getSmt(opr, variables, loc, b));
          variables.push_back({getCombValue(op, loc, b), op.getResult(0)});
        }
      }
    }
    if(!found){
      for(auto &e : variables)
        if (e.first == v)
          updatedVec.push_back(e.second);
    }
  }
  return updatedVec;
}


Transition parseTransition(fsm::TransitionOp t, int from, llvm::SmallVector<std::string> &states, 
    const SmallVector<std::pair<mlir::Value, mlir::Value>> variables, 
    Location &loc, OpBuilder &b, 
    llvm::SmallVector<mlir::Value> vecVal){
  Transition tr = {.from = from, .to = insertStates(states, t.getNextState().str())};
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
  llvm::SmallVector<mlir::Value> val;
  llvm::SmallVector<int> initVal;


  SmallVector<std::pair<mlir::Value, mlir::Value>> variables;
  // everything is an Int (even i1) because we want to have everything in the same structure
  // and we can not mix ints and bools in the same vec
  for (auto a : args){
    auto intVal = b.getType<smt::IntType>();
    argVarTypes.push_back(intVal);
  }

  for (auto variableOp : machineOp.front().getOps<fsm::VariableOp>()) {
    auto intVal = b.getType<smt::IntType>();
    // mlir::Value tmp = b.create<smt::IntConstantOp>(loc, variableOp->getResult(0));
    // val.push_back(tmp);
    argVarTypes.push_back(intVal);
    // int iv = llvm::cast<int>(variableOp.getInitValue());
    // initVal.push_back(iv);
    // variables.push_back({variableOp.getResult(),variableOp.getResult()});
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

  Transition support = {.from = 0, .to = 1, .hasGuard = false, .hasAction = false, .hasOutput = false};

  transitions.push_back(support);

  // todo: populate outputs

  // populate transitions vector

  // add time
  llvm::SmallVector<smt::DeclareFunOp> transitionActive;

  for (auto stateOp : machineOp.front().getOps<fsm::StateOp>()) {
    auto fromState = insertStates(states, stateOp.getName().str());
    for (auto tr: stateOp.getTransitions().front().getOps<fsm::TransitionOp>()){
      auto t = parseTransition(tr, fromState, states, variables, loc, b, argVars);
      transitions.push_back(t);
      // push back function
      // transitionActive.push_back(ValueParamT Elt)
    }
  }

  for(auto [id1, t1] : llvm::enumerate(transitions)){
    for(auto [id2, t2] : llvm::enumerate(transitions)){
      if(id1!=id2 && t1.to == t2.from){
        // head: parse guard regions directly

        // tail 

        // implication
      }
    }
  }
  // get guard1 expression 

  // get guard2 expression 

  // shovel evety


  // llvm::SmallVector<>
  return success();
}

// /**
//  * @brief Nest SMT assertion for all variables in the variables vector
//  */
// expr nestedForall(vector<expr> &solverVars, expr &body, int numArgs,
//                   int numOutputs, z3::context &c) {
//   z3::expr_vector univQ(c);
//   for(int idx = 0; idx < int(int(solverVars.size()))-numOutputs-1; idx++){
//     univQ.push_back(solverVars[idx]);
//   }
//   // quantify next input if present
//   for (int idx = 0; idx < numArgs; idx++){
//     expr tmp = solverVars[idx];
//     if (tmp.is_bool())
//       tmp = c.bool_const((tmp.to_string()+"_p").c_str());
//     else
//       tmp = c.int_const((tmp.to_string()+"_p").c_str());
//     univQ.push_back(tmp);   
//   }
//   univQ.push_back(solverVars[int(solverVars.size())-1]);
//   expr ret = forall(univQ, body);
//   return ret;
// }
// /**
//  * @brief Build Z3 sort for each input argument
//  */
// void populateInvInput(vector<std::pair<expr, mlir::Value>> &variables,
//                       context &c, vector<expr> &solverVars,
//                       vector<Z3_sort> &invInput, int numArgs, int numOutputs) {
//   int i = 0;
//   for (const auto& e : variables) {
//     string name = "var";
//     if (numArgs != 0 && i < numArgs) {
//       name = "input";
//     } else if (numOutputs != 0 && i >= int(variables.size()) - numOutputs) {
//       name = "output";
//     }
//     expr input = c.bool_const((name + to_string(i)).c_str());
//     z3::sort invIn = c.bool_sort();
//     if (e.second.getType().getIntOrFloatBitWidth() > 1) {
//       input = c.int_const((name + to_string(i)).c_str());
//       invIn = c.int_sort();
//     }
//     solverVars.push_back(input);
//     if (V) {
//       llvm::outs() << "solverVars now: " << solverVars[i].to_string() << "\n";
//     }
//     i++;
//     invInput.push_back(invIn);
//   }
// }


//   vector<func_decl> transitionActive;

//   vector<expr> solverVars;

//   vector<Z3_sort> invInput;

//   populateInvInput(variables, c, solverVars, invInput, numArgs, numOutputs);

//   expr time = c.int_const("time");

//   solverVars.push_back(time);

//   Z3_sort timeSort = c.int_sort();

//   invInput.push_back(timeSort);



//   // initialize variables' values

//   vector<expr> solverVarsInit;
//   copy(solverVars.begin(), solverVars.end(), back_inserter(solverVarsInit));
//   for (auto [idx, iv]: llvm::enumerate(initValues)){
//       if(solverVarsInit[numArgs+idx].is_bool())
//         solverVarsInit[numArgs + idx] = c.bool_val(iv);
//       else 
//         solverVarsInit[numArgs + idx] = c.int_val(iv);
//   }

//   // enforce self-loops if none of the guards is respected
//   vector<int> visited;

//   for(auto [idx1, t1]: llvm::enumerate(transitions)){
//     bool found = false;
//     for (auto v: visited)
//       if (t1.from == v)
//         found = true;
//     if (t1.isGuard && !found){
//       visited.push_back(t1.from);
//       vector<z3Fun> tmpGuards;
//       tmpGuards.push_back(t1.guard);  
//       for(auto [idx2, t2]: llvm::enumerate(transitions)){
//         if(idx1!=idx2 && t1.from == t2.from && t2.isGuard)
//           tmpGuards.push_back(t2.guard);
//       }

//       z3Fun tmpG = [tmpGuards, &c](const vector<expr>& vec) {
//         expr neg = c.bool_val(true);
//         for(const auto& tmp: tmpGuards){
//           neg = neg && !tmp(vec);
//         }
//         return neg;
//       };
//       Transition t;
//       t.from = t1.from;
//       t.to = t1.from;
//       t.isGuard = true;
//       t.guard = tmpG;
//       t.isAction = false;
//       t.isOutput = false;
//       transitions.push_back(t);
//     }
//   }

//   // create uninterpreted function vec -> bool for each transition

//   for (const auto& t: transitions){
//     const symbol cc = c.str_symbol(("tr"+to_string(t.from)+to_string(t.to)).c_str());
//     Z3_func_decl inv = Z3_mk_func_decl(c, cc, invInput.size(), invInput.data(), c.bool_sort());
//     func_decl inv2 = func_decl(c, inv);
//     transitionActive.push_back(inv2);
//   }

//   // initial condition (fake transition with no action nor guards)

//   expr tail = solverVars[solverVars.size()-1]==-1;
//   expr head = transitionActive[0](solverVarsInit.size(), solverVarsInit.data());
//   expr body = implies(tail, head);
//   s.add(nestedForall(solverVars, body, numArgs, numOutputs, c));

//   // traverse all transitions and build implications from one to the other 

//   for(auto [idx1, t1]: llvm::enumerate(transitions)){
//     for(auto [idx2, t2]: llvm::enumerate(transitions)){
//       if(t1.to == t2.from && idx1 != idx2){
//         // build implication here (tail = lhs, head = rhs)
//         vector<expr> solverVarsAfter;
//         copy(solverVars.begin(), solverVars.end(), back_inserter(solverVarsAfter));
        
//         if (t1.isAction)
//           solverVarsAfter = t1.action(solverVars);
//         for (int k=0;k<numArgs; k++){
//           if (solverVarsAfter[k].is_bool())
//             solverVarsAfter[k] = c.bool_const((solverVarsAfter[k].to_string()+"_p").c_str());
//           else
//             solverVarsAfter[k] = c.int_const((solverVarsAfter[k].to_string()+"_p").c_str());
//         }
//         solverVarsAfter[solverVarsAfter.size()-1]= solverVarsAfter[solverVarsAfter.size()-1] + 1;
//         expr guard1 = c.bool_val(true);
//         expr guard2 = c.bool_val(true);

//         if(t1.isGuard)
//           guard1 = t1.guard(solverVars);
//         if(t2.isGuard)
//           guard2 = t2.guard(solverVarsAfter);


//         expr tail = transitionActive[idx1](solverVars.size(), solverVars.data()) && guard1 && guard2;
//         expr head = transitionActive[idx2](solverVarsAfter.size(), solverVarsAfter.data());
//         expr body = implies(tail, head);
//         expr imp = nestedForall(solverVars, body, numArgs, numOutputs, c);
//         s.add(imp);
//       }
//     }
//   }
//   vector<func_decl> argInputs;
//   // mutual exclusion 

//   for (auto [idx1, t1]: llvm::enumerate(transitions)){
//     expr tail = transitionActive[idx1](solverVars.size(), solverVars.data());
//     expr head = c.bool_val(true);
//     for(auto [idx2, t2]: llvm::enumerate(transitions)){
//       if (idx1!=idx2)
//         head = head && (!transitionActive[idx2](solverVars.size(), solverVars.data()));
//     }
//     expr body = implies(tail, head);
//     // do not change the 0 here 
//     s.add(nestedForall(solverVars, body, 0, numOutputs, c));
//   }

//   auto r = parseLTL(property, solverVars, stateInv, transitions, transitionActive, numArgs, numOutputs, c);
//   s.add(r);

//   printSolverAssertions(c, s, output, transitionActive, argInputs);
// }

// int main(int argc, char **argv) {
//   string input = argv[1];
//   cout << "input file: " << input << endl;

//   string prop = argv[2];
//   cout << "property file: " << prop << endl;

//   string output = argv[3];
//   cout << "output file: " << output << endl;

//   parseFSM(input, prop, output);

//   return 0;
// }

 
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