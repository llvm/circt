#include "fsm-verification.h"
#include "circt/Dialect/FSM/FSMGraph.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include <z3++.h>
#include <iostream>
#include <vector>
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/raw_ostream.h"
#include "chrono"
#include "fstream"
#include "iostream"


#define VERBOSE 0

using namespace llvm;
using namespace mlir;
using namespace circt;
using namespace std;
using namespace z3;

/**
 * @brief Prints solver assertions
*/
void printSolverAssertions(z3::solver& solver) {
  llvm::outs()<<"---------------------------- SOLVER ----------------------------"<<"\n";
  // llvm::outs()<<solver.to_smt2()<<"\n";

  llvm::outs()<<"------------------------ SOLVER RETURNS ------------------------"<<"\n";
  const auto start{std::chrono::steady_clock::now()};
  llvm::outs()<<solver.check()<<"\n";
  const auto end{std::chrono::steady_clock::now()};

  const std::chrono::duration<double> elapsed_seconds{end - start};


  // llvm::outs()<<"--------------------------- INVARIANT --------------------------"<<"\n";

  // llvm::outs()<<solver.get_model().to_string()<<"\n";
  llvm::outs()<<"-------------------------- END -------------------------------"<<"\n";
  llvm::outs()<<"Time taken: "<<elapsed_seconds.count()<<"s\n";
	ofstream outfile;
	outfile.open("output.txt", ios::app);
	outfile << elapsed_seconds.count() << endl;
	outfile.close();
}

/**
 * @brief Returns list of values to be updated within an action region
*/
vector<mlir::Value> actionsCounter(Region& action){
  vector<mlir::Value> to_update;
  for(auto &op: action.getOps()){
    if (auto updateop = dyn_cast<fsm::UpdateOp>(op)){
      to_update.push_back(updateop.getOperands()[0]);
    }
  }
  return to_update;
}

/**
 * @brief Returns expression from MyExprMap structure
*/
expr findMyExpr(mlir::Value v, MyExprMap expr_map){
  int i=0;
  for(auto vl: expr_map.values){
    if(vl == v){
      return expr_map.exprs[i];
    }
    i++;
  }
  llvm::outs()<<"ERROR: variable "<<v<<" not found in the expression map\n";
}

bool isValInExprMap(mlir::Value v, MyExprMap expr_map){
  for(auto vl: expr_map.values){
    if(vl == v){
      return true;
    }
  }
  return false;;
}

int findMyState(mlir::StringRef s, MyStateInvMap stateInvMap){
  int i=0;
  for(auto st: stateInvMap.stateName){
    if(st == s){
      return stateInvMap.stateID[i];
    }
    i++;
  }
  llvm::outs()<<"ERROR: state "<<s<<" not found in the state invariant map\n";
}

func_decl findMyFun(string s, MyStateInvMapFun *stateInvMap_fun){
  int i=0;
  for(auto st: stateInvMap_fun->stateName){
    if(st == s){
      return stateInvMap_fun->invFun[i];
    }
    i++;
  }
  llvm::outs()<<"ERROR: state "<<s<<" not found in the state invariant map\n";
}


/**
 * @brief Returns expression from Comb dialect operator
*/
expr manage_comb_exp(Operation &op, vector<expr>& vec, z3::context &c){
  if(auto add = dyn_cast<comb::AddOp>(op)){
    return to_expr(c, vec[0] + vec[1]);
  } 
  else if(auto and_op = dyn_cast<comb::AndOp>(op)){
    return expr(vec[0] && vec[1]);
    }
  else if(auto xor_op = dyn_cast<comb::XorOp>(op)){
    return expr(vec[0] ^ vec[1]);
  }
  else if(auto or_op = dyn_cast<comb::OrOp>(op)){
    return expr(vec[0] | vec[1]);
  }
  else if(auto mul_op = dyn_cast<comb::MulOp>(op)){
    return expr(vec[0]* vec[1]);
  }
  else if(auto icmp = dyn_cast<comb::ICmpOp>(op)){
    circt::comb::ICmpPredicate predicate = icmp.getPredicate();
    switch (predicate){
      case circt::comb::ICmpPredicate::eq:
        return expr(vec[0] == vec[1]);
      case circt::comb::ICmpPredicate::ne:
        return expr(vec[0] != vec[1]);
      case circt::comb::ICmpPredicate::slt:
        return expr(vec[0] < vec[1]);
      case circt::comb::ICmpPredicate::sle:
        return expr(vec[0] <= vec[1]);
      case circt::comb::ICmpPredicate::sgt:
        return expr(vec[0] > vec[1]);
      case circt::comb::ICmpPredicate::sge:
        return expr(vec[0] >= vec[1]);
      case circt::comb::ICmpPredicate::ult:
        return expr(vec[0] < vec[1]);
      case circt::comb::ICmpPredicate::ule:
        return expr(vec[0] <= vec[1]);
      case circt::comb::ICmpPredicate::ugt:
        return expr(vec[0] > vec[1]);
      case circt::comb::ICmpPredicate::uge:
        return expr(vec[0] >= vec[1]);
      case circt::comb::ICmpPredicate::ceq:
        return expr(vec[0] == vec[1]);
      case circt::comb::ICmpPredicate::cne:
        return expr(vec[0] != vec[1]);
      case circt::comb::ICmpPredicate::weq: 
        return expr(vec[0] == vec[1]);
      case circt::comb::ICmpPredicate::wne:
        return expr(vec[0] != vec[1]);
    }
  }

  assert(false && "LLVM unreachable");
}

/**
 * @brief Returns expression from densemap or constant operator
*/
expr getExpr(mlir::Value v, MyExprMap expr_map, z3::context& c){
  
  if(isValInExprMap(v, expr_map)){
    return findMyExpr(v, expr_map);
  } else if(auto constop = dyn_cast<hw::ConstantOp>(v.getDefiningOp())){
    if(constop.getType().getIntOrFloatBitWidth()>1)
      return c.int_val(constop.getValue().getSExtValue());
    else
      return c.bool_val(0);
  } else{
    llvm::outs()<<"ERROR: a variable "<<v<<" not found in the expression map\n";
  }
}

/**
 * @brief Returns guard expression for input region
*/
expr getGuardExpr(MyExprMap expr_map, Region& guard, z3::context& c){

  for(auto &op: guard.getOps()){
    if (auto retop = dyn_cast<fsm::ReturnOp>(op)){
      return findMyExpr(retop.getOperand(), expr_map); //expr_map.at(retop.getOperand());
    } 
    vector<expr> vec;
    for (auto operand: op.getOperands()){
      vec.push_back(getExpr(operand, expr_map, c));
    }
    expr_map.exprs.push_back(manage_comb_exp(op, vec, c));
    expr_map.values.push_back(op.getResult(0));  

    // expr_map.insert({op.getResult(0), manage_comb_exp(op, vec, c)});
  }
  return expr(c.bool_const("true"));
}

/**
 * @brief Returns actions for all expressions for the input region
*/
vector<expr> getActionExpr(Region& action, context& c, vector<mlir::Value>* to_update, MyExprMap expr_map){

  vector<expr> updated_vec;
  for (auto v: *to_update){

    bool found = false;
      for(auto &op: action.getOps()){
        if(!found){
          if (auto updateop = dyn_cast<fsm::UpdateOp>(op)){
            if(v == updateop.getOperands()[0]){
              updated_vec.push_back(getExpr(updateop.getOperands()[1], expr_map, c));
              found = true;

            }
          } else {
            vector<expr> vec;
            for (auto operand: op.getOperands()){
              vec.push_back(getExpr(operand, expr_map, c));

            }
            expr_map.exprs.push_back(manage_comb_exp(op, vec, c));
            expr_map.values.push_back(op.getResult(0));
          }
        }

      }
      if(!found){

        updated_vec.push_back(findMyExpr(v, expr_map));
      }
  }
  return updated_vec;
}


int populateArgs(Operation &mod, vector<mlir::Value> *vecVal, MyExprMap *varMap, z3::context &c){
  int numArgs = 0;
  for(Region &rg: mod.getRegions()){
      for(Block &bl: rg){
        for(Operation &op: bl){
          if(auto machine = dyn_cast<fsm::MachineOp>(op)){
            for (Region &rg : op.getRegions()) {
              for (Block &block : rg) {
                for(auto a: block.getArguments()){
                  expr input = c.bool_const(("arg"+to_string(numArgs)).c_str());
                  if(a.getType().getIntOrFloatBitWidth()>1){ 
                    input = c.int_const(("arg"+to_string(numArgs)).c_str());
                  }

                  varMap->exprs.push_back(input);
                  varMap->values.push_back(a);
                  vecVal->push_back(a);
                  numArgs++;
                }
              }
            }
          }
        }
      }
    }
  return numArgs;
}

string getInitialState(Operation &mod){
  for (Region &rg : mod.getRegions()) {
    for (Block &block : rg) {
      for (Operation &op : block) {
        if (auto machine = dyn_cast<fsm::MachineOp>(op)){
          return machine.getInitialState().str();
        }
      }
    }
  }
}

void populateVars(Operation &mod, vector<mlir::Value>* vecVal, MyExprMap * varMap, z3::context &c, int numArgs){
  for(Region &rg: mod.getRegions()){
    for(Block &bl: rg){
      for(Operation &op: bl){
        if(auto machine = dyn_cast<fsm::MachineOp>(op)){
          for (Region &rg : op.getRegions()) {
            for (Block &block : rg) {
              for(Operation &op: block){ 
                if(auto varOp = dyn_cast<fsm::VariableOp>(op)){
                  vecVal->push_back(varOp.getResult());
                  int initValue = varOp.getInitValue().cast<IntegerAttr>().getInt();
                  string varName = varOp.getName().str();
                  if(varOp.getName().str().find("arg") != std::string::npos){
                    // preserved keyword arg for arguments to avoid ambiguity when setting initial state values
                    varName = "var"+to_string(numArgs);
                    numArgs++;
                  }
                  expr input = c.bool_const((varName+"_"+to_string(initValue)).c_str());
                  if(varOp.getResult().getType().getIntOrFloatBitWidth()>1){ 
                    input = c.int_const((varName+"_"+to_string(initValue)).c_str());
                  }
                  varMap->exprs.push_back(input);
                  varMap->values.push_back(varOp.getResult());
                  // varMap->insert({varOp.getResult(), input});
                }
              }
            }
          }
        }
      }
    }
  }
}


void parseFSM(Operation &mod, context &c, MyStateInvMap* stateInvMap, MyStateInvMapOut* stateInvMapOut, vector<transition>* transitions, vector<mlir::Value>* vecVal){
  for (Region &rg: mod.getRegions()){
    for (Block &bl: rg){
      for (Operation &op: bl){
        if (auto machine = dyn_cast<fsm::MachineOp>(op)){
          for (Region &rg : op.getRegions()) {
            for (Block &block : rg) {
              int numState = 0;
              for (Operation &op : block) {
                if (auto state = dyn_cast<fsm::StateOp>(op)){
                  llvm::StringRef currentState = state.getName();
                  // func_decl I = c.function(currentState.str().c_str(), c.int_sort(), c.bool_sort());
                  stateInvMap->stateName.push_back(currentState);
                  stateInvMap->stateID.push_back(numState);
                  // stateInvMap->insert({currentState, numState});
                  numState++;
                  if(VERBOSE){
                    llvm::outs()<<"inserting state "<<currentState<<"\n";
                  }
                  auto regions = state.getRegions();

                  bool existsOutput = false;

                  if(!regions[0]->empty()){
                    existsOutput = true;
                  }
                  Region &outreg = *regions[0];
                  // transitions region
                  for (Block &bl1: *regions[1]){
                    for (Operation &op: bl1.getOperations()){
                      if(auto transop = dyn_cast<fsm::TransitionOp>(op)){


                        transition t;
                        t.from = currentState;
                        t.to = transop.getNextState();
                        t.isGuard = false;
                        t.isOutput = existsOutput;
                        t.isAction = false;
                        auto trRegions = transop.getRegions();
                        string nextState = transop.getNextState().str();                        
                        // guard
                        if(!trRegions[0]->empty()){
                          Region &r = *trRegions[0];
                          z3Fun g = [&r, vecVal, &c](vector<expr> vec) {
                            MyExprMap expr_map_tmp;
                            for(auto [value, expr]: llvm::zip(*vecVal, vec)){
                              expr_map_tmp.exprs.push_back(expr);
                              expr_map_tmp.values.push_back(value);
                              // expr_map_tmp.insert({value, expr});
                            }

                            return getGuardExpr(expr_map_tmp, r, c);
                          };
                          t.guard = g;
                          t.isGuard = true;
                        }
                        // action 
                        if(!trRegions[1]->empty()){
                          vector<mlir::Value> to_update = actionsCounter(*trRegions[1]);
                          Region &r = *trRegions[1];
                          z3FunA a = [&r, vecVal, &c](vector<expr> vec) -> vector<expr> {
                            vector<expr> vec_no_time = vec;
                            expr time = vec_no_time[vec_no_time.size()-1];
                            vec_no_time.pop_back();
                            MyExprMap expr_map_tmp;
                            for(auto [value, expr]: llvm::zip(*vecVal, vec)){
                              expr_map_tmp.exprs.push_back(expr);
                              expr_map_tmp.values.push_back(value);

                            }

                            vector<expr> vec2 =getActionExpr(r, c, vecVal, expr_map_tmp); 

                            vec2.push_back(time);

                            return vec2;
                          };
                          t.action = a;
                          t.isAction = true;
                        }


                        transitions->push_back(t);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

expr nestedForall(vector<expr> solver_vars, expr body, int i){


  if(i==solver_vars.size()-1){ // last element (time) is nested separately as a special case
    return body;
  } else {
    return forall(solver_vars[i], implies(solver_vars[i]>=0 && solver_vars[i]<100, nestedForall(solver_vars, body, i+1)));
  }
}


void populateStateInvMap(MyStateInvMap *stateInvMap, context &c, vector<Z3_sort> *invInput, MyStateInvMapFun *stateInvMap_fun){
  for(auto cs: stateInvMap->stateName){
    const symbol cc = c.str_symbol(cs.str().c_str());
    Z3_func_decl I = Z3_mk_func_decl(c, cc, invInput->size(), invInput->data(), c.bool_sort());
    func_decl I2 = func_decl(c, I);
    llvm::outs()<<"declaring fun "<<(cs.str().c_str())<<"\n";
    stateInvMap_fun->stateName.push_back(cs);
    stateInvMap_fun->invFun.push_back(I2);
    // stateInvMap_fun->insert({cs.first, I2});
  }
}

void populateInvInput(MyExprMap *varMap, context &c, vector<expr> *solverVars, vector<Z3_sort> *invInput){

  int i=0;

  for(auto v: varMap->values){
    expr input = c.bool_const(("arg"+to_string(i)).c_str());
    z3::sort invIn = c.bool_sort();
    if(v.getType().getIntOrFloatBitWidth()>1 ){ 
      input = c.int_const(("arg"+to_string(i)).c_str());
      invIn = c.int_sort(); 
    }
    solverVars->push_back(input);
    if(VERBOSE){
      llvm::outs()<<"solverVars now: "<<solverVars->at(i).to_string()<<"\n";
    }
    i++;
    invInput->push_back(invIn);
  }


}

void parse_fsm(string input_file, int time_bound){

  DialectRegistry registry;
  // clang-format off
  registry.insert<
    comb::CombDialect,
    fsm::FSMDialect,
    hw::HWDialect
  >();



  MyExprMap *exprMap = new MyExprMap();

  MLIRContext context(registry);

  // Parse the MLIR code into a module.
  OwningOpRef<ModuleOp> module = mlir::parseSourceFile<ModuleOp>(input_file, &context);

  Operation& mod = module.get()[0];

  int it = 0;

  z3::context c;


  solver s(c);

	s.set("logic", "HORN");

  MyStateInvMap *stateInvMap = new MyStateInvMap();

  MyExprMap *varMap = new MyExprMap();

  vector<mlir::Value> *vecVal = new vector<mlir::Value>;

  vector<transition> *transitions = new vector<transition>;

  string initialState = getInitialState(mod);

  if(VERBOSE){
    llvm::outs()<<"vb9\n";

    llvm::outs()<<"initial state: "<<initialState<<"\n";
  }


  int numArgs = populateArgs(mod, vecVal, varMap, c);

  if(VERBOSE){
    llvm::outs()<<"vb10\n";

    llvm::outs()<<"number of arguments: "<<numArgs<<"\n";
    for (auto v: *vecVal){
      llvm::outs()<<"argument: "<<v<<"\n";
    }
  }

  populateVars(mod, vecVal, varMap, c, numArgs);


  MyStateInvMapOut *stateInvMapOut = new MyStateInvMapOut();


  populateST(mod, c, stateInvMap, stateInvMapOut, transitions, vecVal);

  // preparing the model

  vector<expr> *solverVars = new vector<expr>;

  vector<Z3_sort> *invInput = new vector<Z3_sort>;

  MyStateInvMapFun *stateInvMap_fun = new MyStateInvMapFun();

  populateInvInput(varMap, c, solverVars, invInput);

  expr time = c.int_const("time");
  z3::sort timeInv = c.int_sort();
  
  solverVars->push_back(time);
  invInput->push_back(timeInv);

  populateStateInvMap(stateInvMap, c, invInput, stateInvMap_fun);

  if(VERBOSE){
    llvm::outs()<<"vb11\n";

    llvm::outs()<<"number of variables + args: "<<solverVars->size()<<"\n";
    for (auto v: *solverVars){
      llvm::outs()<<"variable: "<<v.to_string()<<"\n";
    }
  }



  int j=0;

  vector<expr> *solverVarsInit = new vector<expr>;
  copy(solverVars->begin(), solverVars->end(), back_inserter(*solverVarsInit));  

  for (auto v: varMap->exprs){
    if(v.to_string().find("arg") == std::string::npos && solverVars->size() > 1 && strcmp(v.to_string().c_str(), "time")){
      int init_value = stoi(v.to_string().substr(v.to_string().find("_")+1));
      solverVarsInit->at(j) = c.int_val(init_value);
    }
    j++;
  }

  solverVarsInit->at(solverVarsInit->size()-1) = c.int_val(0);




  // initialize time to 0
  expr body = findMyFun(transitions->at(0).from, stateInvMap_fun)(solverVarsInit->size(), solverVarsInit->data());
  // initial condition
  s.add(nestedForall(*solverVars, body, 0));

  // llvm::outs()<<s.to_smt2()<<"\n";

  vector<int> outputVec;

  for(auto t: *transitions){



    if(VERBOSE){
      llvm::outs()<<"\n\n\ntransition from "<<t.from<<" to "<<t.to<<"\n";
    }
    vector<expr> *solverVarsAfter = new vector<expr>;

    copy(solverVars->begin(), solverVars->end(), back_inserter(*solverVarsAfter));
    solverVarsAfter->at(solverVarsAfter->size()-1) = solverVars->at(solverVars->size()-1)+1;

    if(t.isGuard && t.isAction){
      expr body = implies(findMyFun(t.from, stateInvMap_fun)(solverVars->size(), solverVars->data()) && t.guard(*solverVars), findMyFun(t.to, stateInvMap_fun)(t.action(*solverVarsAfter).size(), t.action(*solverVarsAfter).data()));
      s.add(forall(solverVars->at(solverVars->size()-1), implies((solverVars->at(solverVars->size()-1)>=0 && solverVars->at(solverVars->size()-1)<time_bound), ((nestedForall(*solverVars, body, 0))))));
    } 
    else if (t.isGuard){
      expr body = implies((findMyFun(t.from, stateInvMap_fun)(solverVars->size(), solverVars->data()) && t.guard(*solverVars)), findMyFun(t.to, stateInvMap_fun)(solverVarsAfter->size(), solverVarsAfter->data()));
      s.add(forall(solverVars->at(solverVars->size()-1), implies((solverVars->at(solverVars->size()-1)>=0 && solverVars->at(solverVars->size()-1)<time_bound), ((nestedForall(*solverVars, body,0))))));

    } else if (t.isAction){

      expr body = implies(findMyFun(t.from, stateInvMap_fun)(solverVars->size(), solverVars->data()), findMyFun(t.to, stateInvMap_fun)(t.action(*solverVarsAfter).size(), t.action(*solverVarsAfter).data()));
      s.add(forall(solverVars->at(solverVars->size()-1), implies((solverVars->at(solverVars->size()-1)>=0 && solverVars->at(solverVars->size()-1)<time_bound), ((nestedForall(*solverVars, body, 0))))));

    } else {
      expr body = implies((findMyFun(t.from, stateInvMap_fun)(solverVars->size(), solverVars->data())), findMyFun(t.to, stateInvMap_fun)(solverVarsAfter->size(), solverVarsAfter->data()));
      s.add(forall(solverVars->at(solverVars->size()-1),  implies((solverVars->at(solverVars->size()-1)>=0 && solverVars->at(solverVars->size()-1)<time_bound), ((nestedForall(*solverVars, body, 0))))));

    }

  }
  
  body = !(findMyFun(transitions->at(3).from, stateInvMap_fun)(solverVars->size(), solverVars->data()));

  // llvm::outs()<<"AOOOO "<<(forall(solverVars->at(solverVars->size()-1),  implies((solverVars->at(solverVars->size()-1)>=0 && solverVars->at(solverVars->size()-1)<time_bound), nestedForall(*solverVars,body,0)))).to_string();


  s.add(forall(solverVars->at(solverVars->size()-1),  implies((solverVars->at(solverVars->size()-1)>=0 && solverVars->at(solverVars->size()-1)<time_bound), nestedForall(*solverVars,body,0))));

  // llvm::outs()<<"Additional expr: "<< (not(forall(solverVars->at(solverVars->size()-1),  implies((solverVars->at(solverVars->size()-1)>=0 && solverVars->at(solverVars->size()-1)<time_bound), (implies(body, false) ))))).to_string()<<"\n";

  printSolverAssertions(s);

}

int main(int argc, char **argv){

  string input = argv[1];

  int time = stoi(argv[2]);

  cout << "input file: " << input << endl;

  ofstream outfile;
  outfile.open("output.txt", ios::app);
	outfile << input << endl;
	outfile.close();

  parse_fsm(input, time);

  return 0;

}
