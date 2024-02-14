#include "fsm-verification.h"
#include "circt/Dialect/FSM/FSMGraph.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include <z3++.h>
#include <vector>
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/raw_ostream.h"

#define VERBOSE 1

using namespace llvm;
using namespace mlir;
using namespace circt;
using namespace std;
using namespace z3;

/**
 * @brief Prints solver assertions
*/
void printSolverAssertions(z3::solver& solver) {
  llvm::outs()<<"------------------------ SOLVER ------------------------"<<"\n";
  llvm::outs()<<solver.to_smt2()<<"\n";
  llvm::outs()<<solver.check()<<"\n";

  llvm::outs()<<"------------------------ SOLVER RETURNS ------------------------"<<"\n";
  llvm::outs()<<solver.get_model().to_string()<<"\n";

  llvm::outs()<<"--------------------------------------------------------"<<"\n";

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
 * @brief Returns values from VariableOp operators
*/
// vector<mlir::Value> getVarValues(Operation &op){
//   vector<mlir::Value> vec;
//   op.walk([&](fsm::VariableOp v){
//     vec.push_back(v.getResult());
//   });
//   return vec;
// }

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
    llvm::outs()<<"fake found in expr map\n";
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

// expr getOutputExpr(MyExprMap expr_map, Region& out, z3::context& c){
//     for(auto &op: out.getOps()){
//     if (auto outop = dyn_cast<fsm::OutputOp>(op)){
//       return findMyExpr(outop.getOperands()[0], expr_map); //expr_map.at(retop.getOperand());
//     } 
//     vector<expr> vec;
//     for (auto operand: op.getOperands()){
//       vec.push_back(getExpr(operand, expr_map, c));

//     }

//     expr_map.exprs.push_back(manage_comb_exp(op, vec, c));

//     expr_map.values.push_back(op.getResult(0));  

//     // expr_map.insert({op.getResult(0), manage_comb_exp(op, vec, c)});
//   }
// }

/**
 * @brief Returns guard expression from corresponding region
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

// input: region action, context, vector of values to update
// output: vector of expressions to update

vector<expr> getActionExpr(Region& action, context& c, vector<mlir::Value>* to_update, MyExprMap expr_map){
  if(VERBOSE){
    llvm::outs()<<"action exprMap size "<<expr_map.values.size()<<"\n";
    for(int i=0;i< expr_map.values.size();i++){
      llvm::outs()<<"value "<<expr_map.values[i]<<"\n";
      llvm::outs()<<"expr "<<expr_map.exprs[i].to_string()<<"\n";
    }
  }
  vector<expr> updated_vec;
  for (auto v: *to_update){
    if(VERBOSE){
      llvm::outs()<<"updating "<<v<<"\n";
    } 
    bool found = false;
      for(auto &op: action.getOps()){
        if(!found){
          if (auto updateop = dyn_cast<fsm::UpdateOp>(op)){
            if(v == updateop.getOperands()[0]){
              updated_vec.push_back(getExpr(updateop.getOperands()[1], expr_map, c));
              found = true;
              if(VERBOSE){
                llvm::outs()<<"updated to "<<getExpr(updateop.getOperands()[1], expr_map, c).to_string()<<"\n";
  llvm::outs() << "got here 3\n";

              }
            }
          } else {
            vector<expr> vec;
            for (auto operand: op.getOperands()){
              llvm::outs()<<"operand "<<operand<<"\n";
              vec.push_back(getExpr(operand, expr_map, c));
  llvm::outs() << "got here 2\n";

            }
            expr_map.exprs.push_back(manage_comb_exp(op, vec, c));
            expr_map.values.push_back(op.getResult(0));
            // expr_map.insert({op.getResult(0), manage_comb_exp(op, vec, c)});
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

                  if(VERBOSE){
                    llvm::outs()<<"inserting input "<<a<<"\n";
                    llvm::outs()<<"mapped to "<<input.to_string()<<"\n";
                  }
                  varMap->exprs.push_back(input);
                  varMap->values.push_back(a);
                  // varMap->insert({a, input});
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
                  if(VERBOSE){
                    llvm::outs()<<"inserting variable "<<varOp.getResult()<<"\n";
                    llvm::outs()<<"mapped to "<<(varName+"_"+to_string(initValue)).c_str()<<"\n";
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


void populateST(Operation &mod, context &c, MyStateInvMap* stateInvMap, MyStateInvMapOut* stateInvMapOut, vector<transition>* transitions, vector<mlir::Value>* vecVal){
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
                        t.isOutput = false;
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
                            MyExprMap expr_map_tmp;
                            for(auto [value, expr]: llvm::zip(*vecVal, vec)){
                              expr_map_tmp.exprs.push_back(expr);
                              expr_map_tmp.values.push_back(value);
                              // expr_map_tmp.insert({value, expr});
                              if(VERBOSE){
                                llvm::outs()<<"inserting "<<value<<"\n";
                                llvm::outs()<<"mapped to "<<expr.to_string()<<"\n";
                              }
                            }
                            vector<expr> vec2 = getActionExpr(r, c, vecVal, expr_map_tmp); 
                            return vec2;
                          };
                          t.action = a;
                          t.isAction = true;
                        }

                        // if(existsOutput){

                        //   t.isOutput = true;
                        //   z3Fun o = [&outreg, vecVal, &c](vector<expr> vec) {
                        //     MyExprMap expr_map_tmp;
                        //     for(auto [value, expr]: llvm::zip(*vecVal, vec)){
                        //       expr_map_tmp.exprs.push_back(expr);
                        //       expr_map_tmp.values.push_back(value);
                        //       // expr_map_tmp.insert({value, expr});
                        //     }

                        //     return getOutputExpr(expr_map_tmp, outreg, c);
                        //   };
                        //   t.output = o;
                        // }
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

  llvm::outs()<<"current body, iter "<<i<<" "<<body<<"\n";

  if(i==solver_vars.size()-1){ // last element (time) is nested separately as a special case
    return body;
  } else {
    return forall(solver_vars[i], nestedForall(solver_vars, body, i+1));
  }
}


void populateStateInvMap(MyStateInvMap *stateInvMap, context &c, vector<Z3_sort> *invInput, MyStateInvMapFun *stateInvMap_fun){
  for(auto cs: stateInvMap->stateName){
    const symbol cc = c.str_symbol(cs.str().c_str());
    llvm::outs()<<cs<<"\n";
    Z3_func_decl I = Z3_mk_func_decl(c, cc, invInput->size(), invInput->data(), c.bool_sort());
    func_decl I2 = func_decl(c, I);
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
      llvm::outs()<<solverVars->at(i).to_string()<<"\n";
    }
    i++;
    invInput->push_back(invIn);
  }

  expr time = c.int_const("time");
  z3::sort timeInv = c.int_sort();
  

  solverVars->push_back(time);
  invInput->push_back(timeInv);

}


void parse_fsm(string input_file){

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

  MyStateInvMap *stateInvMap = new MyStateInvMap();

  MyExprMap *varMap = new MyExprMap();

  vector<mlir::Value> *vecVal = new vector<mlir::Value>;

  vector<transition> *transitions = new vector<transition>;

  string initialState = getInitialState(mod);

  if(VERBOSE){
    llvm::outs()<<"initial state: "<<initialState<<"\n";
  }


  int numArgs = populateArgs(mod, vecVal, varMap, c);

  if(VERBOSE){
    llvm::outs()<<"number of arguments: "<<numArgs<<"\n";
    for (auto v: *vecVal){
      llvm::outs()<<"argument: "<<v<<"\n";
    }
  }

  populateVars(mod, vecVal, varMap, c, numArgs);

  if(VERBOSE){
    llvm::outs()<<"number of variables + args: "<<vecVal->size()<<"\n";
    for (auto v: *vecVal){
      llvm::outs()<<"variable: "<<v<<"\n";
    }
  }

  MyStateInvMapOut *stateInvMapOut = new MyStateInvMapOut();


  populateST(mod, c, stateInvMap, stateInvMapOut, transitions, vecVal);



  // preparing the model

  vector<expr> *solverVars = new vector<expr>;

  vector<Z3_sort> *invInput = new vector<Z3_sort>;

  MyStateInvMapFun *stateInvMap_fun = new MyStateInvMapFun();

  populateInvInput(varMap, c, solverVars, invInput);

  populateStateInvMap(stateInvMap, c, invInput, stateInvMap_fun);

  for (auto v: varMap->exprs){
    if(v.to_string().find("arg") == std::string::npos && solverVars->size() > 1){
      int init_value = stoi(v.to_string().substr(v.to_string().find("_")+1));
      expr body = findMyFun(transitions->at(0).from, stateInvMap_fun)(solverVars->at(0), init_value);
      // llvm::outs()<<"initial body "<<body. <<"\n";
      s.add(nestedForall(*solverVars, body, 0));
    }
  }


  llvm:: outs()<<"2 solverVars size "< <   solverVars->size()<<"\n";

  vector<int> outputVec;

  for(auto t: *transitions){

    if(VERBOSE){
      llvm::outs()<<"transition from "<<t.from<<" to "<<t.to<<"\n";
    }

    if(t.isGuard && t.isAction){

      vector<expr> *solverVarsAfter = new vector<expr>;

      copy(solverVars->begin(), solverVars->end(), back_inserter(*solverVarsAfter));

      llvm::outs()<<"solverVarsAfter size "<<solverVarsAfter->size()<<"\n";
      llvm::outs()<<"solverVars size "<<solverVars->size()<<"\n";

      llvm::outs()<<"solverVars contains: \n";
      for(auto v: *solverVars){
        llvm::outs()<<v.to_string()<<"\n";
      }
      llvm::outs()<<"\n";

      llvm::outs()<<"solverVarsAfter contains: \n";
      for(auto v: *solverVarsAfter){
        llvm::outs()<<v.to_string()<<"\n";
      }



      solverVarsAfter->at(solverVarsAfter->size()-1) = solverVars->at(solverVars->size()-1)+1;

      expr body = implies(findMyFun(t.from, stateInvMap_fun)(solverVars->size(), solverVars->data()) && t.guard(*solverVars), findMyFun(t.to, stateInvMap_fun)(t.action(*solverVarsAfter).size(), t.action(*solverVarsAfter).data()));
      s.add(nestedForall(*solverVars, body, 0));
      
    } 
    // else if (t.isGuard){

    //   expr body = implies((findMyFun(t.from, stateInvMap_fun)(solverVars->size(), solverVars->data()) && t.guard(*solverVars)), findMyFun(t.to, stateInvMap_fun)(solverVars->size(), solverVars->data()));
    //   s.add(nestedForall(*solverVars, body, 0));
    // } else if (t.isAction){
    //   expr body = implies(findMyFun(t.from, stateInvMap_fun)(solverVars->size(), solverVars->data()), findMyFun(t.to, stateInvMap_fun)(t.action(*solverVars).size(), t.action(*solverVars).data()));
    //   expr nested = nestedForall(*solverVars, body, 0);
    //   s.add(nestedForall(*solverVars, body, 0));
    // } else {
    //   expr body = implies((findMyFun(t.from, stateInvMap_fun)(solverVars->size(), solverVars->data())), findMyFun(t.to, stateInvMap_fun)(solverVars->size(), solverVars->data()));
    //   s.add(nestedForall(*solverVars, body, 0));
    // }

    // if(t.isOutput){
    //   model m = s.get_model();
    //   outputVec.push_back(m.eval(t.output(*solverVars)));
    //   // outputVec.push_back(t.output(*solverVars).to);
    // }
  }
  

  // expr assertion = implies((findMyFun(transitions->at(0).to, stateInvMap_fun)(solverVars->at(0), solverVars->at(1)) && solverVars->at(1)>5), false);


  // s.add(nestedForall(*solverVars, assertion, 0));



  if(VERBOSE){
    printSolverAssertions(s);
    // llvm::outs()<<"outputVec size "<<outputVec.size()<<"\n";
    // for(auto o: outputVec){
    //   llvm::outs()<<"output "<<o<<"\n";
    // }
  }








}

int main(int argc, char **argv){

  string input = argv[1];

  cout << "input file: " << input << endl;

  parse_fsm(input);

  return 0;

}