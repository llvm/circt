#include "fsm-verification.h"
#include "circt/Dialect/FSM/FSMGraph.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include </Users/luisa/z3/src/api/c++/z3++.h>
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
  llvm::outs()<<"--------------------------------------------------------"<<"\n";
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

/**
 * @brief Returns expression from Comb dialect operator
*/
expr manage_comb_exp(Operation &op, vector<expr> vec, z3::context &c){

  if(auto add = dyn_cast<comb::AddOp>(op)){
    return to_expr(c, vec[0] + vec[1]);
  } 
  else if(auto and_op = dyn_cast<comb::AndOp>(op)){
    return expr(vec[0] && vec[1]);
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
}

/**
 * @brief Returns expression from densemap or constant operator
*/
expr getExpr(mlir::Value v, llvm::DenseMap<mlir::Value, expr> expr_map, z3::context& c){
  if(expr_map.find(v) != expr_map.end()){
    return expr_map.at(v);
  } else if(auto constop = dyn_cast<hw::ConstantOp>(v.getDefiningOp())){
    return c.int_val(constop.getValue().getSExtValue());
  } else{
    llvm::outs()<<"ERROR: variable "<<v<<" not found in the expression map\n";
  }
}

/**
 * @brief Returns guard expression from corresponding region
*/
expr getGuardExpr(llvm::DenseMap<mlir::Value, expr> expr_map, Region& guard, z3::context& c){

  for(auto &op: guard.getOps()){
    if (auto retop = dyn_cast<fsm::ReturnOp>(op)){
      return expr_map.at(retop.getOperand());
    } 
    vector<expr> vec;
    for (auto operand: op.getOperands()){
      vec.push_back(getExpr(operand, expr_map, c));
    }
    expr_map.insert({op.getResult(0), manage_comb_exp(op, vec, c)});
  }
  return expr(c.bool_const("true"));
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

// input: region action, context, vector of values to update
// output: vector of expressions to update

vector<expr> getActionExpr(Region&action, context& c, vector<mlir::Value> to_update, llvm::DenseMap<mlir::Value, expr> expr_map){
  if(VERBOSE){
    llvm::outs()<<"action exprMap size "<<expr_map.size()<<"\n";
    for(auto [value, expr]: expr_map){
      llvm::outs()<<"value "<<value<<"\n";
      llvm::outs()<<"expr "<<expr.to_string()<<"\n";
    }
  }
  vector<expr> updated_vec;
  for (auto v: to_update){
    if(VERBOSE){
      llvm::outs()<<"updating "<<v<<"\n";
    } 
    bool found = false;
      for(auto &op: action.getOps()){
        if (auto updateop = dyn_cast<fsm::UpdateOp>(op)){
          if(v == updateop.getOperands()[0]){
            updated_vec.push_back(getExpr(updateop.getOperands()[1], expr_map, c));
            found = true;
            if(VERBOSE){
              llvm::outs()<<"updated to "<<getExpr(updateop.getOperands()[1], expr_map, c).to_string()<<"\n";
            }
          }
        } else {
          vector<expr> vec;
          for (auto operand: op.getOperands()){
            llvm::outs()<<"operand "<<operand<<"\n";
            vec.push_back(getExpr(operand, expr_map, c));
          }
          expr_map.insert({op.getResult(0), manage_comb_exp(op, vec, c)});
        }
      }
      if(!found){
        updated_vec.push_back(expr_map.at(v));
      }
  }

  return updated_vec;
}


int populateArgs(Operation &mod, vector<mlir::Value> &vecVal, llvm::DenseMap<mlir::Value, expr> &varMap, z3::context &c){
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
                  varMap.insert({a, input});
                  vecVal.push_back(a);
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

void populateVars(Operation &mod, vector<mlir::Value> &vecVal, llvm::DenseMap<mlir::Value, expr> &varMap, z3::context &c, int numArgs){

  for(Region &rg: mod.getRegions()){
    for(Block &bl: rg){
      for(Operation &op: bl){
        if(auto machine = dyn_cast<fsm::MachineOp>(op)){
          for (Region &rg : op.getRegions()) {
            for (Block &block : rg) {
              for(Operation &op: block){ 
                if(auto varOp = dyn_cast<fsm::VariableOp>(op)){

                  vecVal.push_back(varOp.getResult());

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
                  varMap.insert({varOp.getResult(), input});
                }
              }
            }
          }
        }
      }
    }
  }
}


void populateST(Operation &mod, context &c, llvm::DenseMap<StringRef, int> &stateInvMap, vector<transition> &transitions, vector<mlir::Value> vecVal){
  for(Region &rg: mod.getRegions()){
    for(Block &bl: rg){
      for(Operation &op: bl){
        if(auto machine = dyn_cast<fsm::MachineOp>(op)){
          for (Region &rg : op.getRegions()) {
            for (Block &block : rg) {
              int numState = 0;
              for (Operation &op : block) {
                if (auto state = dyn_cast<fsm::StateOp>(op)){
                  llvm::StringRef currentState = state.getName();
                  // func_decl I = c.function(currentState.str().c_str(), c.int_sort(), c.bool_sort());
                  stateInvMap.insert({currentState, numState});
                  numState++;
                  if(VERBOSE){
                    llvm::outs()<<"inserting state "<<currentState<<"\n";
                  }
                  auto regions = state.getRegions();

                  // transitions region
                  for (Block &bl1: *regions[1]){
                    for (Operation &op: bl1.getOperations()){
                      if(auto transop = dyn_cast<fsm::TransitionOp>(op)){
                        transition t;
                        t.from = currentState;
                        t.to = transop.getNextState();
                        t.isGuard = false;
                        t.isAction = false;

                        auto trRegions = transop.getRegions();
                        string nextState = transop.getNextState().str();
                        
                        // guard

                        if(!trRegions[0]->empty()){

                          Region &r = *trRegions[0];

                          z3Fun g = [&r, vecVal, &c](vector<expr> vec) {
                            llvm::DenseMap<mlir::Value, expr> expr_map_tmp;
                            for(auto [value, expr]: llvm::zip(vecVal, vec)){
                              expr_map_tmp.insert({value, expr});
                            }
                            return getGuardExpr(expr_map_tmp, r, c);
                          };

                          t.guard = g;

                          t.isGuard = true;

                          t.guard_reg = trRegions[0];
                        }

                        // action 

                        if(!trRegions[1]->empty()){

                          vector<mlir::Value> to_update = actionsCounter(*trRegions[1]);

                          Region &r = *trRegions[1];

                          z3FunA a = [&r, vecVal, &c](vector<expr> vec) -> vector<expr> {
                            llvm::DenseMap<mlir::Value, expr> expr_map_tmp;
                            for(auto [value, expr]: llvm::zip(vecVal, vec)){
                              expr_map_tmp.insert({value, expr});
                              if(VERBOSE){
                                llvm::outs()<<"inserting "<<value<<"\n";
                                llvm::outs()<<"mapped to "<<expr.to_string()<<"\n";
                              }
                            }
                            vector<expr> vec2 = getActionExpr(r, c, vecVal, expr_map_tmp); 
                            return vec2;
                          };

                          t.action = a;

                          t.action_reg = trRegions[1]; 

                          t.isAction = true;
                        }

                        transitions.push_back(t);
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
  if(i==solver_vars.size()-1){
    return body;
  } else {
    return forall(solver_vars[i], nestedForall(solver_vars, body, i+1));
  }
}


void populateStateInvMap(llvm::DenseMap<StringRef, int> &stateInvMap, context &c, vector<Z3_sort> invInput, llvm::DenseMap<llvm::StringRef, func_decl> &stateInvMap_fun){
  for(auto cs: stateInvMap){
    const symbol cc = c.str_symbol(cs.first.str().c_str());
    llvm::outs()<<cs.first<<"\n";
    Z3_func_decl I = Z3_mk_func_decl(c, cc, invInput.size(), invInput.data(), c.bool_sort());
    func_decl I2 = func_decl(c, I);
    stateInvMap_fun.insert({cs.first, I2});
  }
}

void populateInvInput(llvm::DenseMap<mlir::Value, expr> varMap, context &c, vector<expr> &solverVars, vector<Z3_sort> &invInput){

  int i=0;

  for(auto v: varMap){
    expr input = c.bool_const(("arg"+to_string(i)).c_str());
    z3::sort invIn = c.bool_sort();
    if(v.first.getType().getIntOrFloatBitWidth()>1 ){ 
      llvm::outs()<<"int or float "<<v.first<<"\n";
      input = c.int_const(("arg"+to_string(i)).c_str());
      invIn = c.int_sort(); 
    }
    llvm::outs()<<"solver var "<<i<<": "<<input.to_string()<<"\n";
    solverVars.push_back(input);
    if(VERBOSE){
      llvm::outs()<<solverVars[i].to_string()<<"\n";
    }
    i++;
    invInput.push_back(invIn);
  }


}

void populateSolver(Operation &mod){

  z3::context c;

  solver s(c);

  llvm::DenseMap<mlir::StringRef, int> stateInvMap;

  llvm::DenseMap<mlir::Value, expr> varMap;

  vector<mlir::Value> vecVal;

  vector<transition> transitions;

  string initialState = getInitialState(mod);

  if(VERBOSE){
    llvm::outs()<<"initial state: "<<initialState<<"\n";
  }

  int numArgs = populateArgs(mod, vecVal, varMap, c);

  if(VERBOSE){
    llvm::outs()<<"number of arguments: "<<numArgs<<"\n";
    for (auto v: vecVal){
      llvm::outs()<<"argument: "<<v<<"\n";
    }
  }

  populateVars(mod, vecVal, varMap, c, numArgs);

  if(VERBOSE){
    llvm::outs()<<"number of variables: "<<vecVal.size()<<"\n";
    for (auto v: vecVal){
      llvm::outs()<<"variable: "<<v<<"\n";
    }
  }

  populateST(mod, c, stateInvMap, transitions, vecVal);

  if(VERBOSE){
    for(auto t: transitions){
      llvm::outs()<<"transition from "<<t.from<<" to "<<t.to<<"\n";
      if(t.isGuard){
        llvm::outs()<<"guard: "<<t.guard_reg->front().front()<<"\n";
      }
      if(t.isAction){
        llvm::outs()<<"action: "<<t.action_reg->front().front()<<"\n";
      }
    }
  }

  // preparing the model

  vector<expr> solverVars;

  vector<Z3_sort> invInput;

  llvm::DenseMap<llvm::StringRef, func_decl> stateInvMap_fun;

  populateInvInput(varMap, c, solverVars, invInput);

  populateStateInvMap(stateInvMap, c, invInput, stateInvMap_fun);


  for (auto v: varMap){
    if(v.second.to_string().find("arg") == std::string::npos){
      int init_value = stoi(v.second.to_string().substr(v.second.to_string().find("_")+1));
    }
  }

  for(auto t: transitions){

    if(VERBOSE){
      llvm::outs()<<"transition from "<<t.from<<" to "<<t.to<<"\n";
    }
    int row = stateInvMap.at(t.from);
    int col = stateInvMap.at(t.to);
    if(t.isGuard && t.isAction){
      // func_decl g = stateInvariants_map.at(t.from);
      llvm ::outs()<<"function: "<<stateInvMap_fun.at(t.from).to_string() <<"\n";
      llvm::outs()<<"solver vars: ";
      for(auto v: solverVars){
        llvm::outs()<<v.to_string()<<" ";
      }

      vector<expr> vec = t.action(solverVars);

      expr body = implies(stateInvMap_fun.at(t.from)(solverVars[0], solverVars[1]) && t.guard(solverVars), stateInvMap_fun.at(t.to)(t.action(solverVars)[0], t.action(solverVars)[1]));
      expr nested = nestedForall(solverVars, body, 0);
      // llvm::outs()<<"nested: "<<nested.to_string()<<"\n";
      s.add(nestedForall(solverVars, body, 0));
    } else if (t.isGuard){
      for(auto v: solverVars){
        // expr body = implies((stateInvariants_map.at(t.from)(v) && t.guard(solver_vars)), stateInvariants_map.at(t.to)(v));
        // s.add(forall(v, body));
      }
    } else if (t.isAction){
      int idx_w = 0;
      for (auto v: solverVars){
        // expr body = implies((stateInvariants_map.at(t.from)(v)), stateInvariants_map.at(t.to)(t.action(solver_vars).at(idx_w)));
        // s.add(forall(v, body));
        idx_w++;
      }
    } else {
      for(auto v: solverVars){
        // expr body = implies(stateInvariants_map.at(t.from)(v), stateInvariants_map.at(t.to)(v));
        // s.add(forall(v, body));
      }
    }
  }

  if(VERBOSE){
    printSolverAssertions(s);
  }

}



void parse_fsm(string input_file){

  DialectRegistry registry;
  // clang-format off
  registry.insert<
    comb::CombDialect,
    fsm::FSMDialect,
    hw::HWDialect
  >();


  llvm::DenseMap<mlir::Value, expr> expr_map;

  MLIRContext context(registry);

  // Parse the MLIR code into a module.
  OwningOpRef<ModuleOp> module = mlir::parseSourceFile<ModuleOp>(input_file, &context);

  int it = 0;


  populateSolver(module.get()[0]);

}

int main(int argc, char **argv){

  string input = argv[1];

  cout << "input file: " << input << endl;

  parse_fsm(input);

  return 0;

}