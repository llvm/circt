#include "fsm-verification.h"
#include "circt/Dialect/FSM/FSMGraph.h"
#include </Users/luisa/z3/src/api/c++/z3++.h>
#include "llvm/ADT/DenseMap.h"

#define VERBOSE 1

using namespace llvm;
using namespace mlir;
using namespace circt;
using namespace std;
using namespace z3;

vector<mlir::Value> getVarValues(Operation &op){
  vector<mlir::Value> vec;
  op.walk([&](fsm::VariableOp v){
    vec.push_back(v.getResult());
  });
  return vec;
}


void printMap(llvm::DenseMap<mlir::Value, expr> map){
  llvm::outs()<<"------------------------ EXPRESSION MAP ------------------------"<<"\n";
  for (auto m: map){
    llvm::outs()<<m.first<<"\n";
  }
  llvm::outs()<<"----------------------------------------------------------------"<<"\n";

}

void printTransitions(vector<transition> transitions){
  llvm::outs()<<"------------------------ TRANSITIONS ------------------------"<<"\n";
  for (auto t: transitions){
    llvm::outs()<<"from "<<t.from<<"\n";
    llvm::outs()<<"to "<<t.to<<"\n";
    // llvm::outs()<<"guards: "<<t.guards.size()<<"\n";

    // llvm::outs()<<"actions: "<<t.actions.size()<<"\n";

  }
  llvm::outs()<<"----------------------------------------------------------------"<<"\n";

}



expr manage_comb_exp(Operation &op, vector<expr> vec, z3::context &c){
  // llvm::outs()<<"comb expression "<< op <<"\n";
  // llvm::outs()<<"VEC contains: "<<vec.size()<<"\n";
  // for(auto v: vec){
  //   llvm::outs()<<v.to_string()<<"\n";
  // }
  if(auto add = dyn_cast<comb::AddOp>(op)){
    // int n = add.getNumOperands();
    // auto ops = add.getOperands();
    // mlir::Value new_var = add.getResult();
    // llvm::outs()<<"vec 0 "<<vec[0].get_sort().to_string()<<"\n";
    // llvm::outs()<<"vec 1 "<<vec[1].to_string()<<"\n";

    // expr r = vec[0];
    // for (int i = 1; i < (int)vec.size(); i++)
    //     r = to_expr(c, r + vec[i]);
    // vector<Z3_ast> arr;
    // for (int i = 0; i < (int)vec.size(); i++){
    //   arr.push_back(vec[i]);
    // }
    // // z3Fun G = [](vector<expr> vec) -> expr {
    // //                     return expr(vec[0] + vec[1]);
    // //                   };
    // // llvm::outs()<<"add "<<to_expr(c, vec[0]+vec[1]).to_string()<<"\n";
    // // llvm::outs()<<"add "<<vec[1].to_string()<<"\n";
    // return to_expr(c, Z3_mk_add(c, arr.size(), &arr[0]));
    return to_expr(c, vec[0] + vec[1]);
  } 
  else if(auto and_op = dyn_cast<comb::AndOp>(op)){
    // z3Fun G = [](vector<expr> vec) -> expr {
    //                     return expr(vec[0] && vec[1]);
    //                   };
    // return G;
    return expr(vec[0] && vec[1]);
    }
  else if(auto icmp = dyn_cast<comb::ICmpOp>(op)){
    // TODO switch case
    // llvm::outs()<<"vec 0 "<<vec[0].get_sort().to_string()<<"\n";
    // llvm::outs()<<"vec 1 "<<vec[1].get_sort().to_string()<<"\n";
    return expr(vec[0] == vec[1]);
  }
}

expr getExpr(mlir::Value v, llvm::DenseMap<mlir::Value, expr> expr_map, z3::context& c){
  // llvm::outs()<<"\nget expr "<<v<<"\n";
  if(expr_map.find(v) != expr_map.end()){
    // llvm::outs()<<"found "<<expr_map.at(v).to_string()<<"\n";
    return expr_map.at(v);
  } else if(auto constop = dyn_cast<hw::ConstantOp>(v.getDefiningOp())){
    // llvm::outs()<<"constant "<<constop.getValue().getSExtValue()<<"\n";
    return c.int_val(constop.getValue().getSExtValue());
  } else{
    llvm::outs()<<"ERROR: variable "<<v<<" not found in the expression map\n";
  }
}

expr getGuardExpr(llvm::DenseMap<mlir::Value, expr> expr_map, Region& guard, z3::context& c){
  // llvm::outs()<<"------------------------ GUARD expr_map ------------------------"<<"\n";
  // for (auto m: expr_map){
  //   llvm::outs()<<m.first<<"\n";
  //   llvm::outs()<<m.second.to_string()<<"\n";
  // }
  // llvm::outs()<<"region "<< guard.getOps().empty()<<"\n";
  for(auto &op: guard.getOps()){
    // llvm::outs()<<"ooop "<<op<<"\n";

    if (auto retop = dyn_cast<fsm::ReturnOp>(op)){
      // llvm::outs()<<retop.getOperand()<<"\n";
      return expr_map.at(retop.getOperand());
    } 
    vector<expr> vec;
    for (auto operand: op.getOperands()){
      // llvm::outs()<<"operand "<<operand<<"\n";
      // llvm::outs()<<"expr "<<getExpr(operand, expr_map, c).to_string()<<"\n";
        // llvm::outs()<<"call getExpr 2\n";

      vec.push_back(getExpr(operand, expr_map, c));
    }
    // llvm::outs()<<"op result "<<op.getResult(0) <<"\n";
    // llvm::outs()<<"inserting "<<op.getResult(0)<<"\n";
    //   llvm::outs()<<"mapped to "<<manage_comb_exp(op, vec, c).to_string()<<"\n";
    expr_map.insert({op.getResult(0), manage_comb_exp(op, vec, c)});
  }
  return expr(c.bool_const("true"));
}

vector<mlir::Value> actionsCounter(Region& action){
  vector<mlir::Value> to_update;
  for(auto &op: action.getOps()){
    if (auto updateop = dyn_cast<fsm::UpdateOp>(op)){
      to_update.push_back(updateop.getOperands()[0]);
    }
  }
  return to_update;
}


expr getActionExpr(llvm::DenseMap<mlir::Value, expr> expr_map, Region& action, context& c, int num){
  llvm::outs()<<"\n ACTION expr_map\n ";
  for (auto m: expr_map){
    llvm::outs()<<m.first<<"\n";
    llvm::outs()<<m.second.to_string()<<"\n";
  }
  // z3Fun a;

  int num_up = 0;

  for(auto &op: action.getOps()){
    llvm::outs()<<op<<"\n";
    if (auto updateop = dyn_cast<fsm::UpdateOp>(op)){
      llvm::outs()<<"update "<<updateop<<"\n";
      llvm::outs()<<"operand 0 "<<updateop.getOperands()[0]<<"\n";
      llvm::outs()<<"operand 1 "<<updateop.getOperands()[1]<<"\n";

      llvm::outs()<<"expr "<<getExpr(updateop.getOperands()[1], expr_map, c).to_string()<<"\n";
      // llvm::outs()<<expr_map.find(updateop.getOperands()[0])->first<<"\n";
      // z3Fun a = [&](vector<expr> vec) -> expr {
      //       llvm::DenseMap<mlir::Value, expr> expr_map_tmp;
      //       for(auto [value, expr]: llvm::zip(vecVal, vec)){
      //         expr_map_tmp.insert({value, expr});
      //       }
      //       // expr_map_tmp.find(updateop.getOperands()[0])->second = getExpr(updateop.getOperands()[1], expr_map, c);
      //       return getExpr(updateop.getOperands()[1], expr_map, c);
      //     };
      if(num_up == num){
        llvm::outs()<<"returning "<<getExpr(updateop.getOperands()[1], expr_map, c).to_string()<<"\n";
        return getExpr(updateop.getOperands()[1], expr_map, c);
      }
      num_up++;
      // return a;
    } else {
      llvm::outs()<<"AO op result "<<op <<"\n";
      vector<expr> vec;
      for (auto operand: op.getOperands()){
        vec.push_back(getExpr(operand, expr_map, c));
      }
      expr_map.insert({op.getResult(0), manage_comb_exp(op, vec, c)});
    }
  }
}

void printSolverAssertions(z3::solver& solver) {
  llvm::outs()<<"------------------------ SOLVER ------------------------"<<"\n";
  llvm::outs()<<solver.to_smt2()<<"\n";
  llvm::outs()<<"--------------------------------------------------------"<<"\n";
}

void recOpsMgmt(Operation &mod, context &c, vector<expr> &arguments, llvm::DenseMap<llvm::StringRef, func_decl> &stateInvariants, llvm::DenseMap<mlir::Value, expr> &expr_map, llvm::DenseMap<mlir::Value, expr> &const_map, llvm::DenseMap<mlir::Value, expr> &var_map, vector<mlir::Value> &outputs, string &initialState, vector<transition> &transitions, vector<mlir::Value> &vecVal, solver &s, llvm::DenseMap<llvm::StringRef, int> &state_map){
    
    for (Region &rg : mod.getRegions()) {
    // llvm::outs()<<"region "<<"\n";
    for (Block &block : rg) {
      // store inputs
      int num_args=0;
      // llvm::outs()<<"block "<<"\n";
      for(auto a: block.getArguments()){
        expr input = c.bool_const(("arg"+to_string(num_args)).c_str());
        arguments.push_back(input);
        // llvm::outs()<<"inserting "<<a<<"\n";
        // llvm::outs()<<"mapped to "<<input.to_string()<<"\n";
        expr_map.insert({a, input});
        var_map.insert({a, input});
        num_args++;
        // llvm::outs()<<"input "<<a<<"\n";
      }


      int state_num=0;
      for (Operation &op : block) {
        // llvm::outs()<<"------------------------ INVARIANTS MAP ------------------------"<<"\n";
        // for (auto m: stateInvariants){
        //   llvm::outs()<<m.first<<"\n";
        //   llvm::outs()<<m.second.to_string()<<"\n";
        // }
        // llvm::outs()<<op<<"\n";
        if (auto state = dyn_cast<fsm::StateOp>(op)){
          // llvm::outs()<<"state "<<state<<"\n";
          llvm::StringRef currentState = state.getName();
          // llvm::outs()<<"state "<<currentState.c_str()<<"\n";
          func_decl I = c.function(currentState.str().c_str(), c.int_sort(), c.bool_sort());
          stateInvariants.insert({currentState, I});
          // llvm::outs()<<"inserting "<<currentState<<"\n";
          // llvm::outs()<<"mapped to "<<I.to_string()<<"\n";
          state_map.insert({currentState, state_num});
          state_num++;
          auto regions = state.getRegions();

          int num_op=0;

          // transitions region
          for (Block &bl1: *regions[1]){
            for (Operation &op: bl1.getOperations()){
              if(auto transop = dyn_cast<fsm::TransitionOp>(op)){
                transition t;
                t.from = currentState;
                // char* to = new char[transop.getNextState().size()+1];
                // llvm::outs()<<"NEXT STATE "<<transop.getNextState()<<"\n";
                // strcpy(to, transop.getNextState().str().c_str());
                t.to = transop.getNextState();
                t.isGuard = false;
                t.isAction = false;

                auto trRegions = transop.getRegions();
                string nextState = transop.getNextState().str();
                
                // guard

                if(!trRegions[0]->empty()){

                  // llvm::outs()<<"transition empty "<<trRegions[0]->getOps().empty()<<"\n";
                  
                  // expr ret = getGuardExpr(expr_map, *trRegions[0], c);
                  Region &r = *trRegions[0];

                  z3Fun g = [&](vector<expr> vec) {
                    llvm::DenseMap<mlir::Value, expr> expr_map_tmp;
                    for(auto [value, expr]: llvm::zip(vecVal, vec)){
                      expr_map_tmp.insert({value, expr});
                    }
                    return getGuardExpr(expr_map_tmp, r, c);
                  };

                  // auto x = c.int_const(("x"+to_string(num_op)).c_str());

                  t.guard = g;

                  t.isGuard = true;

                  t.guard_reg = trRegions[0];
                }

                // action 

                if(!trRegions[1]->empty()){

                  vector<mlir::Value> to_update = actionsCounter(*trRegions[1]);

                  llvm::outs()<<"region has "<<to_update.size()<<" updates\n";

                  Region &r = *trRegions[1];

                  for(int nu=0; nu<to_update.size(); nu++){
                    z3FunA a = [&](expr e) -> expr {
                        llvm::DenseMap<mlir::Value, expr> expr_map_tmp;
                        // expr_map_tmp.find(updateop.getOperands()[0])->second = getExpr(updateop.getOperands()[1], expr_map, c);
                        return getActionExpr(expr_map_tmp, r, c, nu);
                      };
                    t.var_updates.insert({to_update[nu], a});
                  }



                  t.isAction = true;
                  // llvm::outs()<<"AAAaction "<<t.action({c.int_const("int"), c.bool_const("const")}).to_string()<<"\n";

                  // t.isAction = true;
                  // t.action = a;
                  // llvm::outs()<<"AAAaction "<<t.action({c.int_const("int"), c.bool_const("const")}).to_string()<<"\n";
                }

                transitions.push_back(t);
                

              } else {
                llvm ::outs()<<"ERROR: transition region should only contain transitions\n";
              }
            }
          }
        } else if(auto var_op = dyn_cast<fsm::VariableOp>(op)){
          // llvm::outs()<<"inserting "<<var_op.getResult()<<"\n";
          // llvm::outs()<<"mapped to "<<c.int_const(var_op.getName().str().c_str()).to_string()<<"\n";
          var_map.insert({var_op.getResult(), c.int_const(var_op.getName().str().c_str())});
          expr_map.insert({var_op.getResult(), c.int_const(var_op.getName().str().c_str())});
        } else if(auto machine = dyn_cast<fsm::MachineOp>(op)){
          initialState = machine.getInitialState();
          vecVal = getVarValues(op);
          recOpsMgmt(op, c, arguments, stateInvariants, expr_map, const_map, var_map, outputs, initialState, transitions, vecVal, s, state_map);
        }
      }
    }
  }

}

void populateSolver(Operation &mod){

  // using transition = std::pair<string, string>;

  // rename_states(cfg);

  z3::context c;

  solver s(c);

  vector<expr> arguments;
  llvm::DenseMap<llvm::StringRef, func_decl> stateInvariants;
  // llvm::DenseMap<string, z3::func_decl> stateInvariants;
  llvm::DenseMap<mlir::Value, expr> expr_map;
  llvm::DenseMap<mlir::Value, expr> const_map;
  llvm::DenseMap<mlir::Value, expr> var_map;
  llvm::DenseMap<llvm::StringRef, int> state_map;

  // llvm::DenseMap<transition, z3Fun> guard_map;
  // llvm::DenseMap<transition, z3Fun> action_map;

  vector<mlir::Value> outputs;
  string initialState;

  vector<transition> transitions;

  vector<mlir::Value> vecVal;


  expr x = c.bv_const("x", 32);

  recOpsMgmt(mod, c, arguments, stateInvariants, expr_map, const_map, var_map, outputs, initialState, transitions, vecVal, s, state_map);

  // printSolverAssertions(s);

  printTransitions(transitions);

  // llvm::outs()<<"------------------------ INVARIANTS MAP ------------------------"<<"\n";
  // for (auto m: stateInvariants){
  //   llvm::outs()<<m.first<<"\n";
  //   llvm::outs()<<m.second.to_string()<<"\n";
  // }

  // first declare variables 

  vector<expr> solver_vars;

  // expr x = c.bv_const("x", 32);
  // expr go = c.bool_const("go");

  int it =0;

  // llvm::outs()<<"map size "<<var_map.size()<<"\n";
  // for(auto var: var_map){
  //   llvm::outs()<<"iter "<<it<<"\n";
  //   llvm::outs()<<"var "<<var.first<<"\n";
  //   auto new_var = c.bv_const(("var"+to_string(it)).c_str(), 32);
  //   it++;
  // }

  // solver_vars.push_back(c.int_const(("var"+to_string(it)).c_str()));
  // solver_vars.push_back(c.bool_const(("var"+to_string(it+1)).c_str()));


  vector< vector <z3Fun> > guards_matrix[state_map.size()][state_map.size()];
  vector< vector <vector<z3Fun> > > actions_matrix[state_map.size()][state_map.size()][state_map.size()];

  llvm::outs()<<"print state map"<<'\n';  
  for(auto s: state_map){
    llvm::outs()<<s.first<<"\n";
    llvm::outs()<<s.second<<"\n";
  }



  llvm::outs()<<"print var_map"<<'\n';
  for(auto e: var_map){
    llvm::outs()<<e.first<<"\n";
    llvm::outs()<<e.second.to_string()<<"\n";
    solver_vars.push_back(e.second);
  }

    llvm::outs()<<"print solver vars"<<'\n';
  for(auto v: solver_vars){
    llvm::outs()<<v.to_string()<<"\n";
  }

  s.add(forall(solver_vars[0], stateInvariants.at(transitions[0].from)(0)));
  int id=0;
  llvm::outs()<<"print update vars"<<'\n';
  for(auto t:transitions){
    llvm::outs()<<"transition "<<id<<"\n";
    for (auto up: t.var_updates){
      z3FunA a = up.second;
      llvm::outs()<<"function "<<a(expr_map.at(up.first)).to_string()<<"\n";
    }

  }
      


  llvm::outs()<<"----------------------------- INSERTIONS -----------------------------\n";
  for(auto t: transitions){
    int row = state_map.at(t.from);
    int col = state_map.at(t.to);
    llvm::outs()<<"from "<<row<<"\n";
    llvm::outs()<<"to "<<col<<"\n";
    if(t.isGuard && t.isAction){
      // llvm::outs()<<"-------- action&guard "<< t.action(solver_vars).to_string() <<"\n";
      for (auto up: t.var_updates){
          z3FunA a = up.second;


          llvm::outs()<<"function "<<a(x).to_string()<<"\n";

        for(auto v: solver_vars){



          // llvm::outs()<<"action z3fun "<<a(solver_varsexpr_map.at(up.first)).to_string()<<"\n";

          s.add(forall(v, implies((stateInvariants.at(t.from)(v) && t.guard(solver_vars)), stateInvariants.at(t.to)(a(expr_map.at(up.first))))));

        }
      }
      
      // s.add(forall(solver_vars[0], implies((stateInvariants.at(t.from)(solver_vars[0]) && t.guard(solver_vars)), stateInvariants.at(t.to)(t.action(solver_vars)))));
    
    } else if (t.isGuard){
      llvm::outs()<<"GUARD "<< t.guard(solver_vars).to_string() <<"\n";
      for(auto v: solver_vars){
        s.add(forall(v, implies((stateInvariants.at(t.from)(v) && t.guard(solver_vars)), stateInvariants.at(t.to)(v))));

      }
      // s.add(forall(solver_vars[0], implies((stateInvariants.at(t.from)(solver_vars[0]) && t.guard(solver_vars)), stateInvariants.at(t.to)(solver_vars[0]))));
    } else if (t.isAction){
      for (auto up: t.var_updates){
        llvm::outs()<<"action expr "<<up.first<<"\n";
        // s.add(forall(solver_vars[0], implies((stateInvariants.at(t.from)(solver_vars[0]) && t.guard(solver_vars)), stateInvariants.at(t.to)(t.action(solver_vars)))));

      }
      // s.add(forall(solver_vars[0], implies(stateInvariants.at(t.from)(solver_vars[0]), stateInvariants.at(t.to)(t.action(solver_vars)))));
    // else if (t.guards.size()>0){
      // llvm::outs()<<"guard "<< t.guards[0](solver_vars).to_string() <<"\n";
      // s.add(forall(solver_vars[0], solver_vars[1], implies((stateInvariants.at(t.from)(solver_vars[0], solver_vars[1]) && t.guards[0](solver_vars)), stateInvariants.at(t.to)(solver_vars[0], solver_vars[1]))));
    //   s.add(forall(solver_vars[0], solver_vars[1], implies(stateInvariants.at(t.from)(solver_vars[0], solver_vars[1]) && t.guards[0](solver_vars), t.guards[1](solver_vars))), stateInvariants.at(t.to)(solver_vars[0], solver_vars[1]));
    // } else if (t.actions.size()>0){
    //   s.add(forall(solver_vars[0], solver_vars[1], implies(stateInvariants.at(t.from)(solver_vars[0], solver_vars[1]), t.guards[1](solver_vars))), stateInvariants.at(t.to)(solver_vars[0], solver_vars[1])&& t.actions[0](solver_vars));
    // } else if (t.actions.size()>0){
    //   s.add(forall(solver_vars[0], solver_vars[1], implies(stateInvariants.at(t.from)(solver_vars[0], solver_vars[1]) && t.guards[0](solver_vars), t.guards[1](solver_vars))), stateInvariants.at(t.to)(solver_vars[0], solver_vars[1]));
    // } else if (t.actions.size()>0){
    //   s.add(forall(solver_vars[0], solver_vars[1], implies(stateInvariants.at(t.from)(solver_vars[0], solver_vars[1]), t.guards[1](solver_vars))), stateInvariants.at(t.to)(solver_vars[0], solver_vars[1])&& t.actions[0](solver_vars));
    // s.add(forall(solver_vars[0], solver_vars[1], implies(stateInvariants.at(t.from)(solver_vars[0], solver_vars[1]), stateInvariants.at(t.to)(t.actions[0](solver_vars)))));  
    } else {
      llvm::outs()<<"inv "<<stateInvariants.at(t.from).to_string()<<"\n";
      s.add(forall(solver_vars[0],  implies(stateInvariants.at(t.from)(solver_vars[0]), stateInvariants.at(t.to)(solver_vars[0]))));
    }
  }

  printSolverAssertions(s);

  



}



void parse_fsm(string input_file){

  DialectRegistry registry;

  // CFG *cfg = new CFG();

  // registerAllDialects(registry);

  // clang-format off
  registry.insert<
    comb::CombDialect,
    fsm::FSMDialect,
    hw::HWDialect
  >();

  llvm::DenseMap<mlir::Value, expr> map;

  // llvm::DenseMap<pair<mlir::Value, mlir::Value>, std::function<vector<mlir::Value>(expr)> > guards;
  llvm::DenseMap<mlir::Value, expr> expr_map;
  llvm::DenseMap<mlir::Value, expr> var_map;

  // cfg->map = map;

  cout << "parsing:\n" << input_file << endl;

  MLIRContext context(registry);

  // Parse the MLIR code into a module.
  OwningOpRef<ModuleOp> module = mlir::parseSourceFile<ModuleOp>(input_file, &context);

  int it = 0;


  populateSolver(module.get()[0]);

}

int main(int argc, char **argv){
  
  InitLLVM y(argc, argv);

  parse_fsm("/Users/luisa/circt/integration_test/Dialect/FSM/variable/top.mlir");

  return 0;

}