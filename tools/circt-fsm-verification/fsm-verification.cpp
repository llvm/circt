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
vector<mlir::Value> getVarValues(Operation &op){
  vector<mlir::Value> vec;
  op.walk([&](fsm::VariableOp v){
    vec.push_back(v.getResult());
  });
  return vec;
}

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
    // TODO switch case
    return expr(vec[0] == vec[1]);
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

/**
 * @brief Returns updates for expressions from corresponding region
*/
expr getActionExpr(llvm::DenseMap<mlir::Value, expr> expr_map, Region& action, context& c, expr v){
  if(VERBOSE){
    llvm::outs()<<"------------------------ ACTION expr_map ------------------------"<<"\n";
    for (auto m: expr_map){
      llvm::outs()<<m.first<<"\n";
      llvm::outs()<<m.second.to_string()<<"\n";
    }
  }
  for(auto &op: action.getOps()){
    if (auto updateop = dyn_cast<fsm::UpdateOp>(op)){
      if(expr_map.find(updateop.getOperands()[0])!= expr_map.end() && (expr_map.at(updateop.getOperands()[0])==v)){
        return getExpr(updateop.getOperands()[1], expr_map, c);
      }
    } else {
      vector<expr> vec;
      for (auto operand: op.getOperands()){
        vec.push_back(getExpr(operand, expr_map, c));
      }
      expr_map.insert({op.getResult(0), manage_comb_exp(op, vec, c)});
    }
  }
  return v;
}


/**
 * @brief FSM operators management
*/
void recOpsMgmt(Operation &mod, context &c, llvm::DenseMap<llvm::StringRef, func_decl> &stateInvariants, llvm::DenseMap<mlir::Value, expr> &expr_map, llvm::DenseMap<mlir::Value, expr> &var_map, vector<mlir::Value> &outputs, string &initialState, vector<transition> &transitions, vector<mlir::Value> &vecVal, solver &s, llvm::DenseMap<llvm::StringRef, int> &state_map){
    
    for (Region &rg : mod.getRegions()) {
    for (Block &block : rg) {
      // store inputs
      int num_args=0;
      for(auto a: block.getArguments()){
        expr input = c.bool_const(("arg"+to_string(num_args)).c_str());

        if(VERBOSE){
          llvm::outs()<<"inserting input "<<a<<"\n";
          llvm::outs()<<"mapped to "<<input.to_string()<<"\n";
        }
        expr_map.insert({a, input});
        var_map.insert({a, input});
        vecVal.push_back(a);
        num_args++;
      }

      int state_num=0;
      for (Operation &op : block) {

        if (auto state = dyn_cast<fsm::StateOp>(op)){
          llvm::StringRef currentState = state.getName();
          func_decl I = c.function(currentState.str().c_str(), c.int_sort(), c.bool_sort());
          stateInvariants.insert({currentState, I});
          if(VERBOSE){
            llvm::outs()<<"inserting state "<<currentState<<"\n";
            llvm::outs()<<"mapped to "<<I.to_string()<<"\n";
          }

          state_map.insert({currentState, state_num});

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

                  z3Fun g = [&](vector<expr> vec) {
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

                  z3FunA a = [&](vector<expr> vec) -> vector<expr> {

                    vector<expr> vec2;

                    llvm::DenseMap<mlir::Value, expr> expr_map_tmp;
                    for(auto [value, expr]: llvm::zip(vecVal, vec)){
                      expr_map_tmp.insert({value, expr});
                    }

                    for(auto v: vec){
                        vec2.push_back(getActionExpr(expr_map_tmp, r, c, v));
                    };

                    return vec2;
                  };

                  t.action = a;

                  t.isAction = true;
                }


                transitions.push_back(t);


              } else {
                llvm ::outs()<<"ERROR: transition region should only contain transitions\n";
              }
            }
          }
        } else if(auto var_op = dyn_cast<fsm::VariableOp>(op)){

          int initial_value = var_op.getInitValue().cast<IntegerAttr>().getInt();
          string var_name = var_op.getName().str();
          if(var_op.getName().str().find("arg") != std::string::npos){
            // preserved keyword arg for arguments to avoid ambiguity when setting initial state values
            var_name = "var"+to_string(num_args);
            num_args++;
          }
          if(VERBOSE){
            llvm::outs()<<"inserting variable "<<var_op.getResult()<<"\n";
            llvm::outs()<<"mapped to "<<(var_name+"_"+to_string(initial_value)).c_str()<<"\n";
          }
          var_map.insert({var_op.getResult(), c.int_const((var_name+"_"+to_string(initial_value)).c_str())});
          expr_map.insert({var_op.getResult(), c.int_const((var_name+"_"+to_string(initial_value)).c_str())});
        } else if(auto machine = dyn_cast<fsm::MachineOp>(op)){
          initialState = machine.getInitialState();
          vector<mlir::Value> vecVal2 = getVarValues(op);
          for(auto v: vecVal2){
            vecVal.push_back(v);
          }

          recOpsMgmt(op, c, stateInvariants, expr_map, var_map, outputs, initialState, transitions, vecVal, s, state_map);

        }
      }
    }
  }


}


void populateSolver(Operation &mod){

  z3::context c;

  solver s(c);

  llvm::DenseMap<llvm::StringRef, func_decl> stateInvariants;
  llvm::DenseMap<mlir::Value, expr> expr_map;
  llvm::DenseMap<mlir::Value, expr> var_map;

  llvm::DenseMap<llvm::StringRef, int> state_map;


  vector<mlir::Value> outputs;
  string initialState;

  vector<transition> transitions;

  vector<mlir::Value> vecVal;

  recOpsMgmt(mod, c, stateInvariants, expr_map, var_map, outputs, initialState, transitions, vecVal, s, state_map);

  vector<expr> solver_vars;



  if(VERBOSE){
    llvm::outs()<<"print state map"<<'\n';  
    for(auto s: state_map){
      llvm::outs()<<s.first<<"\n";
      llvm::outs()<<s.second<<"\n";
    }
    llvm::outs()<<"print solver vars"<<'\n';
    int i=0;
    for(auto v: var_map){
      solver_vars.push_back(c.int_const(v.second.to_string().c_str()));
      llvm::outs()<<solver_vars[i].to_string()<<"\n";
      if(v.second.to_string().find("arg") == std::string::npos){
        // if it is a variable and not an argument
        int init_value = stoi(v.second.to_string().substr(v.second.to_string().find("_")+1));
        // initialize variables initial state
        s.add(forall(solver_vars[i], stateInvariants.at(transitions[0].from)(init_value)));
      }
      i++;
    }
  }



  for(auto t: transitions){

    if(VERBOSE){
      llvm::outs()<<"transition from "<<t.from<<" to "<<t.to<<"\n";
    }
    int row = state_map.at(t.from);
    int col = state_map.at(t.to);
    if(t.isGuard && t.isAction){
      int idx_w = 0;
      for (auto v: solver_vars){
          s.add(forall(v, implies((stateInvariants.at(t.from)(v) && t.guard(solver_vars)), stateInvariants.at(t.to)(t.action(solver_vars).at(idx_w)))));
          idx_w++;
      }
    
    } else if (t.isGuard){
      for(auto v: solver_vars){
        s.add(forall(v, implies((stateInvariants.at(t.from)(v) && t.guard(solver_vars)), stateInvariants.at(t.to)(v))));
      }
    } else if (t.isAction){
      int idx_w = 0;
      for (auto v: solver_vars){
          s.add(forall(v, implies((stateInvariants.at(t.from)(v)), stateInvariants.at(t.to)(t.action(solver_vars).at(idx_w)))));
          idx_w++;
      }
    } else {
      for(auto v: solver_vars)
        s.add(forall(v, implies(stateInvariants.at(t.from)(v), stateInvariants.at(t.to)(v))));
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