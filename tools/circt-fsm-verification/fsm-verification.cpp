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
    llvm::outs()<<"guards: "<<t.guards.size()<<"\n";

    llvm::outs()<<"actions: "<<t.actions.size()<<"\n";

  }
  llvm::outs()<<"----------------------------------------------------------------"<<"\n";

}



expr manage_comb_exp(Operation &op, vector<expr> vec, context &c){
  // llvm::outs()<<"comb expression "<< op <<"\n";
  llvm::outs()<<"VEC contains: "<<vec.size()<<"\n";
  for(auto v: vec){
    llvm::outs()<<v.to_string()<<"\n";
  }
  if(auto add = dyn_cast<comb::AddOp>(op)){
    // int n = add.getNumOperands();
    // auto ops = add.getOperands();
    // mlir::Value new_var = add.getResult();
    llvm::outs()<<"vec 0 "<<vec[0].get_sort().to_string()<<"\n";
    llvm::outs()<<"vec 1 "<<vec[1].to_string()<<"\n";

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

expr getExpr(mlir::Value v, llvm::DenseMap<mlir::Value, expr> expr_map, context& c){
  llvm::outs()<<"get expr "<<v<<"\n";
  if(expr_map.find(v) != expr_map.end()){
    llvm::outs()<<"found "<<expr_map.at(v).to_string()<<"\n";
    return expr_map.at(v);
  } else if(auto constop = dyn_cast<hw::ConstantOp>(v.getDefiningOp())){
    llvm::outs()<<"constant "<<constop.getValue().getSExtValue()<<"\n";
    return c.int_val(constop.getValue().getSExtValue());
  } else{
    llvm::outs()<<"ERROR: variable "<<v<<" not found in the expression map\n";
  }
}

expr getGuardExpr(llvm::DenseMap<mlir::Value, expr> expr_map, Region& guard, context& c){
  for(auto &op: guard.getOps()){
    // llvm::outs()<<"ooop "<<op<<"\n";

    if (auto retop = dyn_cast<fsm::ReturnOp>(op)){
      llvm::outs()<<retop.getOperand()<<"\n";
      return expr_map.at(retop.getOperand());
    } 
    vector<expr> vec;
    for (auto operand: op.getOperands()){
      // llvm::outs()<<"operand "<<operand<<"\n";
      // llvm::outs()<<"expr "<<getExpr(operand, expr_map, c).to_string()<<"\n";
      vec.push_back(getExpr(operand, expr_map, c));
    }
    // llvm::outs()<<"op result "<<op.getResult(0) <<"\n";
    llvm::outs()<<"inserting "<<op.getResult(0)<<"\n";
      llvm::outs()<<"mapped to "<<manage_comb_exp(op, vec, c).to_string()<<"\n";
    expr_map.insert({op.getResult(0), manage_comb_exp(op, vec, c)});
  }
  return expr(c.bool_const("true"));
}


vector<z3Fun> getActionExpr(llvm::DenseMap<mlir::Value, expr> expr_map, Region& action, context& c, vector<mlir::Value> &vecVal){
  vector<z3Fun> actions;
  
  int num_op = 0;
  for(auto &op: action.getOps()){
    llvm::outs()<<op<<"\n";
    if (auto updateop = dyn_cast<fsm::UpdateOp>(op)){
      llvm::outs()<<"update "<<updateop<<"\n";
      expr_map.find(updateop.getOperands()[0])->second = getExpr(updateop.getOperands()[1], expr_map, c);
      z3Fun a = [&](vector<expr> vec) -> expr {
            llvm::DenseMap<mlir::Value, expr> expr_map_tmp;
            for(auto [value, expr]: llvm::zip(vecVal, vec)){
              expr_map_tmp.insert({value, expr});
            }
            return (updateop.getOperands()[1], expr_map, c);
          };
      actions.push_back(a);
    } else {
      // llvm::outs()<<"AO op result "<<op <<"\n";
      vector<expr> vec;
      for (auto operand: op.getOperands()){
        llvm::outs()<<"operand "<<operand<<"\n";
        llvm::outs()<<"expr "<<getExpr(operand, expr_map, c).to_string()<<"\n";
        vec.push_back(getExpr(operand, expr_map, c));
      }
      llvm::outs()<<"inserting "<<op.getResult(0)<<"\n";
      llvm::outs()<<"mapped to "<<manage_comb_exp(op, vec, c).to_string()<<"\n";
      expr_map.insert({op.getResult(0), manage_comb_exp(op, vec, c)});
    }
  }
  return actions;
}

void printSolverAssertions(z3::solver& solver) {
  llvm::outs()<<"------------------------ SOLVER ------------------------"<<"\n";
  llvm::outs()<<solver.to_smt2()<<"\n";
  llvm::outs()<<"--------------------------------------------------------"<<"\n";
}

void recOpsMgmt(Operation &mod, context &c, vector<expr> &arguments, llvm::DenseMap<char*, func_decl> &stateInvariants, llvm::DenseMap<mlir::Value, expr> &expr_map, llvm::DenseMap<mlir::Value, expr> &const_map, llvm::DenseMap<mlir::Value, z3::sort> &var_map, vector<mlir::Value> &outputs, string &initialState, vector<transition> &transitions, vector<mlir::Value> &vecVal, solver &s){
    
    for (Region &rg : mod.getRegions()) {
    // llvm::outs()<<"region "<<"\n";
    for (Block &block : rg) {
      // store inputs
      int num_args=0;
      // llvm::outs()<<"block "<<"\n";
      for(auto a: block.getArguments()){
        expr input = c.int_const(("arg"+to_string(num_args)).c_str());
        arguments.push_back(input);
        llvm::outs()<<"inserting "<<a<<"\n";
        llvm::outs()<<"mapped to "<<input.to_string()<<"\n";
        expr_map.insert({a, input});
        num_args++;
        // llvm::outs()<<"input "<<a<<"\n";
      }


      int state_num=0;
      for (Operation &op : block) {
        // llvm::outs()<<op<<"\n";
        if (auto state = dyn_cast<fsm::StateOp>(op)){
          // llvm::outs()<<"state "<<state<<"\n";
          char* currentState = new char[state.getName().size()+1];
          strcpy(currentState, state.getName().str().c_str());
          // llvm::outs()<<"state "<<currentState.c_str()<<"\n";
          func_decl I = c.function(currentState, c.int_sort(), c.bool_sort());
          stateInvariants.insert({currentState, I});
          // stateInvariants.insert({state_num, I});
          // state_num++;
          auto regions = state.getRegions();

          int num_op=0;

          // transitions region
          for (Block &bl1: *regions[1]){
            for (Operation &op: bl1.getOperations()){
              if(auto transop = dyn_cast<fsm::TransitionOp>(op)){
                transition t;
                t.from = currentState;
                t.to = transop.getNextState().str();

                auto trRegions = transop.getRegions();
                string nextState = transop.getNextState().str();
                
                // guard

                if(!trRegions[0]->empty()){
                  z3Fun g = [&](vector<expr> vec) -> expr {
                    llvm::DenseMap<mlir::Value, expr> expr_map_tmp;
                    for(auto [value, expr]: llvm::zip(vecVal, vec)){
                      expr_map_tmp.insert({value, expr});
                    }
                    return getGuardExpr(expr_map_tmp, *trRegions[0], c);
                  };

                  auto x = c.int_const(("x"+to_string(num_op)).c_str());

                  t.guards.push_back(g);
                }

                

                // llvm::outs()<<"guard "<<g({x}).decl().name().str()<<"\n";

                // s.add(forall(x, g({x})));

                // action

                // z3Fun a = [&](vector<expr> vec) -> expr {
                //       llvm::DenseMap<mlir::Value, expr> expr_map_tmp;
                //       for(auto [value, expr]: llvm::zip(vecVal, vec)){
                //         expr_map_tmp.insert({value, expr});
                //       }
                //       return getActionExpr(expr_map_tmp, *trRegions[1], c);
                //     };

                // s.add(I(a({x})));

                // llvm::outs()<<I(a({x})).to_string()<<"\n";

                // llvm::outs()<<"action "<<a({x}).decl().name().str()<<"\n";

                if(!trRegions[1]->empty()){
                  t.actions = getActionExpr(expr_map, *trRegions[1], c, vecVal);
                }

                

                // s.add(forall(x, I(a({x}))));

                transitions.push_back(t);
                

              } else {
                llvm ::outs()<<"ERROR: transition region should only contain transitions\n";
              }
            }
          }
        } else if(auto const_op = dyn_cast<hw::ConstantOp>(op)){
          // llvm::outs()<<"inserting "<<const_op.getResult()<<"\n";
          // llvm::outs()<<"mapped to "<<c.int_val(const_op->getName().getStringRef().str().c_str()).to_string()<<"\n";
          // const_map.insert({const_op.getResult(), c.int_val(const_op->getName().getStringRef().str().c_str())});
          // expr_map.insert({const_op.getResult(), c.int_val(const_op->getName().getStringRef().str().c_str())});
        } else if(auto var_op = dyn_cast<fsm::VariableOp>(op)){
          llvm::outs()<<"inserting "<<var_op.getResult()<<"\n";
          llvm::outs()<<"mapped to "<<c.int_const(var_op.getName().str().c_str()).to_string()<<"\n";
          var_map.insert({var_op.getResult(), c.int_sort()});
          expr_map.insert({var_op.getResult(), c.int_const(var_op.getName().str().c_str())});
        } else if(auto machine = dyn_cast<fsm::MachineOp>(op)){
          initialState = machine.getInitialState();
          vecVal = getVarValues(op);
          recOpsMgmt(op, c, arguments, stateInvariants, expr_map, const_map, var_map, outputs, initialState, transitions, vecVal, s);
        }
      }
    }
  }

  // llvm::outs()<<s.check()<<"\n";

  // printSolverAssertions(s);


}

void populateSolver(Operation &mod){

  using z3fun = expr (*)(vector<expr>);

  // using transition = std::pair<string, string>;

  // rename_states(cfg);

  z3::context c;

  solver s(c);

  vector<expr> arguments;
  llvm::DenseMap<char*, func_decl> stateInvariants;
  // llvm::DenseMap<string, z3::func_decl> stateInvariants;
  llvm::DenseMap<mlir::Value, expr> expr_map;
  llvm::DenseMap<mlir::Value, expr> const_map;
  llvm::DenseMap<mlir::Value, z3::sort> var_map;
  // llvm::DenseMap<transition, z3Fun> guard_map;
  // llvm::DenseMap<transition, z3Fun> action_map;

  vector<mlir::Value> outputs;
  string initialState;

  vector<transition> transitions;

  vector<mlir::Value> vecVal;




  recOpsMgmt(mod, c, arguments, stateInvariants, expr_map, const_map, var_map, outputs, initialState, transitions, vecVal, s);

  // printSolverAssertions(s);

  printTransitions(transitions);


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

  z3::context c;

  // explore_nested_blocks(module.get()[0], it, cfg, expr_map, c);

  // print_cfg(cfg);

  populateSolver(module.get()[0]);


  // printSolverAssertions(s);

}

int main(int argc, char **argv){
  
  InitLLVM y(argc, argv);

  parse_fsm("/Users/luisa/circt/integration_test/Dialect/FSM/variable/top.mlir");

  return 0;

}