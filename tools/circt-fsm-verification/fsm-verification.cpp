#include "circt/Dialect/FSM/FSMGraph.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Dialect/LTL/LTLDialect.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include <z3++.h>
#include <iostream>
#include <vector>
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "chrono"
#include "fstream"
#include "iostream"
#include "llvm/Support/raw_ostream.h"

#define V 0

using namespace llvm;
using namespace mlir;
using namespace circt;
using namespace std;
using namespace z3; 

using z3Fun = std::function <expr (expr)>;

using z3FunA = std::function <vector<expr> (expr)>;

struct transition{
  int from, to;
  z3Fun guard;
  bool isGuard, isAction, isOutput;
  z3FunA action, output;
};


/**
 * @brief Prints solver assertions
*/
void printSolverAssertions(z3::solver &solver, string output) {

	ofstream outfile;
	outfile.open(output, ios::app);



  if(V){
    llvm::outs()<<"---------------------------- SOLVER ----------------------------"<<"\n";
    llvm::outs()<<solver.to_smt2()<<"\n";
    llvm::outs()<<"------------------------ SOLVER RETURNS ------------------------"<<"\n";
    // llvm::outs()<<solver.check()<<"\n";4
  }
  // const auto start{std::chrono::steady_clock::now()};
  // int sat = solver.check();
  // const auto end{std::chrono::steady_clock::now()};
  outfile <<solver.to_smt2();
  // if(!V)
  //   llvm::outs()<<sat<<"\n";


  // const std::chrono::duration<double> elapsed_seconds{end - start};

  // if(V){
  //   llvm::outs()<<"--------------------------- INVARIANT --------------------------"<<"\n";
  //   llvm::outs()<<solver.get_model().to_string()<<"\n";
  //   llvm::outs()<<"-------------------------- END -------------------------------"<<"\n";
  //   llvm::outs()<<"Time taken: "<<elapsed_seconds.count()<<"s\n";
  // }

	// outfile << elapsed_seconds.count()<<","<<sat << endl;
	outfile.close();
}

void printTransitions(vector<transition> &transitions){
  llvm::outs()<<"\nPRINTING TRANSITIONS\n";
  for(auto t: transitions){
    llvm::outs()<<"\ntransition from "<<t.from<<" to "<<t.to<<"\n";
  }
}

void printExprValMap(vector<std::pair<expr, mlir::Value>> &expr_map){
  llvm::outs()<<"\n\nEXPR-VAL:";
  for(auto e: expr_map){
    llvm::outs()<<"\n\nexpr: "<<e.first.to_string()<<", value: "<<e.second;
  } 
  llvm::outs()<<"END\n";

}

/**
 * @brief Returns FSM's initial state
*/
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
  llvm::errs()<<"Initial state does not exist\n";

}

// /**
//  * @brief Returns list of values to be updated within an action region
// */
// vector<mlir::Value> actionsCounter(Region& action){
//   vector<mlir::Value> to_update;
//   for(auto &op: action.getOps()){
//     if (auto updateop = dyn_cast<fsm::UpdateOp>(op)){
//       to_update.push_back(updateop.getOperands()[0]);
//     }
//   }
//   return to_update;
// }

/**
 * @brief Returns expression from Comb dialect operator
*/
expr manage_comb_exp(Operation &op, vector<expr> &vec, z3::context &c){

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

// /**
//  * @brief Returns expression from densemap or constant operator
// */
// expr getExpr(mlir::Value v, vector<std::pair<expr, mlir::Value>> &expr_map, z3::context &c){

//   for(auto e: expr_map){
//     if (e.second==v)
//       return e.first;
//   }

//   if(auto constop = dyn_cast<hw::ConstantOp>(v.getDefiningOp())){
//     if(constop.getType().getIntOrFloatBitWidth()>1)
//       return c.int_val(constop.getValue().getSExtValue());
//     else
//       return c.bool_val(0);
//   }
//   llvm::errs()<<"Expression not found.";
// }

/**
 * @brief Returns guard expression for input region
*/
expr getGuard(vector<std::pair<mlir::Value, func_decl>> &variablesFunc, Region &guard, z3::context &c, expr time){

  // parse all operations and potential temporary results in an expression map
  // retrieve variables/arguments from their func_decl called at the right time 
  // retrieve the rest from constant values 
  // when parsing retop retrieve expression from map

  vector<std::pair<expr, mlir::Value>> exprMapTmp;


  for(auto &op: guard.getOps()){
    if (auto retop = dyn_cast<fsm::ReturnOp>(op)){
      for(auto e: exprMapTmp){
        if (e.second==retop.getOperand()){

          return e.first;
          
        }
      }
    } 

    vector<expr> tmpVec;

    for (auto operand: op.getOperands()){
      bool f = false;
      // search through variablesFunc

      if (auto blockop = dyn_cast<mlir::BlockArgument>(operand)){
        // block argument
        for(auto vf: variablesFunc){
          if(auto blockvf = dyn_cast<mlir::BlockArgument>(vf.first)){
            if(blockvf==blockop){
              tmpVec.push_back(vf.second(time));
              f=true;
            } 
          }
        }
      } else if (auto constop = dyn_cast<hw::ConstantOp>(operand.getDefiningOp())){

        if(constop.getType().getIntOrFloatBitWidth() > 1)
          tmpVec.push_back(c.int_val(constop.getValue().getSExtValue()));
        else
          tmpVec.push_back(c.bool_val(0));
        f=true;
      } else {
        // normal value
        for(auto vf: variablesFunc){
          if(vf.first==operand){
            tmpVec.push_back(vf.second(time));
            f=true;
          }
        }
      }


      if(!f)
        llvm::outs()<<"\n\nhuge mess here\n\n";
    }


    exprMapTmp.push_back({manage_comb_exp(op, tmpVec, c), op.getResult(0)});

    // printExprValMap(exprMapTmp);

  }
  return expr(c.bool_const("true"));
}

/**
 * @brief Returns output expression for input region
*/
vector<expr> getOutput(vector<std::pair<mlir::Value, func_decl>> &variablesFun, Region &output, context &c, expr time){
  vector<expr> outputVec; 
  vector<std::pair<expr, mlir::Value>> exprMapTmp;

  for(auto &op: output.getOps()){
    if (auto outop = dyn_cast<fsm::OutputOp>(op)){
      for (auto opr: outop.getOperands()){
        bool found = false;
        for(auto e: exprMapTmp){
          if (e.second==opr){
            outputVec.push_back(e.first);
            found = true;
          }
        }
        if(!found){
          for(auto vf: variablesFun){
            if(opr == vf.first){
              outputVec.push_back(vf.second(time));
              found = true;
            }
          }
        }
        if(!found)
          llvm::outs()<<"\n\nhuge mess here too 4\n\n";
      }
      return outputVec;
    } 
    else {
      vector<expr> tmpVec;
      for (auto operand: op.getOperands()){
        bool f = false;
        // search through variablesFunc
        for(auto vf: variablesFun){
          if(vf.first==operand){
            tmpVec.push_back(vf.second(time));
            f=true;
          }
        }
        // otherwise it is a constant
        if(auto constop = dyn_cast<hw::ConstantOp>(operand.getDefiningOp())){
          if(constop.getType().getIntOrFloatBitWidth() > 1)
            tmpVec.push_back(c.int_val(constop.getValue().getSExtValue()));
          else
            tmpVec.push_back(c.bool_val(0));
          f=true;
        }
        if(!f)
          llvm::outs()<<"\n\nhuge mess 2 here\n\n";
      }
      exprMapTmp.push_back({manage_comb_exp(op, tmpVec, c), op.getResult(0)});
    }
  }

}

/**
 * @brief Returns actions for all expressions for the input region
*/

vector<expr> getAction(vector<std::pair<mlir::Value, func_decl>> &variablesFun, Region &action, context &c, expr time){
  vector<expr> updatedVec;
  vector<std::pair<expr, mlir::Value>> exprMapTmp;
  
  for (auto v: variablesFun){
    bool found = false;
    for(auto &op: action.getOps()){
      if(!found){
        if (auto updateop = dyn_cast<fsm::UpdateOp>(op)){
          if(v.first == updateop.getOperands()[0]){

            auto newVal = updateop.getOperands()[1];

            bool f = false;

            for(auto e: exprMapTmp){
              if (e.second==newVal){
                updatedVec.push_back(e.first);
                f = true;
              }
            }
            if (!f){
              if(auto constop = dyn_cast<hw::ConstantOp>(newVal.getDefiningOp())){
                if(constop.getType().getIntOrFloatBitWidth()>1)
                  updatedVec.push_back(c.int_val(constop.getValue().getSExtValue()));
                else
                  updatedVec.push_back(c.bool_val(0));
                f = true;
              }
              
            }
            if(!f)
              llvm::errs()<<"\n\nsome mess here too 3\n\n";
            else 
              found = true;
          }
        } else {
            vector<expr> tmpVec;
            for (auto operand: op.getOperands()){
              bool f = false;
              // search through variablesFunc
              for(auto vf: variablesFun){
                if(vf.first==operand){
                  tmpVec.push_back(vf.second(time));
                  f=true;
                }
              }
              // otherwise it is a constant
              if(auto constop = dyn_cast<hw::ConstantOp>(operand.getDefiningOp())){
                if(constop.getType().getIntOrFloatBitWidth() > 1)
                  tmpVec.push_back(c.int_val(constop.getValue().getSExtValue()));
                else
                  tmpVec.push_back(c.bool_val(0));
                f=true;
              }
              if(!f)
                llvm::outs()<<"\n\nhuge mess 2 here\n\n";
            }
          exprMapTmp.push_back({manage_comb_exp(op, tmpVec, c), op.getResult(0)});
        }
      }
    }
    updatedVec.push_back(v.second(time));
  }
  return updatedVec;
}

/**
 * @brief Parse FSM arguments and add them to the variable map
*/
vector<func_decl> populateArgs(Operation &mod, vector<std::pair<mlir::Value, func_decl>> &variablesFun, z3::context &c){
  vector<func_decl> arguments;
  int numArgs = 0;
  for(Region &rg: mod.getRegions()){
      for(Block &bl: rg){
        for(Operation &op: bl){
          if(auto machine = dyn_cast<fsm::MachineOp>(op)){
            for (Region &rg : op.getRegions()) {
              for (Block &block : rg) {
                for(auto a: block.getArguments()){
                  Z3_sort int_sort = Z3_mk_int_sort(c);
                  Z3_sort domain[1] = { int_sort };
                  // old
                  expr input = c.bool_const(("arg"+to_string(numArgs)).c_str());
                  if(a.getType().getIntOrFloatBitWidth()>1){ 
                    input = c.int_const(("arg"+to_string(numArgs)).c_str());
                    const symbol cc = c.str_symbol(("input-arg"+to_string(numArgs)).c_str());
                    Z3_func_decl I = Z3_mk_func_decl(c, cc, 1, domain, c.int_sort());
                    func_decl I2 = func_decl(c, I);
                    arguments.push_back(I2);
                    numArgs++;
                    variablesFun.push_back({a, I2});
                  } else {
                    input = c.bool_const(("arg"+to_string(numArgs)).c_str());
                    const symbol cc = c.str_symbol(("input-arg"+to_string(numArgs)).c_str());
                    Z3_func_decl I = Z3_mk_func_decl(c, cc, 1, domain, c.bool_sort());
                    func_decl I2 = func_decl(c, I);
                    arguments.push_back(I2);
                    numArgs++;
                    variablesFun.push_back({a, I2});
                  }
                }
                return arguments;
              }
            }
          }
        }
      }
    }
}

vector<func_decl> populateOutputs(Operation &mod, vector<std::pair<mlir::Value, func_decl>> &variablesFun, z3::context &c, MLIRContext &context, OwningOpRef<ModuleOp> &module){
  vector<func_decl> outputs;
  int numOutput = 0;
  for(Region &rg: mod.getRegions()){
    for(Block &bl: rg){
      for(Operation &op: bl){
        if(auto machine = dyn_cast<fsm::MachineOp>(op)){
          for (auto opr: machine.getFunctionType().getResults()) {
            Z3_sort int_sort = Z3_mk_int_sort(c);
            Z3_sort domain[1] = { int_sort };

            OpBuilder builder(&machine.getBody());

            auto loc = builder.getUnknownLoc();

            auto variable = builder.create<fsm::VariableOp>(loc, builder.getIntegerType(opr.getIntOrFloatBitWidth()), IntegerAttr::get(builder.getIntegerType(opr.getIntOrFloatBitWidth()), 0), builder.getStringAttr("outputVal"));

            mlir::Value v = variable.getResult();


            if(opr.getIntOrFloatBitWidth()>1){ 
              expr e = c.int_const(("output-"+to_string(numOutput)).c_str());
              const symbol cc = c.str_symbol(("output-"+to_string(numOutput)).c_str());
              Z3_func_decl I = Z3_mk_func_decl(c, cc, 1, domain, c.int_sort());
              func_decl I2 = func_decl(c, I);
              outputs.push_back(I2);
              variablesFun.push_back({v, I2});
            } else {
              expr e = c.bool_const(("output-"+to_string(numOutput)).c_str());
              const symbol cc = c.str_symbol(("output-"+to_string(numOutput)).c_str());
              Z3_func_decl I = Z3_mk_func_decl(c, cc, 1, domain, c.bool_sort());
              func_decl I2 = func_decl(c, I);
              outputs.push_back(I2);
              variablesFun.push_back({v, I2});
            }
            // is this conceptually correct?
            numOutput++;


          }
        }
      }
    }
  }
  return outputs;
}

/**
 * @brief Parse FSM variables and add them to the variable map
*/
vector<func_decl> populateVars(Operation &mod, vector<std::pair<mlir::Value, func_decl>> &variablesFun, z3::context &c, int numArgs){
  vector<func_decl> variables;
  for(Region &rg: mod.getRegions()){
    for(Block &bl: rg){
      for(Operation &op: bl){
        if(auto machine = dyn_cast<fsm::MachineOp>(op)){
          for (Region &rg : op.getRegions()) {
            for (Block &block : rg) {
              for(Operation &op: block){ 
                if(auto varOp = dyn_cast<fsm::VariableOp>(op)){
                  int initValue = varOp.getInitValue().cast<IntegerAttr>().getInt();
                  string varName = varOp.getName().str();
                  Z3_sort int_sort = Z3_mk_int_sort(c);
                  Z3_sort domain[1] = { int_sort };
                  if(varOp.getName().str().find("arg") != std::string::npos){
                    // reserved keyword arg for arguments to avoid ambiguity when setting initial state values
                    varName = "var"+to_string(numArgs);
                    numArgs++;
                  }
                  if(varOp.getResult().getType().getIntOrFloatBitWidth()>1){ 
                    expr input = c.int_const((varName+"_"+to_string(initValue)).c_str());
                    const symbol cc = c.str_symbol(("var"+to_string(numArgs)+"_"+to_string(initValue)).c_str());
                    Z3_func_decl I = Z3_mk_func_decl(c, cc, 1, domain, c.int_sort());
                    func_decl I2 = func_decl(c, I);
                    variables.push_back(I2);
                    variablesFun.push_back({varOp.getResult(), I2});
                  } else {
                    expr input = c.bool_const((varName+"_"+to_string(initValue)).c_str());
                    const symbol cc = c.str_symbol(("var"+to_string(numArgs)+"_"+to_string(initValue)).c_str());
                    Z3_func_decl I = Z3_mk_func_decl(c, cc, 1, domain, c.bool_sort());
                    func_decl I2 = func_decl(c, I);
                    variables.push_back(I2);
                    variablesFun.push_back({varOp.getResult(), I2});
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return variables;
}

/**
 * @brief Insert state if not present, return position in vector otherwise
*/
int insertState(string state, vector<string> &stateInv){
  int i=0;
  for(auto s: stateInv){
    // return index
    if (s == state)
      return i;
    i++;
  }
  stateInv.push_back(state);
  return stateInv.size()-1;
}

/**
 * @brief Parse FSM states and add them to the state map
*/
void populateST(Operation &mod, context &c, vector<string> &stateInv, vector<transition> &transitions, vector<std::pair<mlir::Value, func_decl>> &variablesFun, int numOutput){
  for (Region &rg: mod.getRegions()){
    for (Block &bl: rg){
      for (Operation &op: bl){
        if (auto machine = dyn_cast<fsm::MachineOp>(op)){
          for (Region &rg : op.getRegions()) {
            for (Block &block : rg) {
              int numState = 0;
              for (Operation &op : block) {
                if (auto state = dyn_cast<fsm::StateOp>(op)){
                  string currentState = state.getName().str();
                  insertState(currentState, stateInv);
                  numState++;
                  if(V){
                    llvm::outs()<<"inserting state "<<currentState<<"\n";
                  }
                  auto regions = state.getRegions();
                  bool existsOutput = false;
                  if(!regions[0]->empty())
                    existsOutput = true;
                  // transitions region
                  for (Block &bl1: *regions[1]){
                    for (Operation &op: bl1.getOperations()){
                      if(auto transop = dyn_cast<fsm::TransitionOp>(op)){
                        transition t;
                        t.from = insertState(currentState, stateInv);
                        t.to = insertState(transop.getNextState().str(), stateInv);
                        t.isGuard = false;
                        t.isAction = false;
                        t.isOutput = false;
                        auto trRegions = transop.getRegions();
                        string nextState = transop.getNextState().str();                        
                        // guard
                        if(!trRegions[0]->empty()){
                          Region &r = *trRegions[0];
                          z3Fun g = [&r, &variablesFun, &c](expr time) {
                            expr guard_expr = getGuard(variablesFun, r, c, time);
                            return guard_expr;
                          };                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
                          t.guard = g;
                          t.isGuard = true;
                        }
                        // action 
                        if(!trRegions[1]->empty()){
                            Region &r = *trRegions[1];
                            // vector<mlir::Value> to_update = actionsCounter(r);
                            z3FunA a = [&r, &variablesFun, &c](expr time) -> vector<expr> {
                              vector<expr> vec2 = getAction(variablesFun, r, c, time); 
                              return vec2;
                            };
                            t.action = a;
                            t.isAction = true;
                        }
                        if(existsOutput){
                            Region &r = *regions[0];
                            z3FunA tf = [&r, &variablesFun, &c](expr time) -> vector<expr> {
                              vector<expr> vec = getOutput(variablesFun, r, c, time);
                              // todo: update output val in vec2
                              return vec;
                            };
                            t.output = tf;
                            t.isOutput = true;
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


/**
 * @brief Build Z3 boolean function for each state in the state map
*/
void populateStateInvMap(vector<string> &stateInv, context &c, vector<Z3_sort> &invInput, vector<func_decl> &stateInvFun){
  for(auto s: stateInv){
    const symbol cc = c.str_symbol(s.c_str());
    Z3_func_decl I = Z3_mk_func_decl(c, cc, invInput.size(), invInput.data(), c.bool_sort());
    func_decl I2 = func_decl(c, I);
    stateInvFun.push_back(I2);
  }
}

/**
 * @brief Build Z3 function for each input argument
*/
void populateInvInput(vector<std::pair<expr, mlir::Value>> &variables, context &c, vector<expr> &solverVars, vector<Z3_sort> &invInput, int numArgs, int numOutputs){

  int i=0;

  for(auto e: variables){
    string name = "var";
    if(numArgs!=0 && i < numArgs){
      name = "input";
    } else if (numOutputs!=0 && i>=variables.size()-numOutputs){
      name = "output";
    }
    expr input = c.bool_const((name+to_string(i)).c_str());
    z3::sort invIn = c.bool_sort();
    if(e.second.getType().getIntOrFloatBitWidth()>1){ 
      input = c.int_const((name+to_string(i)).c_str());
      invIn = c.int_sort(); 
    }
    solverVars.push_back(input);
    if(V){
      llvm::outs()<<"solverVars now: "<<solverVars[i].to_string()<<"\n";
    }
    i++;
    invInput.push_back(invIn);
  }


}

expr parseLTL(string inputFile, vector<string> stateInv, expr_vector &stateExpr, func_decl &timeToState, expr time, vector<std::pair<mlir::Value, func_decl>> &variablesFun){
  DialectRegistry registry;

  registry.insert<ltl::LTLDialect>();

  MLIRContext context(registry);

  int tBound = 30;

  // Parse the MLIR code into a module.
  OwningOpRef<ModuleOp> module = mlir::parseSourceFile<ModuleOp>(inputFile, &context);

  Operation& mod = module.get()[0];

  for (Region &rg: mod.getRegions()){
    for (Block &bl: rg){
      for (Operation &op: bl){ 
        if (auto ev = dyn_cast<ltl::EventuallyOp>(op)){
          auto attr_dict = ev.getOperation()->getAttrs();
          if(attr_dict.size()==1){
            // reachability
            auto a0 = (attr_dict[0].getValue());
            string state;
            raw_string_ostream os1(state);
            a0.print(os1);
            os1.flush();
            state = state.substr(1, state.size() - 2);

            // llvm::outs()<<"\n\n\nTesting reachability of state "<<state;

            expr ret = (forall(time, (timeToState(time)!= stateExpr[insertState(state, stateInv)])));
            return ret;
          } else {
            llvm::outs()<<"Reachability Property can not be parsed."; 
          }
        } else if (auto rep = dyn_cast<ltl::NotOp>(op)){
          auto attr_dict = rep.getOperation()->getAttrs();
          if(attr_dict.size()==3){
            auto a0 = (attr_dict[0].getValue());
            string state;
            raw_string_ostream os1(state);
            a0.print(os1);
            os1.flush();
            state = state.substr(1, state.size() - 2);
            
            auto a1 = (attr_dict[1].getValue());
            string var;
            raw_string_ostream os2(var);
            a1.print(os2);
            os2.flush();
            var = var.substr(1, var.size() - 2);
            int id = stoi(var);

            auto a2 = (attr_dict[2].getValue());
            string val;
            raw_string_ostream os3(val);
            a2.print(os3);
            os3.flush();
            val = val.substr(1, val.size() - 2);
            int v = stoi(val);

            // llvm::outs()<<"\n\n\nTesting value "<<v<<" of variable at index "<<id<<" at state "<<state;



            expr body = (timeToState(time)==stateExpr[insertState(state, stateInv)]) == ((variablesFun[v].second(time))!=id);
            expr ret =(forall(time, body));

            return ret;
          } else {
            llvm::outs()<<"Comb Property can not be parsed.";
          }
        } else if (auto imp = dyn_cast<ltl::ImplicationOp>(op)){
            auto attr_dict = imp.getOperation()->getAttrs();
            if(attr_dict.size()==3){
              // error

              auto a0 = (attr_dict[0].getValue());
              llvm::outs()<<"\n\nattr a: "<<a0;
              string sig;
              raw_string_ostream os1(sig);
              a0.print(os1);
              os1.flush();
              sig = sig.substr(1, sig.size() - 2);
              int signal = stoi(sig);

              auto a1 = (attr_dict[1].getValue());
              llvm::outs()<<"\n\nattr a: "<<a1;

              string var;
              raw_string_ostream os2(var);
              a1.print(os2);
              os2.flush();
              var = var.substr(1, var.size() - 2);
              int input = stoi(var);


              auto a2 = (attr_dict[2].getValue());
              llvm::outs()<<"\n\nattr a: "<<a2;

              string state;
              raw_string_ostream os3(state);
              a2.print(os3);
              os3.flush();
              state = state.substr(1, state.size() - 2);

              expr body = ((timeToState(time)==stateExpr[insertState(state, stateInv)]) && ((variablesFun[input].second(time))!=signal));
              expr ret = (forall(time, (body)));
              return ret;
          } else{
            llvm::outs()<<"Error Management Property can not be parsed.";

          }
        }
      }
    }
  }
  llvm::outs()<<"Property can not be parsed.";
}



/**
 * @brief Parse FSM and build SMT model 
*/
void parse_fsm(string input, string property, string output){

  DialectRegistry registry;

  registry.insert<
    comb::CombDialect,
    fsm::FSMDialect,
    hw::HWDialect
  >();

  MLIRContext context(registry);

  // Parse the MLIR code into a module.
  OwningOpRef<ModuleOp> module = mlir::parseSourceFile<ModuleOp>(input, &context);

  Operation& mod = module.get()[0];

  z3::context c;

  solver s(c);

  vector<string> stateInv;

  vector<std::pair<mlir::Value, func_decl>> variablesFun;

  vector<transition> transitions;
  
  string initialState = getInitialState(mod);

  // initial state is by default associated with id 0

  insertState(initialState, stateInv);

  vector<func_decl> arguments = populateArgs(mod, variablesFun, c);

  vector<func_decl> variables = populateVars(mod, variablesFun, c, arguments.size());

  vector<func_decl> outputs = populateOutputs(mod, variablesFun, c, context, module);

  if(V){
    llvm::outs()<<"initial state: "<<initialState<<"\n";
  }

  populateST(mod, c, stateInv, transitions, variablesFun, outputs.size());


  z3::sort state_sort = c.uninterpreted_sort("STATE");

  expr_vector stateExpr(c);

  for(auto state: stateInv){
    const expr ctmp = c.constant(state.c_str(), state_sort);
    stateExpr.push_back(ctmp);
  }

  expr time = c.int_const("time");

  Z3_sort int_sort = Z3_mk_int_sort(c);


  Z3_sort domain[1] = { int_sort };
  const symbol cc = c.str_symbol(("time-to-state"));
  Z3_func_decl I = Z3_mk_func_decl(c, cc, 1, domain, state_sort);
  func_decl timeToState = func_decl(c, I);

  s.add(timeToState(0)==stateExpr[0]);

  s.add(variablesFun[0].second(0)==0);

  for(int j = arguments.size();  j < variablesFun.size(); j++){
    s.add(variablesFun[j].second(0)==0);

  }

  int tBound = 30;
  
  for (auto t: transitions){
    if(t.isGuard && t.isAction){
      vector<expr> tmpAc = t.action(time);
      expr tmp = (variablesFun[arguments.size()].second(time+1) == tmpAc[arguments.size()]);
      for(int i = arguments.size()+1; i<variablesFun.size(); i++){
        tmp = tmp && (variablesFun[i].second(time+1) == tmpAc[i]);
      }
      expr a = forall(time, implies((timeToState(time)==stateExpr[t.from] && t.guard(time)), (timeToState(time+1)==stateExpr[t.to] && tmp)));
      s.add(a);
    } else if (t.isGuard){
      expr a = forall(time, implies((timeToState(time)==stateExpr[t.from] && t.guard(time)), (timeToState(time+1)==stateExpr[t.to])));
      s.add(a);
    } else if (t.isAction){
      vector<expr> tmpAc = t.action(time);
      expr tmp = (variablesFun[arguments.size()].second(time+1) == tmpAc[arguments.size()]);
      for(int i = arguments.size()+1; i<variablesFun.size(); i++){
        tmp = tmp && (variablesFun[i].second(time+1) == tmpAc[i]);
      }
      expr a = forall(time, implies((timeToState(time)==stateExpr[t.from]), (timeToState(time+1)==stateExpr[t.to] && tmp)));
      s.add(a);

    } else {
      expr a = forall(time, implies((timeToState(time)==stateExpr[t.from]), timeToState(time+1)==stateExpr[t.to]));
      s.add(a);
    }
  }

  expr r = parseLTL(property, stateInv, stateExpr, timeToState, time, variablesFun);

  s.add(r);

  s.add(distinct(stateExpr));

  printSolverAssertions(s, output);

}


int main(int argc, char **argv){
  string input = argv[1];
  cout << "input file: " << input << endl;

  string prop = argv[2];
  cout << "property file: " << prop << endl;

  string output = argv[3];
  cout << "output file: " << output << endl;

  parse_fsm(input, prop, output);

  return 0;
}