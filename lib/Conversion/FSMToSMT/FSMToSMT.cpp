//===- FSMToSMT.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"

#include "mlir/Parser/Parser.h"
#include "circt/Conversion/FSMToSMT.h"
#include "circt/Conversion/HWToSMT.h"
#include "circt/Conversion/CombToSMT.h"
#include "circt/Dialect/SMT/SMTOps.h"
#include "circt/Dialect/SMT/SMTDialect.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Operation.h"


#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <memory>
#include <optional>
#include <variant>
#include <vector>

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

LogicalResult MachineOpConverter::dispatch(){
  b.setInsertionPoint(machineOp);
  auto loc = machineOp.getLoc();
  llvm::outs()<<"am in dispatcher\n";
  
  auto args = machineOp.getArguments();

  llvm::SmallVector<mlir::Type> argVarTypes;
  llvm::SmallVector<mlir::Value> argVars;

  llvm:DenseMap<mlir::Value, mlir::Operation> varExprMap;

  for (auto a : args){
    argVarTypes.push_back(a.getType());
    argVars.push_back(a);
  }

  for (auto variableOp : machineOp.front().getOps<fsm::VariableOp>()) {
    llvm::outs()<<"\nvar\n";
    argVarTypes.push_back(variableOp.getResult().getType());
    argVars.push_back(variableOp.getResult());
  }

  llvm:DenseMap<StringRef, mlir::Value> stateTransitionMap;

  for (auto stateOp : machineOp.front().getOps<fsm::StateOp>()) {
    llvm::outs()<<"\nstate\n";

    auto stateName = stateOp.getSymName();

    llvm::outs()<<stateName;

    auto funcType = b.getFunctionType(argVarTypes, b.getType<smt::BoolType>());
    auto stateFun = b.create<smt::DeclareFunOp>(loc, funcType);

    stateFun.setNamePrefix(stateName);

    stateTransitionMap.insert({stateName, stateFun});
  }



  for (auto stateOp : machineOp.front().getOps<fsm::StateOp>()) {
    // second region (idx 1) always conains the transitions
    for (auto transitionOp : stateOp->getRegion(1).front().getOps<fsm::TransitionOp>()) {

      auto nextState = transitionOp.getNextState();

      llvm::outs()<<"\nto "<<nextState;

      // auto guard = transitionOp->getRegion(0);

      mlir::PassManager pm(transitionOp.getGuard().getContext());

      pm.addPass(createConvertCombToSMT());

      llvm::DenseMap<mlir::Value, mlir::Value> guardMap;

      // ex:
      // %tmp1 = comb.add %cnt, 1
      // %tmp2 = comb.add %tmp1, 1
      // %tmp3 = comb.icmp ule %tmp2 5
      // fsm.return %tmp3

      // fsm.return comb.icmp ule (comb.add (comb.add %cnt 1) 1) 5

      for (auto &op : transitionOp->getRegion(0).getOps()) {
        if (auto retop = dyn_cast<fsm::ReturnOp>(op)) {
          // for (auto e : exprMap) {
          //   if (e.second == retop.getOperand()) {

          //     return e.first;
          //   }
          // }
        }
        llvm::SmallVector<mlir::Value> vec;
        for (auto operand : op.getOperands()) {
          vec.push_back(guardMap.find(operand)->getFirst());
        }
        exprMap.push_back({manageCombExp(op, vec, c), op.getResult(0)});
        else{
          pm.run(&op);
        }
      }

      // auto result = pm.run(guardOp);

      // auto funcType = b.getFunctionType(argVarTypes, b.getType<smt::BoolType>());
      // auto chcFun = b.create<smt::ImpliesOp>(loc, funcType);

    }


  }


}


FailureOr<mlir::Value>
MachineOpConverter::convertArgsAndVars(MachineOp machine, llvm::SmallVector<mlir::Value> vars){
  
    // mlir::ArrayAttr args = machine.getArgAttrsAttr();
    // mlir::ArrayAttr argNames = machine.getArgNamesAttr();

    // llvm::outs()<<"my first pass\n";
    // args.print(llvm::outs());
  
}


// using z3Fun = std::function<expr(SmallVector<expr>)>;

// using z3FunA = std::function<SmallVector<expr>(SmallVector<expr>)>;

// struct transition {
//   int from, to;
//   z3Fun guard;
//   bool isGuard, isAction, isOutput;
//   z3FunA action, output;
// };


// // @brief Prints solver assertions

// void printSolverAssertions(z3::context &c, z3::solver &solver, string output,
//                            SmallVector<func_decl> stateInvFun) {

//   ofstream outfile;
//   outfile.open(output);

//   outfile << "(set-logic HORN)" << std::endl;

//   z3::expr_SmallVector assertions = solver.assertions();

//   llvm::outs() << "assertions size: " << assertions.size() << "\n\n";

//   for (auto fd : stateInvFun) {
//     outfile << fd.to_string() << "\n";
//   }

//   for (unsigned i = 0; i < assertions.size(); ++i) {
//     Z3_ast ast = assertions[i];
//     outfile << "(assert " << Z3_ast_to_string(c, ast) << ")" << std::endl;
//   }
//   outfile << "(check-sat)" << std::endl;

//   outfile.close();
// }


// // @brief Prints FSM transition

// void printTransitions(SmallVector<transition> &transitions) {
//   llvm::outs() << "\nPRINTING TRANSITIONS\n";
//   for (auto t : transitions) {
//     llvm::outs() << "\ntransition from " << t.from << " to " << t.to << "\n";
//   }
// }



// @brief Prints z3::expr-mlir::Value map

// void printExprValMap(SmallVector<std::pair<expr, mlir::Value>> &exprMap) {
//   llvm::outs() << "\n\nEXPR-VAL:";
//   for (auto e : exprMap) {
//     llvm::outs() << "\n\nexpr: " << e.first.to_string()
//                  << ", value: " << e.second;
//   }
//   llvm::outs() << "END\n";
// }

// // @brief Returns FSM's initial state
 
// string getInitialState(Operation &mod) {
//   for (Region &rg : mod.getRegions()) {
//     for (Block &block : rg) {
//       for (Operation &op : block) {
//         if (auto machine = dyn_cast<fsm::MachineOp>(op)) {
//           return machine.getInitialState().str();
//         }
//       }
//     }
//   }
//   llvm::errs() << "Initial state does not exist\n";
// }


// // @brief Returns list of values to be updated within an action region
 
// SmallVector<mlir::Value> actionsCounter(Region &action) {
//   SmallVector<mlir::Value> toUpdate;
//   for (auto &op : action.getOps()) {
//     if (auto updateop = dyn_cast<fsm::UpdateOp>(op)) {
//       toUpdate.push_back(updateop.getOperands()[0]);
//     }
//   }
//   return toUpdate;
// }


// // @brief Returns expression from Comb dialect operator
 
// expr manageCombExp(Operation &op, SmallVector<expr> &vec, z3::context &c) {

//   if (auto add = dyn_cast<comb::AddOp>(op)) {
//     return to_expr(c, vec[0] + vec[1]);
//   } else if (auto and_op = dyn_cast<comb::AndOp>(op)) {
//     return expr(vec[0] && vec[1]);
//   } else if (auto xor_op = dyn_cast<comb::XorOp>(op)) {
//     return expr(vec[0] ^ vec[1]);
//   } else if (auto or_op = dyn_cast<comb::OrOp>(op)) {
//     return expr(vec[0] | vec[1]);
//   } else if (auto mul_op = dyn_cast<comb::MulOp>(op)) {
//     return expr(vec[0] * vec[1]);
//   } else if (auto icmp = dyn_cast<comb::ICmpOp>(op)) {
//     circt::comb::ICmpPredicate predicate = icmp.getPredicate();
//     switch (predicate) {
//     case circt::comb::ICmpPredicate::eq:
//       return expr(vec[0] == vec[1]);
//     case circt::comb::ICmpPredicate::ne:
//       return expr(vec[0] != vec[1]);
//     case circt::comb::ICmpPredicate::slt:
//       return expr(vec[0] < vec[1]);
//     case circt::comb::ICmpPredicate::sle:
//       return expr(vec[0] <= vec[1]);
//     case circt::comb::ICmpPredicate::sgt:
//       return expr(vec[0] > vec[1]);
//     case circt::comb::ICmpPredicate::sge:
//       return expr(vec[0] >= vec[1]);
//     case circt::comb::ICmpPredicate::ult:
//       return expr(vec[0] < vec[1]);
//     case circt::comb::ICmpPredicate::ule:
//       return expr(vec[0] <= vec[1]);
//     case circt::comb::ICmpPredicate::ugt:
//       return expr(vec[0] > vec[1]);
//     case circt::comb::ICmpPredicate::uge:
//       return expr(vec[0] >= vec[1]);
//     case circt::comb::ICmpPredicate::ceq:
//       return expr(vec[0] == vec[1]);
//     case circt::comb::ICmpPredicate::cne:
//       return expr(vec[0] != vec[1]);
//     case circt::comb::ICmpPredicate::weq:
//       return expr(vec[0] == vec[1]);
//     case circt::comb::ICmpPredicate::wne:
//       return expr(vec[0] != vec[1]);
//     }
//   }
//   assert(false && "LLVM unreachable");
// }


// // @brief Returns expression from densemap or constant operator

// expr getExpr(mlir::Value v, SmallVector<std::pair<expr, mlir::Value>> &exprMap,
//              z3::context &c) {

//   for (auto e : exprMap) {
//     if (e.second == v)
//       return e.first;
//   }

//   if (auto constop = dyn_cast<hw::ConstantOp>(v.getDefiningOp())) {
//     if (constop.getType().getIntOrFloatBitWidth() > 1)
//       return c.int_val(constop.getValue().getSExtValue());
//     else
//       return c.bool_val(0);
//   }
//   llvm::errs() << "Expression not found.";
// }


// // @brief Returns guard expression for input region

// expr getGuardExpr(SmallVector<std::pair<expr, mlir::Value>> &exprMap, Region &guard,
//                   z3::context &c) {

//   for (auto &op : guard.getOps()) {
//     if (auto retop = dyn_cast<fsm::ReturnOp>(op)) {
//       for (auto e : exprMap) {
//         if (e.second == retop.getOperand()) {

//           return e.first;
//         }
//       }
//     }
//     SmallVector<expr> vec;
//     for (auto operand : op.getOperands()) {
//       vec.push_back(getExpr(operand, exprMap, c));
//     }

//     exprMap.push_back({manageCombExp(op, vec, c), op.getResult(0)});
//   }
//   return expr(c.bool_const("true"));
// }


// // @brief Returns output expression for input region
 
// SmallVector<expr> getOutputExpr(SmallVector<std::pair<expr, mlir::Value>> &exprMap,
//                            Region &guard, z3::context &c) {
//   SmallVector<expr> outputExp;
//   // printExprValMap(exprMap);
//   for (auto &op : guard.getOps()) {
//     if (auto outop = dyn_cast<fsm::OutputOp>(op)) {
//       for (auto opr : outop.getOperands()) {
//         for (auto e : exprMap) {
//           if (e.second == opr) {
//             llvm::outs() << "\npushing " << e.second;
//             outputExp.push_back(e.first);
//           }
//         }
//       }
//       return outputExp;
//     }
//     SmallVector<expr> vec;
//     for (auto operand : op.getOperands()) {
//       vec.push_back(getExpr(operand, exprMap, c));
//     }
//     exprMap.push_back({manageCombExp(op, vec, c), op.getResult(0)});
//   }
// }


// // @brief Returns actions for all expressions for the input region

// SmallVector<expr> getActionExpr(Region &action, context &c,
//                            SmallVector<mlir::Value> &toUpdate,
//                            SmallVector<std::pair<expr, mlir::Value>> &exprMap) {
//   SmallVector<expr> updatedVec;
//   for (auto v : toUpdate) {
//     bool found = false;
//     for (auto &op : action.getOps()) {
//       if (!found) {
//         if (auto updateop = dyn_cast<fsm::UpdateOp>(op)) {
//           if (v == updateop.getOperands()[0]) {

//             updatedVec.push_back(
//                 getExpr(updateop.getOperands()[1], exprMap, c));

//             found = true;
//           }
//         } else {
//           SmallVector<expr> vec;
//           for (auto operand : op.getOperands()) {

//             vec.push_back(getExpr(operand, exprMap, c));
//           }

//           exprMap.push_back({manageCombExp(op, vec, c), op.getResult(0)});
//         }
//       }
//     }
//     if (!found) {
//       for (auto e : exprMap) {
//         if (e.second == v)
//           updatedVec.push_back(e.first);
//       }
//     }
//   }
//   return updatedVec;
// }


// @brief Parse FSM arguments and add them to the variable map




// @brief Parse FSM output and add them to the variable map

// int populateOutputs(Operation &mod, SmallVector<mlir::Value> &vecVal,
//                     MLIRContext &context, OwningOpRef<ModuleOp> &module) {
//   int numOutput = 0;
//   for (Region &rg : mod.getRegions()) {
//     for (Block &bl : rg) {
//       for (Operation &op : bl) {
//         if (auto machine = dyn_cast<fsm::MachineOp>(op)) {
//           for (auto opr : machine.getFunctionType().getResults()) {
//             // is this conceptually correct?
//             OpBuilder builder(&machine.getBody());

//             auto loc = builder.getUnknownLoc();

//             auto variable = builder.create<fsm::VariableOp>(
//                 loc, builder.getIntegerType(opr.getIntOrFloatBitWidth()),
//                 IntegerAttr::get(
//                     builder.getIntegerType(opr.getIntOrFloatBitWidth()), 0),
//                 builder.getStringAttr("outputVal"));

//             mlir::Value v = variable.getResult();

//             vecVal.push_back(v);
//           }
//         }
//       }
//     }
//   }
//   return numOutput;
// }


// @brief Parse FSM variables and add them to the variable map

// void populateVars(Operation &mod, SmallVector<mlir::Value> &vecVal,
//                   SmallVector<std::pair<expr, mlir::Value>> &variables,
//                   z3::context &c, int numArgs) {
//   for (Region &rg : mod.getRegions()) {
//     for (Block &bl : rg) {
//       for (Operation &op : bl) {
//         if (auto machine = dyn_cast<fsm::MachineOp>(op)) {
//           for (Region &rg : op.getRegions()) {
//             for (Block &block : rg) {
//               for (Operation &op : block) {
//                 if (auto varOp = dyn_cast<fsm::VariableOp>(op)) {
//                   vecVal.push_back(varOp.getResult());
//                 }
//               }
//             }
//           }
//         }
//       }
//     }
//   }
// }


// // @brief Insert state if not present, return position in SmallVector otherwise

// int insertState(string state, SmallVector<string> &stateInv) {
//   int i = 0;
//   for (auto s : stateInv) {
//     // return index
//     if (s == state)
//       return i;
//     i++;
//   }
//   stateInv.push_back(state);
//   return stateInv.size() - 1;
// }


// // @brief Parse FSM states and add them to the state map

// void populateST(Operation &mod, context &c, SmallVector<string> &stateInv,
//                 SmallVector<transition> &transitions, SmallVector<mlir::Value> &vecVal,
//                 int numOutput) {
//   for (Region &rg : mod.getRegions()) {
//     for (Block &bl : rg) {
//       for (Operation &op : bl) {
//         if (auto machine = dyn_cast<fsm::MachineOp>(op)) {
//           for (Region &rg : op.getRegions()) {
//             for (Block &block : rg) {
//               int numState = 0;
//               for (Operation &op : block) {
//                 if (auto state = dyn_cast<fsm::StateOp>(op)) {
//                   string currentState = state.getName().str();
//                   insertState(currentState, stateInv);
//                   numState++;
//                   if (V) {
//                     llvm::outs() << "inserting state " << currentState << "\n";
//                   }
//                   auto regions = state.getRegions();
//                   bool existsOutput = false;
//                   if (!regions[0]->empty())
//                     existsOutput = true;
//                   // transitions region
//                   for (Block &bl1 : *regions[1]) {
//                     for (Operation &op : bl1.getOperations()) {
//                       if (auto transop = dyn_cast<fsm::TransitionOp>(op)) {
//                         transition t;
//                         t.from = insertState(currentState, stateInv);
//                         t.to =
//                             insertState(transop.getNextState().str(), stateInv);
//                         t.isGuard = false;
//                         t.isAction = false;
//                         t.isOutput = false;

//                         auto trRegions = transop.getRegions();
//                         string nextState = transop.getNextState().str();
//                         // guard
//                         if (!trRegions[0]->empty()) {
//                           Region &r = *trRegions[0];
//                           z3Fun g = [&r, &vecVal, &c](SmallVector<expr> vec) {
//                             SmallVector<std::pair<expr, mlir::Value>> exprMapTmp;
//                             for (auto [value, expr] : llvm::zip(vecVal, vec)) {
//                               exprMapTmp.push_back({expr, value});
//                             }
//                             expr guardExpr = getGuardExpr(exprMapTmp, r, c);
//                             return guardExpr;
//                           };
//                           t.guard = g;
//                           t.isGuard = true;
//                         }
//                         // action
//                         if (!trRegions[1]->empty()) {
//                           Region &r = *trRegions[1];
//                           SmallVector<mlir::Value> toUpdate = actionsCounter(r);
//                           z3FunA a = [&r, &vecVal,
//                                       &c](SmallVector<expr> vec) -> SmallVector<expr> {
//                             expr time = vec[vec.size() - 1];
//                             SmallVector<std::pair<expr, mlir::Value>> tmpVar;
//                             for (auto [value, expr] : llvm::zip(vecVal, vec)) {
//                               tmpVar.push_back({expr, value});
//                             }
//                             SmallVector<expr> vec2 =
//                                 getActionExpr(r, c, vecVal, tmpVar);
//                             vec2.push_back(time);
//                             return vec2;
//                           };
//                           t.action = a;
//                           t.isAction = true;
//                         }
//                         if (existsOutput) {
//                           Region &r2 = *regions[0];
//                           z3FunA tf = [&r2, &numOutput, &vecVal,
//                                        &c](SmallVector<expr> vec) -> SmallVector<expr> {
//                             SmallVector<std::pair<expr, mlir::Value>> tmpOut;
//                             for (auto [value, expr] : llvm::zip(vecVal, vec)) {
//                               tmpOut.push_back({expr, value});
//                             }
//                             SmallVector<expr> outputExpr =
//                                 getOutputExpr(tmpOut, r2, c);
//                             for (int j = 0; j < outputExpr.size(); j++) {
//                               vec[vec.size() - 1 - outputExpr.size() + j] =
//                                   outputExpr[j];
//                             }
//                             return vec;
//                           };
//                           t.output = tf;
//                           t.isOutput = true;
//                         }

//                         transitions.push_back(t);
//                       }
//                     }
//                   }
//                 }
//               }
//             }
//           }
//         }
//       }
//     }
//   }
// }


// // @brief Nest SMT assertion for all variables in the variables SmallVector

// expr nestedForall(SmallVector<expr> &solverVars, expr &body, int i,
//                   int numOutputs, z3::context &c) {
//   z3::expr_SmallVector univQ(c);

//   for(int idx = 0; idx < solverVars.size()-1-numOutputs; idx++){
//     univQ.push_back(solverVars[idx]);
//   }
//   univQ.push_back(solverVars[solverVars.size()-1]);

//   expr ret = forall(univQ, body);

//   return ret;
// }


// // @brief Build Z3 boolean function for each state in the state map

// void populateStateInvMap(SmallVector<string> &stateInv, context &c,
//                          SmallVector<Z3_sort> &invInput,
//                          SmallVector<func_decl> &stateInvFun) {
//   for (auto s : stateInv) {
//     const symbol cc = c.str_symbol(s.c_str());
//     Z3_func_decl I =
//         Z3_mk_func_decl(c, cc, invInput.size(), invInput.data(), c.bool_sort());
//     func_decl I2 = func_decl(c, I);
//     stateInvFun.push_back(I2);
//   }
// }


// // @brief Build Z3 sort for each input argument

// void populateInvInput(SmallVector<std::pair<expr, mlir::Value>> &variables,
//                       context &c, SmallVector<expr> &solverVars,
//                       SmallVector<Z3_sort> &invInput, int numArgs, int numOutputs) {

//   int i = 0;

//   for (auto e : variables) {
//     string name = "var";
//     if (numArgs != 0 && i < numArgs) {
//       name = "input";
//     } else if (numOutputs != 0 && i >= variables.size() - numOutputs) {
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


// // @brief Parse LTL dialect file expressing property

// expr parseLTL(string inputFile, SmallVector<expr> &solverVars,
//               SmallVector<string> &stateInv, 
//               SmallVector<func_decl> &stateInvFun, int numArgs, int numOutputs,
//               context &c) {
//   DialectRegistry registry;

//   registry.insert<ltl::LTLDialect>();

//   MLIRContext context(registry);

//   // Parse the MLIR code into a module.
//   OwningOpRef<ModuleOp> module =
//       mlir::parseSourceFile<ModuleOp>(inputFile, &context);

//   Operation &mod = module.get()[0];

//   for (Region &rg : mod.getRegions()) {
//     for (Block &bl : rg) {
//       for (Operation &op : bl) {
//         if (auto ev = dyn_cast<ltl::EventuallyOp>(op)) {
//           auto attrDict = ev.getOperation()->getAttrs();
//           if (attrDict.size() == 1) {
//             // reachability
//             auto a0 = (attrDict[0].getValue());
//             string state;
//             raw_string_ostream os1(state);
//             a0.print(os1);
//             os1.flush();
//             state = state.substr(1, state.size() - 2);

//             if (V)
//               llvm::outs()<<"\n\n\nTesting reachability of state "<<state;

//             // for (int i = 0; i < int(argInputs.size()); i++) {
//             //   solverVars[i] = argInputs[i](solverVars[solverVars.size() - 1]);
//             // }

//             expr body = implies(stateInvFun.at(insertState(state, stateInv))(
//                 solverVars.size(), solverVars.data()), false);

//             expr ret = nestedForall(solverVars, body, numArgs, numOutputs, c);

//             return ret;
//           } else {
//             llvm::outs() << "Reachability Property can not be parsed.";
//           }
//         } else if (auto rep = dyn_cast<ltl::NotOp>(op)) {
//           auto attrDict = rep.getOperation()->getAttrs();
//           if (attrDict.size() == 3) {
//             // combinational
//             auto a0 = (attrDict[0].getValue());
//             string state;
//             raw_string_ostream os1(state);
//             a0.print(os1);
//             os1.flush();
//             state = state.substr(1, state.size() - 2);

//             auto a1 = (attrDict[1].getValue());
//             string var;
//             raw_string_ostream os2(var);
//             a1.print(os2);
//             os2.flush();
//             var = var.substr(1, var.size() - 2);
//             int id = stoi(var);

//             auto a2 = (attrDict[2].getValue());
//             string val;
//             raw_string_ostream os3(val);
//             a2.print(os3);
//             os3.flush();
//             val = val.substr(1, val.size() - 2);
//             int v = stoi(val);

//             expr b1 = implies((stateInvFun[insertState(state, stateInv)](
//                             solverVars.size(), solverVars.data())) &&
//                         (solverVars[v] != id), false);


//             expr r1 = nestedForall(solverVars, b1, numArgs, numOutputs, c);




//             return (r1);
//           } else {
//             llvm::outs() << "Comb Property can not be parsed.";
//           }
//         } else if (auto impl = dyn_cast<ltl::ImplicationOp>(op)) {
//           auto attrDict = impl.getOperation()->getAttrs();
//           if (attrDict.size() == 3) {
//             // error
//             auto a3 = (attrDict[2].getValue());
//             string state;
//             raw_string_ostream os0(state);
//             a3.print(os0);
//             os0.flush();
//             state = state.substr(1, state.size() - 2);

//             llvm::outs()<<"\n\nstate: "<<state;

//             auto a0 = (attrDict[1].getValue());
//             string sig;
//             raw_string_ostream os1(sig);
//             a0.print(os1);
//             os1.flush();
//             sig = sig.substr(1, sig.size() - 2);
//             int signal = stoi(sig);
//             llvm::outs()<<"\n\nsignal: "<<signal;


//             auto a1 = (attrDict[0].getValue());
//             string var;
//             raw_string_ostream os2(var);
//             a1.print(os2);
//             os2.flush();
//             var = var.substr(1, var.size() - 2);
//             int input = stoi(var);
//             llvm::outs()<<"\n\ninput: "<<input;


//             // for (int i = 0; i < int(argInputs.size()); i++) {
//             //   solverVars[i] = argInputs[i](solverVars[solverVars.size() - 1]);
//             // }

//             SmallVector<expr> solverVarsAfter;

//             copy(solverVars.begin(), solverVars.end(),
//                  back_inserter(solverVarsAfter));

//             solverVarsAfter[solverVarsAfter.size() - 1] =
//                 solverVarsAfter[solverVarsAfter.size() - 1] + 1;
//             // for (int i = 0; i < int(argInputs.size()); i++) {
//             //   solverVarsAfter[i] =
//             //       argInputs[i](solverVarsAfter[solverVarsAfter.size() - 1]);
//             // }

//             expr body = implies((solverVars[signal]==input), !(stateInvFun[insertState(state, stateInv)])(solverVarsAfter.size(), solverVarsAfter.data()));

//             llvm::outs()<<body.to_string()<<"\n\n";

//             // expr body = !(stateInvFun[insertState(state,
//             // stateInv)])(solverVarsAfter.size(), solverVarsAfter.data()); expr
//             // ret = (forall(solverVars[solverVars.size()-1],
//             // implies((solverVars[solverVars.size()-1]>0 &&
//             // solverVars[solverVars.size()-1]<time-1 &&
//             // (solverVars[signal]==input) && (stateInvFun[insertState(state,
//             // stateInv)])(solverVars.size(), solverVars.data())),
//             // nestedForall(solverVars, body, numArgs, numOutputs)))); return
//             // ret;

//             return nestedForall(solverVars, body, numArgs, numOutputs, c);

//           } else {
//             llvm::outs() << "Error Management Property can not be parsed.";
//           }
//         }
//       }
//     }
//   }
//   llvm::outs() << "Property can not be parsed.";
//   return c.bool_val(true);
// }


// @brief Parse FSM and build SMT model

// static LogicalResult translateFSM(MLIRContext &context) {

//   DialectRegistry registry;

//   registry.insert<comb::CombDialect, fsm::FSMDialect, hw::HWDialect>();

//   MLIRContext context(registry);

//   // Parse the MLIR code into a module.
//   OwningOpRef<ModuleOp> module =
//       mlir::parseSourceFile<ModuleOp>(input, &context);

//   Operation &mod = module.get()[0];

//   z3::context c;

//   Z3_set_ast_print_mode(c, Z3_PRINT_SMTLIB_FULL);

//   solver s(c);

//   SmallVector<string> stateInv;

//   SmallVector<std::pair<expr, mlir::Value>> variables;

//   SmallVector<mlir::Value> vecVal;

//   SmallVector<transition> transitions;

//   string initialState = getInitialState(mod);

//   // initial state is by default associated with id 0

//   insertState(initialState, stateInv);

//   if (V) {
//     llvm::outs() << "initial state: " << initialState << "\n";
//   }

//   int numArgs = populateArgs(mod, vecVal, variables, c);

//   populateVars(mod, vecVal, variables, c, numArgs);

//   int numOutputs = populateOutputs(mod, vecVal, variables, c, context, module);

//   populateST(mod, c, stateInv, transitions, vecVal, numOutputs);

//   // preparing the model

//   SmallVector<expr> solverVars;

//   SmallVector<Z3_sort> invInput;

//   // SmallVector<func_decl> argInputs;

//   SmallVector<func_decl> stateInvFun;

//   populateInvInput(variables, c, solverVars, invInput, numArgs, numOutputs);

//   expr timeVar = c.int_const("time");
//   z3::sort timeInv = c.int_sort();

//   solverVars.push_back(timeVar);
//   invInput.push_back(timeInv);

//   // generate functions for inputs

//   if (V)
//     llvm::outs() << "number of args: " << numArgs << "\n\n";

//   // for (int i = 0; i < numArgs; i++) {
//   //   const symbol cc = c.str_symbol(("input-arg" + to_string(i)).c_str());
//   //   Z3_func_decl I =
//   //       Z3_mk_func_decl(c, cc, 1, &invInput[invInput.size() - 1], c.int_sort());
//   //   func_decl I2 = func_decl(c, I);
//   //   argInputs.push_back(I2);
//   // }

//   populateStateInvMap(stateInv, c, invInput, stateInvFun);

//   if (V) {
//     llvm::outs() << "number of variables + args: " << solverVars.size() << "\n";
//     for (auto v : solverVars) {
//       llvm::outs() << "variable: " << v.to_string() << "\n";
//     }
//   }

//   int j = 0;

//   SmallVector<expr> solverVarsInit;
//   copy(solverVars.begin(), solverVars.end(), back_inserter(solverVarsInit));

//   for(int i=numArgs; i<int(variables.size()); i++){

//     if (variables[i].second.getType().getIntOrFloatBitWidth() > 1) {
//       int initValue =
//       stoi(variables[i].first.to_string().substr(variables[i].first.to_string().find("_")+1));
//       solverVarsInit[i] = c.int_val(initValue);
//     } else {
//       bool initValue = stoi(variables[i].first.to_string().substr(variables[i].first.to_string().find("_")+1));
//       solverVarsInit.at(i) = c.bool_val(initValue);
//     }
//   }
//   // for (int i = 0; i < numArgs; i++) {
//   //   solverVarsInit[i] = argInputs[i](0);
//   // }
//   if (V) {
//     for (auto sv : solverVarsInit) {
//       llvm::outs() << "\nsvI[i]: " << sv.to_string();
//     }
//     llvm::outs() << "\n\n";
//   }

//   if(V)
//     printTransitions(transitions);

//   expr initState =
//       (stateInvFun[0](solverVarsInit.size(), solverVarsInit.data()));
//   expr body =
//       implies((solverVarsInit[solverVarsInit.size() - 1] == 0),
//               initState);

//   // initial condition
//   s.add(nestedForall(solverVars, body, 0, numOutputs, c));

//   for (auto t : transitions) {

//     SmallVector<expr> solverVarsAfter;

//     // for (int i = 0; i < int(argInputs.size()); i++) {
//     //   solverVars[i] = argInputs[i](solverVars[solverVars.size() - 1]);
//     // }

//     copy(solverVars.begin(), solverVars.end(), back_inserter(solverVarsAfter));
//     solverVarsAfter.at(solverVarsAfter.size() - 1) =
//         solverVars[solverVars.size() - 1] + 1;

//     // for (int i = 0; i < int(argInputs.size()); i++) {
//     //   solverVarsAfter[i] =
//     //       argInputs[i](solverVarsAfter.at(solverVarsAfter.size() - 1));
//     // }

//     if (t.isOutput) {
//       if (t.isGuard && t.isAction) {
//         expr body = implies(
//             (stateInvFun[t.from](solverVars.size(), solverVars.data())) &&
//                 t.guard(solverVars),
//             stateInvFun[t.to](t.output(t.action(solverVarsAfter)).size(),
//                               t.output(t.action(solverVarsAfter)).data()));
//         s.add(nestedForall(solverVars, body, 0, numOutputs, c));
//       } else if (t.isGuard) {
//         expr body = implies(
//             (stateInvFun[t.from](solverVars.size(), solverVars.data()) &&
//              t.guard(solverVars)),
//             stateInvFun[t.to](t.output((solverVarsAfter)).size(),
//                               t.output((solverVarsAfter)).data()));
//         s.add(nestedForall(solverVars, body, 0, numOutputs, c));
//       } else if (t.isAction) {
//         expr body = implies(
//             stateInvFun[t.from](solverVars.size(), solverVars.data()),
//             stateInvFun[t.to](t.output(t.action(solverVarsAfter)).size(),
//                               t.output(t.action(solverVarsAfter)).data()));
//         s.add(nestedForall(solverVars, body, 0, numOutputs, c));
//       } else {
//         expr body =
//             implies((stateInvFun[t.from](solverVars.size(), solverVars.data())),
//                     stateInvFun[t.to](t.output(solverVarsAfter).size(),
//                                       t.output(solverVarsAfter).data()));
//         s.add(nestedForall(solverVars, body, 0, numOutputs, c));
//       }
//     } else {
//       if (t.isGuard && t.isAction) {
//         expr body = implies(
//             (stateInvFun[t.from](solverVars.size(), solverVars.data())) &&
//                 t.guard(solverVars),
//             stateInvFun[t.to](t.action(solverVarsAfter).size(),
//                               t.action(solverVarsAfter).data()));
//         s.add(nestedForall(solverVars, body, 0, numOutputs, c));
//       } else if (t.isGuard) {
//         expr body = implies(
//             (stateInvFun[t.from](solverVars.size(), solverVars.data()) &&
//              t.guard(solverVars)),
//             stateInvFun[t.to]((solverVarsAfter).size(),
//                               (solverVarsAfter).data()));
//         s.add(nestedForall(solverVars, body, 0, numOutputs, c));
//       } else if (t.isAction) {
//         expr body =
//             implies(stateInvFun[t.from](solverVars.size(), solverVars.data()),
//                     stateInvFun[t.to](t.action(solverVarsAfter).size(),
//                                       t.action(solverVarsAfter).data()));
//         s.add(nestedForall(solverVars, body, 0, numOutputs, c));
//       } else {
//         expr body =
//             implies((stateInvFun[t.from](solverVars.size(), solverVars.data())),
//                     stateInvFun[t.to]((solverVarsAfter).size(),
//                                       (solverVarsAfter).data()));
//         s.add(nestedForall(solverVars, body, 0, numOutputs, c));
//       }
//     }
//   }

//   // for(auto state1 : stateInvFun){
//   //   expr mutualExc = c.bool_val(true);
//   //   for(auto state2: stateInvFun){
//   //     if(state1.to_string() != state2.to_string()){
//   //       mutualExc = mutualExc && !state2(solverVars.size(), solverVars.data());
//   //     }
//   //   }
//   //   expr stateMutualExc = implies(state1(solverVars.size(), solverVars.data()), mutualExc);
//   //   s.add(nestedForall(solverVars, stateMutualExc, numArgs, numOutputs, c));
//   // }

//   expr r = parseLTL(property, solverVars, stateInv, stateInvFun,
//                     0, numOutputs, c);

//   s.add(r);

//   printSolverAssertions(c, s, output, stateInvFun);

//   return success();
// }

// // / The entry point for the `circt-fsmt` tool:
// // / parses the input file and calls the `convertFSM` function to do the actual work.

// int main(int argc, char **argv) {

//   llvm::InitLLVM y(argc, argv);

//   cl::ParseCommandLineOptions(
//       argc, argv,
//       "circt-fsmt - FSM representation model checker\n\n"
//       "\tThis tool verifies a property over the FSM hardware abstraction.\n");

//   llvm::setBugReportMsg(circt::circtBugReportMsg);

//   DialectRegistry registry;
//   registry.insert<circt::comb::CombDialect, circt::hw::HWDialect,
//                   circt::smt::SMTDialect, circt::fsm::FSMDialect,
//                   mlir::LLVM::LLVMDialect, mlir::BuiltinDialect>();
//   mlir::func::registerInlinerExtension(registry);
//   mlir::registerBuiltinDialectTranslation(registry);
//   mlir::registerLLVMDialectTranslation(registry);
//   MLIRContext context(registry);

//   llvm::SourceMgr sourceMgr;
//   SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

//   context.printOpOnDiagnostic(false);

//   // exit(failed(translateFSM(context)));

//   // translateFSM(input, prop, output);

//   return 0;
// }
namespace{
struct FSMToSMTPass : public circt::impl::ConvertFSMToSMTBase<FSMToSMTPass> {
  void runOnOperation() override;
};


void FSMToSMTPass::runOnOperation() {

  auto module = getOperation();
  auto loc = module.getLoc();
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

  // for(auto machineOp: machineOps){

  //   smt::SolverOp solver = target.create<smt::SolverOp>(loc, b.getI1Type(), ValueRange{});

  //   mlir::ArrayAttr args = machineOp.getArgAttrsAttr();
  //   mlir::ArrayAttr argNames = machineOp.getArgNamesAttr();
  //   for(auto a: args)
  //     llvm::outs()<<a;
  //   // int numArgs = populateArgs(mod, vecVal);

  // }

  // parse variables, arguments, outputs separately

  // then for each state (thus, for each transition)
  // output the corresponding assertion


  



  // populateVars(mod, vecVal, numArgs);

  // int numOutputs = populateOutputs(mod, vecVal, context, module);


  // SmallVector<string> stateInv;

  // string initialState = getInitialState(mod);

  // // initial state is by default associated with id 0

  // insertState(initialState, stateInv);

  // populateST(mod, c, stateInv, transitions, vecVal, numOutputs);

  // // preparing the model

  // SmallVector<expr> solverVars;

  // SmallVector<Z3_sort> invInput;

  // SmallVector<func_decl> stateInvFun;

  // populateInvInput(variables, c, solverVars, invInput, numArgs, numOutputs);

  // expr timeVar = c.int_const("time");
  // z3::sort timeInv = c.int_sort();

  // solverVars.push_back(timeVar);
  // invInput.push_back(timeInv);

  // populateStateInvMap(stateInv, c, invInput, stateInvFun);

  // if (V) {
  //   llvm::outs() << "number of variables + args: " << solverVars.size() << "\n";
  //   for (auto v : solverVars) {
  //     llvm::outs() << "variable: " << v.to_string() << "\n";
  //   }
  // }

  // int j = 0;

  // SmallVector<expr> solverVarsInit;
  // copy(solverVars.begin(), solverVars.end(), back_inserter(solverVarsInit));

  // for(int i=numArgs; i<int(variables.size()); i++){

  //   if (variables[i].second.getType().getIntOrFloatBitWidth() > 1) {
  //     int initValue =
  //     stoi(variables[i].first.to_string().substr(variables[i].first.to_string().find("_")+1));
  //     solverVarsInit[i] = c.int_val(initValue);
  //   } else {
  //     bool initValue = stoi(variables[i].first.to_string().substr(variables[i].first.to_string().find("_")+1));
  //     solverVarsInit.at(i) = c.bool_val(initValue);
  //   }
  // }
  // for (int i = 0; i < numArgs; i++) {
  //   solverVarsInit[i] = argInputs[i](0);
  // }
  // expr initState =
  //     (stateInvFun[0](solverVarsInit.size(), solverVarsInit.data()));
  // expr body =
  //     implies((solverVarsInit[solverVarsInit.size() - 1] == 0),
  //             initState);

  // initial condition
  // s.add(nestedForall(solverVars, body, 0, numOutputs, c));

  // for (auto t : transitions) {

//   SmallVector<expr> solverVarsAfter;

//     copy(solverVars.begin(), solverVars.end(), back_inserter(solverVarsAfter));
//     solverVarsAfter.at(solverVarsAfter.size() - 1) =
//         solverVars[solverVars.size() - 1] + 1;

//     if (t.isOutput) {
//       if (t.isGuard && t.isAction) {
//         expr body = implies(
//             (stateInvFun[t.from](solverVars.size(), solverVars.data())) &&
//                 t.guard(solverVars),
//             stateInvFun[t.to](t.output(t.action(solverVarsAfter)).size(),
//                               t.output(t.action(solverVarsAfter)).data()));
//         s.add(nestedForall(solverVars, body, 0, numOutputs, c));
//       } else if (t.isGuard) {
//         expr body = implies(
//             (stateInvFun[t.from](solverVars.size(), solverVars.data()) &&
//              t.guard(solverVars)),
//             stateInvFun[t.to](t.output((solverVarsAfter)).size(),
//                               t.output((solverVarsAfter)).data()));
//         s.add(nestedForall(solverVars, body, 0, numOutputs, c));
//       } else if (t.isAction) {
//         expr body = implies(
//             stateInvFun[t.from](solverVars.size(), solverVars.data()),
//             stateInvFun[t.to](t.output(t.action(solverVarsAfter)).size(),
//                               t.output(t.action(solverVarsAfter)).data()));
//         s.add(nestedForall(solverVars, body, 0, numOutputs, c));
//       } else {
//         expr body =
//             implies((stateInvFun[t.from](solverVars.size(), solverVars.data())),
//                     stateInvFun[t.to](t.output(solverVarsAfter).size(),
//                                       t.output(solverVarsAfter).data()));
//         s.add(nestedForall(solverVars, body, 0, numOutputs, c));
//       }
//     } else {
//       if (t.isGuard && t.isAction) {
//         expr body = implies(
//             (stateInvFun[t.from](solverVars.size(), solverVars.data())) &&
//                 t.guard(solverVars),
//             stateInvFun[t.to](t.action(solverVarsAfter).size(),
//                               t.action(solverVarsAfter).data()));
//         s.add(nestedForall(solverVars, body, 0, numOutputs, c));
//       } else if (t.isGuard) {
//         expr body = implies(
//             (stateInvFun[t.from](solverVars.size(), solverVars.data()) &&
//              t.guard(solverVars)),
//             stateInvFun[t.to]((solverVarsAfter).size(),
//                               (solverVarsAfter).data()));
//         s.add(nestedForall(solverVars, body, 0, numOutputs, c));
//       } else if (t.isAction) {
//         expr body =
//             implies(stateInvFun[t.from](solverVars.size(), solverVars.data()),
//                     stateInvFun[t.to](t.action(solverVarsAfter).size(),
//                                       t.action(solverVarsAfter).data()));
//         s.add(nestedForall(solverVars, body, 0, numOutputs, c));
//       } else {
//         expr body =
//             implies((stateInvFun[t.from](solverVars.size(), solverVars.data())),
//                     stateInvFun[t.to]((solverVarsAfter).size(),
//                                       (solverVarsAfter).data()));
//         s.add(nestedForall(solverVars, body, 0, numOutputs, c));
//       }
//     }
//   }

//   // for(auto state1 : stateInvFun){
//   //   expr mutualExc = c.bool_val(true);
//   //   for(auto state2: stateInvFun){
//   //     if(state1.to_string() != state2.to_string()){
//   //       mutualExc = mutualExc && !state2(solverVars.size(), solverVars.data());
//   //     }
//   //   }
//   //   expr stateMutualExc = implies(state1(solverVars.size(), solverVars.data()), mutualExc);
//   //   s.add(nestedForall(solverVars, stateMutualExc, numArgs, numOutputs, c));
//   // }

//   expr r = parseLTL(property, solverVars, stateInv, stateInvFun,
//                     0, numOutputs, c);

//   s.add(r);

//   printSolverAssertions(c, s, output, stateInvFun);

//   return success();
// }

/// The entry point for the `circt-fsmt` tool:
/// parses the input file and calls the `convertFSM` function to do the actual work.

// int main(int argc, char **argv) {

//   llvm::InitLLVM y(argc, argv);

//   cl::ParseCommandLineOptions(
//       argc, argv,
//       "circt-fsmt - FSM representation model checker\n\n"
//       "\tThis tool verifies a property over the FSM hardware abstraction.\n");

//   llvm::setBugReportMsg(circt::circtBugReportMsg);

//   DialectRegistry registry;
//   registry.insert<circt::comb::CombDialect, circt::hw::HWDialect,
//                   circt::smt::SMTDialect, circt::fsm::FSMDialect,
//                   mlir::LLVM::LLVMDialect, mlir::BuiltinDialect>();
//   mlir::func::registerInlinerExtension(registry);
//   mlir::registerBuiltinDialectTranslation(registry);
//   mlir::registerLLVMDialectTranslation(registry);
//   MLIRContext context(registry);

//   llvm::SourceMgr sourceMgr;
//   SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

//   context.printOpOnDiagnostic(false);

//   // exit(failed(translateFSM(context)));

//   // translateFSM(input, prop, output);

//   return 0;
// }
}
} // namespace

std::unique_ptr<mlir::Pass> circt::createConvertFSMToSMTPass() {
  return std::make_unique<FSMToSMTPass>();
}