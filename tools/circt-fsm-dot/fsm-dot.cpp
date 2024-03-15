#include "fsm-dot.hpp"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include <iostream>
#include <vector>
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


void parseFSM(string input_file, string output_file){

  DialectRegistry registry;
  // clang-format off
  registry.insert<
    comb::CombDialect,
    fsm::FSMDialect,
    hw::HWDialect
  >();
  
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
                          string g;
                          for(auto &op: r.getOps()){
                            if (auto retop = dyn_cast<fsm::ReturnOp>(op)){
                              g = retop.to_string()  //expr_map.at(retop.getOperand());
                            } 
                          }
                          t.guard = g;
                          t.isGuard = true;
                        }
                        // action 
                        if(!trRegions[1]->empty()){
                          Region &r = *trRegions[1];
                          for(auto &op: r.getOps()){
                            if(!found){
                              if (auto updateop = dyn_cast<fsm::UpdateOp>(op)){
                                if(v == updateop.getOperands()[0]){
                                  updated_vec.push_back(getExpr(updateop.getOperands()[1], expr_map, c));
                                }
                              } else {
                                vector<expr> vec;
                                for (auto operand: op.getOperands()){
                                  t.action.push_back(op.to_string());

                                }
                              }
                            }

                          }
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

int main(int argc, char **argv){

  string input = argv[1];

  string output = argg[2];

  parseFSM(input, output);

  return 0;

}
