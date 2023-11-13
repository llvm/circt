#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Dialect/FSM/FSMPasses.h"
#include "circt/Dialect/FSM/FSMTypes.h"
#include "circt/Dialect/FSM/FSMGraph.h"
#include "circt/Dialect/FSM/FSMDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/InitAllDialects.h"
#include "circt/InitAllPasses.h"
#include "circt/Support/Passes.h"
#include "circt/Support/Version.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "circt/InitAllDialects.h"
#include "circt/InitAllPasses.h"
#include "circt/Support/LoweringOptions.h"
#include "circt/Support/Version.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWReductions.h"
#include "circt/InitAllDialects.h"
#include "circt/Reduce/GenericReductions.h"
#include "circt/Reduce/Tester.h"
#include "circt/Support/Version.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace mlir;
using namespace circt;
using namespace std;

struct variable{
  string name;
  int initValue;
};

struct inputs{
  string name;
};

struct outputs{
  string name;
};

struct transition{
  string from;
  string to;
  std::vector<std::string > *var_updates;
  std::vector<std::string > *guards;
};


struct CFG{
  string initialState;
  std::vector<variable> variables;
  std::vector<transition *> transitions;
  std::vector<outputs> outputs;
  std::vector<inputs> inputs;
};

void print_cfg(CFG *cfg);

void store_variable_declaration(Operation &op, CFG *cfg);

string manage_comb_exp(Operation &op);

void manage_output_region(Region &region, CFG *cfg);

void manage_guard_region(Region &region, transition *t);

void manage_action_region(Region &region, transition *t);

void manage_transitions_region(Region &region, string current_state, CFG *cfg);

void manage_state(Operation &op, CFG *cfg);

void explore_nested_blocks(Operation &op, int level, CFG *cfg);

void parse_fsm(string input_file);

void manage_block_in_out(Block &op, CFG *cfg);
