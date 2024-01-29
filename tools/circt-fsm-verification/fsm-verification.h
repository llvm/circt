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
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
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
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include </Users/luisa/z3/src/api/c++/z3++.h>

using namespace llvm;
using namespace mlir;
using namespace circt;
using namespace std;
using namespace z3; 

using z3Fun = std::function <expr (vector<expr>)>;

using z3FunA = std::function <vector<expr> (vector<expr>)>;

struct transition{
  string from;
  string to;
  z3Fun guard;
  mlir::Region *guard_reg;
  bool isGuard;
  mlir::Region *action_reg;
  z3FunA action;
  bool isAction;
};

struct MyExprMap{
  vector<expr> exprs;
  vector<mlir::Value> values;
};

struct MyStateInvMap{
  vector<mlir::StringRef> stateName;
  vector<int> stateID;
};

struct MyStateInvMapFun{
  vector<mlir::StringRef> stateName;
  vector<func_decl> invFun;
};




void printSolverAssertions(z3::solver& solver);