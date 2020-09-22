//===- handshake-runner.cpp -----------------------------------------------===//
//
// Copyright 2019 The CIRCT Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

// Tool which executes a restricted form of the standard dialect, and
// the handshake dialect.

#include <stdio.h>

#include <algorithm>
#include <list>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Any.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"

#include "circt/Dialect/Handshake/HandshakeOps.h"

#define DEBUG_TYPE "runner"

#define INDEX_WIDTH 32

using namespace llvm;
using namespace mlir;
using namespace circt;
using cl::opt;

static cl::OptionCategory mainCategory("Application options");

static opt<std::string> inputFileName(cl::Positional, cl::desc("<input file>"),
                                      cl::init("-"), cl::cat(mainCategory));
static cl::list<std::string> inputArgs(cl::Positional, cl::desc("<input args>"),
                                       cl::ZeroOrMore, cl::cat(mainCategory));

static opt<std::string>
    toplevelFunction("toplevelFunction", cl::Optional,
                     cl::desc("The toplevel function to execute"),
                     cl::init("main"), cl::cat(mainCategory));

static opt<bool> runStats("runStats", cl::Optional,
                          cl::desc("Print Execution Statistics"),
                          cl::init(false), cl::cat(mainCategory));

STATISTIC(instructionsExecuted, "Instructions Executed");
STATISTIC(simulatedTime, "Simulated Time");
// static int instructionsExecuted = 0;

void executeOp(mlir::ConstantIndexOp op, std::vector<Any> &in,
               std::vector<Any> &out) {
  auto attr = op.getAttrOfType<mlir::IntegerAttr>("value");
  out[0] = attr.getValue().sextOrTrunc(INDEX_WIDTH);
}
void executeOp(mlir::ConstantIntOp op, std::vector<Any> &in,
               std::vector<Any> &out) {
  auto attr = op.getAttrOfType<mlir::IntegerAttr>("value");
  out[0] = attr.getValue();
}
void executeOp(mlir::AddIOp op, std::vector<Any> &in, std::vector<Any> &out) {
  out[0] = any_cast<APInt>(in[0]) + any_cast<APInt>(in[1]);
}
void executeOp(mlir::AddFOp op, std::vector<Any> &in, std::vector<Any> &out) {
  out[0] = any_cast<APFloat>(in[0]) + any_cast<APFloat>(in[1]);
}
void executeOp(mlir::CmpIOp op, std::vector<Any> &in, std::vector<Any> &out) {
  APInt in0 = any_cast<APInt>(in[0]);
  APInt in1 = any_cast<APInt>(in[1]);
  APInt out0(1, mlir::applyCmpPredicate(op.getPredicate(), in0, in1));
  out[0] = out0;
}
void executeOp(mlir::CmpFOp op, std::vector<Any> &in, std::vector<Any> &out) {
  APFloat in0 = any_cast<APFloat>(in[0]);
  APFloat in1 = any_cast<APFloat>(in[1]);
  APInt out0(1, mlir::applyCmpPredicate(op.getPredicate(), in0, in1));
  out[0] = out0;
}
void executeOp(mlir::SubIOp op, std::vector<Any> &in, std::vector<Any> &out) {
  out[0] = any_cast<APInt>(in[0]) - any_cast<APInt>(in[1]);
}
void executeOp(mlir::SubFOp op, std::vector<Any> &in, std::vector<Any> &out) {
  out[0] = any_cast<APFloat>(in[0]) + any_cast<APFloat>(in[1]);
}
void executeOp(mlir::MulIOp op, std::vector<Any> &in, std::vector<Any> &out) {
  out[0] = any_cast<APInt>(in[0]) * any_cast<APInt>(in[1]);
}
void executeOp(mlir::MulFOp op, std::vector<Any> &in, std::vector<Any> &out) {
  out[0] = any_cast<APFloat>(in[0]) * any_cast<APFloat>(in[1]);
}
void executeOp(mlir::SignedDivIOp op, std::vector<Any> &in,
               std::vector<Any> &out) {
  assert(any_cast<APInt>(in[1]).getZExtValue() && "Division By Zero!");
  out[0] = any_cast<APInt>(in[0]).sdiv(any_cast<APInt>(in[1]));
}
void executeOp(mlir::UnsignedDivIOp op, std::vector<Any> &in,
               std::vector<Any> &out) {
  assert(any_cast<APInt>(in[1]).getZExtValue() && "Division By Zero!");
  out[0] = any_cast<APInt>(in[0]).udiv(any_cast<APInt>(in[1]));
}
void executeOp(mlir::DivFOp op, std::vector<Any> &in, std::vector<Any> &out) {
  out[0] = any_cast<APFloat>(in[0]) / any_cast<APFloat>(in[1]);
}
void executeOp(mlir::IndexCastOp op, std::vector<Any> &in,
               std::vector<Any> &out) {
  out[0] = in[0];
}
void executeOp(mlir::SignExtendIOp op, std::vector<Any> &in,
               std::vector<Any> &out) {
  int width = op.getType().getIntOrFloatBitWidth();
  out[0] = any_cast<APInt>(in[0]).sext(width);
}
void executeOp(mlir::ZeroExtendIOp op, std::vector<Any> &in,
               std::vector<Any> &out) {
  int width = op.getType().getIntOrFloatBitWidth();
  out[0] = any_cast<APInt>(in[0]).zext(width);
}

// Allocate a new matrix with dimensions given by the type, in the
// given store.  Return the pseuddo-pointer to the new matrix in the
// store (i.e. the first dimension index)
unsigned allocateMemRef(mlir::MemRefType type, std::vector<Any> &in,
                        std::vector<std::vector<Any>> &store,
                        std::vector<double> &storeTimes) {
  ArrayRef<int64_t> shape = type.getShape();
  int allocationSize = 1;
  unsigned count = 0;
  for (int64_t dim : shape) {
    if (dim > 0)
      allocationSize *= dim;
    else {
      assert(count < in.size());
      allocationSize *= any_cast<APInt>(in[count++]).getSExtValue();
    }
  }
  unsigned ptr = store.size();
  store.resize(ptr + 1);
  storeTimes.resize(ptr + 1);
  store[ptr].resize(allocationSize);
  storeTimes[ptr] = 0.0;
  mlir::Type elementType = type.getElementType();
  int width = elementType.getIntOrFloatBitWidth();
  for (int i = 0; i < allocationSize; i++) {
    if (elementType.isa<mlir::IntegerType>()) {
      store[ptr][i] = APInt(width, 0);
    } else if (elementType.isa<mlir::FloatType>()) {
      store[ptr][i] = APFloat(0.0);
    } else {
      llvm_unreachable("Unknown result type!\n");
    }
  }
  return ptr;
}

void executeOp(mlir::LoadOp op, std::vector<Any> &in, std::vector<Any> &out,
               std::vector<std::vector<Any>> &store) {
  ArrayRef<int64_t> shape = op.getMemRefType().getShape();
  unsigned address = 0;
  for (unsigned i = 0; i < shape.size(); i++) {
    address = address * shape[i] + any_cast<APInt>(in[i + 1]).getZExtValue();
  }
  unsigned ptr = any_cast<unsigned>(in[0]);
  assert(ptr < store.size());
  auto &ref = store[ptr];
  assert(address < ref.size());
  //  LLVM_DEBUG(dbgs() << "Load " << ref[address] << " from " << ptr << "[" <<
  //  address << "]\n");
  Any result = ref[address];
  out[0] = result;
}

void executeOp(mlir::StoreOp op, std::vector<Any> &in, std::vector<Any> &out,
               std::vector<std::vector<Any>> &store) {
  ArrayRef<int64_t> shape = op.getMemRefType().getShape();
  unsigned address = 0;
  for (unsigned i = 0; i < shape.size(); i++) {
    address = address * shape[i] + any_cast<APInt>(in[i + 2]).getZExtValue();
  }
  unsigned ptr = any_cast<unsigned>(in[1]);
  assert(ptr < store.size());
  auto &ref = store[ptr];
  //  LLVM_DEBUG(dbgs() << "Store " << in[0] << " to " << ptr << "[" << address
  //  << "]\n");
  assert(address < ref.size());
  ref[address] = in[0];
}

void executeOp(handshake::ForkOp op, std::vector<Any> &in,
               std::vector<Any> &out) {
  for (unsigned i = 0; i < out.size(); i++) {
    out[i] = in[0];
  }
}
void executeOp(handshake::JoinOp op, std::vector<Any> &in,
               std::vector<Any> &out) {
  out[0] = in[0];
}
void executeOp(handshake::ConstantOp op, std::vector<Any> &in,
               std::vector<Any> &out) {
  auto attr = op.getAttrOfType<mlir::IntegerAttr>("value");
  out[0] = attr.getValue();
}
void executeOp(handshake::StoreOp op, std::vector<Any> &in,
               std::vector<Any> &out, std::vector<std::vector<Any>> &store) {
  // Forward the address and data to the memory op.
  out[0] = in[0];
  out[1] = in[1];
}
void executeOp(handshake::BranchOp op, std::vector<Any> &in,
               std::vector<Any> &out) {
  out[0] = in[0];
}

void debugArg(const std::string &head, mlir::Value op, const APInt &value,
              double time) {
  LLVM_DEBUG(dbgs() << "  " << head << ":  " << op << " = " << value
                    << " (APInt<" << value.getBitWidth() << ">) @" << time
                    << "\n");
}

void debugArg(const std::string &head, mlir::Value op, const APFloat &value,
              double time) {
  LLVM_DEBUG(dbgs() << "  " << head << ":  " << op << " = ";
             value.print(dbgs()); dbgs() << " ("
                                         << "float"
                                         << ") @" << time << "\n");
}

void debugArg(const std::string &head, mlir::Value op, const Any &value,
              double time) {
  if (any_isa<APInt>(value)) {
    debugArg(head, op, any_cast<APInt>(value), time);
  } else if (any_isa<APFloat>(value)) {
    debugArg(head, op, any_cast<APFloat>(value), time);
  } else if (any_isa<unsigned>(value)) {
    // Represents an allocated buffer.
    LLVM_DEBUG(dbgs() << "  " << head << ":  " << op << " = Buffer "
                      << any_cast<unsigned>(value) << "\n");
  } else {
    llvm_unreachable("unknown type");
  }
}
Any readValueWithType(mlir::Type type, std::string &in) {
  std::stringstream arg(in);
  if (type.isIndex()) {
    int x;
    arg >> x;
    int width = INDEX_WIDTH;
    APInt aparg(width, x);
    return aparg;
  } else if (type.isa<mlir::IntegerType>()) {
    int x;
    arg >> x;
    int width = type.getIntOrFloatBitWidth();
    APInt aparg(width, x);
    return aparg;
    // } else if (type.isF16()) {
    //   half x;
    //   arg >> x;
    //   APFloat aparg(x);
    //   return aparg;
  } else if (type.isF32()) {
    float x;
    arg >> x;
    APFloat aparg(x);
    return aparg;
  } else if (type.isF64()) {
    double x;
    arg >> x;
    APFloat aparg(x);
    return aparg;
  } else {
    assert(0 && "unknown argument type!\n");
  }
}

std::string printAnyValueWithType(mlir::Type type, Any &value) {
  std::stringstream out;
  if (type.isa<mlir::IntegerType>() || type.isa<mlir::IndexType>()) {
    out << any_cast<APInt>(value).getSExtValue();
    return out.str();
  } else if (type.isa<mlir::FloatType>()) {
    out << any_cast<APFloat>(value).convertToDouble();
    return out.str();
  } else if (type.isa<mlir::NoneType>()) {
    return "none";
  } else {
    llvm_unreachable("Unknown result type!");
  }
}

void scheduleIfNeeded(std::list<mlir::Operation *> &readyList,
                      llvm::DenseMap<mlir::Value, Any> &valueMap,
                      mlir::Operation *op) {
  if (std::find(readyList.begin(), readyList.end(), op) == readyList.end()) {
    readyList.push_back(op);
  }
}
void scheduleUses(std::list<mlir::Operation *> &readyList,
                  llvm::DenseMap<mlir::Value, Any> &valueMap,
                  mlir::Value value) {
  for (auto &use : value.getUses()) {
    scheduleIfNeeded(readyList, valueMap, use.getOwner());
  }
}

bool executeStdOp(mlir::Operation &op, std::vector<Any> &inValues,
                  std::vector<Any> &outValues) {
  if (auto Op = dyn_cast<mlir::ConstantIndexOp>(op))
    executeOp(Op, inValues, outValues);
  else if (auto Op = dyn_cast<mlir::ConstantIntOp>(op))
    executeOp(Op, inValues, outValues);
  else if (auto Op = dyn_cast<mlir::AddIOp>(op))
    executeOp(Op, inValues, outValues);
  else if (auto Op = dyn_cast<mlir::AddFOp>(op))
    executeOp(Op, inValues, outValues);
  else if (auto Op = dyn_cast<mlir::SubIOp>(op))
    executeOp(Op, inValues, outValues);
  else if (auto Op = dyn_cast<mlir::SubFOp>(op))
    executeOp(Op, inValues, outValues);
  else if (auto Op = dyn_cast<mlir::CmpIOp>(op))
    executeOp(Op, inValues, outValues);
  else if (auto Op = dyn_cast<mlir::CmpFOp>(op))
    executeOp(Op, inValues, outValues);
  else if (auto Op = dyn_cast<mlir::MulIOp>(op))
    executeOp(Op, inValues, outValues);
  else if (auto Op = dyn_cast<mlir::MulFOp>(op))
    executeOp(Op, inValues, outValues);
  else if (auto Op = dyn_cast<mlir::UnsignedDivIOp>(op))
    executeOp(Op, inValues, outValues);
  else if (auto Op = dyn_cast<mlir::SignedDivIOp>(op))
    executeOp(Op, inValues, outValues);
  else if (auto Op = dyn_cast<mlir::DivFOp>(op))
    executeOp(Op, inValues, outValues);
  else if (auto Op = dyn_cast<mlir::IndexCastOp>(op))
    executeOp(Op, inValues, outValues);
  else if (auto Op = dyn_cast<mlir::SignExtendIOp>(op))
    executeOp(Op, inValues, outValues);
  else if (auto Op = dyn_cast<mlir::ZeroExtendIOp>(op))
    executeOp(Op, inValues, outValues);
  else
    return false;
  return true;
}

void executeFunction(mlir::FuncOp &toplevel,
                     llvm::DenseMap<mlir::Value, Any> &valueMap,
                     llvm::DenseMap<mlir::Value, double> &timeMap,
                     std::vector<Any> &results,
                     std::vector<double> &resultTimes,
                     std::vector<std::vector<Any>> &store,
                     std::vector<double> &storeTimes) {
  mlir::Block &entryBlock = toplevel.getBody().front();
  // An iterator which walks over the instructions.
  mlir::Block::iterator instIter = entryBlock.begin();

  // Main executive loop.  Start at the first instruction of the entry
  // block.  Fetch and execute instructions until we hit a terminator.
  while (true) {
    mlir::Operation &op = *instIter;
    int i = 0;
    std::vector<Any> inValues(op.getNumOperands());
    std::vector<Any> outValues(op.getNumResults());
    LLVM_DEBUG(dbgs() << "OP:  " << op.getName() << "\n");
    double time = 0.0;
    for (mlir::Value in : op.getOperands()) {
      inValues[i] = valueMap[in];
      time = std::max(time, timeMap[in]);
      LLVM_DEBUG(debugArg("IN", in, inValues[i], timeMap[in]));
      i++;
    }
    if (executeStdOp(op, inValues, outValues)) {
    } else if (auto Op = dyn_cast<mlir::AllocOp>(op)) {
      outValues[0] = allocateMemRef(Op.getType(), inValues, store, storeTimes);
      unsigned ptr = any_cast<unsigned>(outValues[0]);
      storeTimes[ptr] = time;
    } else if (auto Op = dyn_cast<mlir::LoadOp>(op)) {
      executeOp(Op, inValues, outValues, store);
      unsigned ptr = any_cast<unsigned>(inValues[0]);
      double storeTime = storeTimes[ptr];
      LLVM_DEBUG(dbgs() << "STORE: " << storeTime << "\n");
      time = std::max(time, storeTime);
      storeTimes[ptr] = time;
    } else if (auto Op = dyn_cast<mlir::StoreOp>(op)) {
      executeOp(Op, inValues, outValues, store);
      unsigned ptr = any_cast<unsigned>(inValues[1]);
      double storeTime = storeTimes[ptr];
      LLVM_DEBUG(dbgs() << "STORE: " << storeTime << "\n");
      time = std::max(time, storeTime);
      storeTimes[ptr] = time;
      // } else if (auto Op = dyn_cast<mlir::hpx::ExecutableOpInterface>(op)) {
      //   std::vector<APInt> inInts(op.getNumOperands());
      //   std::vector<APInt> outInts(op.getNumResults());
      //   for (unsigned i = 0; i < op.getNumOperands(); i++)
      //     inInts[i] = any_cast<APInt>(inValues[i]);
      //   Op.execute(inInts, outInts);
      //   for (unsigned i = 0; i < op.getNumResults(); i++)
      //     outValues[i] = outInts[i];
      //   if (!isa<mlir::hpx::SliceOp>(op) && !isa<mlir::hpx::UnsliceOp>(op))
      //     time += 1;
    } else if (auto Op = dyn_cast<mlir::BranchOp>(op)) {
      mlir::Block *dest = Op.getDest();
      unsigned arg = 0;
      for (mlir::Value out : dest->getArguments()) {
        LLVM_DEBUG(debugArg("ARG", out, inValues[arg], time));
        valueMap[out] = inValues[arg];
        timeMap[out] = time;
        arg++;
      }
      instIter = dest->begin();
      continue;
    } else if (auto Op = dyn_cast<mlir::CondBranchOp>(op)) {
      APInt condition = any_cast<APInt>(inValues[0]);
      mlir::Block *dest;
      std::vector<Any> inArgs;
      double time = 0.0;
      if (condition != 0) {
        dest = Op.getTrueDest();
        inArgs.resize(Op.getNumTrueOperands());
        for (mlir::Value in : Op.getTrueOperands()) {
          inArgs[i] = valueMap[in];
          time = std::max(time, timeMap[in]);
          LLVM_DEBUG(debugArg("IN", in, inArgs[i], timeMap[in]));
          i++;
        }
      } else {
        dest = Op.getFalseDest();
        inArgs.resize(Op.getNumFalseOperands());
        for (mlir::Value in : Op.getFalseOperands()) {
          inArgs[i] = valueMap[in];
          time = std::max(time, timeMap[in]);
          LLVM_DEBUG(debugArg("IN", in, inArgs[i], timeMap[in]));
          i++;
        }
      }
      int arg = 0;
      for (mlir::Value out : dest->getArguments()) {
        LLVM_DEBUG(debugArg("ARG", out, inArgs[arg], time));
        valueMap[out] = inArgs[arg];
        timeMap[out] = time;
        arg++;
      }
      instIter = dest->begin();
      continue;
    } else if (auto Op = dyn_cast<mlir::ReturnOp>(op)) {
      for (unsigned i = 0; i < results.size(); i++) {
        results[i] = inValues[i];
        resultTimes[i] = timeMap[Op.getOperand(i)];
      }
      return;
    } else if (auto Op = dyn_cast<mlir::CallOpInterface>(op)) {
      // implement function calls.
      mlir::Operation *calledOp = Op.resolveCallable();
      if (auto funcOp = dyn_cast<mlir::FuncOp>(calledOp)) {
        mlir::FunctionType ftype = funcOp.getType();
        unsigned inputs = ftype.getNumInputs();
        unsigned outputs = ftype.getNumResults();
        llvm::DenseMap<mlir::Value, Any> newValueMap;
        llvm::DenseMap<mlir::Value, double> newTimeMap;
        std::vector<Any> results(outputs);
        std::vector<double> resultTimes(outputs);
        std::vector<std::vector<Any>> store;
        std::vector<double> storeTimes;
        mlir::Block &entryBlock = funcOp.getBody().front();
        mlir::Block::BlockArgListType blockArgs = entryBlock.getArguments();

        for (unsigned i = 0; i < inputs; i++) {
          newValueMap[blockArgs[i]] = inValues[i];
          newTimeMap[blockArgs[i]] = timeMap[op.getOperand(i)];
        }
        executeFunction(funcOp, newValueMap, newTimeMap, results, resultTimes,
                        store, storeTimes);
        i = 0;
        for (mlir::Value out : op.getResults()) {
          valueMap[out] = results[i];
          timeMap[out] = resultTimes[i];
          i++;
        }
        instIter++;
        continue;
      } else {
        llvm_unreachable("Callable was not a Function!\n");
      }
    } else {
      llvm_unreachable("Unknown operation!\n");
    }
    i = 0;
    for (mlir::Value out : op.getResults()) {
      LLVM_DEBUG(debugArg("OUT", out, outValues[i], time));
      valueMap[out] = outValues[i];
      timeMap[out] = time + 1;
      i++;
    }
    instIter++;
    instructionsExecuted++;
  }
}

void executeHandshakeFunction(handshake::FuncOp &toplevel,
                              llvm::DenseMap<mlir::Value, Any> &valueMap,
                              llvm::DenseMap<mlir::Value, double> &timeMap,
                              std::vector<Any> &results,
                              std::vector<double> &resultTimes,
                              std::vector<std::vector<Any>> &store,
                              std::vector<double> &storeTimes) {
  mlir::Block &entryBlock = toplevel.getBody().front();
  // The arguments of the entry block.
  mlir::Block::BlockArgListType blockArgs = entryBlock.getArguments();
  // A list of operations which might be ready to execute.
  std::list<mlir::Operation *> readyList;

  for (unsigned i = 0; i < blockArgs.size(); i++) {
    scheduleUses(readyList, valueMap, blockArgs[i]);
  }
#define EXTRA_DEBUG
  while (true) {
#ifdef EXTRA_DEBUG
    LLVM_DEBUG(
        for (auto t
             : readyList) { dbgs() << "READY: " << *t << "\n"; } dbgs()
            << "Live: " << valueMap.size() << "\n";
        for (auto t
             : valueMap) {
          debugArg("Value:", t.first, t.second, 0.0);
          //        dbgs() << "Value: " << *(t.first) << " " << t.second <<
          //        "\n";
        });
#endif
    assert(readyList.size() > 0);
    mlir::Operation &op = *readyList.front();
    readyList.pop_front();

    /*    for(mlir::Value out : op.getResults()) {
      if(valueMap.count(out) != 0) {
        LLVM_DEBUG(dbgs() << "OP:  " << op << "\n");
        for(auto t : valueMap) {
          dbgs() << "Value: " << *(t.first) << " " << t.second << "\n";
        }
        assert(false);
      }
      }*/

    // Special handling for non-total functions.
    if (auto Op = dyn_cast<handshake::ControlMergeOp>(op)) {
      bool found = false;
      int i = 0;
      LLVM_DEBUG(dbgs() << "OP:  " << op << "\n");
      for (mlir::Value in : op.getOperands()) {
        if (valueMap.count(in) == 1) {
          assert(!found && "More than one valid input to CMerge!");
          auto t = valueMap[in];
          valueMap[op.getResult(0)] = t;
          timeMap[op.getResult(0)] = timeMap[in];
          scheduleUses(readyList, valueMap, op.getResult(0));

          valueMap[op.getResult(1)] = APInt(INDEX_WIDTH, i);
          timeMap[op.getResult(1)] = timeMap[in];
          scheduleUses(readyList, valueMap, op.getResult(1));

          // Consume the inputs.
          valueMap.erase(in);

          found = true;
          LLVM_DEBUG(debugArg("IN", in, t, timeMap[in]));
        }
        i++;
      }
      assert(found && "No valid input to CMerge!");
      continue;
    }

    if (auto Op = dyn_cast<handshake::MergeOp>(op)) {
      // Almost the same as CMerge above.
      bool found = false;
      int i = 0;
      LLVM_DEBUG(dbgs() << "OP:  " << op << "\n");
      for (mlir::Value in : op.getOperands()) {
        if (valueMap.count(in) == 1) {
          assert(!found && "More than one valid input to Merge!");
          auto t = valueMap[in];
          valueMap[op.getResult(0)] = t;
          timeMap[op.getResult(0)] = timeMap[in];
          scheduleUses(readyList, valueMap, op.getResult(0));

          // Consume the inputs.
          valueMap.erase(in);

          found = true;
          LLVM_DEBUG(debugArg("IN", in, t, timeMap[in]));
        }
        i++;
      }
      assert(found && "No valid input to Merge!");
      continue;
    }

    // Special handling for non-total functions.
    if (auto Op = dyn_cast<handshake::MuxOp>(op)) {
      mlir::Value control = op.getOperand(0);
      if (valueMap.count(control) == 0) {
        // it's not ready.  Reschedule it.
#ifdef EXTRA_DEBUG
        LLVM_DEBUG(dbgs() << "OP:  " << op << "\n");
        LLVM_DEBUG(dbgs() << "Rescheduling control...\n");
#endif
        readyList.push_back(&op);
        continue;
      }
      auto controlValue = valueMap[control];
      auto controlTime = timeMap[control];
      mlir::Value in = any_cast<APInt>(controlValue) == 0 ? op.getOperand(1)
                                                          : op.getOperand(2);
      if (valueMap.count(in) == 0) {
        // it's not ready.  Reschedule it.
#ifdef EXTRA_DEBUG
        LLVM_DEBUG(dbgs() << "Rescheduling data("
                          << any_cast<APInt>(controlValue) << ")...\n");
#endif
        readyList.push_back(&op);
        continue;
      }
      auto inValue = valueMap[in];
      auto inTime = timeMap[in];
      LLVM_DEBUG(dbgs() << "OP:  " << op << "\n");
      LLVM_DEBUG(debugArg("IN", control, controlValue, controlTime));
      LLVM_DEBUG(debugArg("IN", in, inValue, inTime));
      double time = std::max(controlTime, inTime);
      valueMap[op.getResult(0)] = inValue;
      timeMap[op.getResult(0)] = time;
      scheduleUses(readyList, valueMap, op.getResult(0));

      // Consume the inputs.
      valueMap.erase(control);
      valueMap.erase(in);
      continue;
    }

    // Special handling for non-total functions.
    if (auto Op = dyn_cast<handshake::LoadOp>(op)) {
      mlir::Value address = op.getOperand(0);
      mlir::Value data = op.getOperand(1);
      mlir::Value nonce = op.getOperand(2);
      mlir::Value addressOut = op.getResult(1);
      mlir::Value dataOut = op.getResult(0);
      if ((valueMap.count(address) && !valueMap.count(nonce)) ||
          (!valueMap.count(address) && valueMap.count(nonce)) ||
          (!valueMap.count(address) && !valueMap.count(nonce) &&
           !valueMap.count(data))) {
        // it's not ready.  Reschedule it.
#ifdef EXTRA_DEBUG
        LLVM_DEBUG(dbgs() << "OP:  " << op << "\n");
        LLVM_DEBUG(dbgs() << "Rescheduling...\n");
#endif
        readyList.push_back(&op);
        continue;
      }
      if (valueMap.count(address) && valueMap.count(nonce)) {
        auto addressValue = valueMap[address];
        auto addressTime = timeMap[address];
        auto nonceValue = valueMap[nonce];
        auto nonceTime = timeMap[nonce];
        LLVM_DEBUG(dbgs() << "OP:  " << op << "\n");
        LLVM_DEBUG(debugArg("Address", address, addressValue, addressTime));
        LLVM_DEBUG(debugArg("Nonce", nonce, nonceValue, nonceTime));
        valueMap[addressOut] = addressValue;
        double time = std::max(addressTime, nonceTime);
        timeMap[addressOut] = time;
        scheduleUses(readyList, valueMap, addressOut);
        // Consume the inputs.
        valueMap.erase(address);
        valueMap.erase(nonce);
      } else if (valueMap.count(data)) {
        auto dataValue = valueMap[data];
        auto dataTime = timeMap[data];
        LLVM_DEBUG(dbgs() << "OP:  " << op << "\n");
        LLVM_DEBUG(debugArg("Data", data, dataValue, dataTime));
        valueMap[dataOut] = dataValue;
        timeMap[dataOut] = dataTime;
        scheduleUses(readyList, valueMap, dataOut);
        // Consume the inputs.
        valueMap.erase(data);
      } else {
        llvm_unreachable("why?");
      }
      continue;
    }
    // Special handling for non-total functions.
    if (auto Op = dyn_cast<handshake::MemoryOp>(op)) {
      static llvm::DenseMap<unsigned, unsigned> idToBuffer;
      int opIndex = 0;
      bool notReady = false;
      LLVM_DEBUG(dbgs() << "OP:  " << op << "\n");
      unsigned id = Op.getID(); // The ID of this memory.
      if (!idToBuffer.count(id)) {
        auto memreftype = Op.getMemRefType();
        std::vector<Any> nothing;
        std::string x;
        unsigned buffer =
            allocateMemRef(memreftype, nothing, store, storeTimes);
        idToBuffer[id] = buffer;
      }
      unsigned buffer = idToBuffer[id];

      for (unsigned i = 0; i < Op.getStCount().getZExtValue(); i++) {
        mlir::Value data = op.getOperand(opIndex++);
        mlir::Value address = op.getOperand(opIndex++);
        mlir::Value nonceOut = op.getResult(Op.getLdCount().getZExtValue() + i);
        if ((!valueMap.count(data) || !valueMap.count(address))) {
          notReady = true;
          continue;
        }
        auto addressValue = valueMap[address];
        auto addressTime = timeMap[address];
        auto dataValue = valueMap[data];
        auto dataTime = timeMap[data];
        LLVM_DEBUG(debugArg("Store", address, addressValue, addressTime));
        LLVM_DEBUG(debugArg("StoreData", data, dataValue, dataTime));

        assert(buffer < store.size());
        auto &ref = store[buffer];
        //  LLVM_DEBUG(dbgs() << "Store " << in[0] << " to " << ptr << "[" <<
        //  address << "]\n");
        unsigned offset = any_cast<APInt>(addressValue).getZExtValue();
        assert(offset < ref.size());
        ref[offset] = dataValue;

        // Implicit none argument
        APInt apnonearg(1, 0);
        valueMap[nonceOut] = apnonearg;
        double time = std::max(addressTime, dataTime);
        timeMap[nonceOut] = time;
        scheduleUses(readyList, valueMap, nonceOut);
        // Consume the inputs.
        valueMap.erase(data);
        valueMap.erase(address);
      }

      for (unsigned i = 0; i < Op.getLdCount().getZExtValue(); i++) {
        mlir::Value address = op.getOperand(opIndex++);
        mlir::Value dataOut = op.getResult(i);
        mlir::Value nonceOut = op.getResult(Op.getLdCount().getZExtValue() +
                                            Op.getStCount().getZExtValue() + i);
        if (!valueMap.count(address)) {
          notReady = true;
          continue;
        }
        auto addressValue = valueMap[address];
        auto addressTime = timeMap[address];
        LLVM_DEBUG(debugArg("Load:", address, addressValue, addressTime));
        assert(buffer < store.size());
        auto &ref = store[buffer];
        //  LLVM_DEBUG(dbgs() << "Store " << in[0] << " to " << ptr << "[" <<
        //  address << "]\n");
        unsigned offset = any_cast<APInt>(addressValue).getZExtValue();
        assert(offset < ref.size());

        valueMap[dataOut] = ref[offset];
        timeMap[dataOut] = addressTime;
        // Implicit none argument
        APInt apnonearg(1, 0);
        valueMap[nonceOut] = apnonearg;
        timeMap[nonceOut] = addressTime;
        scheduleUses(readyList, valueMap, dataOut);
        scheduleUses(readyList, valueMap, nonceOut);
        // Consume the inputs.
        valueMap.erase(address);
      }
      if (notReady) {
        // it's not ready.  Reschedule it.
#ifdef EXTRA_DEBUG
        LLVM_DEBUG(dbgs() << "OP:  " << op << "\n");
        LLVM_DEBUG(dbgs() << "Rescheduling...\n");
#endif
        readyList.push_back(&op);
        continue;
      }
      continue;
    }

    if (auto Op = dyn_cast<handshake::ConditionalBranchOp>(op)) {
      mlir::Value control = op.getOperand(0);
      if (valueMap.count(control) == 0) {
        // it's not ready.  Reschedule it.
#ifdef EXTRA_DEBUG
        LLVM_DEBUG(dbgs() << "Rescheduling control...\n");
#endif
        readyList.push_back(&op);
        continue;
      }
      auto controlValue = valueMap[control];
      auto controlTime = timeMap[control];
      mlir::Value in = op.getOperand(1);
      if (valueMap.count(in) == 0) {
        // it's not ready.  Reschedule it.
#ifdef EXTRA_DEBUG
        LLVM_DEBUG(dbgs() << "Rescheduling data...\n");
#endif
        readyList.push_back(&op);
        continue;
      }
      auto inValue = valueMap[in];
      auto inTime = timeMap[in];
      LLVM_DEBUG(dbgs() << "OP:  " << op << "\n");
      LLVM_DEBUG(debugArg("IN", control, controlValue, controlTime));
      LLVM_DEBUG(debugArg("IN", in, inValue, inTime));
      mlir::Value out = any_cast<APInt>(controlValue) != 0 ? op.getResult(0)
                                                           : op.getResult(1);
      double time = std::max(controlTime, inTime);
      valueMap[out] = inValue;
      timeMap[out] = time;
      scheduleUses(readyList, valueMap, out);

      // Consume the inputs.
      valueMap.erase(control);
      valueMap.erase(in);

      continue;
    }

    int i = 0;
    std::vector<Any> inValues(op.getNumOperands());
    std::vector<Any> outValues(op.getNumResults());
    bool reschedule = false;
    LLVM_DEBUG(dbgs() << "OP: (" << op.getNumOperands() << "->"
                      << op.getNumResults() << ")" << op << "\n");
    double time;
    for (mlir::Value in : op.getOperands()) {
      if (valueMap.count(in) == 0) {
        reschedule = true;
        continue;
      }
      inValues[i] = valueMap[in];
      time = std::max(time, timeMap[in]);
      LLVM_DEBUG(debugArg("IN", in, inValues[i], timeMap[in]));
      i++;
    }
    if (reschedule) {
      LLVM_DEBUG(dbgs() << "Rescheduling data...\n");
      readyList.push_back(&op);
      continue;
    }
    // Consume the inputs.
    for (mlir::Value in : op.getOperands()) {
      valueMap.erase(in);
    }
    if (executeStdOp(op, inValues, outValues)) {
      // } else if (auto Op = dyn_cast<mlir::hpx::ExecutableOpInterface>(op)) {
      //   std::vector<APInt> inInts(op.getNumOperands());
      //   std::vector<APInt> outInts(op.getNumResults());
      //   for (unsigned i = 0; i < op.getNumOperands(); i++) {
      //     assert(inValues[i].hasValue());
      //     inInts[i] = any_cast<APInt>(inValues[i]);
      //   }
      //   Op.execute(inInts, outInts);
      //   for (unsigned i = 0; i < op.getNumResults(); i++)
      //     outValues[i] = outInts[i];
      //   if (!isa<mlir::hpx::SliceOp>(op) && !isa<mlir::hpx::UnsliceOp>(op))
      //     time += 1;
    } else if (auto Op = dyn_cast<handshake::StartOp>(op)) {
    } else if (auto Op = dyn_cast<handshake::EndOp>(op)) {
    } else if (auto Op = dyn_cast<handshake::SinkOp>(op)) {
    } else if (auto Op = dyn_cast<handshake::ForkOp>(op))
      executeOp(Op, inValues, outValues);
    else if (auto Op = dyn_cast<handshake::JoinOp>(op))
      executeOp(Op, inValues, outValues);
    else if (auto Op = dyn_cast<handshake::ConstantOp>(op))
      executeOp(Op, inValues, outValues);
    else if (auto Op = dyn_cast<handshake::StoreOp>(op))
      executeOp(Op, inValues, outValues, store);
    else if (auto Op = dyn_cast<handshake::BranchOp>(op))
      executeOp(Op, inValues, outValues);
    else if (auto Op = dyn_cast<handshake::ReturnOp>(op)) {
      for (unsigned i = 0; i < results.size(); i++) {
        results[i] = inValues[i];
        resultTimes[i] = timeMap[Op.getOperand(i)];
      }
      return;
      //} else {
      // implement function calls.
    } else {
      llvm_unreachable("Unknown operation!\n");
    }
    i = 0;
    for (mlir::Value out : op.getResults()) {
      LLVM_DEBUG(debugArg("OUT", out, outValues[i], time));
      assert(outValues[i].hasValue());
      valueMap[out] = outValues[i];
      timeMap[out] = time + 1;
      scheduleUses(readyList, valueMap, out);

      i++;
    }
    instructionsExecuted++;
  }
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(
      argc, argv,
      "MLIR Standard dialect runner\n\n"
      "This application executes a function in the given MLIR module\n"
      "Arguments to the function are passed on the command line and\n"
      "results are returned on stdout.\n"
      "Memref types are specified as a comma-separated list of values.\n");

  auto file_or_err = MemoryBuffer::getFileOrSTDIN(inputFileName.c_str());
  if (std::error_code error = file_or_err.getError()) {
    errs() << argv[0] << ": could not open input file '" << inputFileName
           << "': " << error.message() << "\n";
    return 1;
  }

  // Load the MLIR module.
  mlir::MLIRContext context;
  context.loadDialect<StandardOpsDialect, handshake::HandshakeOpsDialect>();
  SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(*file_or_err), SMLoc());
  mlir::OwningModuleRef module(mlir::parseSourceFile(source_mgr, &context));
  if (!module)
    return 1;

  mlir::Operation *mainP = module->lookupSymbol(toplevelFunction);
  // The toplevel function can accept any number of operands, and returns
  // any number of results.
  if (!mainP) {
    errs() << "Toplevel function " << toplevelFunction << " not found!\n";
    return 1;
  }

  // The store associates each allocation in the program
  // (represented by a int) with a vector of values which can be
  // accessed by it.  Currently values are assumed to be an integer.
  std::vector<std::vector<Any>> store;
  std::vector<double> storeTimes;

  // The valueMap associates each SSA statement in the program
  // (represented by a Value*) with it's corresponding value.
  // Currently the value is assumed to be an integer.
  llvm::DenseMap<mlir::Value, Any> valueMap;

  // The timeMap associates each value with the time it was created.
  llvm::DenseMap<mlir::Value, double> timeMap;

  // We need three things in a function-type independent way.
  // The type signature of the function.
  mlir::FunctionType ftype;
  // The arguments of the entry block.
  mlir::Block::BlockArgListType blockArgs;

  // The number of inputs to the function in the IR.
  unsigned inputs;
  unsigned outputs;
  // The number of 'real' inputs.  This avoids the dummy input
  // associated with the handshake control logic for handshake
  // functions.
  unsigned realInputs;
  unsigned realOutputs;

  if (mlir::FuncOp toplevel =
          module->lookupSymbol<mlir::FuncOp>(toplevelFunction)) {
    ftype = toplevel.getType();
    mlir::Block &entryBlock = toplevel.getBody().front();
    blockArgs = entryBlock.getArguments();

    // Get the primary inputs of toplevel off the command line.
    inputs = ftype.getNumInputs();
    realInputs = inputs;
    outputs = ftype.getNumResults();
    realOutputs = outputs;
  } else if (handshake::FuncOp toplevel =
                 module->lookupSymbol<handshake::FuncOp>(toplevelFunction)) {
    ftype = toplevel.getType();
    mlir::Block &entryBlock = toplevel.getBody().front();
    blockArgs = entryBlock.getArguments();

    // Get the primary inputs of toplevel off the command line.
    inputs = ftype.getNumInputs();
    realInputs = inputs - 1;
    outputs = ftype.getNumResults();
    realOutputs = outputs - 1;
    if (inputs == 0) {
      errs() << "Function " << toplevelFunction << " is expected to have "
             << "at least one dummy argument.\n";
      return 1;
    }
    if (outputs == 0) {
      errs() << "Function " << toplevelFunction << " is expected to have "
             << "at least one dummy result.\n";
      return 1;
    }
    // Implicit none argument
    APInt apnonearg(1, 0);
    valueMap[blockArgs[blockArgs.size() - 1]] = apnonearg;
  } else {
    llvm_unreachable("Function not supported.\n");
  }

  if (inputArgs.size() != realInputs) {
    errs() << "Toplevel function " << toplevelFunction << " has " << realInputs
           << " actual arguments, but " << inputArgs.size()
           << " arguments were provided on the command line.\n";
    return 1;
  }

  for (unsigned i = 0; i < realInputs; i++) {
    mlir::Type type = ftype.getInput(i);
    if (type.isa<mlir::MemRefType>()) {
      // We require this memref type to be fully specified.
      auto memreftype = type.dyn_cast<mlir::MemRefType>();
      std::vector<Any> nothing;
      std::string x;
      unsigned buffer = allocateMemRef(memreftype, nothing, store, storeTimes);
      valueMap[blockArgs[i]] = buffer;
      timeMap[blockArgs[i]] = 0.0;
      int i = 0;
      std::stringstream arg(inputArgs[i]);
      while (!arg.eof()) {
        getline(arg, x, ',');
        store[buffer][i++] = readValueWithType(memreftype.getElementType(), x);
      }
    } else {
      Any value = readValueWithType(type, inputArgs[i]);
      valueMap[blockArgs[i]] = value;
      timeMap[blockArgs[i]] = 0.0;
    }
  }

  std::vector<Any> results(realOutputs);
  std::vector<double> resultTimes(realOutputs);
  if (mlir::FuncOp toplevel =
          module->lookupSymbol<mlir::FuncOp>(toplevelFunction)) {
    executeFunction(toplevel, valueMap, timeMap, results, resultTimes, store,
                    storeTimes);
  } else if (handshake::FuncOp toplevel =
                 module->lookupSymbol<handshake::FuncOp>(toplevelFunction)) {
    executeHandshakeFunction(toplevel, valueMap, timeMap, results, resultTimes,
                             store, storeTimes);
  }
  double time = 0.0;
  for (unsigned i = 0; i < results.size(); i++) {
    mlir::Type t = ftype.getResult(i);
    outs() << printAnyValueWithType(t, results[i]) << " ";
    time = std::max(resultTimes[i], time);
  }
  // Go back through the arguments and output any memrefs.
  for (unsigned i = 0; i < realInputs; i++) {
    mlir::Type type = ftype.getInput(i);
    if (type.isa<mlir::MemRefType>()) {
      // We require this memref type to be fully specified.
      auto memreftype = type.dyn_cast<mlir::MemRefType>();
      unsigned buffer = any_cast<unsigned>(valueMap[blockArgs[i]]);
      auto elementType = memreftype.getElementType();
      for (int j = 0; j < memreftype.getNumElements(); j++) {
        if (j != 0)
          outs() << ",";
        outs() << printAnyValueWithType(elementType, store[buffer][j]);
      }
      outs() << " ";
    }
  }
  outs() << "\n";

  simulatedTime += (int)time;

  return 0;
}
