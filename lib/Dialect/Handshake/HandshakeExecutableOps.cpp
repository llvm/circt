//===- HandshakeExecutableOps.cpp - Handshake executable Operations -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of execution semantics for Handshake
// operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/Any.h"
#include "llvm/Support/Debug.h"

using namespace circt;
using namespace circt::handshake;

#define INDEX_WIDTH 32

// Convert ValueRange to vectors
static std::vector<mlir::Value> toVector(mlir::ValueRange range) {
  return std::vector<mlir::Value>(range.begin(), range.end());
}

// Returns whether the precondition holds for a general op to execute
static bool isReadyToExecute(ArrayRef<mlir::Value> ins,
                             ArrayRef<mlir::Value> outs,
                             llvm::DenseMap<mlir::Value, llvm::Any> &valueMap) {
  for (auto in : ins)
    if (valueMap.count(in) == 0)
      return false;

  for (auto out : outs)
    if (valueMap.count(out) > 0)
      return false;

  return true;
}

// Fetch values from the value map and consume them
static std::vector<llvm::Any>
fetchValues(ArrayRef<mlir::Value> values,
            llvm::DenseMap<mlir::Value, llvm::Any> &valueMap) {
  std::vector<llvm::Any> ins;
  for (auto &value : values) {
    assert(valueMap[value].hasValue());
    ins.push_back(valueMap[value]);
    valueMap.erase(value);
  }
  return ins;
}

// Store values to the value map
static void storeValues(std::vector<llvm::Any> &values,
                        ArrayRef<mlir::Value> outs,
                        llvm::DenseMap<mlir::Value, llvm::Any> &valueMap) {
  assert(values.size() == outs.size());
  for (unsigned long i = 0; i < outs.size(); ++i)
    valueMap[outs[i]] = values[i];
}

// Update the time map after the execution
static void updateTime(ArrayRef<mlir::Value> ins, ArrayRef<mlir::Value> outs,
                       llvm::DenseMap<mlir::Value, double> &timeMap,
                       double latency) {
  double time = 0;
  for (auto &in : ins)
    time = std::max(time, timeMap[in]);
  time += latency;
  for (auto &out : outs)
    timeMap[out] = time;
}

static bool tryToExecute(Operation *op,
                         llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                         llvm::DenseMap<mlir::Value, double> &timeMap,
                         std::vector<mlir::Value> &scheduleList,
                         double latency) {
  auto ins = toVector(op->getOperands());
  auto outs = toVector(op->getResults());

  if (isReadyToExecute(ins, outs, valueMap)) {
    auto in = fetchValues(ins, valueMap);
    std::vector<llvm::Any> out(outs.size());
    auto generalOp = dyn_cast<GeneralOpInterface>(op);
    if (!generalOp)
      op->emitError("Undefined execution for the current op");
    generalOp.execute(in, out);
    storeValues(out, outs, valueMap);
    updateTime(ins, outs, timeMap, latency);
    scheduleList = outs;
    return true;
  } else
    return false;
}

namespace circt {
namespace handshake {

bool ForkOp::tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                        llvm::DenseMap<unsigned, unsigned> & /*memoryMap*/,
                        llvm::DenseMap<mlir::Value, double> &timeMap,
                        std::vector<std::vector<llvm::Any>> & /*store*/,
                        std::vector<mlir::Value> &scheduleList) {
  return tryToExecute(getOperation(), valueMap, timeMap, scheduleList, 1);
}

bool MergeOp::tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                         llvm::DenseMap<unsigned, unsigned> & /*memoryMap*/,
                         llvm::DenseMap<mlir::Value, double> &timeMap,
                         std::vector<std::vector<llvm::Any>> & /*store*/,
                         std::vector<mlir::Value> &scheduleList) {
  auto op = getOperation();
  bool found = false;
  int i = 0;
  for (mlir::Value in : op->getOperands()) {
    if (valueMap.count(in) == 1) {
      if (found)
        op->emitError("More than one valid input to Merge!");
      auto t = valueMap[in];
      valueMap[op->getResult(0)] = t;
      timeMap[op->getResult(0)] = timeMap[in];
      // Consume the inputs.
      valueMap.erase(in);
      found = true;
    }
    i++;
  }
  if (!found)
    op->emitError("No valid input to Merge!");
  scheduleList.push_back(getResult());
  return true;
}
bool MuxOp::tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                       llvm::DenseMap<unsigned, unsigned> & /*memoryMap*/,
                       llvm::DenseMap<mlir::Value, double> &timeMap,
                       std::vector<std::vector<llvm::Any>> & /*store*/,
                       std::vector<mlir::Value> &scheduleList) {
  auto op = getOperation();
  mlir::Value control = op->getOperand(0);
  if (valueMap.count(control) == 0)
    return false;
  auto controlValue = valueMap[control];
  auto controlTime = timeMap[control];
  mlir::Value in = llvm::any_cast<APInt>(controlValue) == 0 ? op->getOperand(1)
                                                            : op->getOperand(2);
  if (valueMap.count(in) == 0)
    return false;
  auto inValue = valueMap[in];
  auto inTime = timeMap[in];
  double time = std::max(controlTime, inTime);
  valueMap[op->getResult(0)] = inValue;
  timeMap[op->getResult(0)] = time;

  // Consume the inputs.
  valueMap.erase(control);
  valueMap.erase(in);
  scheduleList.push_back(getResult());
  return true;
}
bool ControlMergeOp::tryExecute(
    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> & /*memoryMap*/,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> & /*store*/,
    std::vector<mlir::Value> &scheduleList) {
  auto op = getOperation();
  bool found = false;
  int i = 0;
  for (mlir::Value in : op->getOperands()) {
    if (valueMap.count(in) == 1) {
      if (found)
        op->emitError("More than one valid input to CMerge!");
      auto t = valueMap[in];
      valueMap[op->getResult(0)] = t;
      timeMap[op->getResult(0)] = timeMap[in];

      valueMap[op->getResult(1)] = APInt(INDEX_WIDTH, i);
      timeMap[op->getResult(1)] = timeMap[in];

      // Consume the inputs.
      valueMap.erase(in);

      found = true;
    }
    i++;
  }
  if (!found)
    op->emitError("No valid input to CMerge!");
  scheduleList = toVector(op->getResults());
  return true;
}

void BranchOp::execute(std::vector<llvm::Any> &ins,
                       std::vector<llvm::Any> &outs) {
  outs[0] = ins[0];
}

bool BranchOp::tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                          llvm::DenseMap<unsigned, unsigned> & /*memoryMap*/,
                          llvm::DenseMap<mlir::Value, double> &timeMap,
                          std::vector<std::vector<llvm::Any>> & /*store*/,
                          std::vector<mlir::Value> &scheduleList) {
  return tryToExecute(getOperation(), valueMap, timeMap, scheduleList, 0);
}

bool ConditionalBranchOp::tryExecute(
    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> & /*memoryMap*/,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> & /*store*/,
    std::vector<mlir::Value> &scheduleList) {
  auto op = getOperation();
  mlir::Value control = op->getOperand(0);
  if (valueMap.count(control) == 0)
    return false;
  auto controlValue = valueMap[control];
  auto controlTime = timeMap[control];
  mlir::Value in = op->getOperand(1);
  if (valueMap.count(in) == 0)
    return false;
  auto inValue = valueMap[in];
  auto inTime = timeMap[in];
  mlir::Value out = llvm::any_cast<APInt>(controlValue) != 0 ? op->getResult(0)
                                                             : op->getResult(1);
  double time = std::max(controlTime, inTime);
  valueMap[out] = inValue;
  timeMap[out] = time;
  scheduleList.push_back(out);

  // Consume the inputs.
  valueMap.erase(control);
  valueMap.erase(in);
  return true;
}
bool StartOp::tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> & /*valueMap*/,
                         llvm::DenseMap<unsigned, unsigned> & /*memoryMap*/,
                         llvm::DenseMap<mlir::Value, double> & /*timeMap*/,
                         std::vector<std::vector<llvm::Any>> & /*store*/,
                         std::vector<mlir::Value> & /*scheduleList*/) {
  assert(false && "StartOp's should never exist in a real program due to being "
                  "purely lowering helper operations.");
  return true;
}
bool EndOp::tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> & /*valueMap*/,
                       llvm::DenseMap<unsigned, unsigned> & /*memoryMap*/,
                       llvm::DenseMap<mlir::Value, double> & /*timeMap*/,
                       std::vector<std::vector<llvm::Any>> & /*store*/,
                       std::vector<mlir::Value> & /*scheduleList*/) {
  assert(false && "EndOp's should never exist in a real program due to being "
                  "purely lowering helper operations.");
  return true;
}

bool SinkOp::tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                        llvm::DenseMap<unsigned, unsigned> & /*memoryMap*/,
                        llvm::DenseMap<mlir::Value, double> & /*timeMap*/,
                        std::vector<std::vector<llvm::Any>> & /*store*/,
                        std::vector<mlir::Value> & /*scheduleList*/) {
  valueMap.erase(getOperand());
  return true;
}

void ConstantOp::execute(std::vector<llvm::Any> & /*ins*/,
                         std::vector<llvm::Any> &outs) {
  auto attr = (*this)->getAttrOfType<mlir::IntegerAttr>("value");
  outs[0] = attr.getValue();
}

bool ConstantOp::tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                            llvm::DenseMap<unsigned, unsigned> & /*memoryMap*/,
                            llvm::DenseMap<mlir::Value, double> &timeMap,
                            std::vector<std::vector<llvm::Any>> & /*store*/,
                            std::vector<mlir::Value> &scheduleList) {
  return tryToExecute(getOperation(), valueMap, timeMap, scheduleList, 0);
}

bool ExternalMemoryOp::tryExecute(
    llvm::DenseMap<mlir::Value, llvm::Any> & /*valueMap*/,
    llvm::DenseMap<unsigned, unsigned> & /*memoryMap*/,
    llvm::DenseMap<mlir::Value, double> & /*timeMap*/,
    std::vector<std::vector<llvm::Any>> & /*store*/,
    std::vector<mlir::Value> & /*scheduleList*/) {
  // todo(mortbopet): implement execution of ExternalMemoryOp's.
  assert(false && "implement me");
  return 0;
}

bool MemoryOp::tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                          llvm::DenseMap<unsigned, unsigned> &memoryMap,
                          llvm::DenseMap<mlir::Value, double> &timeMap,
                          std::vector<std::vector<llvm::Any>> &store,
                          std::vector<mlir::Value> &scheduleList) {
  auto op = getOperation();
  int opIndex = 0;
  bool notReady = false;
  unsigned buffer = memoryMap[id()];

  for (unsigned i = 0; i < stCount(); i++) {
    mlir::Value data = op->getOperand(opIndex++);
    mlir::Value address = op->getOperand(opIndex++);
    mlir::Value nonceOut = op->getResult(ldCount() + i);
    if ((!valueMap.count(data) || !valueMap.count(address))) {
      notReady = true;
      continue;
    }
    auto addressValue = valueMap[address];
    auto addressTime = timeMap[address];
    auto dataValue = valueMap[data];
    auto dataTime = timeMap[data];

    assert(buffer < store.size());
    auto &ref = store[buffer];
    unsigned offset = llvm::any_cast<APInt>(addressValue).getZExtValue();
    assert(offset < ref.size());
    ref[offset] = dataValue;

    // Implicit none argument
    APInt apnonearg(1, 0);
    valueMap[nonceOut] = apnonearg;
    double time = std::max(addressTime, dataTime);
    timeMap[nonceOut] = time;
    scheduleList.push_back(nonceOut);
    // Consume the inputs.
    valueMap.erase(data);
    valueMap.erase(address);
  }

  for (unsigned i = 0; i < ldCount(); i++) {
    mlir::Value address = op->getOperand(opIndex++);
    mlir::Value dataOut = op->getResult(i);
    mlir::Value nonceOut = op->getResult(ldCount() + stCount() + i);
    if (!valueMap.count(address)) {
      notReady = true;
      continue;
    }
    auto addressValue = valueMap[address];
    auto addressTime = timeMap[address];
    assert(buffer < store.size());
    auto &ref = store[buffer];
    unsigned offset = llvm::any_cast<APInt>(addressValue).getZExtValue();
    assert(offset < ref.size());

    valueMap[dataOut] = ref[offset];
    timeMap[dataOut] = addressTime;
    // Implicit none argument
    APInt apnonearg(1, 0);
    valueMap[nonceOut] = apnonearg;
    timeMap[nonceOut] = addressTime;
    scheduleList.push_back(dataOut);
    scheduleList.push_back(nonceOut);
    // Consume the inputs.
    valueMap.erase(address);
  }
  return (notReady) ? false : true;
}

bool LoadOp::tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                        llvm::DenseMap<unsigned, unsigned> & /*memoryMap*/,
                        llvm::DenseMap<mlir::Value, double> &timeMap,
                        std::vector<std::vector<llvm::Any>> & /*store*/,
                        std::vector<mlir::Value> &scheduleList) {
  auto op = getOperation();
  mlir::Value address = op->getOperand(0);
  mlir::Value data = op->getOperand(1);
  mlir::Value nonce = op->getOperand(2);
  mlir::Value addressOut = op->getResult(1);
  mlir::Value dataOut = op->getResult(0);
  if ((valueMap.count(address) && !valueMap.count(nonce)) ||
      (!valueMap.count(address) && valueMap.count(nonce)) ||
      (!valueMap.count(address) && !valueMap.count(nonce) &&
       !valueMap.count(data)))
    return false;
  if (valueMap.count(address) && valueMap.count(nonce)) {
    auto addressValue = valueMap[address];
    auto addressTime = timeMap[address];
    auto nonceValue = valueMap[nonce];
    auto nonceTime = timeMap[nonce];
    valueMap[addressOut] = addressValue;
    double time = std::max(addressTime, nonceTime);
    timeMap[addressOut] = time;
    scheduleList.push_back(addressOut);
    // Consume the inputs.
    valueMap.erase(address);
    valueMap.erase(nonce);
  } else if (valueMap.count(data)) {
    auto dataValue = valueMap[data];
    auto dataTime = timeMap[data];
    valueMap[dataOut] = dataValue;
    timeMap[dataOut] = dataTime;
    scheduleList.push_back(dataOut);
    // Consume the inputs.
    valueMap.erase(data);
  } else {
    llvm_unreachable("why?");
  }
  return true;
}

bool StoreOp::tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                         llvm::DenseMap<unsigned, unsigned> & /*memoryMap*/,
                         llvm::DenseMap<mlir::Value, double> &timeMap,
                         std::vector<std::vector<llvm::Any>> & /*store*/,
                         std::vector<mlir::Value> &scheduleList) {
  return tryToExecute(getOperation(), valueMap, timeMap, scheduleList, 1);
}

void JoinOp::execute(std::vector<llvm::Any> &ins,
                     std::vector<llvm::Any> &outs) {
  outs[0] = ins[0];
}

bool JoinOp::tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                        llvm::DenseMap<unsigned, unsigned> & /*memoryMap*/,
                        llvm::DenseMap<mlir::Value, double> &timeMap,
                        std::vector<std::vector<llvm::Any>> & /*store*/,
                        std::vector<mlir::Value> &scheduleList) {
  return tryToExecute(getOperation(), valueMap, timeMap, scheduleList, 1);
}

void StoreOp::execute(std::vector<llvm::Any> &ins,
                      std::vector<llvm::Any> &outs) {
  // Forward the address and data to the memory op.
  outs[0] = ins[1];
  outs[1] = ins[0];
}

void ForkOp::execute(std::vector<llvm::Any> &ins,
                     std::vector<llvm::Any> &outs) {
  for (auto &out : outs)
    out = ins[0];
}

} // namespace handshake
} // namespace circt
