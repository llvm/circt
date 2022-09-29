//===- LowerExtmemToHW.cpp - lock functions pass ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the lower extmem pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace handshake;
using namespace mlir;
namespace {
using NamedType = std::pair<StringAttr, Type>;
struct HandshakeMemType {
  llvm::SmallVector<NamedType> inputTypes, outputTypes;
};

struct LoadNames {
  StringAttr dataIn;
  StringAttr addrOut;

  static LoadNames get(MLIRContext *ctx, unsigned idx) {
    return {StringAttr::get(ctx, "ld" + std::to_string(idx) + ".data"),
            StringAttr::get(ctx, "ld" + std::to_string(idx) + ".addr")};
  }
};

struct StoreNames {
  StringAttr doneIn;
  StringAttr out;

  static StoreNames get(MLIRContext *ctx, unsigned idx) {
    return {StringAttr::get(ctx, "st" + std::to_string(idx) + ".done"),
            StringAttr::get(ctx, "st" + std::to_string(idx))};
  }
};

} // namespace

static HandshakeMemType getMemTypeForExtmem(Value v) {
  auto *ctx = v.getContext();
  assert(v.getType().isa<mlir::MemRefType>() && "Value is not a memref type");
  auto extmemOp = cast<handshake::ExternalMemoryOp>(*v.getUsers().begin());
  HandshakeMemType memType;
  llvm::SmallVector<hw::detail::FieldInfo> inFields, outFields;

  // Add load ports.
  for (auto [i, ldif] : llvm::enumerate(extmemOp.getLoadPorts())) {
    auto names = LoadNames::get(ctx, i);
    memType.inputTypes.push_back({names.dataIn, ldif.dataOut.getType()});
    memType.outputTypes.push_back({names.addrOut, ldif.addressIn.getType()});
  }

  // Add store ports.
  for (auto [i, stif] : llvm::enumerate(extmemOp.getStorePorts())) {
    auto names = StoreNames::get(ctx, i);

    // Incoming store data and address
    llvm::SmallVector<hw::StructType::FieldInfo> storeOutFields;
    storeOutFields.push_back(
        {StringAttr::get(ctx, "data"), stif.dataIn.getType()});
    storeOutFields.push_back(
        {StringAttr::get(ctx, "addr"), stif.addressIn.getType()});
    auto inType = hw::StructType::get(ctx, storeOutFields);
    memType.outputTypes.push_back({names.out, inType});
    memType.inputTypes.push_back({names.doneIn, stif.doneOut.getType()});
  }
  return memType;
}

namespace {

struct HandshakeLowerExtmemToHWPass
    : public HandshakeLowerExtmemToHWBase<HandshakeLowerExtmemToHWPass> {
  void runOnOperation() override {
    auto op = getOperation();
    for (auto func : op.getOps<handshake::FuncOp>()) {
      if (failed(LowerExtmemToHW(func))) {
        signalPassFailure();
        return;
      }
    }
  };

  LogicalResult LowerExtmemToHW(handshake::FuncOp func);
};

static Value plumbLoadPort(Location loc, OpBuilder &b,
                           const handshake::ExtMemLoadInterface &ldif,
                           Value loadData) {
  // We need to feed both the load data and the load done outputs.
  // Fork the extracted load data into two, and 'join' the second one to
  // generate a none-typed output to drive the load done.
  auto dataFork = b.create<ForkOp>(loc, loadData, 2);

  auto dataOut = dataFork.getResult()[0];
  llvm::SmallVector<Value> joinArgs = {dataFork.getResult()[1]};
  auto dataDone = b.create<JoinOp>(loc, joinArgs);

  ldif.dataOut.replaceAllUsesWith(dataOut);
  ldif.doneOut.replaceAllUsesWith(dataDone);

  // Return load address, to be fed to the top-level output.
  return ldif.addressIn;
}

static Value plumbStorePort(Location loc, OpBuilder &b,
                            const handshake::ExtMemStoreInterface &stif,
                            Value done, Type outType) {
  stif.doneOut.replaceAllUsesWith(done);
  // Return the store and data to be fed to the top-level output.
  llvm::SmallVector<Value> structArgs = {stif.dataIn, stif.addressIn};
  return b
      .create<hw::StructCreateOp>(loc, outType.cast<hw::StructType>(),
                                  structArgs)
      .getResult();
}

static void appendToStringArrayAttr(Operation *op, StringRef attrName,
                                    StringRef attrVal) {
  auto *ctx = op->getContext();
  llvm::SmallVector<Attribute> newArr;
  llvm::copy(op->getAttrOfType<ArrayAttr>(attrName).getValue(),
             std::back_inserter(newArr));
  newArr.push_back(StringAttr::get(ctx, attrVal));
  op->setAttr(attrName, ArrayAttr::get(ctx, newArr));
}

static void insertInStringArrayAttr(Operation *op, StringRef attrName,
                                    StringRef attrVal, unsigned idx) {
  auto *ctx = op->getContext();
  llvm::SmallVector<Attribute> newArr;
  llvm::copy(op->getAttrOfType<ArrayAttr>(attrName).getValue(),
             std::back_inserter(newArr));
  newArr.insert(newArr.begin() + idx, StringAttr::get(ctx, attrVal));
  op->setAttr(attrName, ArrayAttr::get(ctx, newArr));
}

static void eraseFromArrayAttr(Operation *op, StringRef attrName,
                               unsigned idx) {
  auto *ctx = op->getContext();
  llvm::SmallVector<Attribute> newArr;
  llvm::copy(op->getAttrOfType<ArrayAttr>(attrName).getValue(),
             std::back_inserter(newArr));
  newArr.erase(newArr.begin() + idx);
  op->setAttr(attrName, ArrayAttr::get(ctx, newArr));
}

LogicalResult
HandshakeLowerExtmemToHWPass::LowerExtmemToHW(handshake::FuncOp func) {
  // Gather memref ports to be converted.
  llvm::DenseMap<unsigned, Value> memrefArgs;
  for (auto [i, arg] : llvm::enumerate(func.getArguments())) {
    if (arg.getType().isa<MemRefType>())
      memrefArgs[i] = arg;
  }

  if (memrefArgs.empty())
    return success(); // nothing to do.

  OpBuilder b(func);
  for (auto it : memrefArgs) {
    // Do not use structured bindings for 'it' - cannot reference inside lambda.
    unsigned i = it.first;
    auto arg = it.second;
    auto loc = arg.getLoc();
    // Get the attached extmemory external module.
    auto extmemOp = cast<handshake::ExternalMemoryOp>(*arg.getUsers().begin());
    OpBuilder b(extmemOp);
    b.setInsertionPoint(extmemOp);

    // Add memory input - this is the output of the extmemory op.
    auto memIOTypes = getMemTypeForExtmem(arg);

    auto oldReturnOp =
        cast<handshake::ReturnOp>(func.getBody().front().getTerminator());
    llvm::SmallVector<Value> newReturnOperands = oldReturnOp.getOperands();
    unsigned addedInPorts = 0;
    auto memName = func.getArgName(i);
    auto addArgRes = [&](unsigned id, NamedType &argType, NamedType &resType) {
      // Function argument
      unsigned newArgIdx = i + addedInPorts;
      func.insertArgument(newArgIdx, argType.second, {}, arg.getLoc());
      insertInStringArrayAttr(func, "argNames",
                              memName.str() + "_" + argType.first.str(),
                              newArgIdx);
      auto newInPort = func.getArgument(newArgIdx);
      ++addedInPorts;

      // Function result.
      func.insertResult(func.getNumResults(), resType.second, {});
      appendToStringArrayAttr(func, "resNames",
                              memName.str() + "_" + resType.first.str());
      return newInPort;
    };

    // Plumb load ports.
    unsigned portIdx = 0;
    for (auto loadPort : extmemOp.getLoadPorts()) {
      auto newInPort = addArgRes(loadPort.index, memIOTypes.inputTypes[portIdx],
                                 memIOTypes.outputTypes[portIdx]);
      newReturnOperands.push_back(plumbLoadPort(loc, b, loadPort, newInPort));
      ++portIdx;
    }

    // Plumb store ports.
    for (auto storePort : extmemOp.getStorePorts()) {
      auto newInPort =
          addArgRes(storePort.index, memIOTypes.inputTypes[portIdx],
                    memIOTypes.outputTypes[portIdx]);
      newReturnOperands.push_back(
          plumbStorePort(loc, b, storePort, newInPort,
                         memIOTypes.outputTypes[portIdx].second));
      ++portIdx;
    }

    // Replace the return op of the function with a new one that returns the
    // memory output struct.
    b.setInsertionPoint(oldReturnOp);
    b.create<ReturnOp>(arg.getLoc(), newReturnOperands);
    oldReturnOp.erase();

    // Erase the extmemory operation since I/O plumbing has replaced all of its
    // results.
    extmemOp.erase();

    // Erase the original memref argument of the top-level i/o now that it's
    // use has been removed.
    func.eraseArgument(i + addedInPorts);
    eraseFromArrayAttr(func, "argNames", i + addedInPorts);
  }

  return success();
}

} // namespace

std::unique_ptr<mlir::Pass>
circt::handshake::createHandshakeLowerExtmemToHWPass() {
  return std::make_unique<HandshakeLowerExtmemToHWPass>();
}
