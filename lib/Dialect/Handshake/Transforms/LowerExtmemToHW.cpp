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

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "circt/Dialect/Handshake/HandshakeUtils.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

namespace circt {
namespace handshake {
#define GEN_PASS_DEF_HANDSHAKELOWEREXTMEMTOHW
#include "circt/Dialect/Handshake/HandshakePasses.h.inc"
} // namespace handshake
} // namespace circt

using namespace circt;
using namespace handshake;
using namespace mlir;
namespace {
using NamedType = std::pair<StringAttr, Type>;
struct HandshakeMemType {
  llvm::SmallVector<NamedType> inputTypes, outputTypes;
  MemRefType memRefType;
  unsigned loadPorts, storePorts;
};

struct LoadName {
  StringAttr dataIn;
  StringAttr addrOut;

  static LoadName get(MLIRContext *ctx, unsigned idx) {
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

static Type indexToMemAddr(Type t, MemRefType memRef) {
  assert(isa<IndexType>(t) && "Expected index type");
  auto shape = memRef.getShape();
  assert(shape.size() == 1 && "Expected 1D memref");
  unsigned addrWidth = llvm::Log2_64_Ceil(shape[0]);
  return IntegerType::get(t.getContext(), addrWidth);
}

static HandshakeMemType getMemTypeForExtmem(Value v) {
  auto *ctx = v.getContext();
  assert(isa<mlir::MemRefType>(v.getType()) && "Value is not a memref type");
  auto extmemOp = cast<handshake::ExternalMemoryOp>(*v.getUsers().begin());
  HandshakeMemType memType;
  llvm::SmallVector<hw::detail::FieldInfo> inFields, outFields;

  // Add memory type.
  memType.memRefType = cast<MemRefType>(v.getType());
  memType.loadPorts = extmemOp.getLdCount();
  memType.storePorts = extmemOp.getStCount();

  // Add load ports.
  for (auto [i, ldif] : llvm::enumerate(extmemOp.getLoadPorts())) {
    auto names = LoadName::get(ctx, i);
    memType.inputTypes.push_back({names.dataIn, ldif.dataOut.getType()});
    memType.outputTypes.push_back(
        {names.addrOut,
         indexToMemAddr(ldif.addressIn.getType(), memType.memRefType)});
  }

  // Add store ports.
  for (auto [i, stif] : llvm::enumerate(extmemOp.getStorePorts())) {
    auto names = StoreNames::get(ctx, i);

    // Incoming store data and address
    llvm::SmallVector<hw::StructType::FieldInfo> storeOutFields;
    storeOutFields.push_back(
        {StringAttr::get(ctx, "address"),
         indexToMemAddr(stif.addressIn.getType(), memType.memRefType)});
    storeOutFields.push_back(
        {StringAttr::get(ctx, "data"), stif.dataIn.getType()});
    auto inType = hw::StructType::get(ctx, storeOutFields);
    memType.outputTypes.push_back({names.out, inType});
    memType.inputTypes.push_back({names.doneIn, stif.doneOut.getType()});
  }

  return memType;
}

namespace {
struct HandshakeLowerExtmemToHWPass
    : public circt::handshake::impl::HandshakeLowerExtmemToHWBase<
          HandshakeLowerExtmemToHWPass> {

  HandshakeLowerExtmemToHWPass(std::optional<bool> createESIWrapper) {
    if (createESIWrapper)
      this->createESIWrapper = *createESIWrapper;
  }

  void runOnOperation() override {
    auto op = getOperation();
    for (auto func : op.getOps<handshake::FuncOp>()) {
      if (failed(lowerExtmemToHW(func))) {
        signalPassFailure();
        return;
      }
    }
  };

  LogicalResult lowerExtmemToHW(handshake::FuncOp func);
  LogicalResult
  wrapESI(handshake::FuncOp func, hw::ModulePortInfo origPorts,
          const std::map<unsigned, HandshakeMemType> &argReplacements);
};

LogicalResult HandshakeLowerExtmemToHWPass::wrapESI(
    handshake::FuncOp func, hw::ModulePortInfo origPorts,
    const std::map<unsigned, HandshakeMemType> &argReplacements) {
  auto *ctx = func.getContext();
  OpBuilder b(func);
  auto loc = func.getLoc();

  // Create external module which will match the interface of 'func' after it's
  // been lowered to HW.
  b.setInsertionPoint(func);
  auto newPortInfo = handshake::getPortInfoForOpTypes(
      func, func.getArgumentTypes(), func.getResultTypes());
  auto extMod = hw::HWModuleExternOp::create(
      b, loc, StringAttr::get(ctx, "__" + func.getName() + "_hw"), newPortInfo);

  // Add an attribute to the original handshake function to indicate that it
  // needs to resolve to extMod in a later pass.
  func->setAttr(kPredeclarationAttr,
                FlatSymbolRefAttr::get(ctx, extMod.getName()));

  // Create wrapper module. This will have the same ports as the original
  // module, sans the replaced arguments.
  auto wrapperModPortInfo = origPorts;
  llvm::SmallVector<unsigned> argReplacementsIdxs;
  llvm::transform(argReplacements, std::back_inserter(argReplacementsIdxs),
                  [](auto &pair) { return pair.first; });
  for (auto i : llvm::reverse(argReplacementsIdxs))
    wrapperModPortInfo.eraseInput(i);
  auto wrapperMod = hw::HWModuleOp::create(
      b, loc, StringAttr::get(ctx, func.getName() + "_esi_wrapper"),
      wrapperModPortInfo);
  Value clk = wrapperMod.getBodyBlock()->getArgument(
      wrapperMod.getBodyBlock()->getNumArguments() - 2);
  Value rst = wrapperMod.getBodyBlock()->getArgument(
      wrapperMod.getBodyBlock()->getNumArguments() - 1);
  SmallVector<Value> clkRes = {clk, rst};

  b.setInsertionPointToStart(wrapperMod.getBodyBlock());
  BackedgeBuilder bb(b, loc);

  // Create backedges for the results of the external module. These will be
  // replaced by the service instance requests if associated with a memory.
  llvm::SmallVector<Backedge> backedges;
  for (auto resType : extMod.getOutputTypes())
    backedges.push_back(bb.get(resType));

  // Maintain which index we're currently at in the lowered handshake module's
  // return.
  unsigned resIdx = origPorts.sizeOutputs();

  // Maintain the arguments which each memory will add to the inner module
  // instance.
  llvm::SmallVector<llvm::SmallVector<Value>> instanceArgsForMem;

  for (auto [i, memType] : argReplacements) {

    b.setInsertionPoint(wrapperMod);
    // Create a memory service declaration for each memref argument that was
    // served.
    auto origPortInfo = origPorts.atInput(i);
    auto memrefShape = memType.memRefType.getShape();
    auto dataType = memType.memRefType.getElementType();
    assert(memrefShape.size() == 1 && "Only 1D memrefs are supported");
    unsigned memrefSize = memrefShape[0];
    auto memServiceDecl = esi::RandomAccessMemoryDeclOp::create(
        b, loc, origPortInfo.name, TypeAttr::get(dataType),
        b.getI64IntegerAttr(memrefSize));
    esi::ServicePortInfo writePortInfo = memServiceDecl.writePortInfo();
    esi::ServicePortInfo readPortInfo = memServiceDecl.readPortInfo();

    SmallVector<Value> instanceArgsFromThisMem;

    // Create service requests. This MUST follow the order of which ports were
    // added in other parts of this pass (load ports first, then store ports).
    b.setInsertionPointToStart(wrapperMod.getBodyBlock());

    // Load ports:
    for (unsigned i = 0; i < memType.loadPorts; ++i) {
      auto req = esi::RequestConnectionOp::create(
          b, loc, readPortInfo.type, readPortInfo.port,
          esi::AppIDAttr::get(ctx, b.getStringAttr("load"), resIdx));
      auto reqUnpack = esi::UnpackBundleOp::create(
          b, loc, req.getToClient(), ValueRange{backedges[resIdx]});
      instanceArgsFromThisMem.push_back(
          reqUnpack.getToChannels()
              [esi::RandomAccessMemoryDeclOp::RespDirChannelIdx]);
      ++resIdx;
    }

    // Store ports:
    for (unsigned i = 0; i < memType.storePorts; ++i) {
      auto req = esi::RequestConnectionOp::create(
          b, loc, writePortInfo.type, writePortInfo.port,
          esi::AppIDAttr::get(ctx, b.getStringAttr("store"), resIdx));
      auto reqUnpack = esi::UnpackBundleOp::create(
          b, loc, req.getToClient(), ValueRange{backedges[resIdx]});
      instanceArgsFromThisMem.push_back(
          reqUnpack.getToChannels()
              [esi::RandomAccessMemoryDeclOp::RespDirChannelIdx]);
      ++resIdx;
    }

    instanceArgsForMem.emplace_back(std::move(instanceArgsFromThisMem));
  }

  // Stitch together arguments from the top-level ESI wrapper and the instance
  // arguments generated from the service requests.
  llvm::SmallVector<Value> instanceArgs;

  // Iterate over the arguments of the original handshake.func and determine
  // whether to grab operands from the arg replacements or the wrapper module.
  unsigned wrapperArgIdx = 0;

  for (unsigned i = 0, e = func.getNumArguments(); i < e; i++) {
    // Arg replacement indices refer to the original handshake.func argument
    // index.
    if (argReplacements.count(i)) {
      // This index was originally a memref - pop the instance arguments for the
      // next-in-line memory and add them.
      auto &memArgs = instanceArgsForMem.front();
      instanceArgs.append(memArgs.begin(), memArgs.end());
      instanceArgsForMem.erase(instanceArgsForMem.begin());
    } else {
      // Add the argument from the wrapper mod. This is maintained by its own
      // counter (memref arguments are removed, so if there was an argument at
      // this point, it needs to come from the wrapper module).
      instanceArgs.push_back(
          wrapperMod.getBodyBlock()->getArgument(wrapperArgIdx++));
    }
  }

  // Add any missing arguments from the wrapper module (this will be clock and
  // reset)
  for (; wrapperArgIdx < wrapperMod.getBodyBlock()->getNumArguments();
       ++wrapperArgIdx)
    instanceArgs.push_back(
        wrapperMod.getBodyBlock()->getArgument(wrapperArgIdx));

  // Instantiate the inner module.
  auto instance =
      hw::InstanceOp::create(b, loc, extMod, func.getName(), instanceArgs);

  // And resolve the backedges.
  for (auto [res, be] : llvm::zip(instance.getResults(), backedges))
    be.setValue(res);

  // Finally, grab the (non-memory) outputs from the inner module and return
  // them through the wrapper.
  auto outputOp =
      cast<hw::OutputOp>(wrapperMod.getBodyBlock()->getTerminator());
  b.setInsertionPoint(outputOp);
  hw::OutputOp::create(
      b, outputOp.getLoc(),
      instance.getResults().take_front(wrapperMod.getNumOutputPorts()));
  outputOp.erase();

  return success();
}

// Truncates the index-typed 'v' into an integer-type of the same width as the
// 'memref' argument.
// Uses arith operations since these are supported in the HandshakeToHW
// lowering.
static Value truncateToMemoryWidth(Location loc, OpBuilder &b, Value v,
                                   MemRefType memRefType) {
  assert(isa<IndexType>(v.getType()) && "Expected an index-typed value");
  auto addrWidth = llvm::Log2_64_Ceil(memRefType.getShape().front());
  return arith::IndexCastOp::create(b, loc, b.getIntegerType(addrWidth), v);
}

static Value plumbLoadPort(Location loc, OpBuilder &b,
                           handshake::MemLoadInterface &ldif, Value loadData,
                           MemRefType memrefType) {
  // We need to feed both the load data and the load done outputs.
  // Fork the extracted load data into two, and 'join' the second one to
  // generate a none-typed output to drive the load done.
  auto dataFork = ForkOp::create(b, loc, loadData, 2);

  auto dataOut = dataFork.getResult()[0];
  llvm::SmallVector<Value> joinArgs = {dataFork.getResult()[1]};
  auto dataDone = JoinOp::create(b, loc, joinArgs);

  ldif.dataOut.replaceAllUsesWith(dataOut);
  ldif.doneOut.replaceAllUsesWith(dataDone);

  // Return load address, to be fed to the top-level output, truncated to the
  // width of the memory that is accessed.
  return truncateToMemoryWidth(loc, b, ldif.addressIn, memrefType);
}

static Value plumbStorePort(Location loc, OpBuilder &b,
                            handshake::MemStoreInterface &stif, Value done,
                            Type outType, MemRefType memrefType) {
  stif.doneOut.replaceAllUsesWith(done);
  // Return the store address and data to be fed to the top-level output.
  // Address is truncated to the width of the memory that is accessed.
  llvm::SmallVector<Value> structArgs = {
      truncateToMemoryWidth(loc, b, stif.addressIn, memrefType), stif.dataIn};

  return hw::StructCreateOp::create(b, loc, cast<hw::StructType>(outType),
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

struct ArgTypeReplacement {
  unsigned index;
  TypeRange ins;
  TypeRange outs;
};

LogicalResult
HandshakeLowerExtmemToHWPass::lowerExtmemToHW(handshake::FuncOp func) {
  // Gather memref ports to be converted. This is an ordered map, and will be
  // iterated from lo to hi indices.
  std::map<unsigned, Value> memrefArgs;
  for (auto [i, arg] : llvm::enumerate(func.getArguments()))
    if (isa<MemRefType>(arg.getType()))
      memrefArgs[i] = arg;

  if (memrefArgs.empty())
    return success(); // nothing to do.

  // Record which arg indices were replaces with handshake memory ports.
  // This is an ordered map, and will be iterated from lo to hi indices.
  std::map<unsigned, HandshakeMemType> argReplacements;

  // Record the hw.module i/o of the original func (used for ESI wrapper).
  auto origPortInfo = handshake::getPortInfoForOpTypes(
      func, func.getArgumentTypes(), func.getResultTypes());

  OpBuilder b(func);
  for (auto it : memrefArgs) {
    // Do not use structured bindings for 'it' - cannot reference inside lambda.
    unsigned i = it.first;
    auto arg = it.second;
    auto loc = arg.getLoc();
    // Get the attached extmemory external module.
    auto extmemOp = cast<handshake::ExternalMemoryOp>(*arg.getUsers().begin());
    b.setInsertionPoint(extmemOp);

    // Add memory input - this is the output of the extmemory op.
    auto memIOTypes = getMemTypeForExtmem(arg);
    MemRefType memrefType = cast<MemRefType>(arg.getType());

    auto oldReturnOp =
        cast<handshake::ReturnOp>(func.getBody().front().getTerminator());
    llvm::SmallVector<Value> newReturnOperands = oldReturnOp.getOperands();
    unsigned addedInPorts = 0;
    auto memName = func.getArgName(i);
    auto addArgRes = [&](unsigned id, NamedType &argType,
                         NamedType &resType) -> FailureOr<Value> {
      // Function argument
      unsigned newArgIdx = i + addedInPorts;
      if (failed(
              func.insertArgument(newArgIdx, argType.second, {}, arg.getLoc())))
        return failure();
      insertInStringArrayAttr(func, "argNames",
                              memName.str() + "_" + argType.first.str(),
                              newArgIdx);
      auto newInPort = func.getArgument(newArgIdx);
      ++addedInPorts;

      // Function result.
      if (failed(func.insertResult(func.getNumResults(), resType.second, {})))
        return failure();
      appendToStringArrayAttr(func, "resNames",
                              memName.str() + "_" + resType.first.str());
      return newInPort;
    };

    // Plumb load ports.
    unsigned portIdx = 0;
    for (auto loadPort : extmemOp.getLoadPorts()) {
      auto newInPort = addArgRes(loadPort.index, memIOTypes.inputTypes[portIdx],
                                 memIOTypes.outputTypes[portIdx]);
      if (failed(newInPort))
        return failure();
      newReturnOperands.push_back(
          plumbLoadPort(loc, b, loadPort, *newInPort, memrefType));
      ++portIdx;
    }

    // Plumb store ports.
    for (auto storePort : extmemOp.getStorePorts()) {
      auto newInPort =
          addArgRes(storePort.index, memIOTypes.inputTypes[portIdx],
                    memIOTypes.outputTypes[portIdx]);
      if (failed(newInPort))
        return failure();
      newReturnOperands.push_back(
          plumbStorePort(loc, b, storePort, *newInPort,
                         memIOTypes.outputTypes[portIdx].second, memrefType));
      ++portIdx;
    }

    // Replace the return op of the function with a new one that returns the
    // memory output struct.
    b.setInsertionPoint(oldReturnOp);
    ReturnOp::create(b, arg.getLoc(), newReturnOperands);
    oldReturnOp.erase();

    // Erase the extmemory operation since I/O plumbing has replaced all of its
    // results.
    extmemOp.erase();

    // Erase the original memref argument of the top-level i/o now that it's
    // use has been removed.
    if (failed(func.eraseArgument(i + addedInPorts)))
      return failure();
    eraseFromArrayAttr(func, "argNames", i + addedInPorts);

    argReplacements[i] = memIOTypes;
  }

  if (createESIWrapper)
    if (failed(wrapESI(func, origPortInfo, argReplacements)))
      return failure();

  return success();
}

} // namespace

std::unique_ptr<mlir::Pass>
circt::handshake::createHandshakeLowerExtmemToHWPass(
    std::optional<bool> createESIWrapper) {
  return std::make_unique<HandshakeLowerExtmemToHWPass>(createESIWrapper);
}
