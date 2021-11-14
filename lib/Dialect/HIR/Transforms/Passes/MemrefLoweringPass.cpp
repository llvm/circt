//=========- MemrefLoweringPass.cpp - Lower memref type---===//
//
// This file implements lowering pass for memref type.
//
//===----------------------------------------------------------------------===//

#include "MemrefLoweringUtils.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/DenseSet.h"
#include <iostream>
using namespace circt;
using namespace hir;
namespace {

/// This class lowers a hir::MemrefType to multiple buses.
/// For each addr, read and write, there are two buses - data_bus and en_bus.
class MemrefLoweringPass : public hir::MemrefLoweringBase<MemrefLoweringPass> {
public:
  void runOnOperation() override;

private:
  void insertBusArguments(hir::FuncLike);
  void removeMemrefArguments(hir::FuncLike);
  LogicalResult visitOp(hir::FuncOp);
  LogicalResult visitOp(hir::FuncExternOp);
  LogicalResult visitOp(hir::AllocaOp);
  LogicalResult visitOp(hir::LoadOp);
  LogicalResult visitOp(hir::StoreOp);
  LogicalResult visitOp(hir::CallOp);
  LogicalResult visitOp(hir::MemrefExtractOp);
  MemrefPortInterface defineBusesForMemrefPort(OpBuilder &, hir::MemrefType,
                                               Attribute,
                                               llvm::Optional<StringRef> name);
  LogicalResult createBusInstantiationsAndCallOp(hir::AllocaOp);
  void initUnConnectedPorts(hir::FuncOp);

private:
  MemrefInfo memrefInfo;
  SmallVector<Operation *, 10> opsToErase;
  OpBuilder *topLevelBuilder;
};
} // end anonymous namespace

//------------------------------------------------------------------------------
// Helper functions.
//------------------------------------------------------------------------------

static mlir::FlatSymbolRefAttr createUniqueFuncName(MLIRContext *context,
                                                    StringRef name,
                                                    ArrayRef<int64_t> params) {
  std::string newName(name);
  newName += "_";
  for (size_t i = 0; i < params.size(); i++) {
    if (i > 0)
      newName += "x";
    newName += std::to_string(params[i]);
  }
  return FlatSymbolRefAttr::get(context, newName);
}

Value createLinearAddr(OpBuilder &builder, Location loc,
                       ArrayRef<Value> indices) {
  if (indices.size() == 0)
    return Value();

  SmallVector<Type, 4> idxTypes;
  for (auto idx : indices)
    idxTypes.push_back(idx.getType());

  return builder.create<comb::ConcatOp>(loc, indices).getResult();
}

MemrefPortInterface MemrefLoweringPass::defineBusesForMemrefPort(
    OpBuilder &builder, hir::MemrefType memrefTy, Attribute port,
    llvm::Optional<StringRef> name) {
  MemrefPortInterface portInterface;

  auto *context = memrefTy.getContext();
  Type enableTy = hir::BusTensorType::get(context, memrefTy.getNumBanks(),
                                          builder.getI1Type());

  size_t addrWidth = 0;
  for (int64_t dimSize : memrefTy.filterShape(ADDR)) {
    addrWidth += helper::clog2(dimSize);
  }

  Type addrTy = hir::BusTensorType::get(context, memrefTy.getNumBanks(),
                                        builder.getIntegerType(addrWidth));
  Type dataTy = hir::BusTensorType::get(context, memrefTy.getNumBanks(),
                                        memrefTy.getElementType());

  if (memrefTy.getNumElementsPerBank() > 1) {
    portInterface.addrEnableBusTensor =
        topLevelBuilder->create<hir::BusTensorOp>(builder.getUnknownLoc(),
                                                  enableTy);
    portInterface.addrDataBusTensor = topLevelBuilder->create<hir::BusTensorOp>(
        builder.getUnknownLoc(), addrTy);
    if (name) {
      helper::setNames(portInterface.addrEnableBusTensor.getDefiningOp(),
                       {name.getValue().str() + "_addr_en"});
      helper::setNames(portInterface.addrDataBusTensor.getDefiningOp(),
                       {name.getValue().str() + "_addr_data"});
    }
  }

  if (auto rdLatency = helper::getRdLatency(port)) {
    portInterface.rdEnableBusTensor = topLevelBuilder->create<hir::BusTensorOp>(
        builder.getUnknownLoc(), enableTy);
    portInterface.rdDataBusTensor = topLevelBuilder->create<hir::BusTensorOp>(
        builder.getUnknownLoc(), dataTy);
    portInterface.rdLatency = rdLatency.getValue();
    if (name) {
      helper::setNames(portInterface.rdEnableBusTensor.getDefiningOp(),
                       {name.getValue().str() + "_rd_en"});
      helper::setNames(portInterface.rdDataBusTensor.getDefiningOp(),
                       {name.getValue().str() + "_rd_data"});
    }
  }

  if (helper::isWrite(port)) {
    portInterface.wrEnableBusTensor = topLevelBuilder->create<hir::BusTensorOp>(
        builder.getUnknownLoc(), enableTy);
    portInterface.wrDataBusTensor = topLevelBuilder->create<hir::BusTensorOp>(
        builder.getUnknownLoc(), dataTy);
    if (name) {
      helper::setNames(portInterface.wrEnableBusTensor.getDefiningOp(),
                       {name.getValue().str() + "_wr_en"});
      helper::setNames(portInterface.wrDataBusTensor.getDefiningOp(),
                       {name.getValue().str() + "_wr_data"});
    }
  }
  return portInterface;
}

LogicalResult
MemrefLoweringPass::createBusInstantiationsAndCallOp(hir::AllocaOp op) {
  Value tstartRegion = op->getParentRegion()->getArguments().back();
  hir::MemrefType memrefTy = op.getType().dyn_cast<hir::MemrefType>();
  ArrayAttr ports = op.ports();
  OpBuilder builder(op);
  builder.setInsertionPoint(op);
  SmallVector<MemrefPortInterface> memrefPortInterfaces;

  for (auto port : ports) {
    auto portInterface = defineBusesForMemrefPort(
        builder, memrefTy, port, helper::getOptionalName(op, 0));
    memrefPortInterfaces.push_back(portInterface);
  }

  SmallVector<Value> inputBuses;
  SmallVector<StringRef> inputBusNames;
  SmallVector<DictionaryAttr> inputBusAttrs;
  SmallVector<Type> inputBusTypes;
  auto sendAttr = helper::getDictionaryAttr(builder, "hir.bus.ports",
                                            builder.getStrArrayAttr({"send"}));
  auto recvAttr = helper::getDictionaryAttr(builder, "hir.bus.ports",
                                            builder.getStrArrayAttr({"recv"}));
  for (size_t portNum = 0; portNum < memrefPortInterfaces.size(); portNum++) {
    std::string portPrefix = "p" + std::to_string(portNum) + "_";
    auto portInterface = memrefPortInterfaces[portNum];
    if (portInterface.addrEnableBusTensor) {
      inputBuses.push_back(portInterface.addrEnableBusTensor);
      inputBusTypes.push_back(portInterface.addrEnableBusTensor.getType());
      inputBusAttrs.push_back(recvAttr);
      inputBusNames.push_back(portPrefix + "addr_en");
      inputBuses.push_back(portInterface.addrDataBusTensor);
      inputBusTypes.push_back(portInterface.addrDataBusTensor.getType());
      inputBusAttrs.push_back(recvAttr);
      inputBusNames.push_back(portPrefix + "addr_data");
    }

    if (portInterface.rdEnableBusTensor) {
      inputBuses.push_back(portInterface.rdEnableBusTensor);
      inputBusTypes.push_back(portInterface.rdEnableBusTensor.getType());
      inputBusAttrs.push_back(recvAttr);
      inputBusNames.push_back(portPrefix + "rd_en");
      inputBuses.push_back(portInterface.rdDataBusTensor);
      inputBusTypes.push_back(portInterface.rdDataBusTensor.getType());
      inputBusAttrs.push_back(sendAttr);
      inputBusNames.push_back(portPrefix + "rd_data");
    }

    if (portInterface.wrEnableBusTensor) {
      inputBuses.push_back(portInterface.wrEnableBusTensor);
      inputBusTypes.push_back(portInterface.wrEnableBusTensor.getType());
      inputBusAttrs.push_back(recvAttr);
      inputBusNames.push_back(portPrefix + "wr_en");
      inputBuses.push_back(portInterface.wrDataBusTensor);
      inputBusTypes.push_back(portInterface.wrDataBusTensor.getType());
      inputBusAttrs.push_back(recvAttr);
      inputBusNames.push_back(portPrefix + "wr_data");
    }
  }
  Type funcTy = hir::FuncType::get(builder.getContext(), inputBusTypes,
                                   inputBusAttrs, {}, {});

  auto elementWidth = helper::getBitWidth(memrefTy.getElementType()).getValue();
  auto addrWidth = helper::clog2(memrefTy.getNumElementsPerBank());
  auto tensorSize = memrefTy.getNumBanks();

  auto memModuleName = createUniqueFuncName(
      builder.getContext(), op.mem_type().str(),
      {memrefTy.getNumBanks(), memrefTy.getNumElementsPerBank(), elementWidth});
  auto instanceName = helper::getOptionalName(op, 0);
  auto instanceNameAttr = instanceName
                              ? builder.getStringAttr(instanceName.getValue())
                              : StringAttr();
  auto callOp = builder.create<hir::CallOp>(
      builder.getUnknownLoc(), SmallVector<Type>(), instanceNameAttr,
      memModuleName, TypeAttr::get(funcTy), inputBuses, tstartRegion,
      IntegerAttr());

  auto params = builder.getDictionaryAttr(
      {builder.getNamedAttr("ELEMENT_WIDTH",
                            builder.getI64IntegerAttr(elementWidth)),
       builder.getNamedAttr("ADDR_WIDTH", builder.getI64IntegerAttr(addrWidth)),
       builder.getNamedAttr("TENSOR_SIZE",
                            builder.getI64IntegerAttr(tensorSize))});

  callOp->setAttr("params", params);

  helper::declareExternalFuncForCall(callOp, inputBusNames);

  memrefInfo.map(op.res(), emitMemoryInterfacesForEachPortBank(
                               builder, memrefTy, memrefPortInterfaces));
  return success();
}

static void initUnconnectedMemoryInterface(OpBuilder &builder,
                                           MemoryInterface memoryInterface) {
  auto enTy = hir::BusType::get(builder.getContext(), builder.getI1Type());
  if (memoryInterface.hasAddrBus()) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(memoryInterface.getAddrEnBus().getDefiningOp());
    auto c0 = builder.create<hw::ConstantOp>(
        builder.getUnknownLoc(), IntegerAttr::get(builder.getI1Type(), 0));
    builder.getUnknownLoc(),
        memoryInterface.getAddrEnBus().replaceAllUsesWith(
            builder.create<hir::CastOp>(builder.getUnknownLoc(), enTy, c0));
  }
  if (memoryInterface.hasRdBus()) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(memoryInterface.getRdEnBus().getDefiningOp());
    auto c0 = builder.create<hw::ConstantOp>(
        builder.getUnknownLoc(), IntegerAttr::get(builder.getI1Type(), 0));
    memoryInterface.getRdEnBus().replaceAllUsesWith(
        builder.create<hir::CastOp>(builder.getUnknownLoc(), enTy, c0));
  }
  if (memoryInterface.hasWrBus()) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(memoryInterface.getWrEnBus().getDefiningOp());
    auto c0 = builder.create<hw::ConstantOp>(
        builder.getUnknownLoc(), IntegerAttr::get(builder.getI1Type(), 0));
    memoryInterface.getWrEnBus().replaceAllUsesWith(
        builder.create<hir::CastOp>(builder.getUnknownLoc(), enTy, c0));
  }
}

void MemrefLoweringPass::initUnConnectedPorts(hir::FuncOp op) {
  OpBuilder builder(op);
  auto *returnOperation = &op.body().front().back();
  builder.setInsertionPoint(returnOperation);
  // for (auto memoryInterface : memrefInfo.getAllMemoryInterfaces()) {
  //  initUnconnectedMemoryInterface(builder, memoryInterface);
  //}
}

SmallVector<Value> filterMemrefArgs(Block::BlockArgListType args) {
  SmallVector<Value> memrefArgs;
  for (auto arg : args)
    if (arg.getType().isa<hir::MemrefType>())
      memrefArgs.push_back(arg);
  return memrefArgs;
}

size_t insertBusTypesAndAttrsForMemrefPort(
    size_t loc, size_t port, Block &bb,
    SmallVectorImpl<std::string> &inputNames,
    SmallVectorImpl<DictionaryAttr> &inputAttrs, hir::MemrefType memrefTy,
    DictionaryAttr portDict, MemrefPortInterface &portInterface) {
  auto *context = memrefTy.getContext();
  Builder builder(context);
  Type enableTy = hir::BusTensorType::get(context, memrefTy.getNumBanks(),
                                          IntegerType::get(context, 1));

  size_t addrWidth = 0;
  for (int64_t dimSize : memrefTy.filterShape(ADDR)) {
    addrWidth += helper::clog2(dimSize);
  }

  Type addrTy = hir::BusTensorType::get(context, memrefTy.getNumBanks(),
                                        builder.getIntegerType(addrWidth));
  Type dataTy = hir::BusTensorType::get(context, memrefTy.getNumBanks(),
                                        memrefTy.getElementType());

  DictionaryAttr sendAttr = helper::getDictionaryAttr(
      context, "hir.bus.ports",
      ArrayAttr::get(context, StringAttr::get(context, "send")));
  DictionaryAttr recvAttr = helper::getDictionaryAttr(
      context, "hir.bus.ports",
      ArrayAttr::get(context, StringAttr::get(context, "recv")));

  std::string memName =
      inputNames[loc] + std::string("_p") + std::to_string(port);
  if (memrefTy.getNumElementsPerBank() > 1) {
    portInterface.addrEnableBusTensor = bb.insertArgument(loc, enableTy);
    inputAttrs.insert(inputAttrs.begin() + loc, sendAttr);
    inputNames.insert(inputNames.begin() + loc++,
                      memName + std::string("_addr_en"));
    portInterface.addrDataBusTensor = bb.insertArgument(loc, addrTy);
    inputAttrs.insert(inputAttrs.begin() + loc, sendAttr);
    inputNames.insert(inputNames.begin() + loc++, memName + "_addr_data");
  }

  if (auto rdLatency = helper::getRdLatency(portDict)) {
    portInterface.rdEnableBusTensor = bb.insertArgument(loc, enableTy);
    inputAttrs.insert(inputAttrs.begin() + loc, sendAttr);
    inputNames.insert(inputNames.begin() + loc++, memName + "_rd_en");
    portInterface.rdDataBusTensor = bb.insertArgument(loc, dataTy);
    inputAttrs.insert(inputAttrs.begin() + loc, recvAttr);
    inputNames.insert(inputNames.begin() + loc++, memName + "_rd_data");
    portInterface.rdLatency = rdLatency.getValue();
  }

  if (helper::isWrite(portDict)) {
    portInterface.wrEnableBusTensor = bb.insertArgument(loc, enableTy);
    inputAttrs.insert(inputAttrs.begin() + loc, sendAttr);
    inputNames.insert(inputNames.begin() + loc++, memName + "_wr_en");
    portInterface.wrDataBusTensor = bb.insertArgument(loc, dataTy);
    inputAttrs.insert(inputAttrs.begin() + loc, sendAttr);
    inputNames.insert(inputNames.begin() + loc++, memName + "_wr_data");
  }
  return loc;
}

void addBusAttrsPerPort(size_t i, SmallVectorImpl<DictionaryAttr> &attrs,
                        hir::MemrefType memrefTy, DictionaryAttr port) {
  auto *context = memrefTy.getContext();
  if (memrefTy.getNumElementsPerBank() > 1) {
    attrs.push_back(helper::getDictionaryAttr(
        context, "hir.bus.ports",
        ArrayAttr::get(context, StringAttr::get(context, "send"))));
  }
  if (helper::getRdLatency(port)) {
    attrs.push_back(helper::getDictionaryAttr(
        context, "hir.bus.ports",
        ArrayAttr::get(context, StringAttr::get(context, "send"))));
    attrs.push_back(helper::getDictionaryAttr(
        context, "hir.bus.ports",
        ArrayAttr::get(context, StringAttr::get(context, "recv"))));
  }
  if (helper::isWrite(port)) {
    attrs.push_back(helper::getDictionaryAttr(
        context, "hir.bus.ports",
        ArrayAttr::get(context, StringAttr::get(context, "send"))));
    attrs.push_back(helper::getDictionaryAttr(
        context, "hir.bus.ports",
        ArrayAttr::get(context, StringAttr::get(context, "send"))));
  }
}

void MemrefLoweringPass::insertBusArguments(hir::FuncLike op) {
  OpBuilder builder(op);
  builder.setInsertionPointToStart(&op.getFuncBody().front());
  auto funcTy = op.getFuncType();
  auto &bb = op.getFuncBody().front();
  SmallVector<DictionaryAttr> inputAttrs;
  SmallVector<std::string> inputNames;

  // initialize with old attrs.
  for (auto attr : funcTy.getInputAttrs())
    inputAttrs.push_back(attr);

  if (op->getAttrOfType<ArrayAttr>("argNames")) {
    for (auto attr : op->getAttrOfType<ArrayAttr>("argNames"))
      inputNames.push_back(attr.dyn_cast<StringAttr>().getValue().str());
  }

  // insert new types to bb args and new attrs to inputAttrs.
  for (int i = bb.getNumArguments() - 2 /*last arg is timetype*/; i >= 0; i--) {
    Value arg = bb.getArgument(i);
    if (auto memrefTy = arg.getType().dyn_cast<hir::MemrefType>()) {
      auto ports =
          helper::extractMemrefPortsFromDict(funcTy.getInputAttrs()[i]);
      SmallVector<MemrefPortInterface> memrefPortInterfaces;
      size_t insertBefore = i;
      for (size_t port = 0; port < ports.getValue().size(); port++) {
        auto portDict = ports.getValue()[port].dyn_cast<DictionaryAttr>();
        assert(portDict);
        MemrefPortInterface portInterface;
        insertBefore = insertBusTypesAndAttrsForMemrefPort(
            insertBefore, port, bb, inputNames, inputAttrs, memrefTy, portDict,
            portInterface);
        memrefPortInterfaces.push_back(portInterface);
      }
      memrefInfo.map(arg, emitMemoryInterfacesForEachPortBank(
                              builder, memrefTy, memrefPortInterfaces));
    }
  }

  op.updateArguments(inputAttrs);
  SmallVector<Attribute> inputNamesRef;
  for (size_t i = 0; i < inputNames.size(); i++) {
    auto name = builder.getStringAttr(inputNames[i]);
    inputNamesRef.push_back(name);
  }
  op->setAttr("argNames", builder.getArrayAttr(inputNamesRef));
}

void MemrefLoweringPass::removeMemrefArguments(hir::FuncLike op) {
  OpBuilder builder(op);
  auto funcTy = op.getFuncType();
  SmallVector<Type> inputTypes;
  SmallVector<DictionaryAttr> inputAttrs;
  auto &bb = op.getFuncBody().front();
  SmallVector<Attribute> inputNames;
  // initialize with old attrs.
  size_t i;
  for (i = 0; i < funcTy.getInputAttrs().size(); i++) {
    auto attr = funcTy.getInputAttrs()[i];
    inputAttrs.push_back(attr);
    inputNames.push_back(op.getInputNames()[i]);
  }
  inputNames.push_back(op.getInputNames()[i]);

  for (int i = bb.getNumArguments() - 2 /*last arg is timetype*/; i >= 0; i--) {
    if (bb.getArgument(i).getType().isa<hir::MemrefType>()) {
      bb.eraseArgument(i);
      inputAttrs.erase(inputAttrs.begin() + i);
      inputNames.erase(inputNames.begin() + i);
    }
  }
  op.updateArguments(inputAttrs);
  op->setAttr("argNames", ArrayAttr::get(builder.getContext(), inputNames));
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// MemrefLoweringPass methods.
//------------------------------------------------------------------------------

LogicalResult MemrefLoweringPass::visitOp(hir::LoadOp op) {
  auto memrefTy = op.mem().getType().dyn_cast<hir::MemrefType>();
  auto bank = helper::calcLinearIndex(op.filterIndices(BANK),
                                      memrefTy.filterShape(BANK))
                  .getValue();
  auto *memoryInterface =
      memrefInfo.getInterface(op.mem(), op.port().getValue(), bank);

  mlir::OpBuilder builder(op.getContext());
  builder.setInsertionPoint(op);

  // Insert logic to send address and valid signals to address bus.
  Value addr = createLinearAddr(builder, op.getLoc(), op.filterIndices(ADDR));

  if (memoryInterface->hasAddrBus() &&
      failed(memoryInterface->emitAddrSendLogic(builder, op.getLoc(), addr,
                                                op.tstart(), op.offsetAttr())))
    return failure();

  auto loadValue = memoryInterface->emitRdRecvLogic(
      builder, op.getLoc(), op.tstart(), op.offsetAttr());
  if (!loadValue)
    return failure();
  op.replaceAllUsesWith(loadValue);
  opsToErase.push_back(op);
  return success();
}

LogicalResult MemrefLoweringPass::visitOp(hir::StoreOp op) {

  auto memrefTy = op.mem().getType().dyn_cast<hir::MemrefType>();
  auto bank = helper::calcLinearIndex(op.filterIndices(BANK),
                                      memrefTy.filterShape(BANK))
                  .getValue();
  auto *memoryInterface =
      memrefInfo.getInterface(op.mem(), op.port().getValue(), bank);

  mlir::OpBuilder builder(op.getContext());
  builder.setInsertionPoint(op);

  // Insert logic to send address and valid signals to address bus.
  Value addr = createLinearAddr(builder, op.getLoc(), op.filterIndices(ADDR));

  if (memoryInterface->hasAddrBus() &&
      failed(memoryInterface->emitAddrSendLogic(builder, op.getLoc(), addr,
                                                op.tstart(), op.offsetAttr())))
    return failure();

  // Insert logic to send wr valid and data.
  if (memoryInterface->hasWrBus() &&
      failed(memoryInterface->emitWrSendLogic(builder, op.getLoc(), op.value(),
                                              op.tstart(), op.offsetAttr())))
    return failure();

  opsToErase.push_back(op);
  return success();
}

LogicalResult MemrefLoweringPass::visitOp(hir::CallOp op) {
  auto funcTy = op.getFuncType();
  SmallVector<Value> operands;
  SmallVector<Type> inputTypes;
  SmallVector<DictionaryAttr> inputAttrs;
  OpBuilder builder(op);
  builder.setInsertionPoint(op);

  bool hasMemrefArg = false;
  for (size_t i = 0; i < funcTy.getInputTypes().size(); i++) {
    auto ty = funcTy.getInputTypes()[i];
    auto operand = op.operands()[i];
    if (!ty.isa<hir::MemrefType>()) {
      operands.push_back(operand);
      inputTypes.push_back(funcTy.getInputTypes()[i]);
      inputAttrs.push_back(funcTy.getInputAttrs()[i]);
      continue;
    }
    hasMemrefArg = true;
    auto numPorts = memrefInfo.getNumPorts(operand);
    auto ports = helper::extractMemrefPortsFromDict(funcTy.getInputAttrs()[i])
                     .getValue();
    if (ports.size() != numPorts) {
      return op->emitError("ports.size()!=numPorts.")
                 .attachNote(operand.getDefiningOp()->getLoc())
             << "Memref defined here.";
    }
    emitCallOpOperandsForMemrefPort(builder, memrefInfo, operand, operands,
                                    inputTypes, inputAttrs);
  }
  if (!hasMemrefArg)
    return success();
  auto newFuncTy =
      hir::FuncType::get(builder.getContext(), inputTypes, inputAttrs,
                         funcTy.getResultTypes(), funcTy.getResultAttrs());
  auto newCallOp = builder.create<hir::CallOp>(
      op.getLoc(), op.getResultTypes(), op.instance_nameAttr(), op.calleeAttr(),
      TypeAttr::get(newFuncTy), operands, op.tstart(), op.offsetAttr());
  op.replaceAllUsesWith(newCallOp);
  opsToErase.push_back(op);
  return success();
}

LogicalResult MemrefLoweringPass::visitOp(hir::FuncExternOp op) {
  insertBusArguments(op);
  removeMemrefArguments(op);
  memrefInfo.clear();
  return success();
}

LogicalResult MemrefLoweringPass::visitOp(hir::AllocaOp op) {
  // Add comment.
  OpBuilder builder(op);
  builder.setInsertionPoint(op);
  if (failed(createBusInstantiationsAndCallOp(op)))
    return failure();
  opsToErase.push_back(op);
  return success();
}

LogicalResult MemrefLoweringPass::visitOp(hir::MemrefExtractOp op) {
  // port 0 of the op.res() is mapped to op.port() of op.mem().
  memrefInfo.mapPort(op.res(), op.mem(), op.port().getValue());

  opsToErase.push_back(op);
  return success();
}

LogicalResult MemrefLoweringPass::visitOp(hir::FuncOp funcOp) {
  topLevelBuilder = new OpBuilder(funcOp);
  topLevelBuilder->setInsertionPointToStart(&funcOp.body().front());
  insertBusArguments(funcOp);
  WalkResult result = funcOp.walk([this](Operation *operation) -> WalkResult {
    if (auto op = dyn_cast<hir::AllocaOp>(operation)) {
      if (failed(visitOp(op)))
        return WalkResult::interrupt();
    } else if (auto op = dyn_cast<hir::MemrefExtractOp>(operation)) {
      if (failed(visitOp(op)))
        return WalkResult::interrupt();
    } else if (auto op = dyn_cast<hir::LoadOp>(operation)) {
      if (failed(visitOp(op)))
        return WalkResult::interrupt();
    } else if (auto op = dyn_cast<hir::StoreOp>(operation)) {
      if (failed(visitOp(op)))
        return WalkResult::interrupt();
    } else if (auto op = dyn_cast<hir::CallOp>(operation)) {
      if (failed(visitOp(op)))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    return failure();
  }

  initUnConnectedPorts(funcOp);
  helper::eraseOps(opsToErase);
  memrefInfo.clear();
  removeMemrefArguments(funcOp);
  return success();
}

void MemrefLoweringPass::runOnOperation() {
  mlir::ModuleOp moduleOp = getOperation();
  LogicalResult result = success();
  for (auto &operation : moduleOp) {
    if (auto funcOp = dyn_cast<hir::FuncOp>(operation)) {
      if (failed(visitOp(funcOp)))
        result = failure();
    }
    if (auto funcExternOp = dyn_cast<hir::FuncExternOp>(operation)) {
      if (failed(visitOp(funcExternOp)))
        result = failure();
    }
  }
  if (failed(result))
    signalPassFailure();
}

namespace circt {
namespace hir {
std::unique_ptr<OperationPass<mlir::ModuleOp>> createMemrefLoweringPass() {
  return std::make_unique<MemrefLoweringPass>();
}
} // namespace hir
} // namespace circt
