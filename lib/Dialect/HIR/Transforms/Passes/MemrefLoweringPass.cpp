//=========- MemrefLoweringPass.cpp - Lower memref type---===//
//
// This file implements lowering pass for memref type.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseSet.h"
#include <iostream>
using namespace circt;
using namespace hir;
namespace {
struct MemrefPortInterface {
  Value addrDataBusTensor;
  Value addrEnableBusTensor;
  Value rdEnableBusTensor;
  int64_t rdLatency = -1;
  Value rdDataBusTensor;
  Value wrEnableBusTensor;
  Value wrDataBusTensor;
};

class MapMemrefToPortInterfaces {
public:
  void insert(Value mem, SmallVector<MemrefPortInterface> memPortInterfaces) {
    assert(mem.getType().isa<hir::MemrefType>());
    assert(mapMemref2idx.find(mem) == mapMemref2idx.end());
    SmallVector<uint64_t> portLocs;
    for (auto portInterface : memPortInterfaces) {
      portLocs.push_back(portInterfaceList.size());
      portInterfaceList.push_back(portInterface);
    }
    mapMemref2idx[mem] = portLocs;
  }

  void remap(Value mem, Value originalMem, uint64_t port) {
    assert(mem.getType().isa<hir::MemrefType>());
    assert(originalMem.getType().isa<hir::MemrefType>());
    assert(mapMemref2idx.find(mem) == mapMemref2idx.end());
    assert(mapMemref2idx.find(originalMem) != mapMemref2idx.end());
    mapMemref2idx[mem].push_back(mapMemref2idx[originalMem][port]);
  }

  uint64_t getNumPorts(Value mem) { return mapMemref2idx[mem].size(); }

  MemrefPortInterface &get(Value mem, uint64_t port) {
    return portInterfaceList[mapMemref2idx[mem][port]];
  }
  ArrayRef<MemrefPortInterface> getAllPortInterfaces() {
    return portInterfaceList;
  }
  void clear() {
    mapMemref2idx.clear();
    portInterfaceList.clear();
  }

private:
  DenseMap<Value, SmallVector<uint64_t>> mapMemref2idx;
  SmallVector<MemrefPortInterface> portInterfaceList;
};

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
  Value getEnableSendBusTensor(OpBuilder &, Value &, Value, IntegerAttr,
                               std::string);
  Value getDataSendBusTensor(OpBuilder &, Value &, Value, Value, IntegerAttr,
                             std::string);
  Value getEnableSendBus(OpBuilder &, Value &, uint64_t, uint64_t, Value,
                         IntegerAttr, std::string);
  Value getDataSendBus(OpBuilder &, Value &, Value, uint64_t, uint64_t, Value,
                       IntegerAttr, std::string);
  MemrefPortInterface defineBusesForMemrefPort(OpBuilder &, hir::MemrefType,
                                               Attribute,
                                               llvm::Optional<StringRef> name);
  LogicalResult createBusInstantiationsAndCallOp(hir::AllocaOp);
  void initUnConnectedPorts(hir::FuncOp);

private:
  MapMemrefToPortInterfaces mapMemrefToPortInterfaces;
  SmallVector<Operation *, 10> opsToErase;
  OpBuilder *topLevelBuilder;
};
} // end anonymous namespace

//------------------------------------------------------------------------------
// Helper functions.
//------------------------------------------------------------------------------
Operation *declareExternalFuncForCall(hir::CallOp callOp,
                                      SmallVector<StringRef> inputNames,
                                      SmallVector<StringRef> resultNames = {}) {
  if (callOp.getCalleeDecl())
    return NULL;
  OpBuilder builder(callOp);
  auto moduleOp = callOp->getParentOfType<ModuleOp>();
  builder.setInsertionPointToStart(&moduleOp.body().front());

  auto declOp = builder.create<hir::FuncExternOp>(
      builder.getUnknownLoc(), callOp.calleeAttr().getAttr(),
      TypeAttr::get(callOp.getFuncType()));

  declOp.getFuncBody().push_back(new Block);
  OpBuilder declOpBuilder(declOp);
  FuncExternOp::ensureTerminator(declOp.getFuncBody(), declOpBuilder,
                                 builder.getUnknownLoc());
  // declOp.getFuncBody().front();
  assert(inputNames.size() == callOp.getFuncType().getInputTypes().size());
  inputNames.push_back("t");
  declOp->setAttr("argNames", builder.getStrArrayAttr(inputNames));

  if (resultNames.size() > 0) {
    assert(resultNames.size() == callOp.getFuncType().getResultTypes().size());
    declOp->setAttr("resultNames", builder.getStrArrayAttr(resultNames));
  }

  return declOp;
}

Type getTensorElementType(Value tensor) {
  auto ty = tensor.getType().dyn_cast<mlir::TensorType>();
  assert(ty);
  return ty.getElementType();
}

mlir::FlatSymbolRefAttr createUniqueFuncName(MLIRContext *context,
                                             StringRef name,
                                             ArrayRef<uint64_t> params) {
  std::string newName(name);
  newName += "_";
  for (uint64_t i = 0; i < params.size(); i++) {
    if (i > 0)
      newName += "x";
    newName += std::to_string(params[i]);
  }
  return FlatSymbolRefAttr::get(context, newName);
}

Value createAddrTuple(OpBuilder &builder, Location loc,
                      ArrayRef<Value> indices) {
  if (indices.size() == 0)
    return Value();

  SmallVector<Type, 4> idxTypes;
  for (auto idx : indices)
    idxTypes.push_back(idx.getType());

  return builder
      .create<hir::CreateTupleOp>(loc, builder.getTupleType(idxTypes), indices)
      .getResult();
}

mlir::TensorType buildBusTensor(MLIRContext *context, ArrayRef<int64_t> shape,
                                SmallVector<Type> busElementTypes) {
  SmallVector<BusDirection> busDirections;
  // Only same direction is allowed inside a tensor<bus>.
  busDirections.append(busElementTypes.size(), BusDirection::SAME);

  return mlir::RankedTensorType::get(
      shape, hir::BusType::get(context, busElementTypes, busDirections));
}

MemrefPortInterface MemrefLoweringPass::defineBusesForMemrefPort(
    OpBuilder &builder, hir::MemrefType memrefTy, Attribute port,
    llvm::Optional<StringRef> name) {
  MemrefPortInterface portInterface;

  auto *context = memrefTy.getContext();
  Type enableTy = buildBusTensor(context, memrefTy.filterShape(BANK),
                                 {IntegerType::get(context, 1)});

  SmallVector<Type> addrTupleElementTypes;
  uint64_t addrWidth = 0;
  for (uint64_t dimSize : memrefTy.filterShape(ADDR)) {
    addrTupleElementTypes.push_back(
        IntegerType::get(context, helper::clog2(dimSize)));
    addrWidth += helper::clog2(dimSize);
  }

  Type addrTupleTy =
      buildBusTensor(context, memrefTy.filterShape(BANK),
                     {TupleType::get(context, addrTupleElementTypes)});
  Type dataTy = buildBusTensor(context, memrefTy.filterShape(BANK),
                               {memrefTy.getElementType()});

  if (memrefTy.getNumElementsPerBank() > 1) {
    portInterface.addrEnableBusTensor =
        topLevelBuilder->create<hir::BusOp>(builder.getUnknownLoc(), enableTy);
    portInterface.addrDataBusTensor = topLevelBuilder->create<hir::BusOp>(
        builder.getUnknownLoc(), addrTupleTy);
    if (name) {
      helper::setNames(portInterface.addrEnableBusTensor.getDefiningOp(),
                       {name.getValue().str() + "_addr_en"});
      helper::setNames(portInterface.addrDataBusTensor.getDefiningOp(),
                       {name.getValue().str() + "_addr_data"});
    }
  }

  if (auto rdLatency = helper::getRdLatency(port)) {
    portInterface.rdEnableBusTensor =
        topLevelBuilder->create<hir::BusOp>(builder.getUnknownLoc(), enableTy);
    portInterface.rdDataBusTensor =
        topLevelBuilder->create<hir::BusOp>(builder.getUnknownLoc(), dataTy);
    portInterface.rdLatency = rdLatency.getValue();
    if (name) {
      helper::setNames(portInterface.rdEnableBusTensor.getDefiningOp(),
                       {name.getValue().str() + "_rd_en"});
      helper::setNames(portInterface.rdDataBusTensor.getDefiningOp(),
                       {name.getValue().str() + "_rd_data"});
    }
  }

  if (helper::isWrite(port)) {
    portInterface.wrEnableBusTensor =
        topLevelBuilder->create<hir::BusOp>(builder.getUnknownLoc(), enableTy);
    portInterface.wrDataBusTensor =
        topLevelBuilder->create<hir::BusOp>(builder.getUnknownLoc(), dataTy);
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
  for (uint64_t portNum = 0; portNum < memrefPortInterfaces.size(); portNum++) {
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

  auto callOp = builder.create<hir::CallOp>(
      builder.getUnknownLoc(), SmallVector<Type>(), memModuleName,
      TypeAttr::get(funcTy), inputBuses, tstartRegion, IntegerAttr());

  declareExternalFuncForCall(callOp, inputBusNames);

  auto params = builder.getDictionaryAttr(
      {builder.getNamedAttr("ELEMENT_WIDTH",
                            builder.getI64IntegerAttr(elementWidth)),
       builder.getNamedAttr("ADDR_WIDTH", builder.getI64IntegerAttr(addrWidth)),
       builder.getNamedAttr("TENSOR_SIZE",
                            builder.getI64IntegerAttr(tensorSize))});

  callOp->setAttr("params", params);

  mapMemrefToPortInterfaces.insert(op.res(), memrefPortInterfaces);
  return success();
}

void initUnconnectedEnBusTensor(OpBuilder &builder, Value busTensor) {
  if (!busTensor)
    return;
  auto *context = builder.getContext();
  DictionaryAttr sendAttr = helper::getDictionaryAttr(
      context, "hir.bus.ports",
      ArrayAttr::get(context, StringAttr::get(context, "send")));
  auto funcTy =
      hir::FuncType::get(context, {busTensor.getType()}, {sendAttr}, {}, {});
  auto tstartRegion = busTensor.getParentRegion()->getArguments().back();

  auto width = builder.getI64IntegerAttr(
      helper::getBitWidth(
          busTensor.getType().dyn_cast<mlir::TensorType>().getElementType())
          .getValue());
  auto numBanks = builder.getI64IntegerAttr(
      busTensor.getType().dyn_cast<mlir::TensorType>().getNumElements());
  auto params =
      builder.getDictionaryAttr({builder.getNamedAttr("WIDTH", width),
                                 builder.getNamedAttr("NUM_BANKS", numBanks)});
  auto callOp = builder.create<hir::CallOp>(
      builder.getUnknownLoc(), SmallVector<Type>(),
      createUniqueFuncName(
          builder.getContext(), "hir_en_init",
          {(uint64_t)numBanks.getInt(), (uint64_t)width.getInt()}),
      TypeAttr::get(funcTy), SmallVector<Value>({busTensor}), tstartRegion,
      IntegerAttr());
  callOp->setAttr("params", params);
  declareExternalFuncForCall(callOp, {"en_bus_tensor"});
}

void MemrefLoweringPass::initUnConnectedPorts(hir::FuncOp op) {
  OpBuilder builder(op);
  auto *returnOperation = &op.body().front().back();
  builder.setInsertionPoint(returnOperation);
  for (auto portInterface : mapMemrefToPortInterfaces.getAllPortInterfaces()) {
    initUnconnectedEnBusTensor(builder, portInterface.addrEnableBusTensor);
    initUnconnectedEnBusTensor(builder, portInterface.rdEnableBusTensor);
    initUnconnectedEnBusTensor(builder, portInterface.wrEnableBusTensor);
  }
}

SmallVector<Value> filterMemrefArgs(Block::BlockArgListType args) {
  SmallVector<Value> memrefArgs;
  for (auto arg : args)
    if (arg.getType().isa<hir::MemrefType>())
      memrefArgs.push_back(arg);
  return memrefArgs;
}

size_t insertBusTypesAndAttrsForMemrefPort(
    size_t i, Block &bb, SmallVectorImpl<std::string> &inputNames,
    SmallVectorImpl<DictionaryAttr> &inputAttrs, hir::MemrefType memrefTy,
    DictionaryAttr port, MemrefPortInterface &portInterface) {
  auto *context = memrefTy.getContext();
  Type enableTy = buildBusTensor(context, memrefTy.filterShape(BANK),
                                 {IntegerType::get(context, 1)});

  SmallVector<Type> addrTupleElementTypes;

  for (uint64_t dimSize : memrefTy.filterShape(ADDR)) {
    addrTupleElementTypes.push_back(
        IntegerType::get(context, helper::clog2(dimSize)));
  }

  Type addrTupleTy =
      buildBusTensor(context, memrefTy.filterShape(BANK),
                     {TupleType::get(context, addrTupleElementTypes)});
  Type dataTy = buildBusTensor(context, memrefTy.filterShape(BANK),
                               {memrefTy.getElementType()});

  DictionaryAttr sendAttr = helper::getDictionaryAttr(
      context, "hir.bus.ports",
      ArrayAttr::get(context, StringAttr::get(context, "send")));
  DictionaryAttr recvAttr = helper::getDictionaryAttr(
      context, "hir.bus.ports",
      ArrayAttr::get(context, StringAttr::get(context, "recv")));

  std::string memName;
  if (memrefTy.getNumElementsPerBank() > 1) {
    portInterface.addrEnableBusTensor = bb.insertArgument(i, enableTy);
    inputAttrs.insert(inputAttrs.begin() + i, sendAttr);
    memName = inputNames[i];
    inputNames.insert(inputNames.begin() + i++,
                      memName + std::string("_addr_en"));
    portInterface.addrDataBusTensor = bb.insertArgument(i, addrTupleTy);
    inputAttrs.insert(inputAttrs.begin() + i, sendAttr);
    memName = inputNames[i];
    inputNames.insert(inputNames.begin() + i++, memName + "_addr_data");
  }

  if (auto rdLatency = helper::getRdLatency(port)) {
    portInterface.rdEnableBusTensor = bb.insertArgument(i, enableTy);
    inputAttrs.insert(inputAttrs.begin() + i, sendAttr);
    inputNames.insert(inputNames.begin() + i++, memName + "_rd_en");
    portInterface.rdDataBusTensor = bb.insertArgument(i, dataTy);
    inputAttrs.insert(inputAttrs.begin() + i, recvAttr);
    inputNames.insert(inputNames.begin() + i++, memName + "_rd_data");
    portInterface.rdLatency = rdLatency.getValue();
  }

  if (helper::isWrite(port)) {
    portInterface.wrEnableBusTensor = bb.insertArgument(i, enableTy);
    inputAttrs.insert(inputAttrs.begin() + i, sendAttr);
    inputNames.insert(inputNames.begin() + i++, memName + "_wr_en");
    portInterface.wrDataBusTensor = bb.insertArgument(i, dataTy);
    inputAttrs.insert(inputAttrs.begin() + i, sendAttr);
    inputNames.insert(inputNames.begin() + i++, memName + "_wr_data");
  }
  return i;
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
      for (auto port : ports.getValue()) {
        auto portDict = port.dyn_cast<DictionaryAttr>();
        assert(portDict);
        MemrefPortInterface portInterface;
        insertBefore = insertBusTypesAndAttrsForMemrefPort(
            insertBefore, bb, inputNames, inputAttrs, memrefTy, portDict,
            portInterface);
        memrefPortInterfaces.push_back(portInterface);
      }
      mapMemrefToPortInterfaces.insert(arg, memrefPortInterfaces);
    }
  }
  op.updateArguments(inputAttrs);
  SmallVector<Attribute> inputNamesRef;
  for (uint64_t i = 0; i < inputNames.size(); i++) {
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
  uint64_t i;
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

Value extractBusFromTensor(OpBuilder &builder, Value busTensor,
                           ArrayRef<uint64_t> indices, ArrayAttr ports) {
  auto tensorTy = busTensor.getType().dyn_cast<mlir::TensorType>();
  assert(tensorTy);
  auto busTy = tensorTy.getElementType().dyn_cast<hir::BusType>();
  assert(busTy);
  SmallVector<Value> cIndices;
  for (auto idx : indices) {
    cIndices.push_back(builder.create<mlir::ConstantOp>(
        builder.getUnknownLoc(), builder.getIndexType(),
        builder.getIndexAttr(idx)));
  }
  return builder.create<hir::TensorExtractOp>(builder.getUnknownLoc(), busTy,
                                              busTensor, cIndices, ports);
}

void createValidCombineCallOp(OpBuilder &builder, Value destBusTensor,
                              ArrayRef<uint64_t> bankIndices, Value busTensor,
                              Value tstartRegion) {
  Value bus = extractBusFromTensor(builder, destBusTensor, bankIndices,
                                   builder.getStrArrayAttr({"send"}));
  auto sendAttr = helper::getDictionaryAttr(builder, "hir.bus.ports",
                                            builder.getStrArrayAttr({"send"}));
  auto recvAttr = helper::getDictionaryAttr(builder, "hir.bus.ports",
                                            builder.getStrArrayAttr({"recv"}));

  Type funcTy = hir::FuncType::get(builder.getContext(),
                                   {bus.getType(), busTensor.getType()},
                                   {sendAttr, recvAttr}, {}, {});

  auto tensorTy = busTensor.getType().dyn_cast<mlir::TensorType>();
  auto callOp = builder.create<hir::CallOp>(
      builder.getUnknownLoc(), SmallVector<Type>(),
      createUniqueFuncName(builder.getContext(), "hir_en_combine_x",
                           tensorTy.getNumElements()),
      TypeAttr::get(funcTy), SmallVector<Value>({bus, busTensor}), tstartRegion,
      IntegerAttr());
  declareExternalFuncForCall(callOp, {"combined_en_bus", "en_bus_tensor"});

  auto params = builder.getDictionaryAttr({builder.getNamedAttr(
      "TENSOR_SIZE", builder.getI64IntegerAttr(tensorTy.getNumElements()))});
  callOp->setAttr("params", params);
}

void createBusBroadcastCallOp(OpBuilder &builder, Value sourceBusTensor,
                              ArrayRef<uint64_t> bankIndices,
                              Value destBusValidTensor, Value destBusTensor,
                              Value tstartRegion) {
  Value sourceBus = extractBusFromTensor(builder, sourceBusTensor, bankIndices,
                                         builder.getStrArrayAttr({"recv"}));
  auto sendAttr = helper::getDictionaryAttr(builder, "hir.bus.ports",
                                            builder.getStrArrayAttr({"send"}));
  auto recvAttr = helper::getDictionaryAttr(builder, "hir.bus.ports",
                                            builder.getStrArrayAttr({"recv"}));

  Type funcTy =
      hir::FuncType::get(builder.getContext(),
                         {destBusTensor.getType(), destBusValidTensor.getType(),
                          sourceBus.getType()},
                         {sendAttr, recvAttr, recvAttr}, {}, {});

  uint64_t tensorSize =
      destBusTensor.getType().dyn_cast<mlir::TensorType>().getNumElements();
  auto elementWidth = helper::getBitWidth(
      destBusTensor.getType().dyn_cast<mlir::TensorType>().getElementType());

  auto callOp = builder.create<hir::CallOp>(
      builder.getUnknownLoc(), SmallVector<Type>(),
      createUniqueFuncName(builder.getContext(), "hir_bus_broadcast",
                           {elementWidth.getValue(), tensorSize}),
      TypeAttr::get(funcTy),
      SmallVector<Value>({destBusTensor, destBusValidTensor, sourceBus}),
      tstartRegion, IntegerAttr());

  declareExternalFuncForCall(
      callOp, {"dest_bus_tensor", "dest_bus_valid_tensor", "source_bus"});

  auto params = builder.getDictionaryAttr(
      {builder.getNamedAttr("TENSOR_SIZE",
                            builder.getI64IntegerAttr(tensorSize)),
       builder.getNamedAttr("ELEMENT_WIDTH", builder.getI64IntegerAttr(
                                                 elementWidth.getValue()))});
  callOp->setAttr("params", params);
}

void createBusMuxCallOp(OpBuilder &builder, Value destBusTensor,
                        ArrayRef<uint64_t> bankIndices, Value busValidTensor,
                        Value busTensor, Value tstartRegion) {
  Value bus = extractBusFromTensor(builder, destBusTensor, bankIndices,
                                   builder.getStrArrayAttr({"send"}));
  auto sendAttr = helper::getDictionaryAttr(builder, "hir.bus.ports",
                                            builder.getStrArrayAttr({"send"}));
  auto recvAttr = helper::getDictionaryAttr(builder, "hir.bus.ports",
                                            builder.getStrArrayAttr({"recv"}));

  Type funcTy = hir::FuncType::get(
      builder.getContext(),
      {bus.getType(), busValidTensor.getType(), busTensor.getType()},
      {sendAttr, recvAttr, recvAttr}, {}, {});

  uint64_t tensorSize =
      busTensor.getType().dyn_cast<mlir::TensorType>().getNumElements();
  auto elementWidth = helper::getBitWidth(
      busTensor.getType().dyn_cast<mlir::TensorType>().getElementType());
  auto callOp = builder.create<hir::CallOp>(
      builder.getUnknownLoc(), SmallVector<Type>(),
      createUniqueFuncName(builder.getContext(), "hir_bus_mux",
                           {elementWidth.getValue(), tensorSize}),
      TypeAttr::get(funcTy),
      SmallVector<Value>({bus, busValidTensor, busTensor}), tstartRegion,
      IntegerAttr());
  declareExternalFuncForCall(callOp, {"out", "bus_valid_tensor", "bus_tensor"});

  callOp->setAttr("TENSOR_SIZE", builder.getI64IntegerAttr(tensorSize));
  callOp->setAttr("ELEMENT_WIDTH",
                  builder.getI64IntegerAttr(elementWidth.getValue()));
}

Value MemrefLoweringPass::getEnableSendBusTensor(OpBuilder &builder,
                                                 Value &originalEnableBusTensor,
                                                 Value tstart,
                                                 IntegerAttr offsetAttr,
                                                 std::string name) {
  auto forkedEnableBusTensor = topLevelBuilder->create<hir::BusOp>(
      builder.getUnknownLoc(), originalEnableBusTensor.getType());
  helper::setNames(forkedEnableBusTensor, {name + "_forked"});
  auto newEnableBusTensor = topLevelBuilder->create<hir::BusOp>(
      builder.getUnknownLoc(), forkedEnableBusTensor.getType());
  helper::setNames(newEnableBusTensor, {name + "_buses"});

  auto sendAttr = helper::getDictionaryAttr(builder, "hir.bus.ports",
                                            builder.getStrArrayAttr({"send"}));
  auto recvAttr = helper::getDictionaryAttr(builder, "hir.bus.ports",
                                            builder.getStrArrayAttr({"recv"}));
  auto funcTy = hir::FuncType::get(builder.getContext(),
                                   {originalEnableBusTensor.getType(),
                                    forkedEnableBusTensor.getType(),
                                    newEnableBusTensor.getType()},
                                   {sendAttr, recvAttr, recvAttr}, {}, {});

  auto width = builder.getI64IntegerAttr(
      helper::getBitWidth(originalEnableBusTensor.getType()
                              .dyn_cast<mlir::TensorType>()
                              .getElementType())
          .getValue());
  auto numBanks = builder.getI64IntegerAttr(originalEnableBusTensor.getType()
                                                .dyn_cast<mlir::TensorType>()
                                                .getNumElements());

  auto combineEnBus = builder.create<hir::CallOp>(
      builder.getUnknownLoc(), SmallVector<Type>(),
      createUniqueFuncName(
          builder.getContext(), "combine_en_bus_tensor",
          {(uint64_t)width.getInt(), (uint64_t)numBanks.getInt()}),
      TypeAttr::get(funcTy),
      SmallVector<Value>(
          {originalEnableBusTensor, forkedEnableBusTensor, newEnableBusTensor}),
      tstart, offsetAttr);
  declareExternalFuncForCall(
      combineEnBus,
      {"original_en_bus_tensor", "forked_en_bus_tensor", "new_en_bus_tensor"});

  auto params =
      builder.getDictionaryAttr({builder.getNamedAttr("WIDTH", width),
                                 builder.getNamedAttr("NUM_BANKS", numBanks)});
  combineEnBus->setAttr("params", params);
  originalEnableBusTensor = newEnableBusTensor;
  return forkedEnableBusTensor;
}

Value MemrefLoweringPass::getDataSendBusTensor(
    OpBuilder &builder, Value &originalDataBusTensor, Value enableBusTensor,
    Value tstart, IntegerAttr offsetAttr, std::string name) {
  auto forkedDataBusTensor = topLevelBuilder->create<hir::BusOp>(
      builder.getUnknownLoc(), originalDataBusTensor.getType());
  helper::setNames(forkedDataBusTensor, {name + "_forked"});
  auto newDataBusTensor = topLevelBuilder->create<hir::BusOp>(
      builder.getUnknownLoc(), forkedDataBusTensor.getType());
  helper::setNames(newDataBusTensor, {name + "_buses"});

  auto sendAttr = helper::getDictionaryAttr(builder, "hir.bus.ports",
                                            builder.getStrArrayAttr({"send"}));
  auto recvAttr = helper::getDictionaryAttr(builder, "hir.bus.ports",
                                            builder.getStrArrayAttr({"recv"}));
  auto funcTy = hir::FuncType::get(
      builder.getContext(),
      {originalDataBusTensor.getType(), enableBusTensor.getType(),
       forkedDataBusTensor.getType(), newDataBusTensor.getType()},
      {sendAttr, recvAttr, recvAttr, recvAttr}, {}, {});

  auto width = builder.getI64IntegerAttr(
      helper::getBitWidth(originalDataBusTensor.getType()
                              .dyn_cast<mlir::TensorType>()
                              .getElementType())
          .getValue());
  auto numBanks = builder.getI64IntegerAttr(originalDataBusTensor.getType()
                                                .dyn_cast<mlir::TensorType>()
                                                .getNumElements());
  auto combineDataBus = builder.create<hir::CallOp>(
      builder.getUnknownLoc(), SmallVector<Type>(),
      createUniqueFuncName(
          builder.getContext(), "combine_data_bus_tensor",
          {(uint64_t)width.getInt(), (uint64_t)numBanks.getInt()}),

      TypeAttr::get(funcTy),
      SmallVector<Value>({originalDataBusTensor, enableBusTensor,
                          forkedDataBusTensor, newDataBusTensor}),
      tstart, offsetAttr);

  declareExternalFuncForCall(combineDataBus,
                             {"original_data_bus_tensor", "en_bus_tensor",
                              "forked_data_bus_tensor", "new_data_bus_tensor"});

  auto params =
      builder.getDictionaryAttr({builder.getNamedAttr("WIDTH", width),
                                 builder.getNamedAttr("NUM_BANKS", numBanks)});
  combineDataBus->setAttr("params", params);
  originalDataBusTensor = newDataBusTensor;
  return forkedDataBusTensor;
}

Value MemrefLoweringPass::getEnableSendBus(OpBuilder &builder,
                                           Value &enableBusTensor,
                                           uint64_t bank, uint64_t numBanks,
                                           Value tstart, IntegerAttr offsetAttr,
                                           std::string name) {
  auto forkedEnableBus = topLevelBuilder->create<hir::BusOp>(
      builder.getUnknownLoc(), getTensorElementType(enableBusTensor));
  helper::setNames(forkedEnableBus, {name + "_forked"});
  auto newEnableBusTensor = topLevelBuilder->create<hir::BusOp>(
      builder.getUnknownLoc(), enableBusTensor.getType());
  helper::setNames(newEnableBusTensor, {name + "_buses"});

  auto sendAttr = helper::getDictionaryAttr(builder, "hir.bus.ports",
                                            builder.getStrArrayAttr({"send"}));
  auto recvAttr = helper::getDictionaryAttr(builder, "hir.bus.ports",
                                            builder.getStrArrayAttr({"recv"}));
  auto funcTy =
      hir::FuncType::get(builder.getContext(),
                         {enableBusTensor.getType(), forkedEnableBus.getType(),
                          newEnableBusTensor.getType()},
                         {sendAttr, recvAttr, recvAttr}, {}, {});

  auto combineEnBus = builder.create<hir::CallOp>(
      builder.getUnknownLoc(), SmallVector<Type>(),
      createUniqueFuncName(builder.getContext(), "combine_en_bus", numBanks),

      TypeAttr::get(funcTy),
      SmallVector<Value>(
          {enableBusTensor, forkedEnableBus, newEnableBusTensor}),
      tstart, offsetAttr);

  declareExternalFuncForCall(
      combineEnBus, {"original_en_bus_tensor", "forked_en_bus", "new_en_bus"});

  auto params = builder.getDictionaryAttr(
      {builder.getNamedAttr("BANK", builder.getI64IntegerAttr(bank)),
       builder.getNamedAttr("NUM_BANKS", builder.getI64IntegerAttr(numBanks))});
  combineEnBus->setAttr("params", params);
  enableBusTensor = newEnableBusTensor;
  return forkedEnableBus;
}

Value MemrefLoweringPass::getDataSendBus(OpBuilder &builder,
                                         Value &dataBusTensor, Value enableBus,
                                         uint64_t bank, uint64_t numBanks,
                                         Value tstart, IntegerAttr offsetAttr,
                                         std::string name) {
  auto forkedDataBus = topLevelBuilder->create<hir::BusOp>(
      builder.getUnknownLoc(), getTensorElementType(dataBusTensor));
  helper::setNames(forkedDataBus, {name + "_forked"});

  auto newDataBusTensor = topLevelBuilder->create<hir::BusOp>(
      builder.getUnknownLoc(), dataBusTensor.getType());
  helper::setNames(newDataBusTensor, {name + "_buses"});
  auto sendAttr = helper::getDictionaryAttr(builder, "hir.bus.ports",
                                            builder.getStrArrayAttr({"send"}));
  auto recvAttr = helper::getDictionaryAttr(builder, "hir.bus.ports",
                                            builder.getStrArrayAttr({"recv"}));
  auto funcTy =
      hir::FuncType::get(builder.getContext(),
                         {dataBusTensor.getType(), enableBus.getType(),
                          forkedDataBus.getType(), newDataBusTensor.getType()},
                         {sendAttr, recvAttr, recvAttr, recvAttr}, {}, {});

  auto width = builder.getI64IntegerAttr(
      helper::getBitWidth(
          dataBusTensor.getType().dyn_cast<mlir::TensorType>().getElementType())
          .getValue());

  auto combineData = builder.create<hir::CallOp>(
      builder.getUnknownLoc(), SmallVector<Type>(),
      createUniqueFuncName(builder.getContext(), "combine_data_bus",
                           {numBanks, (uint64_t)width.getInt()}),
      TypeAttr::get(funcTy),
      SmallVector<Value>(
          {dataBusTensor, enableBus, forkedDataBus, newDataBusTensor}),
      tstart, offsetAttr);

  declareExternalFuncForCall(combineData,
                             {"original_data_bus_tensor", "en_bus_tensor",
                              "forked_data_bus", "new_data_bus_tensor"});

  auto params = builder.getDictionaryAttr(
      {builder.getNamedAttr("BANK", builder.getI64IntegerAttr(bank)),
       builder.getNamedAttr("NUM_BANKS", builder.getI64IntegerAttr(numBanks)),
       builder.getNamedAttr("ELEMENT_WIDTH", width)});
  combineData->setAttr("params", params);
  dataBusTensor = newDataBusTensor;
  return forkedDataBus;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// MemrefLoweringPass methods.
//------------------------------------------------------------------------------

LogicalResult MemrefLoweringPass::visitOp(hir::LoadOp op) {
  mlir::OpBuilder builder(op.getContext());
  builder.setInsertionPoint(op);
  auto memrefTy = op.mem().getType().dyn_cast<hir::MemrefType>();
  int64_t bank = helper::calcLinearIndex(op.filterIndices(BANK),
                                         memrefTy.filterShape(BANK))
                     .getValue();

  builder.create<hir::CommentOp>(op.getLoc(),
                                 builder.getStringAttr("LoadOp start"));
  if (!op.port().hasValue())
    return op.emitError() << "MemrefLoweringPass requires port number";
  auto &portInterface =
      mapMemrefToPortInterfaces.get(op.mem(), op.port().getValue());

  Value c1 = builder
                 .create<mlir::ConstantOp>(
                     builder.getUnknownLoc(), builder.getI1Type(),
                     builder.getIntegerAttr(builder.getIntegerType(1), 1))
                 .getResult();

  if (portInterface.addrEnableBusTensor) {
    assert(portInterface.addrDataBusTensor);
    auto addrEnableBus = getEnableSendBus(
        builder, portInterface.addrEnableBusTensor, bank,
        memrefTy.getNumBanks(), op.tstart(), op.offsetAttr(), "addr_en");
    builder.create<hir::CommentOp>(
        builder.getUnknownLoc(),
        builder.getStringAttr("Send 1 to the addrEnableBus."));

    builder
        .create<hir::SendOp>(builder.getUnknownLoc(), c1, addrEnableBus,
                             builder.getI64IntegerAttr(0), op.tstart(),
                             op.offsetAttr())
        ->setAttr("default", builder.getI64IntegerAttr(0));

    auto addrDataBus = getDataSendBus(
        builder, portInterface.addrDataBusTensor, addrEnableBus, bank,
        memrefTy.getNumBanks(), op.tstart(), op.offsetAttr(), "addr_data");

    builder.create<hir::CommentOp>(
        builder.getUnknownLoc(),
        builder.getStringAttr("Create a tuple of the addresses corresponding "
                              "to the ADDR dims."));
    Value addrTuple =
        createAddrTuple(builder, op.getLoc(), op.filterIndices(ADDR));

    builder.create<hir::CommentOp>(
        builder.getUnknownLoc(),
        builder.getStringAttr("Send the address tuple to the addrBus."));
    // Send the address tuple to the addrBus.
    builder.create<hir::SendOp>(op.getLoc(), addrTuple, addrDataBus,
                                builder.getI64IntegerAttr(0), op.tstart(),
                                op.offsetAttr());
  }

  assert(portInterface.rdLatency >= 0);
  IntegerAttr recvOffset =
      (portInterface.rdLatency == 0)
          ? op.offsetAttr()
          : builder.getI64IntegerAttr(op.offset().getValueOr(0) +
                                      portInterface.rdLatency);

  auto rdEnableBus = getEnableSendBus(builder, portInterface.rdEnableBusTensor,
                                      bank, memrefTy.getNumBanks(), op.tstart(),
                                      op.offsetAttr(), "rd_en");
  builder.create<hir::CommentOp>(
      builder.getUnknownLoc(),
      builder.getStringAttr("Send 1 to the rdEnableBus."));
  builder
      .create<hir::SendOp>(builder.getUnknownLoc(), c1, rdEnableBus,
                           builder.getI64IntegerAttr(0), op.tstart(),
                           op.offsetAttr())
      ->setAttr("default", builder.getI64IntegerAttr(0));

  builder.create<hir::CommentOp>(
      builder.getUnknownLoc(),
      builder.getStringAttr("Extract the rdDataBus for this use of memref."));
  auto rdDataBus = builder.create<hir::TensorExtractOp>(
      builder.getUnknownLoc(),
      getTensorElementType(portInterface.rdDataBusTensor),
      portInterface.rdDataBusTensor, op.filterIndices(BANK),
      builder.getStrArrayAttr({"send"}));
  helper::setNames(rdDataBus, {"rd_data_bus"});

  Operation *receiveOp = builder.create<hir::RecvOp>(
      op.getLoc(), op.res().getType(), rdDataBus, builder.getI64IntegerAttr(0),
      op.tstart(), recvOffset);

  if (op->hasAttrOfType<ArrayAttr>("names"))
    receiveOp->setAttr("names", op->getAttr("names"));
  op.replaceAllUsesWith(receiveOp);
  builder.create<hir::CommentOp>(op.getLoc(),
                                 builder.getStringAttr("LoadOp end"));
  opsToErase.push_back(op);
  return success();
}

LogicalResult MemrefLoweringPass::visitOp(hir::StoreOp op) {
  mlir::OpBuilder builder(op.getContext());
  builder.setInsertionPoint(op);
  auto memrefTy = op.mem().getType().dyn_cast<hir::MemrefType>();
  uint64_t bank = helper::calcLinearIndex(op.filterIndices(BANK),
                                          memrefTy.filterShape(BANK))
                      .getValue();

  builder.create<hir::CommentOp>(op.getLoc(),
                                 builder.getStringAttr("StoreOp start"));
  if (!op.port().hasValue())
    return op.emitError() << "MemrefLoweringPass requires port number";
  auto &portInterface =
      mapMemrefToPortInterfaces.get(op.mem(), op.port().getValue());

  Value c1 = builder
                 .create<mlir::ConstantOp>(
                     builder.getUnknownLoc(), builder.getI1Type(),
                     builder.getIntegerAttr(builder.getIntegerType(1), 1))
                 .getResult();

  if (portInterface.addrEnableBusTensor) {
    assert(portInterface.addrDataBusTensor);
    auto addrEnableBus = getEnableSendBus(
        builder, portInterface.addrEnableBusTensor, bank,
        memrefTy.getNumBanks(), op.tstart(), op.offsetAttr(), "addr_en");

    builder.create<hir::CommentOp>(
        builder.getUnknownLoc(),
        builder.getStringAttr("Send 1 to the addrEnableBus."));

    builder
        .create<hir::SendOp>(builder.getUnknownLoc(), c1, addrEnableBus,
                             builder.getI64IntegerAttr(0), op.tstart(),
                             op.offsetAttr())
        ->setAttr("default", builder.getI64IntegerAttr(0));

    auto addrDataBus = getDataSendBus(
        builder, portInterface.addrDataBusTensor, addrEnableBus, bank,
        memrefTy.getNumBanks(), op.tstart(), op.offsetAttr(), "addr_data");

    builder.create<hir::CommentOp>(
        builder.getUnknownLoc(),
        builder.getStringAttr("Create a tuple of the addresses corresponding "
                              "to the ADDR dims."));

    Value addrTuple =
        createAddrTuple(builder, op.getLoc(), op.filterIndices(ADDR));

    builder.create<hir::CommentOp>(
        builder.getUnknownLoc(),
        builder.getStringAttr("Send the address tuple to the addrBus."));
    builder.create<hir::SendOp>(op.getLoc(), addrTuple, addrDataBus,
                                builder.getI64IntegerAttr(0), op.tstart(),
                                op.offsetAttr());
  }
  auto wrEnableBus = getEnableSendBus(builder, portInterface.wrEnableBusTensor,
                                      bank, memrefTy.getNumBanks(), op.tstart(),
                                      op.offsetAttr(), "wr_en");
  builder.create<hir::CommentOp>(
      builder.getUnknownLoc(),
      builder.getStringAttr("Send 1 to the wrEnableBus."));
  builder
      .create<hir::SendOp>(builder.getUnknownLoc(), c1, wrEnableBus,
                           builder.getI64IntegerAttr(0), op.tstart(),
                           op.offsetAttr())
      ->setAttr("default", builder.getI64IntegerAttr(0));
  builder.create<hir::CommentOp>(builder.getUnknownLoc(),
                                 builder.getStringAttr("Create wrDataBus."));
  auto wrDataBus = getDataSendBus(
      builder, portInterface.wrDataBusTensor, portInterface.wrEnableBusTensor,
      bank, memrefTy.getNumBanks(), op.tstart(), op.offsetAttr(), "wr_data");

  builder.create<hir::CommentOp>(
      builder.getUnknownLoc(),
      builder.getStringAttr("Send data to wrDataBus."));
  builder.create<hir::SendOp>(builder.getUnknownLoc(), op.value(), wrDataBus,
                              builder.getI64IntegerAttr(0), op.tstart(),
                              op.offsetAttr());

  builder.create<hir::CommentOp>(op.getLoc(),
                                 builder.getStringAttr("StoreOp end"));
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
  auto sendAttr = helper::getDictionaryAttr(builder, "hir.bus.ports",
                                            builder.getStrArrayAttr({"send"}));
  auto recvAttr = helper::getDictionaryAttr(builder, "hir.bus.ports",
                                            builder.getStrArrayAttr({"recv"}));
  bool hasMemrefArg = false;
  for (uint64_t i = 0; i < funcTy.getInputTypes().size(); i++) {
    auto ty = funcTy.getInputTypes()[i];
    auto operand = op.operands()[i];
    if (!ty.isa<hir::MemrefType>()) {
      operands.push_back(operand);
      inputTypes.push_back(funcTy.getInputTypes()[i]);
      inputAttrs.push_back(funcTy.getInputAttrs()[i]);
      continue;
    }
    hasMemrefArg = true;
    auto numPorts = mapMemrefToPortInterfaces.getNumPorts(operand);
    auto ports = helper::extractMemrefPortsFromDict(funcTy.getInputAttrs()[i])
                     .getValue();
    assert(ports.size() == numPorts);

    for (uint64_t j = 0; j < ports.size(); j++) {
      auto &portInterface = mapMemrefToPortInterfaces.get(operand, j);
      if (portInterface.addrEnableBusTensor) {
        auto addrEnableBusTensor =
            getEnableSendBusTensor(builder, portInterface.addrEnableBusTensor,
                                   op.tstart(), op.offsetAttr(), "addr_en");
        operands.push_back(addrEnableBusTensor);
        inputTypes.push_back(addrEnableBusTensor.getType());
        inputAttrs.push_back(sendAttr);
      }
      if (portInterface.addrDataBusTensor) {
        auto addrDataBusTensor = getDataSendBusTensor(
            builder, portInterface.addrDataBusTensor, operands.back(),
            op.tstart(), op.offsetAttr(), "addr_data");
        operands.push_back(addrDataBusTensor);
        inputTypes.push_back(addrDataBusTensor.getType());
        inputAttrs.push_back(sendAttr);
      }
      if (portInterface.rdEnableBusTensor) {
        auto forkedRdEnableBusTensor =
            getEnableSendBusTensor(builder, portInterface.rdEnableBusTensor,
                                   op.tstart(), op.offsetAttr(), "rd_en");
        operands.push_back(forkedRdEnableBusTensor);
        inputTypes.push_back(forkedRdEnableBusTensor.getType());
        inputAttrs.push_back(sendAttr);
      }
      if (portInterface.rdDataBusTensor) {
        auto rdDataBusTensor = getDataSendBusTensor(
            builder, portInterface.rdDataBusTensor, operands.back(),
            op.tstart(), op.offsetAttr(), "rd_data");
        operands.push_back(rdDataBusTensor);
        inputTypes.push_back(rdDataBusTensor.getType());
        inputAttrs.push_back(recvAttr);
      }
      if (portInterface.wrEnableBusTensor) {
        auto forkedWrEnableBusTensor =
            getEnableSendBusTensor(builder, portInterface.wrEnableBusTensor,
                                   op.tstart(), op.offsetAttr(), "wr_en");
        operands.push_back(forkedWrEnableBusTensor);
        inputTypes.push_back(forkedWrEnableBusTensor.getType());
        inputAttrs.push_back(sendAttr);
      }
      if (portInterface.wrDataBusTensor) {
        auto wrDataBusTensor = getDataSendBusTensor(
            builder, portInterface.wrDataBusTensor, operands.back(),
            op.tstart(), op.offsetAttr(), "wr_data");
        operands.push_back(wrDataBusTensor);
        inputTypes.push_back(wrDataBusTensor.getType());
        inputAttrs.push_back(sendAttr);
      }
    }
  }
  if (!hasMemrefArg)
    return success();
  auto newFuncTy =
      hir::FuncType::get(builder.getContext(), inputTypes, inputAttrs,
                         funcTy.getResultTypes(), funcTy.getResultAttrs());
  auto newCallOp = builder.create<hir::CallOp>(
      op.getLoc(), op.getResultTypes(), op.calleeAttr(),
      TypeAttr::get(newFuncTy), operands, op.tstart(), op.offsetAttr());
  op.replaceAllUsesWith(newCallOp);
  opsToErase.push_back(op);
  return success();
}

LogicalResult MemrefLoweringPass::visitOp(hir::FuncExternOp op) {
  insertBusArguments(op);
  removeMemrefArguments(op);
  mapMemrefToPortInterfaces.clear();
  return success();
}

LogicalResult MemrefLoweringPass::visitOp(hir::AllocaOp op) {
  // Add comment.
  OpBuilder builder(op);
  builder.setInsertionPoint(op);
  builder.create<hir::CommentOp>(op.getLoc(),
                                 builder.getStringAttr("AllocaOp start"));
  if (failed(createBusInstantiationsAndCallOp(op)))
    return failure();
  builder.create<hir::CommentOp>(op.getLoc(),
                                 builder.getStringAttr("AllocaOp end"));
  opsToErase.push_back(op);
  return success();
}

LogicalResult MemrefLoweringPass::visitOp(hir::MemrefExtractOp op) {
  // port 0 of the op.res() is mapped to op.port() of op.mem().
  mapMemrefToPortInterfaces.remap(op.res(), op.mem(), op.port().getValue());

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
  mapMemrefToPortInterfaces.clear();
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
