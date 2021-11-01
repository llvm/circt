//=========- MemrefLoweringPass.cpp - Lower memref type---===//
//
// This file implements lowering pass for memref type.
//
//===----------------------------------------------------------------------===//

#include "MemrefLoweringUtils.h"
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

Type getTensorElementType(Value tensor) {
  auto ty = tensor.getType().dyn_cast<mlir::TensorType>();
  assert(ty);
  return ty.getElementType();
}

static mlir::FlatSymbolRefAttr createUniqueFuncName(MLIRContext *context,
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
                                Type busType) {
  return mlir::RankedTensorType::get(shape,
                                     hir::BusType::get(context, busType));
}

MemrefPortInterface MemrefLoweringPass::defineBusesForMemrefPort(
    OpBuilder &builder, hir::MemrefType memrefTy, Attribute port,
    llvm::Optional<StringRef> name) {
  MemrefPortInterface portInterface;

  auto *context = memrefTy.getContext();
  Type enableTy =
      buildBusTensor(context, memrefTy.filterShape(BANK), builder.getI1Type());

  SmallVector<Type> addrTupleElementTypes;
  uint64_t addrWidth = 0;
  for (uint64_t dimSize : memrefTy.filterShape(ADDR)) {
    addrTupleElementTypes.push_back(
        IntegerType::get(context, helper::clog2(dimSize)));
    addrWidth += helper::clog2(dimSize);
  }

  Type addrTupleTy =
      buildBusTensor(context, memrefTy.filterShape(BANK),
                     TupleType::get(context, addrTupleElementTypes));
  Type dataTy = buildBusTensor(context, memrefTy.filterShape(BANK),
                               memrefTy.getElementType());

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

  auto params = builder.getDictionaryAttr(
      {builder.getNamedAttr("ELEMENT_WIDTH",
                            builder.getI64IntegerAttr(elementWidth)),
       builder.getNamedAttr("ADDR_WIDTH", builder.getI64IntegerAttr(addrWidth)),
       builder.getNamedAttr("TENSOR_SIZE",
                            builder.getI64IntegerAttr(tensorSize))});

  callOp->setAttr("params", params);

  helper::declareExternalFuncForCall(callOp, inputBusNames);

  mapMemrefToPortInterfaces.insert(op.res(), memrefPortInterfaces);
  return success();
}

static void initUnconnectedEnBusTensor(OpBuilder &builder, Value busTensor,
                                       Value tstart) {
  if (!busTensor)
    return;
  auto busTy =
      busTensor.getType().dyn_cast<mlir::TensorType>().getElementType();
  auto dataTy =
      helper::getElementType(busTensor.getType()).dyn_cast<IntegerType>();
  assert(dataTy);
  auto c0 = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), IntegerAttr::get(dataTy, 0));
  auto zeroBus = builder.create<hir::BusOp>(builder.getUnknownLoc(), busTy);
  builder
      .create<hir::SendOp>(builder.getUnknownLoc(), c0, zeroBus, tstart,
                           builder.getI64IntegerAttr(0))
      ->setAttr("default", IntegerAttr::get(dataTy, 0));
  auto zeroBusTensor = builder.create<hir::BusBroadcastOp>(
      builder.getUnknownLoc(), busTensor.getType(), zeroBus);
  builder.create<hir::BusAssignOp>(builder.getUnknownLoc(), busTensor,
                                   zeroBusTensor);
}

void MemrefLoweringPass::initUnConnectedPorts(hir::FuncOp op) {
  OpBuilder builder(op);
  auto *returnOperation = &op.body().front().back();
  builder.setInsertionPoint(returnOperation);
  for (auto portInterface : mapMemrefToPortInterfaces.getAllPortInterfaces()) {
    initUnconnectedEnBusTensor(builder, portInterface.addrEnableBusTensor,
                               op.getRegionTimeVar());
    initUnconnectedEnBusTensor(builder, portInterface.rdEnableBusTensor,
                               op.getRegionTimeVar());
    initUnconnectedEnBusTensor(builder, portInterface.wrEnableBusTensor,
                               op.getRegionTimeVar());
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
    size_t loc, size_t port, Block &bb,
    SmallVectorImpl<std::string> &inputNames,
    SmallVectorImpl<DictionaryAttr> &inputAttrs, hir::MemrefType memrefTy,
    DictionaryAttr portDict, MemrefPortInterface &portInterface) {
  auto *context = memrefTy.getContext();
  Builder builder(context);
  Type enableTy = buildBusTensor(context, memrefTy.filterShape(BANK),
                                 IntegerType::get(context, 1));

  SmallVector<Type> addrTupleElementTypes;

  for (uint64_t dimSize : memrefTy.filterShape(ADDR)) {
    addrTupleElementTypes.push_back(
        IntegerType::get(context, helper::clog2(dimSize)));
  }

  Type addrTupleTy =
      buildBusTensor(context, memrefTy.filterShape(BANK),
                     TupleType::get(context, addrTupleElementTypes));
  Type dataTy = buildBusTensor(context, memrefTy.filterShape(BANK),
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
    portInterface.addrDataBusTensor = bb.insertArgument(loc, addrTupleTy);
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
  helper::declareExternalFuncForCall(
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

  helper::declareExternalFuncForCall(
      combineDataBus, {"original_data_bus_tensor", "en_bus_tensor",
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

  helper::declareExternalFuncForCall(
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

  helper::declareExternalFuncForCall(
      combineData, {"original_data_bus_tensor", "en_bus_tensor",
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
  auto &portInterface =
      mapMemrefToPortInterfaces.get(op.mem(), op.port().getValue());

  mlir::OpBuilder builder(op.getContext());
  builder.setInsertionPoint(op);

  // Insert logic to send address and valid signals to address bus.
  Value addrTuple =
      createAddrTuple(builder, op.getLoc(), op.filterIndices(ADDR));

  if (portInterface.addrDataBusTensor) {
    assert(portInterface.addrEnableBusTensor);
    portInterface.addrEnableBusTensor = insertEnableSendLogic(
        builder, portInterface.addrEnableBusTensor, op.tstart(),
        op.offsetAttr(), op.filterIndices(BANK));
    portInterface.addrDataBusTensor = insertDataSendLogic(
        builder, addrTuple, portInterface.addrDataBusTensor, op.tstart(),
        op.offsetAttr(), op.filterIndices(BANK));
  }
  // Insert logic to send rd valid signal and receive the data after rdLatency.
  assert(portInterface.rdEnableBusTensor);
  portInterface.rdEnableBusTensor = insertEnableSendLogic(
      builder, portInterface.rdEnableBusTensor, op.tstart(), op.offsetAttr(),
      op.filterIndices(BANK));

  auto loadValue = insertDataRecvLogic(
      builder, op.getLoc(), op->getAttrOfType<mlir::ArrayAttr>("names"),
      portInterface.rdLatency, portInterface.rdDataBusTensor, op.tstart(),
      op.offsetAttr(), op.filterIndices(BANK));

  op.replaceAllUsesWith(loadValue);
  opsToErase.push_back(op);
  return success();
}

LogicalResult MemrefLoweringPass::visitOp(hir::StoreOp op) {
  auto &portInterface =
      mapMemrefToPortInterfaces.get(op.mem(), op.port().getValue());

  mlir::OpBuilder builder(op.getContext());
  builder.setInsertionPoint(op);

  // Insert logic to send address and valid signals to address bus.
  Value addrTuple =
      createAddrTuple(builder, op.getLoc(), op.filterIndices(ADDR));

  if (portInterface.addrDataBusTensor) {
    assert(portInterface.addrEnableBusTensor);
    portInterface.addrEnableBusTensor = insertEnableSendLogic(
        builder, portInterface.addrEnableBusTensor, op.tstart(),
        op.offsetAttr(), op.filterIndices(BANK));
    portInterface.addrDataBusTensor = insertDataSendLogic(
        builder, addrTuple, portInterface.addrDataBusTensor, op.tstart(),
        op.offsetAttr(), op.filterIndices(BANK));
  }
  // Insert logic to send wr valid and data.
  assert(portInterface.wrEnableBusTensor);
  portInterface.wrEnableBusTensor = insertEnableSendLogic(
      builder, portInterface.wrEnableBusTensor, op.tstart(), op.offsetAttr(),
      op.filterIndices(BANK));

  portInterface.wrDataBusTensor =
      insertDataSendLogic(builder, op.value(), portInterface.wrDataBusTensor,
                          op.tstart(), op.offsetAttr(), op.filterIndices(BANK));

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
  if (failed(createBusInstantiationsAndCallOp(op)))
    return failure();
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
