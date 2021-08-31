//=========- MemrefLoweringPass.cpp - Lower memref type---===//
//
// This file implements lowering pass for memref type.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include <iostream>
using namespace circt;
using namespace hir;
namespace {
enum MemrefPortKind { RD = 0, WR = 1, RW = 2 };
struct MemrefPortInterface {
  Value addrDataBusTensor;
  Value addrEnableBusTensor;
  Value rdEnableBusTensor;
  int64_t rdLatency = -1;
  Value rdDataBusTensor;
  Value wrEnableBusTensor;
  Value wrDataBusTensor;
};

class MultiDimCounter {
public:
  MultiDimCounter(ArrayRef<int64_t> shape) : shape(shape) {
    indices = SmallVector<uint64_t>(shape.size(), 0);
  }
  ArrayRef<uint64_t> getIndices() { return indices; }
  LogicalResult inc() {
    if (indices.size() == 0)
      return failure();
    return inc(indices.size() - 1);
  }
  uint64_t getLinearIdx() {
    if (indices.size() == 0)
      return 0;
    uint64_t linearIdx = indices[indices.size() - 1];
    for (int i = indices.size() - 2; i >= 0; i--) {
      linearIdx += indices[i] * (uint64_t)shape[i + 1];
    }
    uint64_t size = 1;
    for (auto s : shape) {
      size *= s;
    }
    if (linearIdx == 12)
      assert(linearIdx < size);
    return linearIdx;
  }

private:
  LogicalResult inc(uint64_t dim) {
    assert(dim < indices.size());
    assert(shape[dim] > 0);
    if (indices[dim] < (uint64_t)shape[dim] - 1) {
      indices[dim]++;
      return success();
    }

    if (dim == 0)
      return failure();
    indices[dim] = 0;
    return inc(dim - 1);
  }

  ArrayRef<int64_t> shape;
  SmallVector<uint64_t> indices;
};

/// This class lowers a hir::MemrefType to multiple buses.
/// For each addr, read and write, there are two buses - data_bus and en_bus.
class MemrefLoweringPass : public hir::MemrefLoweringBase<MemrefLoweringPass> {
public:
  void runOnOperation() override;

private:
  void insertBusArguments();
  LogicalResult visitOp(hir::FuncOp);
  LogicalResult visitOp(hir::AllocaOp);
  LogicalResult visitOp(hir::LoadOp);
  LogicalResult visitOp(hir::StoreOp);
  LogicalResult visitOp(hir::CallOp);
  LogicalResult visitOp(hir::MemrefExtractOp);
  Value getEnableSendBusTensor(OpBuilder &, Value &, Value, IntegerAttr,
                               std::string);
  Value getDataSendBusTensor(OpBuilder &, Value &, Value, Value, IntegerAttr,
                             std::string);
  Value getEnableSendBus(OpBuilder &, Value &, uint64_t, Value, IntegerAttr,
                         std::string);
  Value getDataSendBus(OpBuilder &, Value &, Value, uint64_t, Value,
                       IntegerAttr, std::string);
  void defineBusesForMemrefPort(OpBuilder &, hir::MemrefType, Attribute,
                                MemrefPortInterface &);
  LogicalResult createBusInstantiationsAndCallOp(
      hir::AllocaOp, DenseMap<Value, SmallVector<MemrefPortInterface>> &,
      Value);

private:
  DenseMap<Value, SmallVector<MemrefPortInterface>> mapMemrefPortToBuses;
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

void MemrefLoweringPass::defineBusesForMemrefPort(
    OpBuilder &builder, hir::MemrefType memrefTy, Attribute port,
    MemrefPortInterface &portInterface) {

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
        topLevelBuilder->create<hir::BusInstantiateOp>(builder.getUnknownLoc(),
                                                       enableTy);
    portInterface.addrDataBusTensor =
        topLevelBuilder->create<hir::BusInstantiateOp>(builder.getUnknownLoc(),
                                                       addrTupleTy);
  }

  if (auto rdLatency = helper::getRdLatency(port)) {
    portInterface.rdEnableBusTensor =
        topLevelBuilder->create<hir::BusInstantiateOp>(builder.getUnknownLoc(),
                                                       enableTy);
    portInterface.rdDataBusTensor =
        topLevelBuilder->create<hir::BusInstantiateOp>(builder.getUnknownLoc(),
                                                       dataTy);
    portInterface.rdLatency = rdLatency.getValue();
  }

  if (helper::isWrite(port)) {
    portInterface.wrEnableBusTensor =
        topLevelBuilder->create<hir::BusInstantiateOp>(builder.getUnknownLoc(),
                                                       enableTy);
    portInterface.wrDataBusTensor =
        topLevelBuilder->create<hir::BusInstantiateOp>(builder.getUnknownLoc(),
                                                       dataTy);
  }
}
LogicalResult MemrefLoweringPass::createBusInstantiationsAndCallOp(
    hir::AllocaOp op,
    DenseMap<Value, SmallVector<MemrefPortInterface>> &mapMemrefPortToBuses,
    Value tstartRegion) {
  hir::MemrefType memrefTy = op.getType().dyn_cast<hir::MemrefType>();
  ArrayAttr ports = op.ports();
  OpBuilder builder(op);
  builder.setInsertionPoint(op);
  SmallVector<MemrefPortInterface> memrefPortInterfaces;

  for (auto port : ports) {
    MemrefPortInterface portInterface;
    defineBusesForMemrefPort(builder, memrefTy, port, portInterface);
    memrefPortInterfaces.push_back(portInterface);
  }

  SmallVector<Value> inputBuses;
  SmallVector<DictionaryAttr> inputBusAttrs;
  SmallVector<Type> inputBusTypes;
  auto sendAttr = helper::getDictionaryAttr(builder, "hir.bus.ports",
                                            builder.getStrArrayAttr({"send"}));
  auto recvAttr = helper::getDictionaryAttr(builder, "hir.bus.ports",
                                            builder.getStrArrayAttr({"recv"}));
  for (auto portInterface : memrefPortInterfaces) {
    if (portInterface.addrEnableBusTensor) {
      inputBuses.push_back(portInterface.addrEnableBusTensor);
      inputBusTypes.push_back(portInterface.addrEnableBusTensor.getType());
      inputBusAttrs.push_back(recvAttr);
      inputBuses.push_back(portInterface.addrDataBusTensor);
      inputBusTypes.push_back(portInterface.addrDataBusTensor.getType());
      inputBusAttrs.push_back(recvAttr);
    }

    if (portInterface.rdEnableBusTensor) {
      inputBuses.push_back(portInterface.rdEnableBusTensor);
      inputBusTypes.push_back(portInterface.rdEnableBusTensor.getType());
      inputBusAttrs.push_back(recvAttr);
      inputBuses.push_back(portInterface.rdDataBusTensor);
      inputBusTypes.push_back(portInterface.rdDataBusTensor.getType());
      inputBusAttrs.push_back(sendAttr);
    }

    if (portInterface.wrEnableBusTensor) {
      inputBuses.push_back(portInterface.wrEnableBusTensor);
      inputBusTypes.push_back(portInterface.wrEnableBusTensor.getType());
      inputBusAttrs.push_back(recvAttr);
      inputBuses.push_back(portInterface.wrDataBusTensor);
      inputBusTypes.push_back(portInterface.wrDataBusTensor.getType());
      inputBusAttrs.push_back(recvAttr);
    }
  }
  Type funcTy = hir::FuncType::get(builder.getContext(), inputBusTypes,
                                   inputBusAttrs, {}, {});

  auto callOp = builder.create<hir::CallOp>(
      builder.getUnknownLoc(), SmallVector<Type>(),
      FlatSymbolRefAttr::get(builder.getContext(), op.mem_type()),
      TypeAttr::get(funcTy), inputBuses, tstartRegion, IntegerAttr());

  callOp->setAttr(
      "ELEMENT_WIDTH",
      builder.getI64IntegerAttr(
          helper::getBitWidth(memrefTy.getElementType()).getValue()));

  callOp->setAttr("ADDR_WIDTH", builder.getI64IntegerAttr(helper::clog2(
                                    memrefTy.getNumElementsPerBank())));
  callOp->setAttr("NUM_ELEMENTS",
                  builder.getI64IntegerAttr(memrefTy.getNumElementsPerBank()));

  mapMemrefPortToBuses[op.res()] = memrefPortInterfaces;
  return success();
}

SmallVector<Value> filterMemrefArgs(Block::BlockArgListType args) {
  SmallVector<Value> memrefArgs;
  for (auto arg : args)
    if (arg.getType().isa<hir::MemrefType>())
      memrefArgs.push_back(arg);
  return memrefArgs;
}

size_t insertBusTypesAndAttrsForMemrefPort(
    size_t i, Block &bb, SmallVectorImpl<DictionaryAttr> &inputAttrs,
    hir::MemrefType memrefTy, DictionaryAttr port,
    MemrefPortInterface &portInterface) {
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

  if (memrefTy.getNumElementsPerBank() > 1) {
    portInterface.addrEnableBusTensor = bb.insertArgument(i, enableTy);
    inputAttrs.insert(inputAttrs.begin() + i++, sendAttr);
    portInterface.addrDataBusTensor = bb.insertArgument(i, addrTupleTy);
    inputAttrs.insert(inputAttrs.begin() + i++, sendAttr);
  }

  if (auto rdLatency = helper::getRdLatency(port)) {
    portInterface.rdEnableBusTensor = bb.insertArgument(i, enableTy);
    inputAttrs.insert(inputAttrs.begin() + i++, sendAttr);
    portInterface.rdDataBusTensor = bb.insertArgument(i, dataTy);
    inputAttrs.insert(inputAttrs.begin() + i++, recvAttr);
    portInterface.rdLatency = rdLatency.getValue();
  }

  if (helper::isWrite(port)) {
    portInterface.wrEnableBusTensor = bb.insertArgument(i, enableTy);
    inputAttrs.insert(inputAttrs.begin() + i++, sendAttr);
    portInterface.wrDataBusTensor = bb.insertArgument(i, dataTy);
    inputAttrs.insert(inputAttrs.begin() + i++, sendAttr);
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

void MemrefLoweringPass::insertBusArguments() {

  hir::FuncOp op = getOperation();
  auto funcTy = op.funcTy().dyn_cast<hir::FuncType>();
  auto &bb = op.getFuncBody().front();
  SmallVector<DictionaryAttr> inputAttrs;

  // initialize with old attrs.
  for (auto attr : funcTy.getInputAttrs())
    inputAttrs.push_back(attr);

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
            insertBefore, bb, inputAttrs, memrefTy, portDict, portInterface);
        memrefPortInterfaces.push_back(portInterface);
      }
      mapMemrefPortToBuses[arg] = memrefPortInterfaces;
    }
  }
  op.updateArguments(inputAttrs);
}

void removeMemrefArguments(hir::FuncOp op) {
  auto funcTy = op.funcTy().dyn_cast<hir::FuncType>();
  SmallVector<Type> inputTypes;
  SmallVector<DictionaryAttr> inputAttrs;
  auto &bb = op.getFuncBody().front();

  // initialize with old attrs.
  for (auto attr : funcTy.getInputAttrs())
    inputAttrs.push_back(attr);

  for (int i = bb.getNumArguments() - 2 /*last arg is timetype*/; i >= 0; i--) {
    if (bb.getArgument(i).getType().isa<hir::MemrefType>()) {
      bb.eraseArgument(i);
      inputAttrs.erase(inputAttrs.begin() + i);
    } else {
    }
  }
  op.updateArguments(inputAttrs);
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
void initUnconnectedValidBus(OpBuilder &builder, Value busTensor,
                             ArrayRef<uint64_t> indices) {
  if (!busTensor)
    return;

  assert(busTensor.getType()
             .dyn_cast<mlir::TensorType>()
             .getElementType()
             .dyn_cast<hir::BusType>()
             .getElementTypes()
             .size() == 1);

  Value validBus = extractBusFromTensor(builder, busTensor, indices,
                                        builder.getStrArrayAttr({"send"}));
  builder.create<hir::AssignOp>(builder.getUnknownLoc(),
                                builder.getI64IntegerAttr(0), validBus);
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

  auto callOp = builder.create<hir::CallOp>(
      builder.getUnknownLoc(), SmallVector<Type>(),
      FlatSymbolRefAttr::get(builder.getContext(), "hir_valid_combine"),
      TypeAttr::get(funcTy), SmallVector<Value>({bus, busTensor}), tstartRegion,
      IntegerAttr());
  auto tensorTy = busTensor.getType().dyn_cast<mlir::TensorType>();
  callOp->setAttr("TENSOR_SIZE",
                  builder.getI64IntegerAttr(tensorTy.getNumElements()));
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

  auto callOp = builder.create<hir::CallOp>(
      builder.getUnknownLoc(), SmallVector<Type>(),
      FlatSymbolRefAttr::get(builder.getContext(), "hir_bus_broadcast"),
      TypeAttr::get(funcTy),
      SmallVector<Value>({destBusTensor, destBusValidTensor, sourceBus}),
      tstartRegion, IntegerAttr());

  auto tensorSize =
      destBusTensor.getType().dyn_cast<mlir::TensorType>().getNumElements();
  auto elementWidth = helper::getBitWidth(
      destBusTensor.getType().dyn_cast<mlir::TensorType>().getElementType());

  callOp->setAttr("TENSOR_SIZE", builder.getI64IntegerAttr(tensorSize));
  callOp->setAttr("ELEMENT_WIDTH",
                  builder.getI64IntegerAttr(elementWidth.getValue()));
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

  auto callOp = builder.create<hir::CallOp>(
      builder.getUnknownLoc(), SmallVector<Type>(),
      FlatSymbolRefAttr::get(builder.getContext(), "hir_bus_mux"),
      TypeAttr::get(funcTy),
      SmallVector<Value>({bus, busValidTensor, busTensor}), tstartRegion,
      IntegerAttr());
  auto tensorSize =
      busTensor.getType().dyn_cast<mlir::TensorType>().getNumElements();
  auto elementWidth = helper::getBitWidth(
      busTensor.getType().dyn_cast<mlir::TensorType>().getElementType());

  callOp->setAttr("TENSOR_SIZE", builder.getI64IntegerAttr(tensorSize));
  callOp->setAttr("ELEMENT_WIDTH",
                  builder.getI64IntegerAttr(elementWidth.getValue()));
}

Value MemrefLoweringPass::getEnableSendBusTensor(OpBuilder &builder,
                                                 Value &originalEnableBusTensor,
                                                 Value tstart,
                                                 IntegerAttr offsetAttr,
                                                 std::string name) {
  auto forkedEnableBusTensor = topLevelBuilder->create<hir::BusInstantiateOp>(
      builder.getUnknownLoc(), originalEnableBusTensor.getType());
  helper::setNames(forkedEnableBusTensor, {name + "_forked"});
  auto newEnableBusTensor = topLevelBuilder->create<hir::BusInstantiateOp>(
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

  auto combineEnBus = builder.create<hir::CallOp>(
      builder.getUnknownLoc(), SmallVector<Type>(),
      FlatSymbolRefAttr::get(builder.getContext(), "combine_en_bus_tensor"),
      TypeAttr::get(funcTy),
      SmallVector<Value>(
          {originalEnableBusTensor, forkedEnableBusTensor, newEnableBusTensor}),
      tstart, offsetAttr);

  auto width = builder.getI64IntegerAttr(
      helper::getBitWidth(originalEnableBusTensor.getType()
                              .dyn_cast<mlir::TensorType>()
                              .getElementType())
          .getValue());
  auto numBanks = builder.getI64IntegerAttr(originalEnableBusTensor.getType()
                                                .dyn_cast<mlir::TensorType>()
                                                .getNumElements());
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
  auto forkedDataBusTensor = topLevelBuilder->create<hir::BusInstantiateOp>(
      builder.getUnknownLoc(), originalDataBusTensor.getType());
  helper::setNames(forkedDataBusTensor, {name + "_forked"});
  auto newDataBusTensor = topLevelBuilder->create<hir::BusInstantiateOp>(
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

  auto combineDataBus = builder.create<hir::CallOp>(
      builder.getUnknownLoc(), SmallVector<Type>(),
      FlatSymbolRefAttr::get(builder.getContext(), "combine_data_bus_tensor"),
      TypeAttr::get(funcTy),
      SmallVector<Value>({originalDataBusTensor, enableBusTensor,
                          forkedDataBusTensor, newDataBusTensor}),
      tstart, offsetAttr);

  auto width = builder.getI64IntegerAttr(
      helper::getBitWidth(originalDataBusTensor.getType()
                              .dyn_cast<mlir::TensorType>()
                              .getElementType())
          .getValue());
  auto numBanks = builder.getI64IntegerAttr(originalDataBusTensor.getType()
                                                .dyn_cast<mlir::TensorType>()
                                                .getNumElements());
  auto params =
      builder.getDictionaryAttr({builder.getNamedAttr("WIDTH", width),
                                 builder.getNamedAttr("NUM_BANKS", numBanks)});
  combineDataBus->setAttr("params", params);
  originalDataBusTensor = newDataBusTensor;
  return forkedDataBusTensor;
}

Value MemrefLoweringPass::getEnableSendBus(OpBuilder &builder,
                                           Value &enableBusTensor,
                                           uint64_t bank, Value tstart,
                                           IntegerAttr offsetAttr,
                                           std::string name) {
  auto forkedEnableBus = topLevelBuilder->create<hir::BusInstantiateOp>(
      builder.getUnknownLoc(), getTensorElementType(enableBusTensor));
  helper::setNames(forkedEnableBus, {name + "_forked"});
  auto newEnableBusTensor = topLevelBuilder->create<hir::BusInstantiateOp>(
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
      FlatSymbolRefAttr::get(builder.getContext(), "combine_en_bus"),
      TypeAttr::get(funcTy),
      SmallVector<Value>(
          {enableBusTensor, forkedEnableBus, newEnableBusTensor}),
      tstart, offsetAttr);
  combineEnBus->setAttr("BANK", builder.getI64IntegerAttr(bank));
  enableBusTensor = newEnableBusTensor;
  return forkedEnableBus;
}

Value MemrefLoweringPass::getDataSendBus(OpBuilder &builder,
                                         Value &dataBusTensor,
                                         Value enableBusTensor, uint64_t bank,
                                         Value tstart, IntegerAttr offsetAttr,
                                         std::string name) {

  auto forkedDataBus = topLevelBuilder->create<hir::BusInstantiateOp>(
      builder.getUnknownLoc(), getTensorElementType(dataBusTensor));
  helper::setNames(forkedDataBus, {name + "_forked"});

  auto newDataBusTensor = topLevelBuilder->create<hir::BusInstantiateOp>(
      builder.getUnknownLoc(), dataBusTensor.getType());
  helper::setNames(newDataBusTensor, {name + "_buses"});
  auto sendAttr = helper::getDictionaryAttr(builder, "hir.bus.ports",
                                            builder.getStrArrayAttr({"send"}));
  auto recvAttr = helper::getDictionaryAttr(builder, "hir.bus.ports",
                                            builder.getStrArrayAttr({"recv"}));
  auto funcTy =
      hir::FuncType::get(builder.getContext(),
                         {dataBusTensor.getType(), enableBusTensor.getType(),
                          forkedDataBus.getType(), newDataBusTensor.getType()},
                         {sendAttr, recvAttr, recvAttr, recvAttr}, {}, {});

  auto combineData = builder.create<hir::CallOp>(
      builder.getUnknownLoc(), SmallVector<Type>(),
      FlatSymbolRefAttr::get(builder.getContext(), "combine_data_bus"),
      TypeAttr::get(funcTy),
      SmallVector<Value>(
          {dataBusTensor, enableBusTensor, forkedDataBus, newDataBusTensor}),
      tstart, offsetAttr);
  combineData->setAttr("BANK", builder.getI64IntegerAttr(bank));
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
  uint64_t bank = helper::calcLinearIndex(op.filterIndices(BANK),
                                          memrefTy.filterShape(BANK))
                      .getValue();

  builder.create<hir::CommentOp>(op.getLoc(),
                                 builder.getStringAttr("LoadOp start"));
  if (!op.port().hasValue())
    return op.emitError() << "MemrefLoweringPass requires port number";
  auto &portInterface = mapMemrefPortToBuses[op.mem()][op.port().getValue()];

  Value c1 = builder
                 .create<mlir::ConstantOp>(
                     builder.getUnknownLoc(), builder.getI1Type(),
                     builder.getIntegerAttr(builder.getIntegerType(1), 1))
                 .getResult();

  if (portInterface.addrEnableBusTensor) {
    assert(portInterface.addrDataBusTensor);
    auto addrEnableBus =
        getEnableSendBus(builder, portInterface.addrEnableBusTensor, bank,
                         op.tstart(), op.offsetAttr(), "addr_en");
    builder.create<hir::CommentOp>(
        builder.getUnknownLoc(),
        builder.getStringAttr("Send 1 to the addrEnableBus."));

    builder.create<hir::SendOp>(builder.getUnknownLoc(), c1, addrEnableBus,
                                builder.getI64IntegerAttr(0), op.tstart(),
                                op.offsetAttr());

    auto addrDataBus =
        getDataSendBus(builder, portInterface.addrDataBusTensor,
                       portInterface.addrEnableBusTensor, bank, op.tstart(),
                       op.offsetAttr(), "addr_data");

    builder.create<hir::CommentOp>(
        builder.getUnknownLoc(),
        builder.getStringAttr(
            "Create a tuple of the addresses corresponding to the ADDR dims."));
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

  auto rdEnableBus =
      getEnableSendBus(builder, portInterface.rdEnableBusTensor, bank,
                       op.tstart(), op.offsetAttr(), "rd_en");
  builder.create<hir::CommentOp>(
      builder.getUnknownLoc(),
      builder.getStringAttr("Send 1 to the rdEnableBus."));
  builder.create<hir::SendOp>(builder.getUnknownLoc(), c1, rdEnableBus,
                              builder.getI64IntegerAttr(0), op.tstart(),
                              op.offsetAttr());

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
  auto &portInterface = mapMemrefPortToBuses[op.mem()][op.port().getValue()];

  Value c1 = builder
                 .create<mlir::ConstantOp>(
                     builder.getUnknownLoc(), builder.getI1Type(),
                     builder.getIntegerAttr(builder.getIntegerType(1), 1))
                 .getResult();

  if (portInterface.addrEnableBusTensor) {
    assert(portInterface.addrDataBusTensor);
    auto addrEnableBus =
        getEnableSendBus(builder, portInterface.addrEnableBusTensor, bank,
                         op.tstart(), op.offsetAttr(), "addr_en");

    builder.create<hir::CommentOp>(
        builder.getUnknownLoc(),
        builder.getStringAttr("Send 1 to the addrEnableBus."));

    builder.create<hir::SendOp>(builder.getUnknownLoc(), c1, addrEnableBus,
                                builder.getI64IntegerAttr(0), op.tstart(),
                                op.offsetAttr());

    auto addrDataBus =
        getDataSendBus(builder, portInterface.addrDataBusTensor,
                       portInterface.addrEnableBusTensor, bank, op.tstart(),
                       op.offsetAttr(), "addr_data");

    builder.create<hir::CommentOp>(
        builder.getUnknownLoc(),
        builder.getStringAttr(
            "Create a tuple of the addresses corresponding to the ADDR dims."));

    Value addrTuple =
        createAddrTuple(builder, op.getLoc(), op.filterIndices(ADDR));

    builder.create<hir::CommentOp>(
        builder.getUnknownLoc(),
        builder.getStringAttr("Send the address tuple to the addrBus."));
    builder.create<hir::SendOp>(op.getLoc(), addrTuple, addrDataBus,
                                builder.getI64IntegerAttr(0), op.tstart(),
                                op.offsetAttr());
  }
  auto wrEnableBus =
      getEnableSendBus(builder, portInterface.wrEnableBusTensor, bank,
                       op.tstart(), op.offsetAttr(), "wr_en");
  builder.create<hir::CommentOp>(
      builder.getUnknownLoc(),
      builder.getStringAttr("Send 1 to the wrEnableBus."));
  builder.create<hir::SendOp>(builder.getUnknownLoc(), c1, wrEnableBus,
                              builder.getI64IntegerAttr(0), op.tstart(),
                              op.offsetAttr());
  builder.create<hir::CommentOp>(builder.getUnknownLoc(),
                                 builder.getStringAttr("Create wrDataBus."));
  auto wrDataBus = getDataSendBus(builder, portInterface.wrDataBusTensor,
                                  portInterface.wrEnableBusTensor, bank,
                                  op.tstart(), op.offsetAttr(), "wr_data");

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
  for (uint64_t i = 0; i < funcTy.getInputTypes().size(); i++) {
    auto ty = funcTy.getInputTypes()[i];
    auto operand = op.operands()[i];
    if (!ty.isa<hir::MemrefType>()) {
      operands.push_back(operand);
      inputTypes.push_back(funcTy.getInputTypes()[i]);
      inputAttrs.push_back(funcTy.getInputAttrs()[i]);
      continue;
    }
    auto &memrefPortInterfaces = mapMemrefPortToBuses[operand];
    auto ports = helper::extractMemrefPortsFromDict(funcTy.getInputAttrs()[i])
                     .getValue();
    assert(ports.size() == memrefPortInterfaces.size());

    for (uint64_t j = 0; j < ports.size(); j++) {
      auto &portInterface = memrefPortInterfaces[j];
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

LogicalResult MemrefLoweringPass::visitOp(hir::AllocaOp op) {
  Value tstartRegion = op->getParentRegion()->getArguments().back();
  // Add comment.
  OpBuilder builder(op);
  builder.setInsertionPoint(op);
  builder.create<hir::CommentOp>(op.getLoc(),
                                 builder.getStringAttr("AllocaOp start"));
  if (failed(createBusInstantiationsAndCallOp(op, mapMemrefPortToBuses,
                                              tstartRegion)))
    return failure();
  builder.create<hir::CommentOp>(op.getLoc(),
                                 builder.getStringAttr("AllocaOp end"));
  opsToErase.push_back(op);
  return success();
}

LogicalResult MemrefLoweringPass::visitOp(hir::MemrefExtractOp op) {
  SmallVector<MemrefPortInterface> memrefPortInterfaces;
  MemrefPortInterface portInterface;
  MemrefPortInterface originalPortInterface =
      mapMemrefPortToBuses[op.mem()][op.port().getValue()];

  portInterface.addrEnableBusTensor = originalPortInterface.addrEnableBusTensor;
  portInterface.addrDataBusTensor = originalPortInterface.addrDataBusTensor;

  if (helper::isRead(op.ports()[0])) {
    portInterface.rdEnableBusTensor = originalPortInterface.rdEnableBusTensor;
    portInterface.rdDataBusTensor = originalPortInterface.rdDataBusTensor;
  }
  if (helper::isWrite(op.ports()[0])) {
    portInterface.wrEnableBusTensor = originalPortInterface.wrEnableBusTensor;
    portInterface.wrDataBusTensor = originalPortInterface.wrDataBusTensor;
  }

  // port 0 of the op.res() is mapped to op.port() of op.mem().
  memrefPortInterfaces.push_back(
      mapMemrefPortToBuses[op.mem()][op.port().getValue()]);
  mapMemrefPortToBuses[op.res()] = memrefPortInterfaces;
  opsToErase.push_back(op);
  return success();
}

void MemrefLoweringPass::runOnOperation() {
  hir::FuncOp funcOp = getOperation();
  topLevelBuilder = new OpBuilder(funcOp);
  topLevelBuilder->setInsertionPointToStart(&funcOp.body().front());
  insertBusArguments();
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
    signalPassFailure();
    return;
  }
  helper::eraseOps(opsToErase);
  // removeMemrefArguments(funcOp);
}

namespace circt {
namespace hir {
std::unique_ptr<OperationPass<hir::FuncOp>> createMemrefLoweringPass() {
  return std::make_unique<MemrefLoweringPass>();
}
} // namespace hir
} // namespace circt
