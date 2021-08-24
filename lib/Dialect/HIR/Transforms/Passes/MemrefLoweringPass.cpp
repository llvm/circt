//=========- MemrefLoweringPass.cpp - Lower memref type---===//
//
// This file implements lowering pass for memref type.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"
#include "circt/Dialect/HIR/Analysis/FanoutAnalysis.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include <iostream>
using namespace circt;
using namespace hir;
namespace {
enum MemrefPortKind { RD = 0, WR = 1, RW = 2 };
struct MemrefPortInterface {
  Value addrBus;
  Value addrEnableBus;
  Value rdEnableBus;
  int64_t rdLatency = -1;
  Value rdDataBus;
  Value wrEnableBus;
  Value wrDataBus;
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
/// This pass lowers a hir::MemrefType to hir::BusType.
/// RD port becomes three separate tensors of (send
/// addr, send rd_valid, recv rd_data).
/// WR port becomes tensors of (send addr, send wr_valid, send wr_data).
/// RW port becomes tensors of (send addr, send wr_valid, send
/// wr_data, send rd_valid, recv rd_data)
class MemrefLoweringPass : public hir::MemrefLoweringBase<MemrefLoweringPass> {
public:
  void runOnOperation() override;

private:
  LogicalResult replaceMemrefArgUses(hir::FuncOp op);

  LogicalResult replacePortUses(OpBuilder &builder, ArrayRef<int64_t> bankShape,
                                MemrefPortInterface portInterface,
                                SmallVector<ListOfUses> uses);
  LogicalResult
  replaceMemrefUses(OpBuilder &builder, Value memref,
                    ArrayRef<MemrefPortInterface> memrefPortInterface,
                    SmallVector<SmallVector<ListOfUses>> &memrefUses);

private:
  LogicalResult visitRegion(Region &);
  LogicalResult dispatch(Operation *);
  LogicalResult visitOp(hir::FuncOp);
  LogicalResult visitOp(hir::AllocaOp);
  LogicalResult visitOp(hir::ForOp);
  LogicalResult visitOp(hir::WhileOp);

private:
  DenseMap<Value, SmallVector<MemrefPortInterface>> mapMemrefPortToBuses;
  llvm::DenseMap<Value, SmallVector<SmallVector<ListOfUses>>> *uses;
  Value tstartRegion;
  SmallVector<Operation *, 10> opsToErase;
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

void defineBusesForMemrefPort(OpBuilder &builder, hir::MemrefType memrefTy,
                              Attribute port,
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
    portInterface.addrEnableBus = builder.create<hir::BusInstantiateOp>(
        builder.getUnknownLoc(), enableTy);
    portInterface.addrBus = builder.create<hir::BusInstantiateOp>(
        builder.getUnknownLoc(), addrTupleTy);
  }

  if (auto rdLatency = helper::getRdLatency(port)) {
    portInterface.rdEnableBus = builder.create<hir::BusInstantiateOp>(
        builder.getUnknownLoc(), enableTy);
    portInterface.rdDataBus =
        builder.create<hir::BusInstantiateOp>(builder.getUnknownLoc(), dataTy);
    portInterface.rdLatency = rdLatency.getValue();
  }

  if (helper::isWrite(port)) {
    portInterface.wrEnableBus = builder.create<hir::BusInstantiateOp>(
        builder.getUnknownLoc(), enableTy);
    portInterface.wrDataBus =
        builder.create<hir::BusInstantiateOp>(builder.getUnknownLoc(), dataTy);
  }
}
LogicalResult createBusInstantiationsAndCallOp(
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
    if (portInterface.addrEnableBus) {
      inputBuses.push_back(portInterface.addrEnableBus);
      inputBusTypes.push_back(portInterface.addrEnableBus.getType());
      inputBusAttrs.push_back(recvAttr);
      inputBuses.push_back(portInterface.addrBus);
      inputBusTypes.push_back(portInterface.addrBus.getType());
      inputBusAttrs.push_back(recvAttr);
    }

    if (portInterface.rdEnableBus) {
      inputBuses.push_back(portInterface.rdEnableBus);
      inputBusTypes.push_back(portInterface.rdEnableBus.getType());
      inputBusAttrs.push_back(recvAttr);
      inputBuses.push_back(portInterface.rdDataBus);
      inputBusTypes.push_back(portInterface.rdDataBus.getType());
      inputBusAttrs.push_back(sendAttr);
    }

    if (portInterface.wrEnableBus) {
      inputBuses.push_back(portInterface.wrEnableBus);
      inputBusTypes.push_back(portInterface.wrEnableBus.getType());
      inputBusAttrs.push_back(recvAttr);
      inputBuses.push_back(portInterface.wrDataBus);
      inputBusTypes.push_back(portInterface.wrDataBus.getType());
      inputBusAttrs.push_back(recvAttr);
    }
  }
  Type funcTy = hir::FuncType::get(builder.getContext(), inputBusTypes,
                                   inputBusAttrs, {}, {});

  // Add comment.
  builder.create<hir::CommentOp>(op.getLoc(),
                                 builder.getStringAttr("AllocaOp start"));

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
  builder.create<hir::CommentOp>(op.getLoc(),
                                 builder.getStringAttr("AllocaOp end"));
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
    portInterface.addrEnableBus = bb.insertArgument(i, enableTy);
    inputAttrs.insert(inputAttrs.begin() + i++, sendAttr);
    portInterface.addrBus = bb.insertArgument(i, addrTupleTy);
    inputAttrs.insert(inputAttrs.begin() + i++, sendAttr);
  }

  if (auto rdLatency = helper::getRdLatency(port)) {
    portInterface.rdEnableBus = bb.insertArgument(i, enableTy);
    inputAttrs.insert(inputAttrs.begin() + i++, sendAttr);
    portInterface.rdDataBus = bb.insertArgument(i, dataTy);
    inputAttrs.insert(inputAttrs.begin() + i++, recvAttr);
    portInterface.rdLatency = rdLatency.getValue();
  }

  if (helper::isWrite(port)) {
    portInterface.wrEnableBus = bb.insertArgument(i, enableTy);
    inputAttrs.insert(inputAttrs.begin() + i++, sendAttr);
    portInterface.wrDataBus = bb.insertArgument(i, dataTy);
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

void insertBusArguments(
    hir::FuncOp op,
    DenseMap<Value, SmallVector<MemrefPortInterface>> &mapMemrefPortToBuses) {
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

/// Connect the valid buses of unused banks to zero and the other buses to 'X'.
void initBusesForNoMemrefUse(OpBuilder &builder, ArrayRef<uint64_t> bankIndices,
                             MemrefPortInterface portInterface) {
  // FIXME: For now we are going to drive only the valid buses. Later use
  // AssignOp with 'X' or 'Z'.

  initUnconnectedValidBus(builder, portInterface.rdEnableBus, bankIndices);
  initUnconnectedValidBus(builder, portInterface.wrEnableBus, bankIndices);
}

MemrefPortInterface defineBusTensors(OpBuilder &builder,
                                     MemrefPortInterface portInterface,
                                     uint64_t numUses) {
  MemrefPortInterface busTensor;
  auto shape = SmallVector<int64_t>({(int64_t)numUses});

  if (portInterface.addrEnableBus)
    busTensor.addrEnableBus = builder.create<hir::BusInstantiateOp>(
        builder.getUnknownLoc(),
        mlir::RankedTensorType::get(
            shape, getTensorElementType(portInterface.addrEnableBus)));

  if (portInterface.addrBus)
    busTensor.addrBus = builder.create<hir::BusInstantiateOp>(
        builder.getUnknownLoc(),
        mlir::RankedTensorType::get(
            shape, getTensorElementType(portInterface.addrBus)));

  if (portInterface.rdEnableBus)
    busTensor.rdEnableBus = builder.create<hir::BusInstantiateOp>(
        builder.getUnknownLoc(),
        mlir::RankedTensorType::get(
            shape, getTensorElementType(portInterface.rdEnableBus)));
  if (portInterface.rdDataBus)
    busTensor.rdDataBus = builder.create<hir::BusInstantiateOp>(
        builder.getUnknownLoc(),
        mlir::RankedTensorType::get(
            shape, getTensorElementType(portInterface.rdDataBus)));
  if (portInterface.wrEnableBus)
    busTensor.wrEnableBus = builder.create<hir::BusInstantiateOp>(
        builder.getUnknownLoc(),
        mlir::RankedTensorType::get(
            shape, getTensorElementType(portInterface.wrEnableBus)));
  if (portInterface.wrDataBus)
    busTensor.wrDataBus = builder.create<hir::BusInstantiateOp>(
        builder.getUnknownLoc(),
        mlir::RankedTensorType::get(
            shape, getTensorElementType(portInterface.wrDataBus)));
  return busTensor;
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

void connectUseAndPortInterfaces(OpBuilder &builder,
                                 ArrayRef<uint64_t> bankIndices,
                                 MemrefPortInterface &useInterface,
                                 MemrefPortInterface portInterface,
                                 Value tstartRegion) {
  // TODO
  // extract-tensor from port interface.
  // call memref_mux on addr, data-out and valid buses.
  // call memref_broadcast on data-in bus.
  if (portInterface.addrEnableBus)
    createValidCombineCallOp(builder, portInterface.addrEnableBus, bankIndices,
                             useInterface.addrEnableBus, tstartRegion);
  if (portInterface.addrBus)
    createBusMuxCallOp(builder, portInterface.addrBus, bankIndices,
                       useInterface.addrEnableBus, useInterface.addrBus,
                       tstartRegion);
  if (portInterface.rdEnableBus) {
    createValidCombineCallOp(builder, portInterface.rdEnableBus, bankIndices,
                             useInterface.rdEnableBus, tstartRegion);
    useInterface.rdLatency = portInterface.rdLatency;

    assert(portInterface.rdDataBus);
    Value delayedRdEnableBus = builder.create<hir::DelayOp>(
        builder.getUnknownLoc(), useInterface.rdEnableBus.getType(),
        useInterface.rdEnableBus,
        builder.getI64IntegerAttr(useInterface.rdLatency), tstartRegion,
        IntegerAttr());
    createBusBroadcastCallOp(builder, portInterface.rdDataBus, bankIndices,
                             delayedRdEnableBus, useInterface.rdDataBus,
                             tstartRegion);
  }

  if (portInterface.wrEnableBus)
    createValidCombineCallOp(builder, portInterface.wrEnableBus, bankIndices,
                             useInterface.wrEnableBus, tstartRegion);
  if (portInterface.wrDataBus)
    createBusMuxCallOp(builder, portInterface.wrDataBus, bankIndices,
                       useInterface.wrEnableBus, useInterface.wrDataBus,
                       tstartRegion);
}

MemrefPortInterface createBusInterfaceForBankUse(
    OpBuilder &builder, MemrefPortInterface portInterface,
    ArrayRef<uint64_t> bankIndices, uint64_t numUses, Value tstartRegion) {
  if (numUses == 0) {
    initBusesForNoMemrefUse(builder, bankIndices, portInterface);

    return MemrefPortInterface();
  }
  auto useInterface = defineBusTensors(builder, portInterface, numUses);

  connectUseAndPortInterfaces(builder, bankIndices, useInterface, portInterface,
                              tstartRegion);
  return useInterface;
}

LogicalResult convertOp(hir::LoadOp op, MemrefPortInterface useInterface,
                        uint64_t useNum) {

  mlir::OpBuilder builder(op.getContext());
  builder.setInsertionPoint(op);

  builder.create<hir::CommentOp>(op.getLoc(),
                                 builder.getStringAttr("LoadOp start"));

  Value cUse = builder
                   .create<mlir::ConstantOp>(op.getLoc(),
                                             IndexType::get(op.getContext()),
                                             builder.getIndexAttr(useNum))
                   .getResult();
  Value c1 = builder
                 .create<mlir::ConstantOp>(
                     op.getLoc(), IntegerType::get(op.getContext(), 1),
                     builder.getIntegerAttr(builder.getIntegerType(1), 1))
                 .getResult();

  if (useInterface.addrEnableBus) {
    assert(useInterface.addrBus);

    // Extract the addrEnableBus for this use of memref.
    auto addrEnableBus = builder.create<hir::TensorExtractOp>(
        builder.getUnknownLoc(),
        getTensorElementType(useInterface.addrEnableBus),
        useInterface.addrEnableBus, cUse, builder.getStrArrayAttr({"send"}));

    // Send 1 to the addrEnableBus.
    builder.create<hir::SendOp>(op.getLoc(), c1, addrEnableBus,
                                builder.getI64IntegerAttr(0), op.tstart(),
                                op.offsetAttr());

    // Extract the addrBus for this use of memref.
    auto addrBus = builder.create<hir::TensorExtractOp>(
        builder.getUnknownLoc(), getTensorElementType(useInterface.addrBus),
        useInterface.addrBus, cUse, builder.getStrArrayAttr({"send"}));

    // Create a tuple of the addresses corresponding to the ADDR dims.
    Value addrTuple =
        createAddrTuple(builder, op.getLoc(), op.filterIndices(ADDR));

    // Send the address tuple to the addrBus.
    builder.create<hir::SendOp>(op.getLoc(), addrTuple, addrBus,
                                builder.getI64IntegerAttr(0), op.tstart(),
                                op.offsetAttr());
  }

  assert(useInterface.rdDataBus);

  // extract the rdEnableBus for this use of memref.
  auto rdEnableBus = builder.create<hir::TensorExtractOp>(
      builder.getUnknownLoc(), getTensorElementType(useInterface.rdEnableBus),
      useInterface.rdEnableBus, cUse, builder.getStrArrayAttr({"send"}));

  // Send 1 to the rdEnableBus.
  builder.create<hir::SendOp>(op.getLoc(), c1, rdEnableBus,
                              builder.getI64IntegerAttr(0), op.tstart(),
                              op.offsetAttr());

  // the time at which the data is received depends on the read-latency of the
  // original memref load operation.
  assert(useInterface.rdLatency >= 0);
  IntegerAttr recvOffset =
      (useInterface.rdLatency == 0)
          ? op.offsetAttr()
          : builder.getI64IntegerAttr(op.offset().getValueOr(0) +
                                      useInterface.rdLatency);

  // Extract the rdDataBus for this use of memref.
  auto rdDataBus = builder.create<hir::TensorExtractOp>(
      builder.getUnknownLoc(), getTensorElementType(useInterface.rdDataBus),
      useInterface.rdDataBus, cUse, builder.getStrArrayAttr({"send"}));

  // Receive the data from the rdDataBus.
  Operation *receiveOp = builder.create<hir::RecvOp>(
      op.getLoc(), op.res().getType(), rdDataBus, builder.getI64IntegerAttr(0),
      op.tstart(), recvOffset);
  if (op->hasAttrOfType<ArrayAttr>("names"))
    receiveOp->setAttr("names", op->getAttr("names"));
  // Remove the LoadOp.
  op.replaceAllUsesWith(receiveOp);
  builder.create<hir::CommentOp>(op.getLoc(),
                                 builder.getStringAttr("LoadOp end"));
  op.erase();
  return success();
}

LogicalResult convertOp(hir::StoreOp op, MemrefPortInterface useInterface,
                        uint64_t useNum) {

  mlir::OpBuilder builder(op.getContext());
  builder.setInsertionPoint(op);
  builder.create<hir::CommentOp>(op.getLoc(),
                                 builder.getStringAttr("StoreOp start"));

  Value cUse = builder
                   .create<mlir::ConstantOp>(op.getLoc(),
                                             IndexType::get(op.getContext()),
                                             builder.getIndexAttr(useNum))
                   .getResult();
  Value c1 = builder
                 .create<mlir::ConstantOp>(
                     op.getLoc(), IntegerType::get(op.getContext(), 1),
                     builder.getIntegerAttr(builder.getIntegerType(1), 1))
                 .getResult();

  if (useInterface.addrEnableBus) {
    assert(useInterface.addrBus);

    // Extract the addrEnableBus for this use of memref.
    auto addrEnableBus = builder.create<hir::TensorExtractOp>(
        builder.getUnknownLoc(),
        getTensorElementType(useInterface.addrEnableBus),
        useInterface.addrEnableBus, cUse, builder.getStrArrayAttr({"send"}));

    // Send 1 to the addrEnableBus.
    builder.create<hir::SendOp>(op.getLoc(), c1, addrEnableBus,
                                builder.getI64IntegerAttr(0), op.tstart(),
                                op.offsetAttr());

    // Extract the addrBus for this use of memref.
    auto addrBus = builder.create<hir::TensorExtractOp>(
        builder.getUnknownLoc(), getTensorElementType(useInterface.addrBus),
        useInterface.addrBus, cUse, builder.getStrArrayAttr({"send"}));

    // Create a tuple of the addresses corresponding to the ADDR dims.
    Value addrTuple =
        createAddrTuple(builder, op.getLoc(), op.filterIndices(ADDR));

    // Send the address tuple to the addrBus.
    builder.create<hir::SendOp>(op.getLoc(), addrTuple, addrBus,
                                builder.getI64IntegerAttr(0), op.tstart(),
                                op.offsetAttr());
  }

  if (!useInterface.wrEnableBus)
    return op->emitError("Could not find useInterface.wrEnableBus.");

  // extract the wrEnableBus for this use of memref.
  auto wrEnableBus = builder.create<hir::TensorExtractOp>(
      builder.getUnknownLoc(), getTensorElementType(useInterface.wrEnableBus),
      useInterface.wrEnableBus, cUse, builder.getStrArrayAttr({"send"}));

  // Send 1 to the wrEnableBus.
  builder.create<hir::SendOp>(op.getLoc(), c1, wrEnableBus,
                              builder.getI64IntegerAttr(0), op.tstart(),
                              op.offsetAttr());

  // Extract the wrDataBus for this use of memref.
  auto wrDataBus = builder.create<hir::TensorExtractOp>(
      builder.getUnknownLoc(), getTensorElementType(useInterface.wrDataBus),
      useInterface.wrDataBus, cUse, builder.getStrArrayAttr({"send"}));

  // Send the $value to the wrDataBus.
  builder.create<hir::SendOp>(op.getLoc(), op.value(), wrDataBus,
                              builder.getI64IntegerAttr(0), op.tstart(),
                              op.offsetAttr());

  // Remove the StoreOp.
  builder.create<hir::CommentOp>(op.getLoc(),
                                 builder.getStringAttr("StoreOp end"));
  op.erase();
  return success();
}

LogicalResult MemrefLoweringPass::replacePortUses(
    OpBuilder &builder, ArrayRef<int64_t> bankShape,
    MemrefPortInterface portInterface, SmallVector<ListOfUses> uses) {
  MultiDimCounter bank(bankShape);
  do {
    uint64_t numUses = uses[bank.getLinearIdx()].size();
    auto useInterface = createBusInterfaceForBankUse(
        builder, portInterface, bank.getIndices(), numUses, tstartRegion);
    for (uint64_t use = 0; use < numUses; use++) {
      if (auto op = dyn_cast<hir::LoadOp>(uses[bank.getLinearIdx()][use])) {
        if (failed(convertOp(op, useInterface, use)))
          return failure();
      } else if (auto op =
                     dyn_cast<hir::StoreOp>(uses[bank.getLinearIdx()][use])) {
        if (failed(convertOp(op, useInterface, use)))
          return failure();
      }
    }
  } while (succeeded(bank.inc()));
  return success();
}

LogicalResult MemrefLoweringPass::replaceMemrefUses(
    OpBuilder &builder, Value memref,
    ArrayRef<MemrefPortInterface> memrefPortInterface,
    SmallVector<SmallVector<ListOfUses>> &memrefUses) {

  auto memrefTy = memref.getType().dyn_cast<hir::MemrefType>();
  for (uint64_t port = 0; port < memrefPortInterface.size(); port++) {
    MemrefPortInterface portInterface = memrefPortInterface[port];
    SmallVector<ListOfUses> portUses = memrefUses[port];
    if (failed(replacePortUses(builder, memrefTy.filterShape(BANK),
                               portInterface, portUses)))
      return failure();
  }
  return success();
}

LogicalResult MemrefLoweringPass::replaceMemrefArgUses(hir::FuncOp op) {
  OpBuilder builder(op);
  auto &entryBlock = op.getFuncBody().front();
  builder.setInsertionPointToStart(&entryBlock);

  // For each memref-port-bank define new tensor<bus> for its uses.
  for (auto memref : filterMemrefArgs(entryBlock.getArguments())) {
    if (failed(replaceMemrefUses(builder, memref, mapMemrefPortToBuses[memref],
                                 (*uses)[memref])))
      return failure();
  }
  return success();
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// MemrefLoweringPass methods.
//------------------------------------------------------------------------------
LogicalResult MemrefLoweringPass::visitOp(hir::FuncOp op) {
  tstartRegion = op.getRegionTimeVar();
  insertBusArguments(op, mapMemrefPortToBuses);

  if (failed(visitRegion(op.getFuncBody())))
    return failure();

  if (failed(replaceMemrefArgUses(op)))
    return failure();

  removeMemrefArguments(op);

  return success();
}

LogicalResult MemrefLoweringPass::visitOp(hir::ForOp op) {
  auto oldTstartRegion = tstartRegion;
  tstartRegion = op.getIterTimeVar();
  if (failed(visitRegion(op.getLoopBody())))
    return failure();
  tstartRegion = oldTstartRegion;
  return success();
}
LogicalResult MemrefLoweringPass::visitOp(hir::WhileOp op) {
  auto oldTstartRegion = tstartRegion;
  tstartRegion = op.getIterTimeVar();
  if (failed(visitRegion(op.body())))
    return failure();
  tstartRegion = oldTstartRegion;
  return success();
}
LogicalResult MemrefLoweringPass::visitOp(hir::AllocaOp op) {
  if (failed(createBusInstantiationsAndCallOp(op, mapMemrefPortToBuses,
                                              tstartRegion)))
    return failure();
  OpBuilder builder(op);
  if (failed(replaceMemrefUses(builder, op.res(),
                               mapMemrefPortToBuses[op.res()],
                               (*uses)[op.res()])))
    return failure();
  opsToErase.push_back(op);
  return success();
}

LogicalResult MemrefLoweringPass::visitRegion(Region &region) {
  assert(region.getBlocks().size() == 1);
  for (auto &operation : region.front()) {
    if (auto op = dyn_cast<hir::FuncOp>(operation)) {
      if (failed(visitOp(op)))
        return failure();
    } else if (auto op = dyn_cast<hir::ForOp>(operation)) {
      if (failed(visitOp(op)))
        return failure();
    } else if (auto op = dyn_cast<hir::WhileOp>(operation)) {
      if (failed(visitOp(op)))
        return failure();
    } else if (auto op = dyn_cast<hir::AllocaOp>(operation)) {
      if (failed(visitOp(op)))
        return failure();
    } else if (auto op = dyn_cast<hir::IfOp>(operation)) {
      auto tstartRegionOld = tstartRegion;
      tstartRegion = op.getRegionTimeVar();
      if (failed(visitRegion(op.if_region())))
        return failure();
      if (failed(visitRegion(op.else_region())))
        return failure();
      tstartRegion = tstartRegionOld;
    } else {
      if (operation.getNumRegions() > 0)
        return operation.emitError()
               << "Unsupported op with region for HIR MemrefLoweringPass.";
    }
  }
  return success();
}

void MemrefLoweringPass::runOnOperation() {
  hir::FuncOp funcOp = getOperation();
  auto memrefUseInfo = MemrefUseInfo(funcOp);
  this->uses = &memrefUseInfo.mapMemref2PerPortPerBankUses;
  if (failed(visitOp(funcOp))) {
    signalPassFailure();
    return;
  }
  helper::eraseOps(opsToErase);
}

namespace circt {
namespace hir {
std::unique_ptr<OperationPass<hir::FuncOp>> createMemrefLoweringPass() {
  return std::make_unique<MemrefLoweringPass>();
}
} // namespace hir
} // namespace circt
