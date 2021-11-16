//=========- MemrefLoweringUtils.cpp - Utils for memref lowering pass---======//
//
// This file implements utilities for memref lowering pass.
//
//===----------------------------------------------------------------------===//

#include "MemrefLoweringUtils.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include <iostream>
using namespace circt;
using namespace hir;

// static Type getBusTypeFormTensor(Value tensor) {
//  return hir::BusType::get(
//      tensor.getContext(),
//      tensor.getType().dyn_cast<hir::BusTensorType>().getElementType());
//}

static Type getBusElementType(Value bus) {
  auto ty = bus.getType().dyn_cast<hir::BusType>();
  assert(ty);
  return ty.getElementType();
}

Value insertDataSendLogic(
    OpBuilder &builder, Location errorLoc,
    std::function<Value(OpBuilder &, Type)> declTopLevelBus, Value data,
    Value rootDataBus, Value tVar, IntegerAttr offsetAttr) {
  assert(offsetAttr.getInt() == 0);
  Value nextDataBus = declTopLevelBus(
      builder, rootDataBus.getType().dyn_cast<hir::BusType>().getElementType());

  assert(rootDataBus.getType().isa<hir::BusType>());

  auto i1BusTy = hir::BusType::get(builder.getContext(), builder.getI1Type());
  auto tBus =
      builder.create<hir::CastOp>(builder.getUnknownLoc(), i1BusTy, tVar);

  auto dataBus = builder.create<hir::CastOp>(builder.getUnknownLoc(),
                                             nextDataBus.getType(), data);
  auto selectedDataBus =
      helper::insertBusSelectLogic(builder, tBus, dataBus, nextDataBus);

  builder.create<hir::BusAssignOp>(builder.getUnknownLoc(), rootDataBus,
                                   selectedDataBus);
  return nextDataBus;
}

Value insertDataRecvLogic(OpBuilder &builder, Location errorLoc, Value dataBus,
                          int64_t rdLatency, Value tVar,
                          IntegerAttr offsetAttr) {

  assert(rdLatency >= 0);
  auto recvOffsetAttr =
      (rdLatency == 0)
          ? offsetAttr
          : builder.getI64IntegerAttr(offsetAttr.getInt() + rdLatency);

  auto receiveOp = builder.create<hir::BusRecvOp>(
      builder.getUnknownLoc(), getBusElementType(dataBus), dataBus, tVar,
      recvOffsetAttr);

  return receiveOp.getResult();
}

MemoryInterface emitMemoryInterface(OpBuilder &builder,
                                    hir::MemrefType memrefTy,
                                    MemrefPortInterface portInterface,
                                    int64_t bank) {
  auto uLoc = builder.getUnknownLoc();
  MemoryInterface memoryInterface(memrefTy);
  Value bankIdx =
      builder.create<mlir::arith::ConstantOp>(uLoc, builder.getIndexAttr(bank));
  if (portInterface.addrEnableBusTensor) {
    auto addrEnBus = emitAddrEnableBus(builder, memrefTy);
    builder.create<hir::BusTensorAssignElementOp>(
        uLoc, portInterface.addrEnableBusTensor, bankIdx, addrEnBus);
    memoryInterface.setAddrEnableBus(addrEnBus);
    assert(portInterface.addrDataBusTensor);
    auto addrDataBus = emitAddrDataBus(builder, memrefTy);
    builder.create<hir::BusTensorAssignElementOp>(
        uLoc, portInterface.addrDataBusTensor, bankIdx, addrDataBus);
    memoryInterface.setAddrDataBus(addrDataBus);
  }
  if (portInterface.rdEnableBusTensor) {
    auto rdEnBus = emitRdEnableBus(builder, memrefTy);
    builder.create<hir::BusTensorAssignElementOp>(
        uLoc, portInterface.rdEnableBusTensor, bankIdx, rdEnBus);
    memoryInterface.setRdEnableBus(rdEnBus);
    assert(portInterface.rdDataBusTensor);
    auto busTy = hir::BusType::get(builder.getContext(),
                                   portInterface.rdDataBusTensor.getType()
                                       .dyn_cast<hir::BusTensorType>()
                                       .getElementType());
    auto rdDataBus = builder.create<hir::BusTensorGetElementOp>(
        uLoc, busTy, portInterface.rdDataBusTensor, bankIdx);
    memoryInterface.setRdDataBus(rdDataBus, portInterface.rdLatency);
  }
  if (portInterface.wrEnableBusTensor) {
    auto wrEnBus = emitWrEnableBus(builder, memrefTy);
    builder.create<hir::BusTensorAssignElementOp>(
        uLoc, portInterface.wrEnableBusTensor, bankIdx, wrEnBus);
    memoryInterface.setWrEnableBus(wrEnBus);
    assert(portInterface.wrDataBusTensor);
    auto wrDataBus = emitWrDataBus(builder, memrefTy);
    builder.create<hir::BusTensorAssignElementOp>(
        uLoc, portInterface.wrDataBusTensor, bankIdx, wrDataBus);
    memoryInterface.setWrDataBus(wrDataBus);
  }

  return memoryInterface;
}

SmallVector<SmallVector<MemoryInterface>> emitMemoryInterfacesForEachPortBank(
    OpBuilder &builder, hir::MemrefType memrefTy,
    SmallVector<MemrefPortInterface> memrefPortInterfaces) {

  int64_t numBanks = memrefTy.getNumBanks();
  SmallVector<SmallVector<MemoryInterface>> mapPortBankToMemoryInterface;

  for (size_t port = 0; port < memrefPortInterfaces.size(); port++) {
    SmallVector<MemoryInterface> mapBankToMemoryInterface;
    for (int64_t bank = 0; bank < numBanks; bank++) {
      auto memoryInterface = emitMemoryInterface(
          builder, memrefTy, memrefPortInterfaces[port], bank);
      mapBankToMemoryInterface.push_back(memoryInterface);
    }
    mapPortBankToMemoryInterface.push_back(mapBankToMemoryInterface);
  }
  return mapPortBankToMemoryInterface;
}

Value getBusFromTensor(OpBuilder &builder, Value busT, int64_t idx) {
  auto uLoc = builder.getUnknownLoc();
  auto numBanks =
      busT.getType().dyn_cast<hir::BusTensorType>().getNumElements();
  SmallVector<Value> cIdx;
  if (numBanks > 1) {
    cIdx.push_back(builder.create<mlir::arith::ConstantOp>(
        uLoc, builder.getIndexAttr(idx)));
  } else {
    cIdx = SmallVector<Value>({});
  }
  auto busTy = hir::BusType::get(
      builder.getContext(),
      busT.getType().dyn_cast<hir::BusTensorType>().getElementType());
  builder.create<hir::CommentOp>(uLoc, "debug start");
  auto out =
      builder.create<hir::BusTensorGetElementOp>(uLoc, busTy, busT, cIdx);
  builder.create<hir::CommentOp>(uLoc, "debug end");
  return out;
}

void emitBusTensorAssignElementLogic(OpBuilder &builder, Value bus, Value busT,
                                     int64_t idx) {
  auto uLoc = builder.getUnknownLoc();
  auto numBanks =
      busT.getType().dyn_cast<hir::BusTensorType>().getNumElements();
  SmallVector<Value> cIdx;
  if (numBanks > 1) {
    cIdx.push_back(builder.create<mlir::arith::ConstantOp>(
        uLoc, builder.getIndexAttr(idx)));
  } else {
    cIdx = SmallVector<Value>({});
  }

  builder.create<hir::BusTensorAssignElementOp>(uLoc, busT, cIdx, bus);
}

void emitCallOpOperandsForMemrefPort(
    OpBuilder &builder, MemrefInfo &memrefInfo, Value mem,
    SmallVectorImpl<Value> &operands, SmallVectorImpl<Type> &inputTypes,
    SmallVectorImpl<DictionaryAttr> &inputAttrs) {
  assert(mem.getType().isa<MemrefType>());
  auto sendAttr = helper::getDictionaryAttr(builder, "hir.bus.ports",
                                            builder.getStrArrayAttr({"send"}));
  auto recvAttr = helper::getDictionaryAttr(builder, "hir.bus.ports",
                                            builder.getStrArrayAttr({"recv"}));
  for (size_t j = 0; j < memrefInfo.getNumPorts(mem); j++) {
    auto duplicate = memrefInfo.emitDuplicatePortInterface(builder, mem, j);
    if (duplicate.addrEnableBusTensor) {
      operands.push_back(duplicate.addrEnableBusTensor);
      inputTypes.push_back(duplicate.addrEnableBusTensor.getType());
      inputAttrs.push_back(sendAttr);

      assert(duplicate.addrDataBusTensor);
      operands.push_back(duplicate.addrDataBusTensor);
      inputTypes.push_back(duplicate.addrDataBusTensor.getType());
      inputAttrs.push_back(sendAttr);
    }

    if (duplicate.rdEnableBusTensor) {
      operands.push_back(duplicate.rdEnableBusTensor);
      inputTypes.push_back(duplicate.rdEnableBusTensor.getType());
      inputAttrs.push_back(sendAttr);

      assert(duplicate.rdDataBusTensor);
      operands.push_back(duplicate.rdDataBusTensor);
      inputTypes.push_back(duplicate.rdDataBusTensor.getType());
      inputAttrs.push_back(recvAttr);
    }

    if (duplicate.wrEnableBusTensor) {
      operands.push_back(duplicate.wrEnableBusTensor);
      inputTypes.push_back(duplicate.wrEnableBusTensor.getType());
      inputAttrs.push_back(sendAttr);

      assert(duplicate.wrDataBusTensor);
      operands.push_back(duplicate.wrDataBusTensor);
      inputTypes.push_back(duplicate.wrDataBusTensor.getType());
      inputAttrs.push_back(sendAttr);
    }
  }
}

Value emitAddrEnableBus(OpBuilder &builder, hir::MemrefType memrefTy) {
  return helper::emitIntegerBusOp(builder, 1);
}
Value emitAddrDataBus(OpBuilder &builder, hir::MemrefType memrefTy) {
  auto width = helper::clog2(memrefTy.getNumElementsPerBank());
  auto i1Bus = helper::emitIntegerBusOp(builder, width);
  return i1Bus;
}
Value emitRdEnableBus(OpBuilder &builder, hir::MemrefType memrefTy) {
  return helper::emitIntegerBusOp(builder, 1);
}
Value emitRdDataBus(OpBuilder &builder, hir::MemrefType memrefTy) {
  auto width = helper::getBitWidth(memrefTy.getElementType()).getValue();
  auto i1Bus = helper::emitIntegerBusOp(builder, width);
  return i1Bus;
}
Value emitWrEnableBus(OpBuilder &builder, hir::MemrefType memrefTy) {
  return helper::emitIntegerBusOp(builder, 1);
}

Value emitWrDataBus(OpBuilder &builder, hir::MemrefType memrefTy) {
  auto width = helper::getBitWidth(memrefTy.getElementType()).getValue();
  auto i1Bus = helper::emitIntegerBusOp(builder, width);
  return i1Bus;
}

std::string createHWMemoryName(llvm::StringRef memKind,
                               hir::MemrefType memrefTy, ArrayAttr memPorts) {
  std::string name =
      memKind.str() + "_" + std::to_string(memrefTy.getNumElementsPerBank()) +
      "x" +
      std::to_string(helper::getBitWidth(memrefTy.getElementType()).getValue());
  for (auto port : memPorts) {
    name += "_";
    if (helper::isRead(port))
      name += "r";
    if (helper::isWrite(port))
      name += "w";
  }
  return name;
}

/// Emit CallOps for each bank of the memref that will instantiate the
/// hw memory, and connect the memory interfaces of all the ports with it.
void emitMemoryInstance(OpBuilder &builder, hir::MemrefType memrefTy,
                        ArrayRef<MemoryInterface> memoryInterfaces,
                        llvm::StringRef memKind, llvm::StringRef memName,
                        llvm::Optional<std::string> instanceName,
                        Value tstart) {

  auto elementWidth = helper::getBitWidth(memrefTy.getElementType()).getValue();
  auto addrWidth = helper::clog2(memrefTy.getNumElementsPerBank());
  auto sendAttr = helper::getDictionaryAttr(builder, "hir.bus.ports",
                                            builder.getStrArrayAttr({"send"}));
  auto recvAttr = helper::getDictionaryAttr(builder, "hir.bus.ports",
                                            builder.getStrArrayAttr({"recv"}));

  // Create the inputs for the call op that instantiates the memory.
  SmallVector<Value> inputBuses;
  SmallVector<std::string> inputBusNames;
  SmallVector<DictionaryAttr> inputBusAttrs;
  SmallVector<Type> inputBusTypes;
  for (size_t port = 0; port < memoryInterfaces.size(); port++) {
    std::string portPrefix = "p" + std::to_string(port) + "_";
    auto memoryInterface = memoryInterfaces[port];
    if (memoryInterface.hasAddrBus()) {
      assert(memrefTy.getNumElementsPerBank() > 1);
      inputBuses.push_back(memoryInterface.getAddrEnBus());
      inputBusTypes.push_back(memoryInterface.getAddrEnBus().getType());
      inputBusAttrs.push_back(recvAttr);
      inputBusNames.push_back(portPrefix + "addr_en");
      inputBuses.push_back(memoryInterface.getAddrDataBus());
      inputBusTypes.push_back(memoryInterface.getAddrDataBus().getType());
      inputBusAttrs.push_back(recvAttr);
      inputBusNames.push_back(portPrefix + "addr_data");
    }
    if (memoryInterface.hasRdBus()) {
      inputBuses.push_back(memoryInterface.getRdEnBus());
      inputBusTypes.push_back(memoryInterface.getRdEnBus().getType());
      inputBusAttrs.push_back(recvAttr);
      inputBusNames.push_back(portPrefix + "rd_en");
      inputBuses.push_back(memoryInterface.getRdDataBus());
      inputBusTypes.push_back(memoryInterface.getRdDataBus().getType());
      inputBusAttrs.push_back(sendAttr);
      inputBusNames.push_back(portPrefix + "rd_data");
    }
    if (memoryInterface.hasWrBus()) {
      inputBuses.push_back(memoryInterface.getWrEnBus());
      inputBusTypes.push_back(memoryInterface.getWrEnBus().getType());
      inputBusAttrs.push_back(recvAttr);
      inputBusNames.push_back(portPrefix + "wr_en");
      inputBuses.push_back(memoryInterface.getWrDataBus());
      inputBusTypes.push_back(memoryInterface.getWrDataBus().getType());
      inputBusAttrs.push_back(recvAttr);
      inputBusNames.push_back(portPrefix + "wr_data");
    }
  }

  Type funcTy = hir::FuncType::get(builder.getContext(), inputBusTypes,
                                   inputBusAttrs, {}, {});
  auto instanceNameAttr = instanceName
                              ? builder.getStringAttr(instanceName.getValue())
                              : StringAttr();
  auto memNameAttr = FlatSymbolRefAttr::get(builder.getContext(), memName);
  auto callOp = builder.create<hir::CallOp>(
      builder.getUnknownLoc(), SmallVector<Type>(), instanceNameAttr,
      memNameAttr, TypeAttr::get(funcTy), inputBuses, tstart,
      builder.getI64IntegerAttr(0));

  auto params = builder.getDictionaryAttr(
      {builder.getNamedAttr("ELEMENT_WIDTH",
                            builder.getI64IntegerAttr(elementWidth)),
       builder.getNamedAttr("ADDR_WIDTH",
                            builder.getI64IntegerAttr(addrWidth))});

  callOp->setAttr("params", params);

  helper::declareExternalFuncForCall(callOp, inputBusNames);
}

Value getRegionTimeVar(Operation *operation) {
  return operation->getParentRegion()->getArguments().back();
}
