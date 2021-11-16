//=========- MemrefLoweringUtils.h - Utils for memref lowering pass---======//
//
// This header declares utilities for memref lowering pass.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "circt/Dialect/HW/HWOps.h"
#include <iostream>
using namespace circt;
using namespace hir;
struct MemrefPortInterface {
  Value addrDataBusTensor;
  Value addrEnableBusTensor;
  Value rdEnableBusTensor;
  int64_t rdLatency = -1;
  Value rdDataBusTensor;
  Value wrEnableBusTensor;
  Value wrDataBusTensor;
};

void emitBusTensorAssignElementLogic(OpBuilder &builder, Value bus, Value busT,
                                     int64_t idx);
Value insertDataSendLogic(OpBuilder &builder, Location errorLoc,
                          std::function<Value(OpBuilder &, Type)>, Value data,
                          Value inDataBus, Value tVar, IntegerAttr offsetAttr);

Value insertDataRecvLogic(OpBuilder &builder, Location loc, Value dataBus,
                          int64_t rdLatency, Value tVar,
                          IntegerAttr offsetAttr);

Value getBusFromTensor(OpBuilder &builder, Value busT, int64_t idx);

template <typename KeyTy, typename ValTy>
class SafeDenseMap {
private:
  DenseMap<KeyTy, ValTy> denseMap;

public:
  ValTy lookup(KeyTy key) {
    auto valPtr = denseMap.find(key);
    assert(valPtr != denseMap.end());
    return valPtr->second;
  }

  ValTy lookupOr(KeyTy key, ValTy defaultVal) {
    auto valPtr = denseMap.find(key);
    if (valPtr != denseMap.end())
      return valPtr->second;
    else {
      return defaultVal;
    }
  }
  void map(KeyTy key, ValTy val) {
    auto valPtr = denseMap.find(key);
    assert(valPtr == denseMap.end());
    denseMap[key] = val;
  }
  void clear() { denseMap.clear(); }
};

class MemoryInterface {
public:
  MemoryInterface(hir::MemrefType memrefTy) : memrefTy(memrefTy) {}

private:
  hir::MemrefType memrefTy;

private:
  Value addrDataBus;
  Value addrEnableBus;
  Value rdEnableBus;
  int64_t rdLatency = -1;
  Value rdDataBus;
  Value wrEnableBus;
  Value wrDataBus;

private:
  Value declTopLevelBus(OpBuilder &builder, Type elementTy) {
    auto &declBlk = builder.getInsertionBlock()
                        ->getParent()
                        ->getParentOfType<hir::FuncOp>()
                        .getFuncBody()
                        .front();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&declBlk);
    auto busTy = hir::BusType::get(builder.getContext(), elementTy);
    return builder.create<hir::BusOp>(builder.getUnknownLoc(), busTy);
  }

public:
  Value getAddrDataBus() {
    assert(addrDataBus);
    return addrDataBus;
  }
  Value getRdDataBus() {
    assert(rdDataBus);
    return rdDataBus;
  }
  Value getWrDataBus() {
    assert(wrDataBus);
    return wrDataBus;
  }
  Value getAddrEnBus() {
    assert(addrEnableBus);
    return addrEnableBus;
  }
  Value getRdEnBus() {
    assert(rdEnableBus);
    return rdEnableBus;
  }
  Value getWrEnBus() {
    assert(wrEnableBus);
    return wrEnableBus;
  }

  bool hasAddrBus() {
    if (addrEnableBus != Value()) {
      assert(memrefTy.getNumElementsPerBank() > 1);
      return true;
    }
    return false;
  }
  bool hasRdBus() { return rdEnableBus != Value(); }
  bool hasWrBus() { return wrEnableBus != Value(); }

public:
  LogicalResult emitAddrSendLogic(OpBuilder &builder, Location errorLoc,
                                  Value addr, Value tstart,
                                  IntegerAttr offsetAttr) {

    assert(hasAddrBus());
    auto c1 = builder.create<hw::ConstantOp>(
        builder.getUnknownLoc(), builder.getI1Type(),
        mlir::IntegerAttr::get(builder.getI1Type(), 1));
    auto declTopLevelBusFunc = [this](OpBuilder &builder, Type elementTy) {
      return this->declTopLevelBus(builder, elementTy);
    };
    setAddrEnableBus(insertDataSendLogic(builder, errorLoc, declTopLevelBusFunc,
                                         c1, addrEnableBus, tstart,
                                         offsetAttr));
    setAddrDataBus(insertDataSendLogic(builder, errorLoc, declTopLevelBusFunc,
                                       addr, addrDataBus, tstart, offsetAttr));

    return success();
  }
  Value emitRdRecvLogic(OpBuilder &builder, Location errorLoc, Value tstart,
                        IntegerAttr offsetAttr) {

    auto declTopLevelBusFunc = [this](OpBuilder &builder, Type elementTy) {
      return this->declTopLevelBus(builder, elementTy);
    };
    auto c1 = builder.create<hw::ConstantOp>(
        builder.getUnknownLoc(), builder.getI1Type(),
        mlir::IntegerAttr::get(builder.getI1Type(), 1));
    setRdEnableBus(insertDataSendLogic(builder, errorLoc, declTopLevelBusFunc,
                                       c1, rdEnableBus, tstart, offsetAttr));

    return insertDataRecvLogic(builder, errorLoc, rdDataBus, rdLatency, tstart,
                               offsetAttr);
  }

  LogicalResult emitWrSendLogic(OpBuilder &builder, Location errorLoc,
                                Value input, Value tstart,
                                IntegerAttr offsetAttr) {

    auto declTopLevelBusFunc = [this](OpBuilder &builder, Type elementTy) {
      return this->declTopLevelBus(builder, elementTy);
    };
    auto c1 = builder.create<hw::ConstantOp>(
        builder.getUnknownLoc(), builder.getI1Type(),
        mlir::IntegerAttr::get(builder.getI1Type(), 1));

    setWrEnableBus(insertDataSendLogic(builder, errorLoc, declTopLevelBusFunc,
                                       c1, getWrEnBus(), tstart, offsetAttr));
    setWrDataBus(insertDataSendLogic(builder, errorLoc, declTopLevelBusFunc,
                                     input, getWrDataBus(), tstart,
                                     offsetAttr));

    return success();
  }

  void attachAddrEnBus(OpBuilder &builder, Value addrEnBus) {
    builder.create<hir::CommentOp>(
        builder.getUnknownLoc(),
        builder.getStringAttr("attachAddrEnBus start"));
    Value prevAddrEnBus = getAddrEnBus();
    Value nextAddrEnBus = declTopLevelBus(
        builder,
        prevAddrEnBus.getType().dyn_cast<hir::BusType>().getElementType());
    assert(prevAddrEnBus.getType()
               .dyn_cast<hir::BusType>()
               .getElementType()
               .getIntOrFloatBitWidth() == 1);
    builder.create<hir::BusAssignOp>(
        builder.getUnknownLoc(), prevAddrEnBus,
        helper::insertBusSelectLogic(builder, addrEnBus, addrEnBus,
                                     nextAddrEnBus));
    assert(hasAddrBus());
    setAddrEnableBus(nextAddrEnBus);
    builder.create<hir::CommentOp>(
        builder.getUnknownLoc(), builder.getStringAttr("attachAddrEnBus end."));
  }

  void attachAddrDataBus(OpBuilder &builder, Value enBus, Value dataBus) {
    Value prevAddrDataBus = getAddrDataBus();
    Value nextAddrDataBus = declTopLevelBus(
        builder,
        prevAddrDataBus.getType().dyn_cast<hir::BusType>().getElementType());

    builder.create<hir::BusAssignOp>(
        builder.getUnknownLoc(), prevAddrDataBus,
        helper::insertBusSelectLogic(builder, enBus, dataBus, nextAddrDataBus));
    setAddrDataBus(nextAddrDataBus);
  }

  void attachRdEnBus(OpBuilder &builder, Value rdEnBus) {
    Value prevRdEnBus = getRdEnBus();
    Value nextRdEnBus = declTopLevelBus(
        builder,
        prevRdEnBus.getType().dyn_cast<hir::BusType>().getElementType());

    assert(prevRdEnBus.getType()
               .dyn_cast<hir::BusType>()
               .getElementType()
               .getIntOrFloatBitWidth() == 1);
    builder.create<hir::BusAssignOp>(
        builder.getUnknownLoc(), prevRdEnBus,
        helper::insertBusSelectLogic(builder, rdEnBus, rdEnBus, nextRdEnBus));
    setRdEnableBus(nextRdEnBus);
  }

  void attachWrEnBus(OpBuilder &builder, Value wrEnBus) {
    Value prevWrEnBus = getWrEnBus();
    Value nextWrEnBus = declTopLevelBus(
        builder,
        prevWrEnBus.getType().dyn_cast<hir::BusType>().getElementType());

    assert(prevWrEnBus.getType()
               .dyn_cast<hir::BusType>()
               .getElementType()
               .getIntOrFloatBitWidth() == 1);
    builder.create<hir::BusAssignOp>(
        builder.getUnknownLoc(), prevWrEnBus,
        helper::insertBusSelectLogic(builder, wrEnBus, wrEnBus, nextWrEnBus));
    setWrEnableBus(nextWrEnBus);
  }

  void attachWrDataBus(OpBuilder &builder, Value enBus, Value dataBus) {
    Value prevWrDataBus = getWrDataBus();
    Value nextWrDataBus = declTopLevelBus(
        builder,
        prevWrDataBus.getType().dyn_cast<hir::BusType>().getElementType());

    builder.create<hir::BusAssignOp>(
        builder.getUnknownLoc(), prevWrDataBus,
        helper::insertBusSelectLogic(builder, enBus, dataBus, nextWrDataBus));
    setWrDataBus(nextWrDataBus);
  }

public:
  void setAddrEnableBus(Value v) {
    assert(v.getType().isa<hir::BusType>());
    assert(v.getType()
               .dyn_cast<hir::BusType>()
               .getElementType()
               .getIntOrFloatBitWidth() == 1);
    if (addrEnableBus)
      assert(addrEnableBus.getType() == v.getType());
    assert(memrefTy.getNumElementsPerBank() > 1);
    addrEnableBus = v;
  }

  void setAddrDataBus(Value v) {
    auto addrWidth = helper::clog2(memrefTy.getNumElementsPerBank());
    assert(v.getType()
               .dyn_cast<hir::BusType>()
               .getElementType()
               .getIntOrFloatBitWidth() == addrWidth);
    if (addrDataBus)
      assert(addrDataBus.getType() == v.getType());
    addrDataBus = v;
  }
  void setRdEnableBus(Value v) {
    assert(v.getType().isa<hir::BusType>());
    assert(v.getType()
               .dyn_cast<hir::BusType>()
               .getElementType()
               .getIntOrFloatBitWidth() == 1);
    if (rdEnableBus)
      assert(rdEnableBus.getType() == v.getType());
    rdEnableBus = v;
  }
  void setRdDataBus(Value v, int64_t rdLatency) {
    assert(v.getType().isa<hir::BusType>());
    if (rdDataBus)
      assert(rdDataBus.getType() == v.getType());
    rdDataBus = v;
    setRdLatency(rdLatency);
  }
  void setWrEnableBus(Value v) {
    assert(v.getType().isa<hir::BusType>());
    assert(v.getType()
               .dyn_cast<hir::BusType>()
               .getElementType()
               .getIntOrFloatBitWidth() == 1);
    if (wrEnableBus)
      if (wrEnableBus.getType() != v.getType()) {
        wrEnableBus.getDefiningOp()->emitError("Type mismatch.")
            << "old type " << wrDataBus.getType() << ", new type"
            << v.getType();
        assert(false);
      }
    wrEnableBus = v;
  }
  void setWrDataBus(Value v) {
    assert(v.getType().isa<hir::BusType>());
    if (wrDataBus)
      assert(wrDataBus.getType() == v.getType());
    wrDataBus = v;
  }
  void setRdLatency(int64_t v) {
    assert(v >= 0);
    rdLatency = v;
  }
};

SmallVector<SmallVector<MemoryInterface>> emitMemoryInterfacesForEachPortBank(
    OpBuilder &builder, hir::MemrefType memrefTy,
    SmallVector<MemrefPortInterface> memrefPortInterfaces);

class MemrefInfo {
public:
  void
  map(Value mem,
      SmallVector<SmallVector<MemoryInterface>> mapPortBankToMemoryInterface) {
    assert(mem.getType().isa<hir::MemrefType>());
    assert(mapPortBankToMemoryInterface.size() > 0 &&
           "There must be atleast one port for a memref.");
    for (size_t port = 0; port < mapPortBankToMemoryInterface.size(); port++) {
      auto mapBankToMemoryInterface = mapPortBankToMemoryInterface[port];
      for (size_t bank = 0; bank < mapBankToMemoryInterface.size(); bank++) {
        auto memoryInterface = mapBankToMemoryInterface[bank];
        mapMemrefPortBank2MemoryInterface.map(std::make_tuple(mem, port, bank),
                                              memoryInterfaces.size());
        memoryInterfaces.push_back(memoryInterface);
      }
    }
    mapMemrefToNumPorts.map(mem, mapPortBankToMemoryInterface.size());
    mapMemrefToNumBanks.map(mem, mapPortBankToMemoryInterface[0].size());
  }

  /// Maps port 0 of mem to originalPort of originalMem.
  void mapPort(Value mem, Value originalMem, int64_t originalPort) {
    assert(mem.getType() == originalMem.getType());
    for (size_t bank = 0; bank < getNumBanks(originalMem); bank++) {
      mapMemrefPortBank2MemoryInterface.map(
          std::make_tuple(mem, 0, bank),
          mapMemrefPortBank2MemoryInterface.lookup(
              std::make_tuple(originalMem, originalPort, bank)));
    }
    mapMemrefToNumPorts.map(mem, 1);
    mapMemrefToNumBanks.map(mem, getNumBanks(originalMem));
  }

  size_t getNumPorts(Value mem) { return mapMemrefToNumPorts.lookup(mem); }
  size_t getNumBanks(Value mem) { return mapMemrefToNumBanks.lookup(mem); }

  MemoryInterface *getInterface(Value mem, int64_t port, int64_t bank) {
    auto loc = mapMemrefPortBank2MemoryInterface.lookup(
        std::make_tuple(mem, port, bank));
    return &memoryInterfaces[loc];
  }

  ArrayRef<MemoryInterface> getAllMemoryInterfaces() {
    return memoryInterfaces;
  }

  void clear() {
    mapMemrefPortBank2MemoryInterface.clear();
    mapMemrefToNumPorts.clear();
    mapMemrefToNumBanks.clear();
    memoryInterfaces.clear();
  }

private:
  SafeDenseMap<std::tuple<Value, int64_t, int64_t>, size_t>
      mapMemrefPortBank2MemoryInterface;
  SmallVector<MemoryInterface> memoryInterfaces;
  SafeDenseMap<Value, int64_t> mapMemrefToNumPorts;
  SafeDenseMap<Value, int64_t> mapMemrefToNumBanks;

public:
  MemrefPortInterface declNewPortInterface(OpBuilder &builder, Value mem,
                                           int64_t port) {
    MemrefPortInterface portInterface;
    auto memrefTy = mem.getType().dyn_cast<hir::MemrefType>();
    auto enableTy = hir::BusTensorType::get(
        builder.getContext(), memrefTy.getNumBanks(), builder.getI1Type());
    auto addrTy =
        hir::BusTensorType::get(builder.getContext(), memrefTy.getNumBanks(),
                                builder.getIntegerType(helper::clog2(
                                    memrefTy.getNumElementsPerBank())));
    auto dataTy =
        hir::BusTensorType::get(builder.getContext(), memrefTy.getNumBanks(),
                                memrefTy.getElementType());
    if (hasAddrBus(mem, port)) {
      assert(memrefTy.getNumElementsPerBank() > 1);
      portInterface.addrEnableBusTensor =
          builder.create<hir::BusTensorOp>(builder.getUnknownLoc(), enableTy);
      portInterface.addrDataBusTensor =
          builder.create<hir::BusTensorOp>(builder.getUnknownLoc(), addrTy);
    }
    if (hasRdBus(mem, port)) {
      portInterface.rdEnableBusTensor =
          builder.create<hir::BusTensorOp>(builder.getUnknownLoc(), enableTy);
      portInterface.rdDataBusTensor =
          builder.create<hir::BusTensorOp>(builder.getUnknownLoc(), dataTy);
    }
    if (hasWrBus(mem, port)) {
      portInterface.wrEnableBusTensor =
          builder.create<hir::BusTensorOp>(builder.getUnknownLoc(), enableTy);
      portInterface.wrDataBusTensor =
          builder.create<hir::BusTensorOp>(builder.getUnknownLoc(), dataTy);
    }
    return portInterface;
  }

  MemrefPortInterface emitDuplicatePortInterface(OpBuilder &builder, Value mem,
                                                 int64_t port) {
    auto duplicatePortInterface = declNewPortInterface(builder, mem, port);
    auto numBanks = getNumBanks(mem);
    for (size_t bank = 0; bank < numBanks; bank++) {
      auto *memoryInterface = getInterface(mem, port, bank);

      if (hasAddrBus(mem, port)) {
        auto duplicateAddrEnBus = getBusFromTensor(
            builder, duplicatePortInterface.addrEnableBusTensor, bank);
        memoryInterface->attachAddrEnBus(builder, duplicateAddrEnBus);

        auto duplicateAddrDataBus = getBusFromTensor(
            builder, duplicatePortInterface.addrDataBusTensor, bank);
        memoryInterface->attachAddrDataBus(builder, duplicateAddrEnBus,
                                           duplicateAddrDataBus);
      }
      if (hasRdBus(mem, port)) {
        auto duplicateRdEnBus = getBusFromTensor(
            builder, duplicatePortInterface.rdEnableBusTensor, bank);
        memoryInterface->attachRdEnBus(builder, duplicateRdEnBus);
        emitBusTensorAssignElementLogic(
            builder, memoryInterface->getRdDataBus(),
            duplicatePortInterface.rdDataBusTensor, bank);
      }
      if (hasWrBus(mem, port)) {
        auto duplicateWrEnBus = getBusFromTensor(
            builder, duplicatePortInterface.wrEnableBusTensor, bank);
        memoryInterface->attachWrEnBus(builder, duplicateWrEnBus);

        auto duplicateWrDataBus = getBusFromTensor(
            builder, duplicatePortInterface.wrDataBusTensor, bank);
        memoryInterface->attachWrDataBus(builder, duplicateWrEnBus,
                                         duplicateWrDataBus);
      }
    }
    return duplicatePortInterface;
  }

  bool hasAddrBus(Value mem, int64_t port) {
    return mem.getType().dyn_cast<hir::MemrefType>().getNumElementsPerBank() >
           1;
  }
  bool hasRdBus(Value mem, int64_t port) {
    return getInterface(mem, port, 0)->hasRdBus();
  }
  bool hasWrBus(Value mem, int64_t port) {
    return getInterface(mem, port, 0)->hasWrBus();
  }
};

void emitCallOpOperandsForMemrefPort(
    OpBuilder &builder, MemrefInfo &memrefInfo, Value mem,
    SmallVectorImpl<Value> &operands, SmallVectorImpl<Type> &inputTypes,
    SmallVectorImpl<DictionaryAttr> &inputAttrs);

Value emitAddrEnableBus(OpBuilder &builder, hir::MemrefType memrefTy);
Value emitAddrDataBus(OpBuilder &builder, hir::MemrefType memrefTy);
Value emitRdEnableBus(OpBuilder &builder, hir::MemrefType memrefTy);
Value emitRdDataBus(OpBuilder &builder, hir::MemrefType memrefTy);
Value emitWrEnableBus(OpBuilder &builder, hir::MemrefType memrefTy);
Value emitWrDataBus(OpBuilder &builder, hir::MemrefType memrefTy);

std::string createHWMemoryName(llvm::StringRef memKind,
                               hir::MemrefType memrefTy, ArrayAttr memPorts);
void emitMemoryInstance(OpBuilder &builder, hir::MemrefType memrefTy,
                        ArrayRef<MemoryInterface> memoryInterfaces,
                        llvm::StringRef memKind, llvm::StringRef memName,
                        llvm::Optional<std::string> instanceName, Value tstart);
Value getRegionTimeVar(Operation *);
