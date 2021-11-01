//=========- MemrefLoweringUtils.cpp - Utils for memref lowering pass---======//
//
// This file implements utilities for memref lowering pass.
//
//===----------------------------------------------------------------------===//

#include "MemrefLoweringUtils.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include <iostream>
using namespace circt;
using namespace hir;
// Helper functions.
static Type getTensorElementType(Value tensor) {
  auto ty = tensor.getType().dyn_cast<mlir::TensorType>();
  assert(ty);
  return ty.getElementType();
}
static Type getBusElementType(Value bus) {
  auto ty = bus.getType().dyn_cast<hir::BusType>();
  assert(ty);
  return ty.getElementType();
}
// hir.func @ADDR_BRAM_2P_BxDxT_loc_0x1(%addr:i<LogD>,
//                                      %addr_data_in:tensor<Bx!hir.bus<LogD>>,
//                                      %addr_en_in:tensor<Bx!hir.bus<i1>>,
//                                      %addr_data_out:tensor<Bx!hir.bus<LogD>>,
//                                      %addr_en_out:tensor<Bx!hir.bus<i1>>)
//                                      at %t{
//
//    %t_bus = hir.bus : !hir.bus<i1>
//    hir.send %c1 to %t_bus at %t {default=0}
//    //Calc addr_data_out bus.
//    %addr_in = hir.tensor.extract %addr_data_in[%c0, %c1]
//    %new_addr = hir.bus : !hir.bus<i<LogD>>
//    hir.send %addr to %new_addr at %t
//    %addr_out = hir.bus.select %t_bus, %new_addr, %addr_in
//    %new_addr_data = hir.tensor.insert %addr_out to %addr_data_in[%c0,%c1]
//    hir.bus.assign %addr_data_out with %new_addr_data
//
//    //Calc addr_en_out bus.
//    %addr_en = hir.tensor.extract %addr_en_in[%c0, %c1]
//    %or = hir.bus.or %t_bus, %addr_en
//    %new_addr_en = hir.tensor.insert %or to %addr_en_in[%c0, %c1]
//    hir.bus.assign %addr_en_out , %new_addr_en
//
//    hir.return
//}

Value insertEnableSendLogic(OpBuilder &builder, Value prevEnableBus, Value tVar,
                            IntegerAttr offsetAttr, ArrayRef<Value> indices) {

  auto nextEnableBus = builder.create<hir::BusOp>(builder.getUnknownLoc(),
                                                  prevEnableBus.getType());
  auto c1 = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), builder.getI1Type(),
      mlir::IntegerAttr::get(builder.getI1Type(), 1));
  auto attrC0 = IntegerAttr::get(c1.getType(), 0);

  if (prevEnableBus.getType().isa<hir::BusType>()) {
    builder.create<hir::SendOp>(builder.getUnknownLoc(), c1, nextEnableBus,
                                tVar, offsetAttr);
    return nextEnableBus;
  }
  auto i1BusTy = hir::BusType::get(builder.getContext(), builder.getI1Type());
  auto tBus = builder.create<hir::BusOp>(builder.getUnknownLoc(), i1BusTy);
  builder
      .create<hir::SendOp>(builder.getUnknownLoc(), c1, tBus, tVar, offsetAttr)
      ->setAttr("default", attrC0);

  auto enable = builder.create<hir::TensorExtractOp>(
      builder.getUnknownLoc(), getTensorElementType(nextEnableBus),
      nextEnableBus, indices);
  auto orBus = builder.create<hir::BusOrOp>(builder.getUnknownLoc(),
                                            enable.getType(), tBus, enable);
  auto newEnableBus = builder.create<hir::TensorInsertOp>(
      builder.getUnknownLoc(), nextEnableBus.getType(), orBus, nextEnableBus,
      indices);
  builder.create<hir::BusAssignOp>(builder.getUnknownLoc(), prevEnableBus,
                                   newEnableBus);
  return nextEnableBus;
}

Value insertDataSendLogic(OpBuilder &builder, Value data, Value prevDataBusT,
                          Value tVar, IntegerAttr offsetAttr,
                          ArrayRef<Value> indices) {
  Value nextDataBusT = builder.create<hir::BusOp>(builder.getUnknownLoc(),
                                                  prevDataBusT.getType());
  auto c1 = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), builder.getI1Type(),
      mlir::IntegerAttr::get(builder.getI1Type(), 1));
  auto attrC0 = IntegerAttr::get(c1.getType(), 0);

  if (prevDataBusT.getType().isa<hir::BusType>()) {
    builder.create<hir::SendOp>(builder.getUnknownLoc(), data, nextDataBusT,
                                tVar, offsetAttr);
    return nextDataBusT;
  }

  auto i1BusTy = hir::BusType::get(builder.getContext(), builder.getI1Type());
  auto tBus = builder.create<hir::BusOp>(builder.getUnknownLoc(), i1BusTy);
  builder
      .create<hir::SendOp>(builder.getUnknownLoc(), c1, tBus, tVar, offsetAttr)
      ->setAttr("default", attrC0);

  auto nextDataBus = builder.create<hir::TensorExtractOp>(
      builder.getUnknownLoc(), getTensorElementType(nextDataBusT), nextDataBusT,
      indices);
  auto dataBus = builder.create<hir::BusOp>(builder.getUnknownLoc(),
                                            nextDataBus.getType());
  builder.create<hir::SendOp>(builder.getUnknownLoc(), data, dataBus, tVar,
                              offsetAttr);
  auto combinedData = builder.create<hir::BusSelectOp>(
      builder.getUnknownLoc(), nextDataBus.getType(), tBus, dataBus,
      nextDataBus);
  auto newDataBus = builder.create<hir::TensorInsertOp>(
      builder.getUnknownLoc(), nextDataBusT.getType(), combinedData,
      nextDataBusT, indices);
  builder.create<hir::BusAssignOp>(builder.getUnknownLoc(), prevDataBusT,
                                   newDataBus);
  return nextDataBusT;
}

Value insertDataRecvLogic(OpBuilder &builder, Location loc, ArrayAttr names,
                          uint64_t rdLatency, Value dataBusT, Value tVar,
                          IntegerAttr offsetAttr, ArrayRef<Value> indices) {

  assert(rdLatency >= 0);
  auto recvOffsetAttr =
      (rdLatency == 0)
          ? offsetAttr
          : builder.getI64IntegerAttr(offsetAttr.getInt() + rdLatency);

  auto dataBus = builder.create<hir::TensorExtractOp>(
      builder.getUnknownLoc(), getTensorElementType(dataBusT), dataBusT);

  auto receiveOp = builder.create<hir::RecvOp>(loc, getBusElementType(dataBus),
                                               dataBus, tVar, recvOffsetAttr);

  if (names)
    receiveOp->setAttr("names", names);
  return receiveOp.getResult();
}
