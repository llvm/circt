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

Value insertDataSendLogic(OpBuilder &builder, Value data, Value rootDataBusT,
                          ArrayRef<Value> indices, Value tVar,
                          IntegerAttr offsetAttr) {
  auto funcOp =
      builder.getInsertionBlock()->getParent()->getParentOfType<hir::FuncOp>();
  OpBuilder funcOpBuilder(funcOp);
  funcOpBuilder.setInsertionPointToStart(&funcOp.getFuncBody().front());
  Value nextDataBusT = funcOpBuilder.create<hir::BusOp>(
      funcOpBuilder.getUnknownLoc(), rootDataBusT.getType());

  auto c1 = builder.create<hw::ConstantOp>(
      builder.getUnknownLoc(), builder.getI1Type(),
      mlir::IntegerAttr::get(builder.getI1Type(), 1));
  auto attrC0 = IntegerAttr::get(c1.getType(), 0);

  assert(rootDataBusT.getType().isa<mlir::TensorType>());

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
  auto selectedData = builder.create<hir::BusSelectOp>(
      builder.getUnknownLoc(), nextDataBus.getType(), tBus, dataBus,
      nextDataBus);
  auto newDataBus = builder.create<hir::TensorInsertOp>(
      builder.getUnknownLoc(), nextDataBusT.getType(), selectedData,
      nextDataBusT, indices);
  builder.create<hir::BusAssignOp>(builder.getUnknownLoc(), rootDataBusT,
                                   newDataBus);
  return nextDataBusT;
}

Value insertDataRecvLogic(OpBuilder &builder, Location loc, ArrayAttr names,
                          Value dataBusT, ArrayRef<Value> indices,
                          uint64_t rdLatency, Value tVar,
                          IntegerAttr offsetAttr) {

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

Value insertDataTensorSendLogic(OpBuilder &builder, Value enableBusT,
                                Value currentDataBusT, Value rootDataBusT) {

  auto funcOp =
      builder.getInsertionBlock()->getParent()->getParentOfType<hir::FuncOp>();
  OpBuilder funcOpBuilder(funcOp);
  funcOpBuilder.setInsertionPointToStart(&funcOp.getFuncBody().front());
  Value nextDataBusT = funcOpBuilder.create<hir::BusOp>(
      funcOpBuilder.getUnknownLoc(), rootDataBusT.getType());

  assert(rootDataBusT.getType().isa<mlir::TensorType>());

  auto selectedDataBusT = builder.create<hir::BusSelectOp>(
      builder.getUnknownLoc(), nextDataBusT.getType(), enableBusT,
      currentDataBusT, nextDataBusT);
  builder.create<hir::BusAssignOp>(builder.getUnknownLoc(), rootDataBusT,
                                   selectedDataBusT);
  return nextDataBusT;
}
