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

void insertEnableSendLogic(OpBuilder &builder, Value inEnableBus,
                           Value outEnableBus, Value tVar,
                           IntegerAttr offsetAttr, ArrayRef<Value> indices) {

  auto c1 = builder.create<mlir::ConstantOp>(
      builder.getUnknownLoc(), builder.getI1Type(),
      mlir::IntegerAttr::get(builder.getI1Type(), 1));
  auto attrC0 = builder.getI64IntegerAttr(0);
  ArrayAttr ports = builder.getStrArrayAttr({"send"});

  if (inEnableBus.getType().isa<hir::BusType>()) {
    builder.create<hir::SendOp>(builder.getUnknownLoc(), c1, outEnableBus,
                                attrC0, tVar, offsetAttr);
    return;
  }
  auto i1BusTy = hir::BusType::get(builder.getContext(), builder.getI1Type());
  auto tBus = builder.create<hir::BusOp>(builder.getUnknownLoc(), i1BusTy);
  builder
      .create<hir::SendOp>(builder.getUnknownLoc(), c1, tBus, attrC0, tVar,
                           offsetAttr)
      ->setAttr("default", attrC0);

  auto enable = builder.create<hir::TensorExtractOp>(
      builder.getUnknownLoc(),
      inEnableBus.getType().dyn_cast<mlir::TensorType>().getElementType(),
      inEnableBus, indices, ports);
  auto orBus = builder.create<hir::BusOrOp>(builder.getUnknownLoc(),
                                            enable.getType(), tBus, enable);
  auto newEnableBus = builder.create<hir::TensorInsertOp>(
      builder.getUnknownLoc(), inEnableBus.getType(), orBus, inEnableBus,
      indices);
  builder.create<hir::BusAssignOp>(builder.getUnknownLoc(), outEnableBus,
                                   newEnableBus);
}

void insertDataSendLogic(OpBuilder &builder, Value data, Value enableBus,
                         Value inDataBus, Value outDataBus, Value tVar,
                         IntegerAttr offsetAttr, ArrayRef<Value> indices) {

  auto c1 = builder.create<mlir::ConstantOp>(
      builder.getUnknownLoc(), builder.getI1Type(),
      mlir::IntegerAttr::get(builder.getI1Type(), 1));
  auto attrC0 = builder.getI64IntegerAttr(0);
  ArrayAttr ports = builder.getStrArrayAttr({"send"});

  if (inDataBus.getType().isa<hir::BusType>()) {
    builder.create<hir::SendOp>(builder.getUnknownLoc(), data, outDataBus,
                                attrC0, tVar, offsetAttr);
    return;
  }

  auto i1BusTy = hir::BusType::get(builder.getContext(), builder.getI1Type());
  auto tBus = builder.create<hir::BusOp>(builder.getUnknownLoc(), i1BusTy);
  builder
      .create<hir::SendOp>(builder.getUnknownLoc(), c1, tBus, attrC0, tVar,
                           offsetAttr)
      ->setAttr("default", attrC0);

  auto inData = builder.create<hir::TensorExtractOp>(
      builder.getUnknownLoc(),
      inDataBus.getType().dyn_cast<mlir::TensorType>().getElementType(),
      inDataBus, indices, ports);
  auto newData =
      builder.create<hir::BusOp>(builder.getUnknownLoc(), inData.getType());
  builder.create<hir::SendOp>(builder.getUnknownLoc(), data, newData, attrC0,
                              tVar, offsetAttr);
  auto outData = builder.create<hir::BusSelectOp>(
      builder.getUnknownLoc(), inData.getType(), tBus, newData, inData);
  auto newDataBus = builder.create<hir::TensorInsertOp>(
      builder.getUnknownLoc(), inDataBus.getType(), outData, inDataBus,
      indices);
  builder.create<hir::BusAssignOp>(builder.getUnknownLoc(), outDataBus,
                                   newDataBus);
}
