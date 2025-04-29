//===- ESIStdServices.cpp - ESI standard services -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESIServices.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

#include <map>
#include <memory>

using namespace circt;
using namespace circt::esi;

/// Utility function to create a req/resp pair bundle service port.
static ServicePortInfo createReqResp(StringAttr sym, Twine name,
                                     StringRef reqName, Type reqType,
                                     StringRef respName, Type respType) {
  assert(reqType && respType);
  auto *ctxt = reqType ? reqType.getContext() : respType.getContext();
  auto bundle = ChannelBundleType::get(
      ctxt,
      {BundledChannel{StringAttr::get(ctxt, reqName), ChannelDirection::from,
                      ChannelType::get(ctxt, reqType)},
       BundledChannel{StringAttr::get(ctxt, respName), ChannelDirection::to,
                      ChannelType::get(ctxt, respType)}},
      /*resettable=false*/ UnitAttr());
  return {hw::InnerRefAttr::get(sym, StringAttr::get(ctxt, name)), bundle};
}

ServicePortInfo RandomAccessMemoryDeclOp::writePortInfo() {
  auto *ctxt = getContext();
  auto addressType = IntegerType::get(ctxt, llvm::Log2_64_Ceil(getDepth()));

  // Write port
  hw::StructType writeType = hw::StructType::get(
      ctxt,
      {hw::StructType::FieldInfo{StringAttr::get(ctxt, "address"), addressType},
       hw::StructType::FieldInfo{StringAttr::get(ctxt, "data"),
                                 getInnerType()}});
  return createReqResp(getSymNameAttr(), "write", "req", writeType, "ack",
                       IntegerType::get(ctxt, 0));
}

ServicePortInfo RandomAccessMemoryDeclOp::readPortInfo() {
  auto *ctxt = getContext();
  auto addressType = IntegerType::get(ctxt, llvm::Log2_64_Ceil(getDepth()));

  return createReqResp(getSymNameAttr(), "read", "address", addressType, "data",
                       getInnerType());
}

void RandomAccessMemoryDeclOp::getPortList(
    SmallVectorImpl<ServicePortInfo> &ports) {
  ports.push_back(writePortInfo());
  ports.push_back(readPortInfo());
}

void CallServiceDeclOp::getPortList(SmallVectorImpl<ServicePortInfo> &ports) {
  auto *ctxt = getContext();
  ports.push_back(ServicePortInfo{
      hw::InnerRefAttr::get(getSymNameAttr(), StringAttr::get(ctxt, "call")),
      ChannelBundleType::get(
          ctxt,
          {BundledChannel{StringAttr::get(ctxt, "arg"), ChannelDirection::from,
                          ChannelType::get(ctxt, AnyType::get(ctxt))},
           BundledChannel{StringAttr::get(ctxt, "result"), ChannelDirection::to,
                          ChannelType::get(ctxt, AnyType::get(ctxt))}},
          /*resettable=*/UnitAttr())});
}

void FuncServiceDeclOp::getPortList(SmallVectorImpl<ServicePortInfo> &ports) {
  auto *ctxt = getContext();
  ports.push_back(ServicePortInfo{
      hw::InnerRefAttr::get(getSymNameAttr(), StringAttr::get(ctxt, "call")),
      ChannelBundleType::get(
          ctxt,
          {BundledChannel{StringAttr::get(ctxt, "arg"), ChannelDirection::to,
                          ChannelType::get(ctxt, AnyType::get(ctxt))},
           BundledChannel{StringAttr::get(ctxt, "result"),
                          ChannelDirection::from,
                          ChannelType::get(ctxt, AnyType::get(ctxt))}},
          /*resettable=*/UnitAttr())});
}

void MMIOServiceDeclOp::getPortList(SmallVectorImpl<ServicePortInfo> &ports) {
  auto *ctxt = getContext();
  // Read only port.
  ports.push_back(ServicePortInfo{
      hw::InnerRefAttr::get(getSymNameAttr(), StringAttr::get(ctxt, "read")),
      ChannelBundleType::get(
          ctxt,
          {BundledChannel{
               StringAttr::get(ctxt, "offset"), ChannelDirection::to,
               ChannelType::get(
                   ctxt,
                   IntegerType::get(
                       ctxt, 32, IntegerType::SignednessSemantics::Unsigned))},
           BundledChannel{StringAttr::get(ctxt, "data"), ChannelDirection::from,
                          ChannelType::get(ctxt, IntegerType::get(ctxt, 64))}},
          /*resettable=*/UnitAttr())});
  // Read-write port.
  auto cmdType = hw::StructType::get(
      ctxt, {
                hw::StructType::FieldInfo{StringAttr::get(ctxt, "write"),
                                          IntegerType::get(ctxt, 1)},
                hw::StructType::FieldInfo{
                    StringAttr::get(ctxt, "offset"),
                    IntegerType::get(
                        ctxt, 32, IntegerType::SignednessSemantics::Unsigned)},
                hw::StructType::FieldInfo{StringAttr::get(ctxt, "data"),
                                          IntegerType::get(ctxt, 64)},
            });
  ports.push_back(ServicePortInfo{
      hw::InnerRefAttr::get(getSymNameAttr(),
                            StringAttr::get(ctxt, "read_write")),
      ChannelBundleType::get(
          ctxt,
          {BundledChannel{StringAttr::get(ctxt, "cmd"), ChannelDirection::to,
                          ChannelType::get(ctxt, cmdType)},
           BundledChannel{StringAttr::get(ctxt, "data"), ChannelDirection::from,
                          ChannelType::get(ctxt, IntegerType::get(ctxt, 64))}},
          /*resettable=*/UnitAttr())});
}

ServicePortInfo HostMemServiceDeclOp::writePortInfo() {
  auto *ctxt = getContext();
  auto addressType =
      IntegerType::get(ctxt, 64, IntegerType::SignednessSemantics::Unsigned);

  // Write port
  hw::StructType writeType = hw::StructType::get(
      ctxt,
      {hw::StructType::FieldInfo{StringAttr::get(ctxt, "address"), addressType},
       hw::StructType::FieldInfo{
           StringAttr::get(ctxt, "tag"),
           IntegerType::get(ctxt, 8,
                            IntegerType::SignednessSemantics::Unsigned)},
       hw::StructType::FieldInfo{StringAttr::get(ctxt, "data"),
                                 AnyType::get(ctxt)}});
  return createReqResp(
      getSymNameAttr(), "write", "req", writeType, "ackTag",
      IntegerType::get(ctxt, 8, IntegerType::SignednessSemantics::Unsigned));
}

ServicePortInfo HostMemServiceDeclOp::readPortInfo() {
  auto *ctxt = getContext();
  auto addressType =
      IntegerType::get(ctxt, 64, IntegerType::SignednessSemantics::Unsigned);

  hw::StructType readReqType = hw::StructType::get(
      ctxt, {
                hw::StructType::FieldInfo{StringAttr::get(ctxt, "address"),
                                          addressType},
                hw::StructType::FieldInfo{
                    StringAttr::get(ctxt, "tag"),
                    IntegerType::get(
                        ctxt, 8, IntegerType::SignednessSemantics::Unsigned)},
            });
  hw::StructType readRespType = hw::StructType::get(
      ctxt, {
                hw::StructType::FieldInfo{
                    StringAttr::get(ctxt, "tag"),
                    IntegerType::get(
                        ctxt, 8, IntegerType::SignednessSemantics::Unsigned)},
                hw::StructType::FieldInfo{StringAttr::get(ctxt, "data"),
                                          AnyType::get(ctxt)},
            });
  return createReqResp(getSymNameAttr(), "read", "req", readReqType, "resp",
                       readRespType);
}

void HostMemServiceDeclOp::getPortList(
    SmallVectorImpl<ServicePortInfo> &ports) {
  ports.push_back(writePortInfo());
  ports.push_back(readPortInfo());
}

void TelemetryServiceDeclOp::getPortList(
    SmallVectorImpl<ServicePortInfo> &ports) {
  auto *ctxt = getContext();
  ports.push_back(ServicePortInfo{
      hw::InnerRefAttr::get(getSymNameAttr(), StringAttr::get(ctxt, "report")),
      ChannelBundleType::get(
          ctxt,
          {BundledChannel{StringAttr::get(ctxt, "get"), ChannelDirection::to,
                          ChannelType::get(ctxt, IntegerType::get(ctxt, 1))},
           BundledChannel{StringAttr::get(ctxt, "data"), ChannelDirection::from,
                          ChannelType::get(ctxt, AnyType::get(ctxt))}},
          /*resettable=*/UnitAttr())});
}
