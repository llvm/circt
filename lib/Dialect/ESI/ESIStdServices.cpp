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
      {BundledChannel{StringAttr::get(ctxt, reqName), ChannelDirection::to,
                      ChannelType::get(ctxt, reqType)},
       BundledChannel{StringAttr::get(ctxt, respName), ChannelDirection::from,
                      ChannelType::get(ctxt, respType)}},
      /*resettable=false*/ UnitAttr());
  return {hw::InnerRefAttr::get(sym, StringAttr::get(ctxt, name)),
          ServicePortInfo::Direction::toServer, bundle};
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
