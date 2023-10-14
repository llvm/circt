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

/// Wrap types in esi channels and return the port info struct.
static ServicePortInfo createReqResp(StringRef name, StringRef reqName,
                                     Type reqType, StringRef respName,
                                     Type respType) {
  assert(reqType || respType);
  auto *ctxt = reqType ? reqType.getContext() : respType.getContext();
  auto bundle = ChannelBundleType::get(
      ctxt,
      {BundledChannel{StringAttr::get(ctxt, reqName), ChannelDirection::to,
                      ChannelType::get(ctxt, reqType)},
       BundledChannel{StringAttr::get(ctxt, respName), ChannelDirection::from,
                      ChannelType::get(ctxt, respType)}},
      /*resettable=false*/ UnitAttr());
  return {StringAttr::get(ctxt, name), ServicePortInfo::Direction::toServer,
          bundle};
}

void RandomAccessMemoryDeclOp::getPortList(
    SmallVectorImpl<ServicePortInfo> &ports) {
  auto *ctxt = getContext();
  auto addressType = IntegerType::get(ctxt, llvm::Log2_64_Ceil(getDepth()));

  // Write port
  hw::StructType writeType = hw::StructType::get(
      ctxt,
      {hw::StructType::FieldInfo{StringAttr::get(ctxt, "address"), addressType},
       hw::StructType::FieldInfo{StringAttr::get(ctxt, "data"),
                                 getInnerType()}});
  ports.push_back(createReqResp("write", "req", writeType, "ack",
                                IntegerType::get(ctxt, 0)));

  // Read port
  ports.push_back(
      createReqResp("read", "address", addressType, "data", getInnerType()));
}
