//===- ESIOps.h - ESI operations --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ESI Ops are defined in tablegen.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_ESIOPS_H
#define CIRCT_DIALECT_ESI_ESIOPS_H

#include "circt/Dialect/ESI/ESIAttributes.h"
#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/ESI/ESITypes.h"

#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/Seq/SeqTypes.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace circt {
namespace esi {
/// Describes a service port. In the unidirection case, either (but not both)
/// type fields will be null.
struct ServicePortInfo {
  hw::InnerRefAttr port;
  ChannelBundleType type;
};

// Check that two types match, allowing for AnyType in 'expected'.
// NOLINTNEXTLINE(misc-no-recursion)
LogicalResult checkInnerTypeMatch(Type expected, Type actual);
/// Check that the channels on two bundles match allowing for AnyType in the
/// 'svc' bundle.
LogicalResult checkBundleTypeMatch(Operation *req,
                                   ChannelBundleType svcBundleType,
                                   ChannelBundleType reqBundleType,
                                   bool skipDirectionCheck);

} // namespace esi
} // namespace circt

#include "circt/Dialect/ESI/ESIInterfaces.h.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/ESI/ESI.h.inc"

#endif
