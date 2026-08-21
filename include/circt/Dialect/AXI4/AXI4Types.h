//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AXI4_AXI4TYPES_H
#define CIRCT_DIALECT_AXI4_AXI4TYPES_H

#include "circt/Dialect/AXI4/AXI4Attributes.h"
#include "circt/Dialect/AXI4/AXI4Dialect.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/AXI4/AXI4Types.h.inc"

namespace circt {
namespace axi4 {

/// The five AXI4 channels.
enum class AXI4Channel { AW, W, B, AR, R };

// Fixed AXI4 field widths (bits).
constexpr unsigned kLenWidth = 8;
constexpr unsigned kSizeWidth = 3;
constexpr unsigned kBurstWidth = 2;
constexpr unsigned kLockWidth = 1;
constexpr unsigned kCacheWidth = 4;
constexpr unsigned kProtWidth = 3;
constexpr unsigned kQosWidth = 4;
constexpr unsigned kRegionWidth = 4;
constexpr unsigned kRespWidth = 2;
constexpr unsigned kLastWidth = 1;

/// Build the `hw.struct` payload type for one channel of an `!axi4.port`.
hw::StructType getChannelPayloadType(PortType port, AXI4Channel channel);

} // namespace axi4
} // namespace circt

#endif // CIRCT_DIALECT_AXI4_AXI4TYPES_H
