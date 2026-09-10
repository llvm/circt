//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Internal API structs providing metadata for traced signals.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_RUNTIME_TRACETAPS_H
#define CIRCT_DIALECT_ARC_RUNTIME_TRACETAPS_H

#include <cstdint>
#include <type_traits>

#pragma pack(push, 1)

struct alignas(8) ArcTraceTap {
  /// Byte offset of the traced value within the model state
  uint64_t stateOffset;
  /// Byte offset to the null terminator of this signal's last alias in the
  /// names array
  uint64_t nameOffset;
  /// Bit width of the traced signal
  uint32_t typeBits;
  /// Padding and reserved for future use
  uint32_t reserved;
};
static_assert(sizeof(ArcTraceTap) == 3 * 8);

struct alignas(8) ArcModelTraceInfo {
  /// Number of trace taps in the array
  uint64_t numTraceTaps;
  /// Array of trace tap information
  struct ArcTraceTap *traceTaps;
  /// Combined list of names and aliases of the trace taps separated by
  /// null terminators
  const char *traceTapNames;
  /// Required capacity in 8 byte increments of the trace buffer
  uint64_t traceBufferCapacity;
};
static_assert(sizeof(ArcModelTraceInfo) == 4 * 8);

namespace circt::arc::runtime {
static constexpr uint32_t defaultTraceBufferCapacity = 256 * 1024;
} // namespace circt::arc::runtime

#pragma pack(pop)

#endif // CIRCT_DIALECT_ARC_RUNTIME_TRACETAPS_H
