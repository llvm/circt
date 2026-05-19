//===- RpcWire.h - Shared cosim wire-format helpers -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Small inline helpers for encoding/decoding the cosim binary data-frame
// layout (`[u16 LE channel_id][payload]`). Used by both RpcServer.cpp and
// RpcClient.cpp so the two ends can't drift apart on wire format. Header is
// private to the CosimRpc library; not installed.
//
//===----------------------------------------------------------------------===//

#ifndef ESI_BACKENDS_RPC_WIRE_H
#define ESI_BACKENDS_RPC_WIRE_H

#include <cstddef>
#include <cstdint>
#include <string>

namespace esi {
namespace cosim {

/// Pack a cosim binary data frame: `[u16 LE channel_id][payload]`.
inline std::string buildDataFrame(uint16_t channelId, const uint8_t *bytes,
                                  size_t size) {
  std::string frame;
  frame.reserve(2 + size);
  uint8_t hdr[2] = {static_cast<uint8_t>(channelId & 0xFF),
                    static_cast<uint8_t>((channelId >> 8) & 0xFF)};
  frame.append(reinterpret_cast<const char *>(hdr), 2);
  frame.append(reinterpret_cast<const char *>(bytes), size);
  return frame;
}

} // namespace cosim
} // namespace esi

#endif // ESI_BACKENDS_RPC_WIRE_H
