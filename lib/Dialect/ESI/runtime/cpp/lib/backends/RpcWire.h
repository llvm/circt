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

#include <libwebsockets.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

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

/// libwebsockets requires `LWS_PRE` bytes of leading slack before every
/// payload so the framing header can be written in place. A `WireFrame`
/// owns its buffer and knows where the actual payload starts and how long
/// it is; `writePtr()` returns the pointer to hand to `lws_write`.
struct WireFrame {
  std::vector<uint8_t> buf;
  size_t payloadSize = 0;
  bool isBinary = false;

  unsigned char *writePtr() {
    return reinterpret_cast<unsigned char *>(buf.data() + LWS_PRE);
  }
};

/// Build a cosim binary data frame in an LWS-ready buffer.
inline WireFrame buildLwsBinaryFrame(uint16_t channelId, const uint8_t *bytes,
                                     size_t size) {
  WireFrame f;
  f.isBinary = true;
  f.payloadSize = 2 + size;
  f.buf.resize(LWS_PRE + f.payloadSize);
  f.buf[LWS_PRE] = static_cast<uint8_t>(channelId & 0xFF);
  f.buf[LWS_PRE + 1] = static_cast<uint8_t>((channelId >> 8) & 0xFF);
  if (size > 0)
    std::memcpy(f.buf.data() + LWS_PRE + 2, bytes, size);
  return f;
}

/// Build a UTF-8 text frame in an LWS-ready buffer.
inline WireFrame buildLwsTextFrame(const std::string &text) {
  WireFrame f;
  f.isBinary = false;
  f.payloadSize = text.size();
  f.buf.resize(LWS_PRE + f.payloadSize);
  if (!text.empty())
    std::memcpy(f.buf.data() + LWS_PRE, text.data(), text.size());
  return f;
}

} // namespace cosim
} // namespace esi

#endif // ESI_BACKENDS_RPC_WIRE_H
