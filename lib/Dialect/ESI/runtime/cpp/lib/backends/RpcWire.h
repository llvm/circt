//===- RpcWire.h - Shared cosim wire-format helpers -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Small inline helpers for encoding/decoding the cosim binary data-frame
// layout (`[u64 LE channel_id][payload]`). Used by both RpcServer.cpp and
// RpcClient.cpp so the two ends can't drift apart on wire format. Header is
// private to the CosimRpc library; not installed.
//
//===----------------------------------------------------------------------===//

#ifndef ESI_BACKENDS_RPC_WIRE_H
#define ESI_BACKENDS_RPC_WIRE_H

#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <string>

namespace esi {
namespace cosim {

/// Pack a cosim binary data frame: `[u64 LE channel_id][payload]`.
inline std::string buildDataFrame(uint64_t channelId, const uint8_t *bytes,
                                  size_t size) {
  std::string frame;
  frame.reserve(8 + size);
  uint8_t hdr[8];
  for (int i = 0; i < 8; ++i)
    hdr[i] = static_cast<uint8_t>((channelId >> (8 * i)) & 0xFF);
  frame.append(reinterpret_cast<const char *>(hdr), 8);
  frame.append(reinterpret_cast<const char *>(bytes), size);
  return frame;
}

/// Size of the binary data-frame header (LE channel_id).
inline constexpr size_t kDataFrameHeaderSize = 8;

/// Parse a cosim binary data frame. On success returns true and sets
/// `channelId` to the decoded id; `payload` / `payloadSize` point into
/// `data` past the header. Returns false if `data` is too short to contain
/// the header.
inline bool parseDataFrame(const std::string &data, uint64_t &channelId,
                           const uint8_t *&payload, size_t &payloadSize) {
  if (data.size() < kDataFrameHeaderSize)
    return false;
  uint64_t id = 0;
  for (int i = 0; i < 8; ++i)
    id |= static_cast<uint64_t>(static_cast<uint8_t>(data[i])) << (8 * i);
  channelId = id;
  payload = reinterpret_cast<const uint8_t *>(data.data()) +
            kDataFrameHeaderSize;
  payloadSize = data.size() - kDataFrameHeaderSize;
  return true;
}

/// Cross-thread error channel for the IXWebSocket network thread.
///
/// IXWebSocket invokes user callbacks from its internal network thread; any
/// exception escaping that callback kills the process. Call sites catch and
/// `record()` the first fault; the next public method on the owning
/// RpcServer/RpcClient calls `check()`, which consumes the stash and rethrows
/// on the user's thread.
class FaultStash {
public:
  /// Stash the first exception we see (subsequent ones are dropped -- the
  /// first is the most informative). Safe to call from any thread.
  void record(std::exception_ptr ep) noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    if (!fault)
      fault = ep;
  }

  /// Consume and rethrow the stored fault on the caller's thread, if any.
  void check() {
    std::exception_ptr ep;
    {
      std::lock_guard<std::mutex> lock(mutex);
      // `std::exception_ptr` has no member `swap()`; use the free function.
      std::swap(ep, fault);
    }
    if (ep)
      std::rethrow_exception(ep);
  }

private:
  std::mutex mutex;
  std::exception_ptr fault;
};

} // namespace cosim
} // namespace esi

#endif // ESI_BACKENDS_RPC_WIRE_H
