//===- LowLevel.h - Cosim low level implementation --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements a bi-directional, thread-safe bridge between the RPC server and
// DPI functions for low level functionality.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_COSIM_LOWLEVEL_H
#define CIRCT_DIALECT_ESI_COSIM_LOWLEVEL_H

#include <mutex>
#include <optional>
#include <queue>
#include <string>

namespace circt {
namespace esi {
namespace cosim {

/// Several of the methods below are inline with the declaration to make them
/// candidates for inlining during compilation. This is particularly important
/// on the simulation side since polling happens at each clock and we do not
/// want to slow down the simulation any more than necessary.
class LowLevel {
public:
  LowLevel();
  ~LowLevel();
  /// Disallow copying. There is only ONE low level object per RPC server, so
  /// copying is almost always a bug.
  LowLevel(const LowLevel &) = delete;

  std::optional<uint32_t> popReadReq() {
    Lock l(m);
    if (readReqQueue.size() == 0)
      return std::nullopt;
    uint32_t address = readReqQueue.front();
    readReqQueue.pop();
    return address;
  }

  void pushReadReq(uint32_t address) {
    Lock l(m);
    readReqQueue.push(address);
  }

  std::optional<std::pair<uint64_t, uint8_t>> popReadResp() {
    Lock l(m);
    if (readRespQueue.size() == 0)
      return {};
    std::pair<uint64_t, uint8_t> dataErr = readRespQueue.front();
    readRespQueue.pop();
    return dataErr;
  }

  void pushReadResp(uint64_t data, uint8_t error) {
    Lock l(m);
    readRespQueue.emplace(data, error);
  }

private:
  using Lock = std::lock_guard<std::mutex>;

  /// This class needs to be thread-safe. All of the mutable member variables
  /// are protected with this object-wide lock. This may be a performance issue
  /// in the future.
  std::mutex m;

  std::queue<uint32_t> readReqQueue;
  std::queue<std::pair<uint64_t, uint8_t>> readRespQueue;
};

} // namespace cosim
} // namespace esi
} // namespace circt

#endif
