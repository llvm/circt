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

template <typename T>
class TSQueue {
  using Lock = std::lock_guard<std::mutex>;

  std::mutex m;
  std::queue<T> q;

public:
  template <typename... E>
  void push(E... t) {
    Lock l(m);
    q.emplace(t...);
  }
  std::optional<T> pop() {
    Lock l(m);
    if (q.size() == 0)
      return std::nullopt;
    auto t = q.front();
    q.pop();
    return t;
  }
};

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

  TSQueue<uint32_t> readReqs;
  TSQueue<std::pair<uint64_t, uint8_t>> readResps;
  TSQueue<std::pair<uint32_t, uint64_t>> writeReqs;
  TSQueue<uint8_t> writeResps;
};

} // namespace cosim
} // namespace esi
} // namespace circt

#endif
