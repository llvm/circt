//===- LowLevel.h - Cosim low level implementation --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COSIM_LOWLEVEL_H
#define COSIM_LOWLEVEL_H

#include <atomic>

#include "cosim/Utils.h"

namespace esi {
namespace cosim {

// Implements a bi-directional, thread-safe bridge between the RPC server and
// DPI functions for low level functionality.
class LowLevel {
public:
  LowLevel() = default;
  ~LowLevel() = default;
  /// Disallow copying. There is only ONE low level object per RPC server, so
  /// copying is almost always a bug.
  LowLevel(const LowLevel &) = delete;

  TSQueue<uint32_t> readReqs;
  TSQueue<std::pair<uint64_t, uint8_t>> readResps;
  std::atomic<unsigned> readsOutstanding = 0;

  TSQueue<std::pair<uint32_t, uint64_t>> writeReqs;
  TSQueue<uint8_t> writeResps;
  std::atomic<unsigned> writesOutstanding = 0;
};

} // namespace cosim
} // namespace esi

#endif
