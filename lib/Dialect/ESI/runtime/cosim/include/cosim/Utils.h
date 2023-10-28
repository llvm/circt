//===- Utils.h - utility code for cosim -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COSIM_UTILS_H
#define COSIM_UTILS_H

#include <mutex>
#include <optional>
#include <queue>

namespace esi {
namespace cosim {

/// Thread safe queue. Just wraps std::queue protected with a lock.
template <typename T>
class TSQueue {
  using Lock = std::lock_guard<std::mutex>;

  std::mutex m;
  std::queue<T> q;

public:
  /// Push onto the queue.
  template <typename... E>
  void push(E... t) {
    Lock l(m);
    q.emplace(t...);
  }

  /// Pop something off the queue but return nullopt if the queue is empty. Why
  /// doesn't std::queue have anything like this?
  std::optional<T> pop() {
    Lock l(m);
    if (q.size() == 0)
      return std::nullopt;
    auto t = q.front();
    q.pop();
    return t;
  }
};

} // namespace cosim
} // namespace esi

#endif
