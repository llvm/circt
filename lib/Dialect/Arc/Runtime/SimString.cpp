//===- SimString.cpp - Simulation string runtime --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements runtime support for dynamic strings in the sim dialect. Functions
// here are plain `extern "C"` leaf utilities; the JIT-binding table at the
// bottom is only compiled into the copy linked into arcilator.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/Runtime/SimString.h"

#include <cstring>

extern "C" int32_t circt_sim_string_len(const char *str) {
  if (!str)
    return 0;
  return static_cast<int32_t>(std::strlen(str));
}

#ifdef ARC_RUNTIME_JIT_BIND
namespace circt {
namespace arc {
namespace runtime {

static const SimStringRuntimeSymbols simStringRuntimeSymbols =
    std::make_tuple(RuntimeSymbol<decltype(&circt_sim_string_len)>{
        "circt_sim_string_len", &circt_sim_string_len});

const SimStringRuntimeSymbols &getSimStringRuntimeSymbols() {
  return simStringRuntimeSymbols;
}

} // namespace runtime
} // namespace arc
} // namespace circt
#endif // ARC_RUNTIME_JIT_BIND
