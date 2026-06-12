//===- SVRuntime.cpp - SystemVerilog execution runtime --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the SystemVerilog execution runtime. See SVRuntime.h. Functions
// here are plain `extern "C"` leaf utilities; the JIT-binding table at the
// bottom is only compiled into the copy linked into arcilator
// (ARC_RUNTIME_JIT_BIND).
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/Runtime/SVRuntime.h"

#include <cstring>

extern "C" int32_t circt_sv_string_len(const char *str) {
  if (!str)
    return 0;
  return static_cast<int32_t>(std::strlen(str));
}

#ifdef ARC_RUNTIME_JIT_BIND
namespace circt {
namespace arc {
namespace runtime {

static const SVRuntimeSymbol svRuntimeSymbols[] = {
    {"circt_sv_string_len", reinterpret_cast<void (*)()>(&circt_sv_string_len)},
    {nullptr, nullptr},
};

const SVRuntimeSymbol *getSVRuntimeSymbols() { return svRuntimeSymbols; }

} // namespace runtime
} // namespace arc
} // namespace circt
#endif // ARC_RUNTIME_JIT_BIND
