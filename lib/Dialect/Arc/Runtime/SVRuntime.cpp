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

#include <cstdlib>
#include <cstring>

extern "C" int32_t circt_sv_string_len(const char *str) {
  if (!str)
    return 0;
  return static_cast<int32_t>(std::strlen(str));
}

extern "C" int32_t circt_sv_strcmp(const char *lhs, const char *rhs) {
  if (!lhs)
    lhs = "";
  if (!rhs)
    rhs = "";
  return std::strcmp(lhs, rhs);
}

extern "C" uint8_t circt_sv_string_getc(const char *str, int32_t idx) {
  if (!str || idx < 0)
    return 0;
  size_t len = std::strlen(str);
  size_t pos = static_cast<size_t>(idx);
  if (pos >= len)
    return 0;
  return static_cast<uint8_t>(static_cast<unsigned char>(str[pos]));
}

extern "C" const char *circt_sv_string_substr(const char *str, int32_t start,
                                              int32_t end) {
  if (!str)
    str = "";
  if (start < 0)
    start = 0;
  if (end < start)
    return "";

  size_t len = std::strlen(str);
  size_t startPos = static_cast<size_t>(start);
  if (startPos >= len)
    return "";

  size_t endPos = static_cast<size_t>(end);
  if (endPos >= len)
    endPos = len - 1;

  size_t outLen = endPos - startPos + 1;
  char *out = static_cast<char *>(std::malloc(outLen + 1));
  if (!out)
    return "";
  std::memcpy(out, str + startPos, outLen);
  out[outLen] = '\0';
  return out;
}

#ifdef ARC_RUNTIME_JIT_BIND
namespace circt {
namespace arc {
namespace runtime {

static const SVRuntimeSymbol svRuntimeSymbols[] = {
    {"circt_sv_string_len", reinterpret_cast<void (*)()>(&circt_sv_string_len)},
    {"circt_sv_strcmp", reinterpret_cast<void (*)()>(&circt_sv_strcmp)},
    {"circt_sv_string_getc",
     reinterpret_cast<void (*)()>(&circt_sv_string_getc)},
    {"circt_sv_string_substr",
     reinterpret_cast<void (*)()>(&circt_sv_string_substr)},
    {nullptr, nullptr},
};

const SVRuntimeSymbol *getSVRuntimeSymbols() { return svRuntimeSymbols; }

} // namespace runtime
} // namespace arc
} // namespace circt
#endif // ARC_RUNTIME_JIT_BIND
