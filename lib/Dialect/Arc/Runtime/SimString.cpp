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

#include <cstdlib>
#include <cstring>
#include <limits>

extern "C" int32_t circt_sim_string_len(const char *str) {
  if (!str)
    return 0;
  return static_cast<int32_t>(std::strlen(str));
}

extern "C" int32_t circt_sim_string_cmp(const char *lhs, const char *rhs) {
  if (!lhs)
    lhs = "";
  if (!rhs)
    rhs = "";
  return std::strcmp(lhs, rhs);
}

extern "C" const char *circt_sim_string_concat(const char *lhs,
                                               const char *rhs) {
  if (!lhs)
    lhs = "";
  if (!rhs)
    rhs = "";

  size_t lhsLen = std::strlen(lhs);
  size_t rhsLen = std::strlen(rhs);
  if (rhsLen > std::numeric_limits<size_t>::max() - lhsLen - 1)
    return "";

  char *out = static_cast<char *>(std::malloc(lhsLen + rhsLen + 1));
  if (!out)
    return "";
  std::memcpy(out, lhs, lhsLen);
  std::memcpy(out + lhsLen, rhs, rhsLen);
  out[lhsLen + rhsLen] = '\0';
  return out;
}

extern "C" const char *circt_sim_string_int_to_string(const void *input,
                                                      uint32_t bitWidth) {
  if (!input || bitWidth == 0)
    return "";

  const auto *bytes = static_cast<const uint8_t *>(input);
  uint32_t byteCount = (bitWidth + 7) / 8;
  uint32_t validBitsInTopByte = bitWidth % 8;
  uint8_t topByteMask =
      validBitsInTopByte == 0
          ? 0xff
          : static_cast<uint8_t>((1u << validBitsInTopByte) - 1u);

  char *out = static_cast<char *>(std::malloc(byteCount + 1));
  if (!out)
    return "";

  uint32_t outLen = 0;
  for (uint32_t index = byteCount; index > 0; --index) {
    uint8_t byte = bytes[index - 1];
    if (index == byteCount)
      byte &= topByteMask;
    if (byte)
      out[outLen++] = static_cast<char>(byte);
  }
  out[outLen] = '\0';
  return out;
}

extern "C" uint8_t circt_sim_string_get_char(const char *str, int32_t idx) {
  if (!str || idx < 0)
    return 0;
  size_t len = std::strlen(str);
  size_t pos = static_cast<size_t>(idx);
  if (pos >= len)
    return 0;
  return static_cast<uint8_t>(static_cast<unsigned char>(str[pos]));
}

extern "C" const char *circt_sim_string_substr(const char *str, int32_t start,
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

static const SimStringRuntimeSymbols simStringRuntimeSymbols = std::make_tuple(
    RuntimeSymbol<decltype(&circt_sim_string_len)>{"circt_sim_string_len",
                                                   &circt_sim_string_len},
    RuntimeSymbol<decltype(&circt_sim_string_cmp)>{"circt_sim_string_cmp",
                                                   &circt_sim_string_cmp},
    RuntimeSymbol<decltype(&circt_sim_string_concat)>{"circt_sim_string_concat",
                                                      &circt_sim_string_concat},
    RuntimeSymbol<decltype(&circt_sim_string_int_to_string)>{
        "circt_sim_string_int_to_string", &circt_sim_string_int_to_string},
    RuntimeSymbol<decltype(&circt_sim_string_get_char)>{
        "circt_sim_string_get_char", &circt_sim_string_get_char},
    RuntimeSymbol<decltype(&circt_sim_string_substr)>{
        "circt_sim_string_substr", &circt_sim_string_substr});

const SimStringRuntimeSymbols &getSimStringRuntimeSymbols() {
  return simStringRuntimeSymbols;
}

} // namespace runtime
} // namespace arc
} // namespace circt
#endif // ARC_RUNTIME_JIT_BIND
