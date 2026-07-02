//===- SimString.h - Simulation string runtime ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares runtime support for dynamic strings in the sim dialect. These leaf
// C functions are statically linked into arcilator and bound to the JIT
// alongside the core Arc runtime.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_RUNTIME_SIMSTRING_H
#define CIRCT_DIALECT_ARC_RUNTIME_SIMSTRING_H

#include "circt/Dialect/Arc/Runtime/RuntimeSymbol.h"

#include <cstdint>
#include <tuple>

extern "C" {

/// Return the length of a NUL-terminated string (a null pointer is treated as
/// the empty string).
int32_t circt_sim_string_len(const char *str);

/// Compare two NUL-terminated strings (each null pointer is treated as the
/// empty string), returning the result of `strcmp`.
int32_t circt_sim_string_cmp(const char *lhs, const char *rhs);

/// Concatenate two NUL-terminated strings (each null pointer is treated as the
/// empty string), returning a freshly allocated NUL-terminated string.
const char *circt_sim_string_concat(const char *lhs, const char *rhs);

/// Convert a little-endian integer byte buffer into a freshly allocated
/// NUL-terminated string using packed-byte string assignment semantics.
const char *circt_sim_string_int_to_string(const void *input,
                                           uint32_t bitWidth);

/// Return the byte at index `idx` of a NUL-terminated string, or 0 if the
/// index is out of range or the string is null.
uint8_t circt_sim_string_get_char(const char *str, int32_t idx);

/// Return the substring `str[start..end]` (inclusive) as a freshly allocated
/// NUL-terminated string. Out-of-range bounds are clamped; an empty result is
/// returned as "".
const char *circt_sim_string_substr(const char *str, int32_t start,
                                    int32_t end);

} // extern "C"

namespace circt {
namespace arc {
namespace runtime {

using SimStringRuntimeSymbols =
    std::tuple<RuntimeSymbol<decltype(&circt_sim_string_len)>,
               RuntimeSymbol<decltype(&circt_sim_string_cmp)>,
               RuntimeSymbol<decltype(&circt_sim_string_concat)>,
               RuntimeSymbol<decltype(&circt_sim_string_int_to_string)>,
               RuntimeSymbol<decltype(&circt_sim_string_get_char)>,
               RuntimeSymbol<decltype(&circt_sim_string_substr)>>;

#ifdef ARC_RUNTIME_JITBIND_FNDECL
/// Return the sim string runtime symbols to register with the JIT.
const SimStringRuntimeSymbols &getSimStringRuntimeSymbols();
#endif // ARC_RUNTIME_JITBIND_FNDECL

} // namespace runtime
} // namespace arc
} // namespace circt

#endif // CIRCT_DIALECT_ARC_RUNTIME_SIMSTRING_H
