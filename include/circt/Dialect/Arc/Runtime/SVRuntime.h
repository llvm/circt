//===- SVRuntime.h - SystemVerilog execution runtime ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares the SystemVerilog execution runtime: leaf C functions that lowered
// models call (via `func.call`) to evaluate dynamic SystemVerilog constructs
// which have no fixed-width hardware lowering (strings, dynamic containers,
// ...). They are statically linked into arcilator and bound to the JIT
// alongside the core Arc runtime.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_RUNTIME_SVRUNTIME_H
#define CIRCT_DIALECT_ARC_RUNTIME_SVRUNTIME_H

#include <cstdint>

extern "C" {

/// Return the length of a NUL-terminated string (a null pointer is treated as
/// the empty string).
int32_t circt_sv_string_len(const char *str);

/// Compare two NUL-terminated strings (each null pointer is treated as the
/// empty string), returning the result of `strcmp`.
int32_t circt_sv_strcmp(const char *lhs, const char *rhs);

/// Return the byte at index `idx` of a NUL-terminated string, or 0 if the
/// index is out of range or the string is null.
uint8_t circt_sv_string_getc(const char *str, int32_t idx);

/// Return the substring `str[start..end]` (inclusive) as a freshly allocated
/// NUL-terminated string. Out-of-range bounds are clamped; an empty result is
/// returned as "".
const char *circt_sv_string_substr(const char *str, int32_t start, int32_t end);

} // extern "C"

namespace circt {
namespace arc {
namespace runtime {

/// A single JIT-bindable SystemVerilog runtime symbol: its name and the address
/// of its `extern "C"` implementation, held as a generic function pointer so
/// `ExecutorAddr::fromPtr` keeps the function-pointer unwrap (e.g. the correct
/// pointer-authentication strip) rather than the data-pointer one.
struct SVRuntimeSymbol {
  const char *name;
  void (*addr)();
};

#ifdef ARC_RUNTIME_JITBIND_FNDECL
/// Return the table of SystemVerilog runtime symbols to register with the JIT,
/// terminated by a `{nullptr, nullptr}` sentinel.
const SVRuntimeSymbol *getSVRuntimeSymbols();
#endif // ARC_RUNTIME_JITBIND_FNDECL

} // namespace runtime
} // namespace arc
} // namespace circt

#endif // CIRCT_DIALECT_ARC_RUNTIME_SVRUNTIME_H
