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

} // extern "C"

namespace circt {
namespace arc {
namespace runtime {

using SimStringRuntimeSymbols =
    std::tuple<RuntimeSymbol<decltype(&circt_sim_string_len)>>;

#ifdef ARC_RUNTIME_JITBIND_FNDECL
/// Return the sim string runtime symbols to register with the JIT.
const SimStringRuntimeSymbols &getSimStringRuntimeSymbols();
#endif // ARC_RUNTIME_JITBIND_FNDECL

} // namespace runtime
} // namespace arc
} // namespace circt

#endif // CIRCT_DIALECT_ARC_RUNTIME_SIMSTRING_H
