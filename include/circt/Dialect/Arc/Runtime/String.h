//===- String.h - Dynamic Strings for the ArcRuntime ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares DynamicString, used by arcRuntimeIR_string*
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_RUNTIME_STRING_H
#define CIRCT_DIALECT_ARC_RUNTIME_STRING_H

#include <cassert>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <type_traits>

namespace circt {
namespace arc {
namespace runtime {

/// A struct to represent Dynamic Strings, modelled after std::string in LLVM
/// These strings follow value semantics, i.e. they are copied on assignment
struct DynamicString {
  DynamicString() {
    size = 0;
    data = nullptr;
  }
  uint64_t size;
  char *data;
};

static_assert(std::is_standard_layout_v<DynamicString>,
              "DynamicString must be standard layout");

} // namespace runtime
} // namespace arc
} // namespace circt

#endif // CIRCT_DIALECT_ARC_RUNTIME_STRING_H
