//===- String.h - Format descriptor for the ArcRuntime ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares String, used by arcRuntimeFormat.
//
// This struct is created during compilation and serialized into the generated
// LLVM IR. It is treated as opaque by the generated LLVM IR, and therefore can
// use implementation-defined layout and padding if needed, as long as the
// compiler used during compilation is that same as that used when compiling
// the runtime.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_RUNTIME_DYNAMIC_STRING_H
#define CIRCT_DIALECT_ARC_RUNTIME_DYNAMIC_STRING_H

#include <cassert>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <type_traits>

namespace circt {
namespace arc {
namespace runtime {

/// A format descriptor, to be given to arcRuntimeFormat.
///
/// arcRuntimeFormat takes an array of FmtDescriptor and a variadic argument
/// list. Each FmtDescriptor describes how to format the corresponding
/// argument. The array is terminated by a FmtDescriptor with action Action_End.
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

#endif // CIRCT_DIALECT_ARC_RUNTIME_DYNAMIC_STRING_H
