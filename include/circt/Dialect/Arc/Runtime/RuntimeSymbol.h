//===- RuntimeSymbol.h - Runtime JIT symbol declarations --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_RUNTIME_RUNTIMESYMBOL_H
#define CIRCT_DIALECT_ARC_RUNTIME_RUNTIMESYMBOL_H

namespace circt {
namespace arc {
namespace runtime {

/// A single JIT-bindable runtime symbol with its actual function pointer type,
/// so `ExecutorAddr::fromPtr` sees the precise function pointer.
template <typename FunctionPtrT>
struct RuntimeSymbol {
  const char *name;
  FunctionPtrT addr;
};

} // namespace runtime
} // namespace arc
} // namespace circt

#endif // CIRCT_DIALECT_ARC_RUNTIME_RUNTIMESYMBOL_H
