//===- Intenal.h - Shared internal runtime utilities-----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides internal utility functions for the ArcRuntime.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_RUNTIME_INTERNAL_H
#define CIRCT_DIALECT_ARC_RUNTIME_INTERNAL_H

#include <cassert>
#include <cstdlib>
#include <iostream>

namespace circt::arc::runtime::impl {

/// Raise an irrecoverable error
[[noreturn]] inline static void fatalError(const char *message) {
  std::cerr << "[ArcRuntime] Internal Error: " << message << std::endl;
  assert(false && "ArcRuntime Internal Error");
  abort();
}

} // namespace circt::arc::runtime::impl

#endif // CIRCT_DIALECT_ARC_RUNTIME_INTERNAL_H
