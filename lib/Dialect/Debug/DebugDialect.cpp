//===- DebugDialect.cpp - Debug dialect implementation --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Debug/DebugDialect.h"
#include "circt/Dialect/Debug/DebugOps.h"

using namespace circt;
using namespace debug;

void DebugDialect::initialize() {
  registerOps();
  registerTypes();
}

// Dialect implementation generated from `DebugDialect.td`
#include "circt/Dialect/Debug/DebugDialect.cpp.inc"
