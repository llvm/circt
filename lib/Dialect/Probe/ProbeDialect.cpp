//===- ProbeDialect.cpp - Probe dialect implementation --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Probe/ProbeDialect.h"
#include "circt/Dialect/Probe/ProbeOps.h"

using namespace circt;
using namespace probe;

void ProbeDialect::initialize() {
  registerOps();
  registerTypes();
}

// Dialect implementation generated from `ProbeDialect.td`
#include "circt/Dialect/Probe/ProbeDialect.cpp.inc"
