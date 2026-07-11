//===- ProbeOps.cpp - Probe dialect operations ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Probe/ProbeOps.h"

using namespace circt;
using namespace probe;

void ProbeDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Probe/Probe.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "circt/Dialect/Probe/Probe.cpp.inc"
