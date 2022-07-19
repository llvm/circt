//===- SystemCTypes.cpp - Implement the SystemC types ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SystemC dialect type system.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SystemC/SystemCTypes.h"
#include "circt/Dialect/SystemC/SystemCDialect.h"

using namespace circt;
using namespace circt::systemc;

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/SystemC/SystemCTypes.cpp.inc"

void SystemCDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/SystemC/SystemCTypes.cpp.inc"
      >();
}
