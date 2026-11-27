//===- SystemCDialect.cpp - Implement the SystemC dialect -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SystemC dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SystemC/SystemCOps.h"

using namespace circt;
using namespace circt::systemc;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void SystemCDialect::initialize() {
  // Register attributes.
  registerAttributes();
  // Register types.
  registerTypes();
}

#include "circt/Dialect/SystemC/SystemCDialect.cpp.inc"
