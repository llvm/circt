//===- CHIRRTLDialect.cpp - Implement the CHIRRTL dialect -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CHIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/CHIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"

using namespace circt;
using namespace chirrtl;

void CHIRRTLDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/CHIRRTL/CHIRRTL.cpp.inc"
      >();
}

#include "circt/Dialect/CHIRRTL/CHIRRTLDialect.cpp.inc"
