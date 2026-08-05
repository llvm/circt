//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AXI4/AXI4Dialect.h"
#include "circt/Dialect/AXI4/AXI4Ops.h"

using namespace circt;
using namespace axi4;

void AXI4Dialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/AXI4/AXI4.cpp.inc"
      >();
}

#include "circt/Dialect/AXI4/AXI4Dialect.cpp.inc"
