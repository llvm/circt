//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Datapath/DatapathDialect.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Datapath/DatapathOps.h"

using namespace circt;
using namespace datapath;

void DatapathDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Datapath/Datapath.cpp.inc"
      >();
}

#include "circt/Dialect/Datapath/DatapathDialect.cpp.inc"
