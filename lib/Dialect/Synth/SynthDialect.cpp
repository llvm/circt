//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/SynthDialect.h"
#include "circt/Dialect/Synth/SynthOps.h"

using namespace circt;
using namespace synth;

void SynthDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Synth/Synth.cpp.inc"
      >();
}

#include "circt/Dialect/Synth/SynthDialect.cpp.inc"
