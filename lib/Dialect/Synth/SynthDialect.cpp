//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/SynthDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"

using namespace circt;
using namespace synth;

void SynthDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Synth/Synth.cpp.inc"
      >();
}

Operation *SynthDialect::materializeConstant(OpBuilder &builder,
                                             Attribute value, Type type,
                                             Location loc) {
  // Integer constants.
  if (auto intType = dyn_cast<IntegerType>(type))
    if (auto attrValue = dyn_cast<IntegerAttr>(value))
      return hw::ConstantOp::create(builder, loc, type, attrValue);
  return nullptr;
}

#include "circt/Dialect/Synth/SynthDialect.cpp.inc"
