//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Support/CustomDirectiveImpl.h"

using namespace circt;
using namespace circt::synth::mig;

#define GET_OP_CLASSES
#include "circt/Dialect/Synth/Synth.cpp.inc"

LogicalResult MajorityInverterOp::verify() {
  if (getNumOperands() % 2 != 1)
    return emitOpError("requires an odd number of operands");

  return success();
}
