//===- SSPOps.cpp - SSP operation implementation --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SSP (static scheduling problem) dialect operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SSP/SSPOps.h"

#include "mlir/IR/Builders.h"

using namespace circt;
using namespace circt::ssp;
using namespace mlir;

//===----------------------------------------------------------------------===//
// InstanceOp
//===----------------------------------------------------------------------===//

RegionKind InstanceOp::getRegionKind(unsigned index) {
  return RegionKind::Graph;
}

//===----------------------------------------------------------------------===//
// TableGen'ed code
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "circt/Dialect/SSP/SSP.cpp.inc"
