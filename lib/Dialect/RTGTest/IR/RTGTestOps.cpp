//===- RTGTestOps.cpp - Implement the RTG operations ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the RTGTest ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTGTest/IR/RTGTestOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/APInt.h"

using namespace circt;
using namespace rtgtest;

//===----------------------------------------------------------------------===//
// ConstantTestOp
//===----------------------------------------------------------------------===//

mlir::OpFoldResult ConstantTestOp::fold(FoldAdaptor adaptor) {
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// GetHartIdOp
//===----------------------------------------------------------------------===//

mlir::OpFoldResult GetHartIdOp::fold(FoldAdaptor adaptor) {
  if (auto cpuAttr = dyn_cast_or_null<CPUAttr>(adaptor.getCpu()))
    return IntegerAttr::get(IndexType::get(getContext()), cpuAttr.getId());
  return {};
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/RTGTest/IR/RTGTest.cpp.inc"
