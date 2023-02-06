//===- ESIFolds.cpp - ESI op folders ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/BuiltinAttributes.h"

using namespace circt;
using namespace circt::esi;

LogicalResult WrapValidReadyOp::fold(FoldAdaptor,
                                     SmallVectorImpl<OpFoldResult> &results) {
  if (!getChanOutput().getUsers().empty())
    return failure();
  results.push_back(mlir::UnitAttr::get(getContext()));
  results.push_back(IntegerAttr::get(IntegerType::get(getContext(), 1), 1));
  return success();
}
