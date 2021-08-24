//===- MSFTAttributes.h - Microsoft dialect attributes ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the MSFT dialect custom attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MSFT_MSFTATTRIBUTES_H
#define CIRCT_DIALECT_MSFT_MSFTATTRIBUTES_H

#include "circt/Dialect/MSFT/MSFTDialect.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/BuiltinAttributes.h"

namespace circt {
namespace msft {
using InstanceIDAttr = mlir::SymbolRefAttr;
using SwitchInstanceCase = std::pair<InstanceIDAttr, Attribute>;
} // namespace msft
} // namespace circt

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/MSFT/MSFTAttributes.h.inc"

#endif // CIRCT_DIALECT_MSFT_MSFTATTRIBUTES_H
