//===- SVAttributes.h - Declare SV dialect attributes ------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SV_SVATTRIBUTES_H
#define CIRCT_DIALECT_SV_SVATTRIBUTES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "circt/Dialect/SV/SVEnums.h.inc"

namespace circt {
namespace sv {

/// Helper functions to handle SV attributes.
bool hasSVAttributes(mlir::Operation *op);
mlir::ArrayAttr getSVAttributes(mlir::Operation *op);
void setSVAttributes(mlir::Operation *op, mlir::Attribute);

} // namespace sv

} // namespace circt

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/SV/SVAttributes.h.inc"

#endif // CIRCT_DIALECT_SV_SVATTRIBUTES_H
