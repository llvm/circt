//===- SMTAttributes.h - Declare SMT dialect attributes ----------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SMT_SMTATTRIBUTES_H
#define CIRCT_DIALECT_SMT_SMTATTRIBUTES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace circt {
namespace smt {
namespace detail {

struct BitVectorAttrStorage;

} // namespace detail
} // namespace smt
} // namespace circt

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/SMT/SMTAttributes.h.inc"

#endif // CIRCT_DIALECT_SMT_SMTATTRIBUTES_H
