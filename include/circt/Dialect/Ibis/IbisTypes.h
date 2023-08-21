//===- IbisTypes.h - Definition of Ibis dialect types ------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_IBIS_IBISTYPES_H
#define CIRCT_DIALECT_IBIS_IBISTYPES_H

#include "circt/Dialect/Ibis/IbisDialect.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"

namespace circt {
namespace ibis {
// Returns true if the given type is an opaque reference to an ibis class.
bool isOpaqueScopeRefType(mlir::Type type);
} // namespace ibis
} // namespace circt

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Ibis/IbisTypes.h.inc"

#endif // CIRCT_DIALECT_IBIS_IBISTYPES_H
