//===- KanagawaTypes.h - Definition of Kanagawa dialect types ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_KANAGAWA_KANAGAWATYPES_H
#define CIRCT_DIALECT_KANAGAWA_KANAGAWATYPES_H

#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/Kanagawa/KanagawaDialect.h"

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"

namespace circt {
namespace kanagawa {
// Returns true if the given type is an opaque reference to an kanagawa class.
bool isOpaqueScopeRefType(mlir::Type type);
} // namespace kanagawa
} // namespace circt

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Kanagawa/KanagawaTypes.h.inc"

#endif // CIRCT_DIALECT_KANAGAWA_KANAGAWATYPES_H
