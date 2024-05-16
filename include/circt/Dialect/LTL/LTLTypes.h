//===- LTLTypes.h - LTL dialect types ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LTL_LTLTYPES_H
#define CIRCT_DIALECT_LTL_LTLTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/LTL/LTLTypes.h.inc"

namespace circt {
namespace ltl {
/// Return true if the specified type is a clocked ltl type. This checks
/// that it is either an ltl.clocked_property, an ltl.clocked_sequence or an
/// ltl.clocked_disabled_property.
bool isClocked(mlir::Type type);

/// Return true if the specified type is a disabled ltl type. This checks
/// that it is either an ltl.disabled_property, or an
/// ltl.clocked_disabled_property.
bool isDisabled(mlir::Type type);

/// Return true if the specified type is a property ltl type. This checks
/// that it is either an ltl.clocked_property, an ltl.disabled_property, an
/// ltl.clocked_disabled_property, or an ltl.property.
bool isProperty(mlir::Type type);

/// Return true if the specified type is a sequence ltl type. This checks
/// that it is either an ltl.sequence, or an ltl.clocked_sequence.
bool isSequence(mlir::Type type);

} // namespace ltl
} // namespace circt

#endif // CIRCT_DIALECT_LTL_LTLTYPES_H
