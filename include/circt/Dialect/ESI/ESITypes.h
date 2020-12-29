//===- ESITypes.h - types for the ESI dialect -------------------*- C++ -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Types for ESI are mostly in tablegen. This file should contain C++ types used
// in MLIR type parameters and other supporting declarations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_ESITYPES_H
#define CIRCT_DIALECT_ESI_ESITYPES_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

#include "ESIDialect.h"

namespace circt {
namespace esi {

/// Every time we implement a breaking change in the schema generation,
/// increment this number. It is a seed for all the schema hashes.
constexpr uint64_t esiCosimSchemaVersion = 1;

/// Get the Cap'nProto schema ID for a type.
uint64_t getCapnpTypeID(mlir::Type);

} // namespace esi
} // namespace circt

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/ESI/ESITypes.h.inc"

#endif
