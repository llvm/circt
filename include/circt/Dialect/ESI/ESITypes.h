//===- ESITypes.h - types for the ESI dialect -------------------*- C++ -*-===//
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

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/ESI/ESITypes.h.inc"

#endif
