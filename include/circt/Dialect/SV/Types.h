//===- SV/Types.h - Declare SV dialect types --------------------*- C++ -*-===//
//
// This file declares the types for the SV dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SV_TYPES_H
#define CIRCT_DIALECT_SV_TYPES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/SV/SVTypes.h.inc"

#endif // CIRCT_DIALECT_SV_TYPES_H
