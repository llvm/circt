//===- Types.h - types for the RTL dialect ----------------------*- C++ -*-===//
//
// Types for the RTL dialect are mostly in tablegen. This file should contain
// C++ types used in MLIR type parameters.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTL_TYPES_H
#define CIRCT_DIALECT_RTL_TYPES_H

#include "mlir/IR/Types.h"

namespace mlir {
class DialectAsmParser;
class DialectAsmPrinter;
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/RTL/RTLTypes.h.inc"

#endif // CIRCT_DIALECT_RTL_TYPES_H
