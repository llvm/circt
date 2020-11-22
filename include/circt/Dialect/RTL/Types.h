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

// FIXME: This is a workaround for MLIR bug #48262
namespace circt {
namespace rtl {
using Location = ::mlir::Location;
}
} // namespace circt

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/RTL/RTLTypes.h.inc"

/// Return true if the specified type can be used as an RTL value type, that is
/// the set of types that can be composed together to represent synthesized,
/// hardware but not marker types like InOutType.
bool isRTLValueType(mlir::Type type);

#endif // CIRCT_DIALECT_RTL_TYPES_H
