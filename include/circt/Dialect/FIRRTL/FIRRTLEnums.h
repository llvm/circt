//===- FIRRTLEnums.h - FIRRTL dialect enums ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains custom FIRRTL Dialect enumerations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLENUMS_H
#define CIRCT_DIALECT_FIRRTL_FIRRTLENUMS_H

#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Support/LLVM.h"

namespace circt {
namespace firrtl {

//===----------------------------------------------------------------------===//
// PortDirections
//===----------------------------------------------------------------------===//

/// This represents the direction of a single port.
enum class Direction { In, Out };

/// Prints the Direction to the stream as either "in" or "out".
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Direction &dir);

namespace direction {

/// Return an output direction if \p isOutput is true, otherwise return an
/// input direction.
static inline Direction get(bool isOutput) { return (Direction)isOutput; }

/// Convert from Direction to bool.  The opposite of get;
static inline bool unGet(Direction dir) { return (bool)dir; }

/// Flip a port direction.
Direction flip(Direction direction);

static inline StringRef toString(Direction direction) {
  return direction == Direction::In ? "in" : "out";
}

static inline StringRef toString(bool direction) {
  return toString(get(direction));
}

/// Return a \p DenseBoolArrayAttr containing the packed representation of an
/// array of directions.
mlir::DenseBoolArrayAttr packAttribute(MLIRContext *context,
                                       ArrayRef<Direction> directions);

/// Return a \p DenseBoolArrayAttr containing the packed representation of an
/// array of directions.
mlir::DenseBoolArrayAttr packAttribute(MLIRContext *context,
                                       ArrayRef<bool> directions);

/// Turn a packed representation of port attributes into a vector that can
/// be worked with.
SmallVector<Direction> unpackAttribute(mlir::DenseBoolArrayAttr directions);

} // namespace direction
} // namespace firrtl
} // namespace circt

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/FIRRTL/FIRRTLEnums.h.inc"

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLENUMS_H
