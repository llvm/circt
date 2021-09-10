//===- FIRRTLOpInterfaces.h - Declare FIRRTL op interfaces ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation interfaces for the FIRRTL IR and supporting
// types.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

namespace circt {
namespace firrtl {

class FIRRTLType;

enum class Direction { In, Out };

template <typename T>
T &operator<<(T &os, const Direction &dir) {
  return os << (dir == Direction::In ? "input" : "output");
}

namespace direction {

/// The key in a module's attribute dictionary used to find the direction.
static const char *const attrKey = "portDirections";

/// Return an output direction if \p isOutput is true, otherwise return an
/// input direction.
inline Direction get(bool isOutput) { return (Direction)isOutput; }

/// Return a \p IntegerAttr containing the packed representation of an array
/// of directions.
IntegerAttr packAttribute(ArrayRef<Direction> a, MLIRContext *b);

/// Turn a packed representation of port attributes into a vector that can
/// be worked with.
SmallVector<Direction> unpackAttribute(Operation *module);

} // namespace direction

/// This holds the name and type that describes the module's ports.
struct ModulePortInfo {
  StringAttr name;
  FIRRTLType type;
  Direction direction;
  Location loc;
  AnnotationSet annotations = AnnotationSet(type.getContext());

  StringRef getName() const { return name ? name.getValue() : ""; }

  /// Return true if this is a simple output-only port.  If you want the
  /// direction of the port, use the \p direction parameter.
  bool isOutput() {
    auto flags = type.getRecursiveTypeProperties();
    return flags.isPassive && !flags.containsAnalog &&
           direction == Direction::Out;
  }

  /// Return true if this is a simple input-only port.  If you want the
  /// direction of the port, use the \p direction parameter.
  bool isInput() {
    auto flags = type.getRecursiveTypeProperties();
    return flags.isPassive && !flags.containsAnalog &&
           direction == Direction::In;
  }

  /// Return true if this is an inout port.  This will be true if the port
  /// contains either bi-directional signals or analog types.
  bool isInOut() { return !isOutput() && !isInput(); }
};
} // namespace firrtl
} // namespace circt

#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.h.inc"
