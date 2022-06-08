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

#ifndef CIRCT_DIALECT_FIRRTL_OP_INTERFACES_H
#define CIRCT_DIALECT_FIRRTL_OP_INTERFACES_H

#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/CommandLine.h"

namespace circt {
namespace firrtl {

class FIRRTLType;

/// This holds the name and type that describes the module's ports.
struct PortInfo {
  StringAttr name;
  FIRRTLType type;
  Direction direction;
  StringAttr sym = {};
  Location loc = UnknownLoc::get(type.getContext());
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

/// Verification hook for verifying module like operations.
LogicalResult verifyModuleLikeOpInterface(FModuleLike module);

class InnerSymbolTable {

public:
  /// Return the name of the attribute used for inner symbol names.
  static StringRef getInnerSymbolAttrName() { return "inner_sym"; }
};

} // namespace firrtl
} // namespace circt

namespace mlir {
namespace OpTrait {
template <typename ConcreteType>
class InnerSymbolTable : public TraitBase<ConcreteType, InnerSymbolTable> {
public:
  static LogicalResult verifyRegionTrait(Operation *op) { return success(); }
};
} // namespace OpTrait
} // namespace mlir

#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.h.inc"
#endif // CIRCT_DIALECT_FIRRTL_OP_INTERFACES_H
