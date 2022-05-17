//===- SVOps.h - Declare SV dialect operations ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the SV dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SV_OPS_H
#define CIRCT_DIALECT_SV_OPS_H

#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace circt {
namespace hw {
class InstanceOp;
class HWSymbolCache;
class InnerRefAttr;
} // namespace hw

namespace sv {

/// Return true if the specified operation is an expression.
bool isExpression(Operation *op);

//===----------------------------------------------------------------------===//
// CaseOp Support
//===----------------------------------------------------------------------===//

/// This describes the bit in a pattern, 0/1/x/z.
enum class CasePatternBit { Zero = 0, One = 1, AnyX = 2, AnyZ = 3 };

/// Return the letter for the specified pattern bit, e.g. "0", "1", "x" or "z".
char getLetter(CasePatternBit bit);

// This is provides convenient access to encode and decode a pattern.
struct CasePattern {
  IntegerAttr attr;

  struct DefaultPatternTag {};

  // Return the number of bits in the pattern.
  size_t getWidth() const { return attr.getValue().getBitWidth() / 2; }

  /// Return the specified bit, bit 0 is the least significant bit.
  CasePatternBit getBit(size_t bitNumber) const;

  /// Return true if this pattern always matches.
  bool isDefault() const;

  /// Return true if this pattern has an X.
  bool hasX() const;

  /// Return true if this pattern has an Z.
  bool hasZ() const;

  /// Get a CasePattern from a specified list of CasePatternBit.  Bits are
  /// specified in most least significant order - element zero is the least
  /// significant bit.
  CasePattern(ArrayRef<CasePatternBit> bits, MLIRContext *context);

  /// Get a CasePattern for the specified constant value.
  CasePattern(const APInt &value, MLIRContext *context);

  /// Get a CasePattern with a correctly encoded attribute.
  CasePattern(IntegerAttr attr) : attr(attr) {}

  /// Get a CasePattern of a default for the specified width.
  CasePattern(size_t width, DefaultPatternTag, MLIRContext *context);
};

// This provides information about one case.
struct CaseInfo {
  CasePattern pattern;
  Block *block;
};

//===----------------------------------------------------------------------===//
// Other Supporting Logic
//===----------------------------------------------------------------------===//

/// Return true if the specified operation is in a procedural region.
LogicalResult verifyInProceduralRegion(Operation *op);
/// Return true if the specified operation is not in a procedural region.
LogicalResult verifyInNonProceduralRegion(Operation *op);

/// Signals that an operations regions are procedural.
template <typename ConcreteType>
class ProceduralRegion
    : public mlir::OpTrait::TraitBase<ConcreteType, ProceduralRegion> {
  static LogicalResult verifyTrait(Operation *op) {
    return mlir::OpTrait::impl::verifyAtLeastNRegions(op, 1);
  }
};

/// This class verifies that the specified op is located in a procedural region.
template <typename ConcreteType>
class ProceduralOp
    : public mlir::OpTrait::TraitBase<ConcreteType, ProceduralOp> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return verifyInProceduralRegion(op);
  }
};

/// This class verifies that the specified op is not located in a procedural
/// region.
template <typename ConcreteType>
class NonProceduralOp
    : public mlir::OpTrait::TraitBase<ConcreteType, NonProceduralOp> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return verifyInNonProceduralRegion(op);
  }
};

/// This class provides a verifier for ops that are expecting their parent
/// to be one of the given parent ops
template <typename ConcreteType>
class VendorExtension
    : public mlir::OpTrait::TraitBase<ConcreteType, VendorExtension> {
public:
  static LogicalResult verifyTrait(Operation *op) { return success(); }
};

} // namespace sv
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/SV/SV.h.inc"

#endif // CIRCT_DIALECT_SV_OPS_H
