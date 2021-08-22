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

#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace circt {
namespace hw {
class InstanceOp;
}
namespace sv {

/// Return true if the specified operation is an expression.
bool isExpression(Operation *op);

//===----------------------------------------------------------------------===//
// CaseZOp Support
//===----------------------------------------------------------------------===//

/// This describes the bit in a pattern, 0/1/x.
enum class CaseZPatternBit { Zero = 0, One = 1, Any = 2 };

/// Return the letter for the specified pattern bit, e.g. "0", "1", "?" or "x".
/// isVerilog indicates whether we should use "?" (verilog syntax) or "x" (mlir
/// operation syntax.
char getLetter(CaseZPatternBit bit, bool isVerilog);

// This is provides convenient access to encode and decode a pattern.
struct CaseZPattern {
  IntegerAttr attr;

  // Return the number of bits in the pattern.
  size_t getWidth() const { return attr.getValue().getBitWidth() / 2; }

  /// Return the specified bit, bit 0 is the least significant bit.
  CaseZPatternBit getBit(size_t bitNumber) const;

  /// Return true if this pattern always matches.
  bool isDefault() const;

  /// Get a CaseZPattern from a specified list of CaseZPatternBit.  Bits are
  /// specified in most least significant order - element zero is the least
  /// significant bit.
  CaseZPattern(ArrayRef<CaseZPatternBit> bits, MLIRContext *context);

  /// Get a CaseZPattern for the specified constant value.
  CaseZPattern(const APInt &value, MLIRContext *context);

  /// Get a CaseZPattern with a correctly encoded attribute.
  CaseZPattern(IntegerAttr attr) : attr(attr) {}

  static CaseZPattern getDefault(unsigned width, MLIRContext *context);
};
// This provides information about one case.
struct CaseZInfo {
  CaseZPattern pattern;
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

} // namespace sv
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/SV/SVEnums.h.inc"
// Clang format shouldn't reorder these headers.
#include "circt/Dialect/SV/SV.h.inc"
#include "circt/Dialect/SV/SVStructs.h.inc"

#endif // CIRCT_DIALECT_SV_OPS_H
