//===- LLHDDialect.h - Declare LLHD dialect operations ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares an MLIR dialect for the LLHD IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LLHD_IR_LLHDDIALECT_H
#define CIRCT_DIALECT_LLHD_IR_LLHDDIALECT_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

#include "circt/Dialect/LLHD/IR/LLHDDialect.h.inc"

namespace circt {
namespace llhd {

namespace detail {
struct SigTypeStorage;
struct TimeAttrStorage;
struct ArrayTypeStorage;
struct PtrTypeStorage;
} // namespace detail

//===----------------------------------------------------------------------===//
// SigType
//===----------------------------------------------------------------------===//

class SigType
    : public mlir::Type::TypeBase<SigType, mlir::Type, detail::SigTypeStorage> {
public:
  using Base::Base;

  /// Get a new instance of llhd sig type
  static SigType get(mlir::Type underlyingType);

  /// The underlying type of the sig type
  Type getUnderlyingType();

  /// Get the keyword for the signal type
  static llvm::StringRef getKeyword() { return "sig"; }
};

//===----------------------------------------------------------------------===//
// TimeType
//===----------------------------------------------------------------------===//

class TimeType : public Type::TypeBase<TimeType, Type, DefaultTypeStorage> {
public:
  using Base::Base;

  /// Get a new instance of type Time
  static TimeType get(MLIRContext *context);

  /// Get the keyword for the time type
  static llvm::StringRef getKeyword() { return "time"; }
};

class PtrType
    : public mlir::Type::TypeBase<PtrType, Type, detail::PtrTypeStorage> {
public:
  using Base::Base;

  /// Get a new instance of llhd Ptr type.
  static PtrType get(Type underlyingType);

  /// The underlying type of the Ptr type.
  Type getUnderlyingType();

  /// Get the keyword for the Ptr type.
  static llvm::StringRef getKeyword() { return "ptr"; }
};

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

class ArrayType
    : public Type::TypeBase<ArrayType, Type, detail::ArrayTypeStorage> {
public:
  using Base::Base;

  /// Get or create a new ArrayType of the provided length and element type.
  /// Assumes the arguments define a well-formed ArrayType.
  static ArrayType get(unsigned length, Type elementType);

  /// Get or create a new ArrayType of the provided length and element type
  /// declared at the given, potentially unknown, location. If the ArrayType
  /// defined by the arguments would be ill-formed, emit errors and return
  /// nullptr-wrapping type.
  static ArrayType getChecked(function_ref<InFlightDiagnostic()> emitError,
                              unsigned length, Type elementType);

  /// Verify the construction of an array type.
  static LogicalResult
  verifyConstructionInvariants(Location loc, unsigned length, Type elementType);

  unsigned getLength() const;
  Type getElementType() const;

  /// Get the keyword for the array type
  static llvm::StringRef getKeyword() { return "array"; }
};

//===----------------------------------------------------------------------===//
// LLHD Attributes
//===----------------------------------------------------------------------===//

class TimeAttr
    : public Attribute::AttrBase<TimeAttr, Attribute, detail::TimeAttrStorage> {
public:
  using Base::Base;
  using ValueType = llvm::ArrayRef<unsigned>;

  /// Get a new instance of Time attribute.
  static TimeAttr get(Type type, llvm::ArrayRef<unsigned> timeValues,
                      llvm::StringRef timeUnit);

  /// Verify construction invariants of a new time attribute.
  static LogicalResult
  verifyConstructionInvariants(Location loc, Type type,
                               llvm::ArrayRef<unsigned> timeValues,
                               llvm::StringRef timeUnit);

  /// Get the time values stored in the attribute.
  llvm::ArrayRef<unsigned> getValue() const;

  /// Get the real time value of the attribute.
  unsigned getTime() const;

  /// Get the delta step value of the attribute.
  unsigned getDelta() const;

  /// Get the epsilon value of the attribute.
  unsigned getEps() const;

  /// Get the real time unit used by the attribute.
  llvm::StringRef getTimeUnit() const;

  /// Get the keyword of the time attribute
  static llvm::StringRef getKeyword() { return "time"; }
};
} // namespace llhd
} // namespace circt

#endif // CIRCT_DIALECT_LLHD_IR_LLHDDIALECT_H
