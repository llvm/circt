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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

namespace circt {
namespace llhd {

namespace detail {
struct SigTypeStorage;
struct TimeAttrStorage;
struct ArrayTypeStorage;
struct PtrTypeStorage;
} // namespace detail

class LLHDDialect : public mlir::Dialect {
public:
  explicit LLHDDialect(mlir::MLIRContext *context);

  /// Returns the prefix used in the textual IR to refer to LLHD operations
  static llvm::StringRef getDialectNamespace() { return "llhd"; }

  /// Parses a type registered to this dialect
  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect
  void printType(mlir::Type type,
                 mlir::DialectAsmPrinter &printer) const override;

  /// Parse an attribute regustered to this dialect
  mlir::Attribute parseAttribute(mlir::DialectAsmParser &parser,
                                 mlir::Type type) const override;

  /// Print an attribute registered to this dialect
  void printAttribute(mlir::Attribute attr,
                      mlir::DialectAsmPrinter &printer) const override;

  mlir::Operation *materializeConstant(mlir::OpBuilder &builder,
                                       mlir::Attribute value, mlir::Type type,
                                       mlir::Location loc) override;
};

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
  mlir::Type getUnderlyingType();

  /// Get the keyword for the signal type
  static llvm::StringRef getKeyword() { return "sig"; }
};

//===----------------------------------------------------------------------===//
// TimeType
//===----------------------------------------------------------------------===//

class TimeType : public mlir::Type::TypeBase<TimeType, mlir::Type,
                                             mlir::DefaultTypeStorage> {
public:
  using Base::Base;

  /// Get a new instance of type Time
  static TimeType get(mlir::MLIRContext *context);

  /// Get the keyword for the time type
  static llvm::StringRef getKeyword() { return "time"; }
};

class PtrType
    : public mlir::Type::TypeBase<PtrType, mlir::Type, detail::PtrTypeStorage> {
public:
  using Base::Base;

  /// Get a new instance of llhd Ptr type.
  static PtrType get(mlir::Type underlyingType);

  /// The underlying type of the Ptr type.
  mlir::Type getUnderlyingType();

  /// Get the keyword for the Ptr type.
  static llvm::StringRef getKeyword() { return "ptr"; }
};

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

class ArrayType : public mlir::Type::TypeBase<ArrayType, mlir::Type,
                                              detail::ArrayTypeStorage> {
public:
  using Base::Base;

  /// Get or create a new ArrayType of the provided length and element type.
  /// Assumes the arguments define a well-formed ArrayType.
  static ArrayType get(unsigned length, mlir::Type elementType);

  /// Get or create a new ArrayType of the provided length and element type
  /// declared at the given, potentially unknown, location. If the ArrayType
  /// defined by the arguments would be ill-formed, emit errors and return
  /// nullptr-wrapping type.
  static ArrayType getChecked(unsigned length, mlir::Type elementType,
                              mlir::Location location);

  /// Verify the construction of an array type.
  static mlir::LogicalResult
  verifyConstructionInvariants(mlir::Location loc, unsigned length,
                               mlir::Type elementType);

  unsigned getLength() const;
  mlir::Type getElementType() const;

  /// Get the keyword for the array type
  static llvm::StringRef getKeyword() { return "array"; }
};

//===----------------------------------------------------------------------===//
// LLHD Attributes
//===----------------------------------------------------------------------===//

class TimeAttr : public mlir::Attribute::AttrBase<TimeAttr, mlir::Attribute,
                                                  detail::TimeAttrStorage> {
public:
  using Base::Base;
  using ValueType = llvm::ArrayRef<unsigned>;

  /// Get a new instance of Time attribute.
  static TimeAttr get(mlir::Type type, llvm::ArrayRef<unsigned> timeValues,
                      llvm::StringRef timeUnit);

  /// Verify construction invariants of a new time attribute.
  static mlir::LogicalResult
  verifyConstructionInvariants(mlir::Location loc, mlir::Type type,
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
