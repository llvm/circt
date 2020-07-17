//===- LLHDDialect.h - Declare LLHD dialect operations ----------*- C++ -*-===//
//
// This file declares an MLIR dialect for the LLHD IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LLHD_IR_LLHDDIALECT_H
#define CIRCT_DIALECT_LLHD_IR_LLHDDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"

namespace mlir {
namespace llhd {
namespace detail {
struct SigTypeStorage;
struct TimeAttrStorage;
struct ArrayTypeStorage;
} // namespace detail

class LLHDDialect : public Dialect {
public:
  explicit LLHDDialect(MLIRContext *context);

  /// Returns the prefix used in the textual IR to refer to LLHD operations
  static StringRef getDialectNamespace() { return "llhd"; }

  /// Parses a type registered to this dialect
  Type parseType(DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect
  void printType(Type type, DialectAsmPrinter &printer) const override;

  /// Parse an attribute regustered to this dialect
  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;

  /// Print an attribute registered to this dialect
  void printAttribute(Attribute attr,
                      DialectAsmPrinter &printer) const override;

  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;
};

//===----------------------------------------------------------------------===//
// LLHD Types
//===----------------------------------------------------------------------===//
namespace LLHDTypes {
enum Kinds {
  Sig = mlir::Type::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
  Time,
  Array,
};
} // namespace LLHDTypes

//===----------------------------------------------------------------------===//
// SigType
//===----------------------------------------------------------------------===//

class SigType
    : public mlir::Type::TypeBase<SigType, mlir::Type, detail::SigTypeStorage> {
public:
  using Base::Base;

  /// Return whether the given kind is of type Sig
  static bool kindof(unsigned kind) { return kind == LLHDTypes::Sig; }

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

  /// Return whether the given kind is of type Time
  static bool kindof(unsigned kind) { return kind == LLHDTypes::Time; }

  /// Get a new instance of type Time
  static TimeType get(MLIRContext *context);

  /// Get the keyword for the time type
  static llvm::StringRef getKeyword() { return "time"; }
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
  static ArrayType getChecked(unsigned length, Type elementType,
                              Location location);

  /// Verify the construction of an array type.
  static LogicalResult
  verifyConstructionInvariants(Location loc, unsigned length, Type elementType);

  unsigned getLength() const;
  Type getElementType() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) { return kind == LLHDTypes::Array; }

  /// Get the keyword for the array type
  static llvm::StringRef getKeyword() { return "array"; }
};

//===----------------------------------------------------------------------===//
// LLHD Attributes
//===----------------------------------------------------------------------===//

namespace LLHDAttrs {
enum Kinds {
  Time = mlir::Attribute::FIRST_PRIVATE_EXPERIMENTAL_0_ATTR,
};
} // namespace LLHDAttrs

class TimeAttr
    : public Attribute::AttrBase<TimeAttr, Attribute, detail::TimeAttrStorage> {
public:
  using Base::Base;
  using ValueType = llvm::ArrayRef<unsigned>;

  /// Returns whether the passed argument is of kind Time.
  static bool kindof(unsigned kind) { return kind == LLHDAttrs::Time; }

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
} // namespace mlir

#endif // CIRCT_DIALECT_LLHD_IR_LLHDDIALECT_H
