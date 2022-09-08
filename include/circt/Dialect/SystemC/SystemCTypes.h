//===- SystemCTypes.h - Declare SystemC dialect types ------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the types for the SystemC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYSTEMC_SYSTEMCTYPES_H
#define CIRCT_DIALECT_SYSTEMC_SYSTEMCTYPES_H

#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"

namespace circt {
namespace systemc {
// Forward declarations.
class IntBaseType;
class UIntBaseType;
class IntType;
class UIntType;
class BigIntType;
class BigUIntType;
class SignedType;
class UnsignedType;
class BitVectorBaseType;
class BitVectorType;
class LogicVectorBaseType;
class LogicVectorType;

namespace detail {
// Forward declarations.
struct IntegerWidthStorage;

/// A struct containing minimal information for a systemc module port. Thus, can
/// be used as parameter in attributes or types.
struct PortInfo {
  mlir::StringAttr name;
  mlir::Type type;
};
} // namespace detail

/// Get the type wrapped by a signal or port (in, inout, out) type.
Type getSignalBaseType(Type type);

///
class ValueBaseType : public Type {
public:
  static bool classof(Type type) {
    return type.isa<SignedType>() || type.isa<UnsignedType>() ||
           type.isa<IntBaseType>() || type.isa<UIntBaseType>() ||
           type.isa<BigIntType>() || type.isa<BigUIntType>() ||
           type.isa<IntType>() || type.isa<UIntType>();
  }

protected:
  using Type::Type;
};

/// Represents limited word-length signed integer in SystemC as described in
/// IEEE 1666-2011 §7.5.4.
class IntBaseType
    : public Type::TypeBase<IntBaseType, ValueBaseType, DefaultTypeStorage> {
public:
  static bool classof(Type type) {
    return type.getTypeID() == getTypeID() || type.isa<IntType>();
  }
  static IntBaseType get(MLIRContext *context);
  static constexpr StringLiteral getMnemonic() { return "int_base"; }

protected:
  using Base::Base;
};

/// Represents limited word-length signed integer in SystemC as described in
/// IEEE 1666-2011 §7.5.4.
class IntType : public Type::TypeBase<IntType, IntBaseType,
                                      systemc::detail::IntegerWidthStorage> {
public:
  static IntType get(MLIRContext *context, unsigned width);
  unsigned getWidth();
  static constexpr StringLiteral getMnemonic() { return "int"; }

protected:
  using Base::Base;
};

/// Represents a limited word-length unsigned integer in SystemC as described
/// inIEEE 1666-2011 §7.5.5.
class UIntBaseType
    : public Type::TypeBase<UIntBaseType, ValueBaseType, DefaultTypeStorage> {
public:
  static bool classof(Type type) {
    return type.getTypeID() == getTypeID() || type.isa<UIntType>();
  }
  static UIntBaseType get(MLIRContext *context);

  static constexpr StringLiteral getMnemonic() { return "uint_base"; }

protected:
  using Base::Base;
};

/// Represents a limited word-length unsigned integer in SystemC as described in
/// IEEE 1666-2011 §7.5.5.
class UIntType : public Type::TypeBase<UIntType, UIntBaseType,
                                       systemc::detail::IntegerWidthStorage> {
public:
  static UIntType get(MLIRContext *context, unsigned width);
  unsigned getWidth();

  static constexpr StringLiteral getMnemonic() { return "uint"; }

protected:
  using Base::Base;
};

/// Represents a finite word-length signed integer in SystemC as described in
/// IEEE 1666-2011 §7.6.5.
class SignedType
    : public Type::TypeBase<SignedType, ValueBaseType, DefaultTypeStorage> {
public:
  static bool classof(Type type) {
    return type.getTypeID() == getTypeID() || type.isa<BigIntType>();
  }
  static SignedType get(MLIRContext *context);

  static constexpr StringLiteral getMnemonic() { return "signed"; }

protected:
  using Base::Base;
};

/// Represents a finite word-length signed integer in SystemC as described in
/// IEEE 1666-2011 §7.6.5.
class BigIntType : public Type::TypeBase<BigIntType, SignedType,
                                         systemc::detail::IntegerWidthStorage> {
public:
  static BigIntType get(MLIRContext *context, unsigned width);
  unsigned getWidth();
  static constexpr StringLiteral getMnemonic() { return "bigint"; }

protected:
  using Base::Base;
};

/// Represents a finite word-length unsigned integer in SystemC as described in
/// IEEE 1666-2011 §7.6.6.
class UnsignedType
    : public Type::TypeBase<UnsignedType, ValueBaseType, DefaultTypeStorage> {
public:
  static bool classof(Type type) {
    return type.getTypeID() == getTypeID() || type.isa<BigUIntType>();
  }
  static UnsignedType get(MLIRContext *context);

  static constexpr StringLiteral getMnemonic() { return "unsigned"; }

protected:
  using Base::Base;
};

/// Represents a limited word-length unsigned integer in SystemC as described in
/// IEEE 1666-2011 §7.5.5.
class BigUIntType
    : public Type::TypeBase<BigUIntType, UnsignedType,
                            systemc::detail::IntegerWidthStorage> {
public:
  static BigUIntType get(MLIRContext *context, unsigned width);
  unsigned getWidth();

  static constexpr StringLiteral getMnemonic() { return "biguint"; }

protected:
  using Base::Base;
};

///
class BitVectorBaseType
    : public Type::TypeBase<BitVectorBaseType, Type, DefaultTypeStorage> {
public:
  static bool classof(Type type) {
    return type.getTypeID() == getTypeID() || type.isa<BitVectorType>();
  }
  static BitVectorBaseType get(MLIRContext *context);
  static constexpr StringLiteral getMnemonic() { return "bv_base"; }

protected:
  using Base::Base;
};

/// Represents a finite word-length bit vector in SystemC as described inIEEE
/// 1666-2011 §7.9.5.
class BitVectorType
    : public Type::TypeBase<BitVectorType, BitVectorBaseType,
                            systemc::detail::IntegerWidthStorage> {
public:
  static BitVectorType get(MLIRContext *context, unsigned width);
  unsigned getWidth();

  static constexpr StringLiteral getMnemonic() { return "bv"; }

protected:
  using Base::Base;
};

///
class LogicVectorBaseType
    : public Type::TypeBase<LogicVectorBaseType, Type, DefaultTypeStorage> {
public:
  static bool classof(Type type) {
    return type.getTypeID() == getTypeID() || type.isa<LogicVectorType>();
  }
  static LogicVectorBaseType get(MLIRContext *context);
  static constexpr StringLiteral getMnemonic() { return "lv_base"; }

protected:
  using Base::Base;
};

///
class LogicVectorType
    : public Type::TypeBase<LogicVectorType, LogicVectorBaseType,
                            systemc::detail::IntegerWidthStorage> {
public:
  static LogicVectorType get(MLIRContext *context, unsigned width);
  unsigned getWidth();

  static constexpr StringLiteral getMnemonic() { return "lv"; }

protected:
  using Base::Base;
};

} // namespace systemc
} // namespace circt

// Include generated types.
#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/SystemC/SystemCTypes.h.inc"

#endif // CIRCT_DIALECT_SYSTEMC_SYSTEMCTYPES_H
