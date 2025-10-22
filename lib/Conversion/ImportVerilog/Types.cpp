//===- Types.cpp - Slang type conversion ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/syntax/AllSyntax.h"

using namespace circt;
using namespace ImportVerilog;
using moore::Domain;

namespace {
struct TypeVisitor {
  Context &context;
  Location loc;
  TypeVisitor(Context &context, Location loc) : context(context), loc(loc) {}

  // Handle simple bit vector types such as `bit`, `int`, or `bit [41:0]`.
  Type getSimpleBitVectorType(const slang::ast::IntegralType &type) {
    return moore::IntType::get(context.getContext(), type.bitWidth,
                               type.isFourState ? Domain::FourValued
                                                : Domain::TwoValued);
  }

  // NOLINTBEGIN(misc-no-recursion)
  Type visit(const slang::ast::VoidType &type) {
    return moore::VoidType::get(context.getContext());
  }

  Type visit(const slang::ast::ScalarType &type) {
    return getSimpleBitVectorType(type);
  }

  Type visit(const slang::ast::FloatingType &type) {
    if (type.floatKind == slang::ast::FloatingType::Kind::RealTime)
      return moore::TimeType::get(context.getContext());
    if (type.floatKind == slang::ast::FloatingType::Kind::Real)
      return moore::RealType::get(context.getContext(), moore::RealWidth::f64);
    return moore::RealType::get(context.getContext(), moore::RealWidth::f32);
  }

  Type visit(const slang::ast::PredefinedIntegerType &type) {
    if (type.integerKind == slang::ast::PredefinedIntegerType::Kind::Time)
      return moore::TimeType::get(context.getContext());
    return getSimpleBitVectorType(type);
  }

  Type visit(const slang::ast::PackedArrayType &type) {
    // Handle simple bit vector types of the form `bit [41:0]`.
    if (type.elementType.as_if<slang::ast::ScalarType>())
      return getSimpleBitVectorType(type);

    // Handle all other packed arrays.
    auto innerType = type.elementType.visit(*this);
    if (!innerType)
      return {};
    // The Slang frontend guarantees the inner type to be packed.
    return moore::ArrayType::get(type.range.width(),
                                 cast<moore::PackedType>(innerType));
  }

  Type visit(const slang::ast::QueueType &type) {
    auto innerType = type.elementType.visit(*this);
    if (!innerType)
      return {};
    return moore::QueueType::get(cast<moore::UnpackedType>(innerType),
                                 type.maxBound);
  }

  Type visit(const slang::ast::AssociativeArrayType &type) {
    auto innerType = type.elementType.visit(*this);
    if (!innerType)
      return {};
    if (!type.indexType) {
      mlir::emitError(
          loc, "unsupported type: associative arrays with wildcard index");
      return {};
    }
    auto indexType = type.indexType->visit(*this);
    if (!indexType)
      return {};
    return moore::AssocArrayType::get(cast<moore::UnpackedType>(innerType),
                                      cast<moore::UnpackedType>(indexType));
  }

  Type visit(const slang::ast::FixedSizeUnpackedArrayType &type) {
    auto innerType = type.elementType.visit(*this);
    if (!innerType)
      return {};
    return moore::UnpackedArrayType::get(type.range.width(),
                                         cast<moore::UnpackedType>(innerType));
  }

  Type visit(const slang::ast::DynamicArrayType &type) {
    auto innerType = type.elementType.visit(*this);
    if (!innerType)
      return {};
    return moore::OpenUnpackedArrayType::get(
        cast<moore::UnpackedType>(innerType));
  }

  // Handle type defs.
  Type visit(const slang::ast::TypeAliasType &type) {
    // Simply return the underlying type.
    return type.targetType.getType().visit(*this);
  }

  // Handle enums.
  Type visit(const slang::ast::EnumType &type) {
    // Simply return the underlying type.
    return type.baseType.visit(*this);
  }

  // Collect the members in a struct or union.
  LogicalResult
  collectMembers(const slang::ast::Scope &structType,
                 SmallVectorImpl<moore::StructLikeMember> &members) {
    for (auto &field : structType.membersOfType<slang::ast::FieldSymbol>()) {
      auto name = StringAttr::get(context.getContext(), field.name);
      auto innerType = context.convertType(*field.getDeclaredType());
      if (!innerType)
        return failure();
      members.push_back({name, cast<moore::UnpackedType>(innerType)});
    }
    return success();
  }

  // Handle packed and unpacked structs.
  Type visit(const slang::ast::PackedStructType &type) {
    SmallVector<moore::StructLikeMember> members;
    if (failed(collectMembers(type, members)))
      return {};
    return moore::StructType::get(context.getContext(), members);
  }

  Type visit(const slang::ast::UnpackedStructType &type) {
    SmallVector<moore::StructLikeMember> members;
    if (failed(collectMembers(type, members)))
      return {};
    return moore::UnpackedStructType::get(context.getContext(), members);
  }

  Type visit(const slang::ast::PackedUnionType &type) {
    SmallVector<moore::StructLikeMember> members;
    if (failed(collectMembers(type, members)))
      return {};
    return moore::UnionType::get(context.getContext(), members);
  }

  Type visit(const slang::ast::UnpackedUnionType &type) {
    SmallVector<moore::StructLikeMember> members;
    if (failed(collectMembers(type, members)))
      return {};
    return moore::UnpackedUnionType::get(context.getContext(), members);
  }

  Type visit(const slang::ast::StringType &type) {
    return moore::StringType::get(context.getContext());
  }

  Type visit(const slang::ast::CHandleType &type) {
    return moore::ChandleType::get(context.getContext());
  }

  Type visit(const slang::ast::ClassType &type) {
    if (auto *lowering = context.declareClass(type)) {
      mlir::StringAttr symName = lowering->op.getSymNameAttr();
      mlir::FlatSymbolRefAttr symRef = mlir::FlatSymbolRefAttr::get(symName);
      return moore::ClassHandleType::get(context.getContext(), symRef);
    }
    return {};
  }

  /// Emit an error for all other types.
  template <typename T>
  Type visit(T &&node) {
    auto d = mlir::emitError(loc, "unsupported type: ")
             << slang::ast::toString(node.kind);
    d.attachNote() << node.template as<slang::ast::Type>().toString();
    return {};
  }
  // NOLINTEND(misc-no-recursion)
};
} // namespace

// NOLINTBEGIN(misc-no-recursion)
Type Context::convertType(const slang::ast::Type &type, LocationAttr loc) {
  if (!loc)
    loc = convertLocation(type.location);
  return type.visit(TypeVisitor(*this, loc));
}

Type Context::convertType(const slang::ast::DeclaredType &type) {
  LocationAttr loc;
  if (auto *ts = type.getTypeSyntax())
    loc = convertLocation(ts->sourceRange().start());
  return convertType(type.getType(), loc);
}
// NOLINTEND(misc-no-recursion)
