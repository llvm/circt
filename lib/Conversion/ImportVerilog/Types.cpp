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
  Type visit(const slang::ast::ScalarType &type) {
    return getSimpleBitVectorType(type);
  }

  Type visit(const slang::ast::FloatingType &type) {
    moore::RealType::Kind kind;
    switch (type.floatKind) {
    case slang::ast::FloatingType::Real:
      kind = moore::RealType::Real;
      break;
    case slang::ast::FloatingType::ShortReal:
      kind = moore::RealType::ShortReal;
      break;
    case slang::ast::FloatingType::RealTime:
      kind = moore::RealType::RealTime;
      break;
    }

    return moore::RealType::get(context.getContext(), kind);
  }

  Type visit(const slang::ast::PredefinedIntegerType &type) {
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
    auto packedInnerType = cast<moore::PackedType>(innerType);
    return moore::PackedRangeDim::get(
        packedInnerType, moore::Range(type.range.left, type.range.right));
  }

  Type visit(const slang::ast::QueueType &type) {
    auto innerType = type.elementType.visit(*this);
    if (!innerType)
      return {};
    return moore::UnpackedQueueDim::get(cast<moore::UnpackedType>(innerType),
                                        type.maxBound);
  }

  Type visit(const slang::ast::AssociativeArrayType &type) {
    auto innerType = type.elementType.visit(*this);
    if (!innerType)
      return {};
    auto indexType = type.indexType->visit(*this);
    if (!indexType)
      return {};
    return moore::UnpackedAssocDim::get(cast<moore::UnpackedType>(innerType),
                                        cast<moore::UnpackedType>(indexType));
  }

  Type visit(const slang::ast::FixedSizeUnpackedArrayType &type) {
    auto innerType = type.elementType.visit(*this);
    if (!innerType)
      return {};
    return moore::UnpackedRangeDim::get(
        cast<moore::UnpackedType>(innerType),
        moore::Range(type.range.left, type.range.right));
  }

  Type visit(const slang::ast::DynamicArrayType &type) {
    auto innerType = type.elementType.visit(*this);
    if (!innerType)
      return {};
    return moore::UnpackedUnsizedDim::get(cast<moore::UnpackedType>(innerType));
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
  LogicalResult collectMembers(const slang::ast::Scope &structType,
                               SmallVectorImpl<moore::StructMember> &members,
                               bool enforcePacked) {
    for (auto &field : structType.membersOfType<slang::ast::FieldSymbol>()) {
      auto name = StringAttr::get(context.getContext(), field.name);
      auto innerType = context.convertType(*field.getDeclaredType());
      if (!innerType)
        return failure();
      // The Slang frontend guarantees the inner type to be packed if the struct
      // is packed.
      assert(!enforcePacked || isa<moore::PackedType>(innerType));
      members.push_back({name, cast<moore::UnpackedType>(innerType)});
    }
    return success();
  }

  // Handle packed and unpacked structs.
  Type visit(const slang::ast::PackedStructType &type) {
    SmallVector<moore::StructMember> members;
    if (failed(collectMembers(type, members, true)))
      return {};
    return moore::PackedStructType::get(context.getContext(),
                                        moore::StructKind::Struct, members);
  }

  Type visit(const slang::ast::UnpackedStructType &type) {
    SmallVector<moore::StructMember> members;
    if (failed(collectMembers(type, members, false)))
      return {};
    return moore::UnpackedStructType::get(context.getContext(),
                                          moore::StructKind::Struct, members);
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
