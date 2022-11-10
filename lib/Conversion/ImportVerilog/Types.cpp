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

namespace {
struct TypeVisitor {
  Context &context;
  Location loc;
  TypeVisitor(Context &context, Location loc) : context(context), loc(loc) {}

  // NOLINTBEGIN(misc-no-recursion)
  Type visit(const slang::ast::ScalarType &type) {
    moore::IntType::Kind kind;
    switch (type.scalarKind) {
    case slang::ast::ScalarType::Bit:
      kind = moore::IntType::Bit;
      break;
    case slang::ast::ScalarType::Logic:
      kind = moore::IntType::Logic;
      break;
    case slang::ast::ScalarType::Reg:
      kind = moore::IntType::Reg;
      break;
    }

    std::optional<moore::Sign> sign =
        type.isSigned ? moore::Sign::Signed : moore::Sign::Unsigned;
    if (sign == moore::IntType::getDefaultSign(kind))
      sign = {};

    return moore::IntType::get(context.getContext(), kind, sign);
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
    moore::IntType::Kind kind;
    switch (type.integerKind) {
    case slang::ast::PredefinedIntegerType::Int:
      kind = moore::IntType::Int;
      break;
    case slang::ast::PredefinedIntegerType::ShortInt:
      kind = moore::IntType::ShortInt;
      break;
    case slang::ast::PredefinedIntegerType::LongInt:
      kind = moore::IntType::LongInt;
      break;
    case slang::ast::PredefinedIntegerType::Integer:
      kind = moore::IntType::Integer;
      break;
    case slang::ast::PredefinedIntegerType::Byte:
      kind = moore::IntType::Byte;
      break;
    case slang::ast::PredefinedIntegerType::Time:
      kind = moore::IntType::Time;
      break;
    }

    std::optional<moore::Sign> sign =
        type.isSigned ? moore::Sign::Signed : moore::Sign::Unsigned;
    if (sign == moore::IntType::getDefaultSign(kind))
      sign = {};

    return moore::IntType::get(context.getContext(), kind, sign);
  }

  Type visit(const slang::ast::PackedArrayType &type) {
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
    auto innerType = type.targetType.getType().visit(*this);
    if (!innerType)
      return {};
    auto loc = context.convertLocation(type.location);
    if (auto packedInnerType = dyn_cast<moore::PackedType>(innerType))
      return moore::PackedNamedType::get(packedInnerType, type.name, loc);
    return moore::UnpackedNamedType::get(cast<moore::UnpackedType>(innerType),
                                         type.name, loc);
  }

  // Handle enums.
  Type visit(const slang::ast::EnumType &type) {
    auto baseType = type.baseType.visit(*this);
    if (!baseType)
      return {};
    return moore::EnumType::get(StringAttr{}, loc,
                                cast<moore::PackedType>(baseType));
  }

  // Collect the members in a struct or union.
  LogicalResult collectMembers(const slang::ast::Scope &structType,
                               SmallVectorImpl<moore::StructMember> &members,
                               bool enforcePacked) {
    for (auto &field : structType.membersOfType<slang::ast::FieldSymbol>()) {
      auto loc = context.convertLocation(field.location);
      auto name = StringAttr::get(context.getContext(), field.name);
      auto innerType = context.convertType(*field.getDeclaredType());
      if (!innerType)
        return failure();
      // The Slang frontend guarantees the inner type to be packed if the struct
      // is packed.
      assert(!enforcePacked || isa<moore::PackedType>(innerType));
      members.push_back({name, loc, cast<moore::UnpackedType>(innerType)});
    }
    return success();
  }

  // Handle packed and unpacked structs.
  Type visit(const slang::ast::PackedStructType &type) {
    auto loc = context.convertLocation(type.location);
    SmallVector<moore::StructMember> members;
    if (failed(collectMembers(type, members, true)))
      return {};
    return moore::PackedStructType::get(moore::StructKind::Struct, members,
                                        StringAttr{}, loc);
  }

  Type visit(const slang::ast::UnpackedStructType &type) {
    auto loc = context.convertLocation(type.location);
    SmallVector<moore::StructMember> members;
    if (failed(collectMembers(type, members, false)))
      return {};
    return moore::UnpackedStructType::get(moore::StructKind::Struct, members,
                                          StringAttr{}, loc);
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
