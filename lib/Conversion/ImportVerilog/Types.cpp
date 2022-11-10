//===- Types.cpp - Slang type conversion ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/ast/ASTVisitor.h"
#include "slang/ast/Symbol.h"
#include "slang/ast/symbols/CompilationUnitSymbols.h"
#include "slang/ast/symbols/InstanceSymbols.h"
#include "slang/ast/symbols/VariableSymbols.h"
#include "slang/ast/types/AllTypes.h"
#include "slang/ast/types/Type.h"
#include "slang/syntax/SyntaxVisitor.h"

using namespace circt;
using namespace ImportVerilog;

namespace {
struct TypeVisitor {
  Context &context;
  Location loc;
  TypeVisitor(Context &context, Location loc) : context(context), loc(loc) {}

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

  Type visit(const slang::ast::PackedArrayType &type) {
    auto innerType = type.elementType.visit(*this);
    if (!innerType)
      return {};
    auto packedInnerType = dyn_cast<moore::PackedType>(innerType);
    if (!packedInnerType) {
      mlir::emitError(loc, "packed array with unpacked elements; ")
          << type.elementType.toString() << " is unpacked";
      return {};
    }
    return moore::PackedRangeDim::get(
        packedInnerType, moore::Range(type.range.left, type.range.right));
  }

  /// Emit an error for all other types.
  template <typename T>
  Type visit(T &&node) {
    mlir::emitError(loc, "unsupported type: ")
        << node.template as<slang::ast::Type>().toString();
    return {};
  }
};
} // namespace

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
