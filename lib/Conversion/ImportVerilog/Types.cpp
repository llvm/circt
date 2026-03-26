//===- Types.cpp - Slang type conversion ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/ast/types/AllTypes.h"
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

  Type visit(const slang::ast::DPIOpenArrayType &type) {
    auto innerType = type.elementType.visit(*this);
    if (!innerType)
      return {};
    if (type.isPacked)
      return moore::OpenArrayType::get(cast<moore::PackedType>(innerType));
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
    if (failed(context.buildClassProperties(type)))
      return {};
    auto *lowering = context.declareClass(type);
    if (!lowering) {
      mlir::emitError(loc) << "no lowering generated for class type `"
                           << type.toString() << "`";
      return {};
    }
    mlir::StringAttr symName = lowering->op.getSymNameAttr();
    mlir::FlatSymbolRefAttr symRef = mlir::FlatSymbolRefAttr::get(symName);
    return moore::ClassHandleType::get(context.getContext(), symRef);
  }

  Type visit(const slang::ast::NullType &type) {
    return moore::NullType::get(context.getContext());
  }

  Type visit(const slang::ast::VirtualInterfaceType &type) {
    auto lowered = context.convertVirtualInterfaceType(type, loc);
    if (failed(lowered))
      return {};
    return *lowered;
  }

  Type visit(const slang::ast::EventType &type) {
    // Treat `event` types as simple `i1` values where an event is signaled by
    // toggling the value.
    return moore::IntType::getInt(context.getContext(), 1);
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

FailureOr<moore::UnpackedStructType> Context::convertVirtualInterfaceType(
    const slang::ast::VirtualInterfaceType &type, Location loc) {
  const slang::ast::InstanceBodySymbol &ifaceBody = type.iface.body;
  const slang::ast::ModportSymbol *modport = type.modport;

  auto &cache = modport ? virtualIfaceModportLowerings[modport]
                        : virtualIfaceLowerings[&ifaceBody];
  if (cache.type)
    return cache.type;

  SmallVector<moore::StructLikeMember> members;
  SmallVector<StringAttr, 8> fieldNames;
  DenseMap<StringAttr, Type> fieldTypes;

  auto addField = [&](StringRef name, const slang::ast::Type &fieldAstType,
                      Location fieldLoc) -> LogicalResult {
    auto nameAttr = builder.getStringAttr(name);

    Type loweredType = convertType(fieldAstType, fieldLoc);
    if (!loweredType)
      return failure();

    auto unpacked = dyn_cast<moore::UnpackedType>(loweredType);
    if (!unpacked) {
      mlir::emitError(fieldLoc)
          << "unsupported virtual interface member type: " << loweredType;
      return failure();
    }

    auto refTy = moore::RefType::get(unpacked);

    if (auto it = fieldTypes.find(nameAttr); it != fieldTypes.end()) {
      if (it->second != refTy) {
        mlir::emitError(fieldLoc) << "virtual interface member `" << name
                                  << "` has conflicting types (" << it->second
                                  << " vs " << refTy << ")";
        return failure();
      }
      return success();
    }

    fieldTypes.try_emplace(nameAttr, refTy);
    members.push_back({nameAttr, refTy});
    fieldNames.push_back(nameAttr);
    return success();
  };

  if (modport) {
    for (auto &member : modport->members()) {
      const auto *mpp = member.as_if<slang::ast::ModportPortSymbol>();
      if (!mpp) {
        auto d = mlir::emitError(convertLocation(member.location))
                 << "unsupported modport member: "
                 << slang::ast::toString(member.kind);
        if (!member.name.empty())
          d << " `" << member.name << "`";
        return failure();
      }
      if (failed(addField(mpp->name, mpp->getType(),
                          convertLocation(mpp->location))))
        return failure();
    }
  } else {
    for (auto *symbol : ifaceBody.getPortList()) {
      if (!symbol)
        continue;
      const auto *port = symbol->as_if<slang::ast::PortSymbol>();
      if (!port) {
        auto d = mlir::emitError(convertLocation(symbol->location))
                 << "unsupported interface port symbol: "
                 << slang::ast::toString(symbol->kind);
        if (!symbol->name.empty())
          d << " `" << symbol->name << "`";
        return failure();
      }
      if (failed(addField(port->name, port->getType(),
                          convertLocation(port->location))))
        return failure();
    }

    for (auto &member : ifaceBody.members()) {
      if (const auto *var = member.as_if<slang::ast::VariableSymbol>()) {
        if (failed(addField(var->name, var->getType(),
                            convertLocation(var->location))))
          return failure();
        continue;
      }
      if (const auto *net = member.as_if<slang::ast::NetSymbol>()) {
        if (failed(addField(net->name, net->getType(),
                            convertLocation(net->location))))
          return failure();
        continue;
      }
      // Skip non-data interface members that do not contribute to the virtual
      // interface handle representation.
      if (member.as_if<slang::ast::ModportSymbol>() ||
          member.as_if<slang::ast::ParameterSymbol>() ||
          member.as_if<slang::ast::TypeParameterSymbol>())
        continue;

      // Bail out loudly on unhandled value symbols to avoid silently dropping
      // interface members that the user may expect to access through a virtual
      // interface.
      if (const auto *value = member.as_if<slang::ast::ValueSymbol>()) {
        auto d = mlir::emitError(convertLocation(value->location))
                 << "unsupported interface member: "
                 << slang::ast::toString(value->kind);
        if (!value->name.empty())
          d << " `" << value->name << "`";
        return failure();
      }
    }
  }

  cache.type = moore::UnpackedStructType::get(getContext(), members);
  cache.fieldNames = fieldNames;
  return cache.type;
}

FailureOr<Value> Context::materializeVirtualInterfaceValue(
    const slang::ast::VirtualInterfaceType &type, Location loc) {
  if (!type.isRealIface) {
    mlir::emitError(loc)
        << "cannot materialize value for non-real virtual interface";
    return failure();
  }

  auto loweredType = convertVirtualInterfaceType(type, loc);
  if (failed(loweredType))
    return failure();

  const slang::ast::InstanceBodySymbol &ifaceBody = type.iface.body;
  const slang::ast::ModportSymbol *modport = type.modport;
  const auto &cache = modport ? virtualIfaceModportLowerings.lookup(modport)
                              : virtualIfaceLowerings.lookup(&ifaceBody);
  if (!cache.type)
    return failure();

  auto *ifaceLowering = interfaceInstances.lookup(&type.iface);
  if (!ifaceLowering) {
    mlir::emitError(loc) << "interface instance `" << type.iface.name
                         << "` was not expanded";
    return failure();
  }

  SmallVector<Value> fields;
  fields.reserve(cache.fieldNames.size());

  auto resolveInterfaceMember = [&](StringAttr nameAttr) -> FailureOr<Value> {
    if (!nameAttr)
      return failure();

    if (Value val = ifaceLowering->expandedMembersByName.lookup(nameAttr))
      return val;

    mlir::emitError(loc) << "unresolved interface member `"
                         << nameAttr.getValue() << "`";
    return failure();
  };

  if (modport) {
    DenseMap<StringAttr, const slang::ast::ModportPortSymbol *> portsByName;
    for (auto &sym : modport->members()) {
      const auto *port = sym.as_if<slang::ast::ModportPortSymbol>();
      if (!port) {
        auto d = mlir::emitError(convertLocation(sym.location))
                 << "unsupported modport member: "
                 << slang::ast::toString(sym.kind);
        if (!sym.name.empty())
          d << " `" << sym.name << "`";
        return failure();
      }
      auto nameAttr = builder.getStringAttr(port->name);
      portsByName.try_emplace(nameAttr, port);
    }

    for (auto nameAttr : cache.fieldNames) {
      const auto *port = portsByName.lookup(nameAttr);
      if (!port) {
        mlir::emitError(loc)
            << "unresolved modport member `" << nameAttr.getValue() << "`";
        return failure();
      }

      if (port->internalSymbol) {
        if (Value val =
                ifaceLowering->expandedMembers.lookup(port->internalSymbol)) {
          fields.push_back(val);
          continue;
        }
        // Fallback to a name-based lookup if the interface expansion recorded
        // the member under a different symbol pointer.
        auto resolved = resolveInterfaceMember(
            builder.getStringAttr(port->internalSymbol->name));
        if (failed(resolved))
          return failure();
        fields.push_back(*resolved);
        continue;
      }

      const auto *connExpr = port->getConnectionExpr();
      if (!connExpr) {
        mlir::emitError(loc) << "modport member `" << nameAttr.getValue()
                             << "` has no connection";
        return failure();
      }

      // Evaluate explicit modport connections in an environment where
      // interface members are in scope as lvalues.
      ValueSymbolScope scope(valueSymbols);
      for (const auto &[sym, value] : ifaceLowering->expandedMembers) {
        const auto *valueSym = sym->as_if<slang::ast::ValueSymbol>();
        if (!valueSym)
          continue;
        valueSymbols.insertIntoScope(valueSymbols.getCurScope(), valueSym,
                                     value);
      }

      Value val = convertLvalueExpression(*connExpr);
      if (!val)
        return failure();
      fields.push_back(val);
    }
  } else {
    for (auto nameAttr : cache.fieldNames) {
      auto val = resolveInterfaceMember(nameAttr);
      if (failed(val))
        return failure();
      fields.push_back(*val);
    }
  }

  return moore::StructCreateOp::create(builder, loc, cache.type, fields)
      .getResult();
}

LogicalResult Context::registerVirtualInterfaceMembers(
    const slang::ast::ValueSymbol &base,
    const slang::ast::VirtualInterfaceType &type, Location loc) {
  auto *scope = virtualIfaceMembers.getCurScope();
  if (!scope) {
    mlir::emitError(loc) << "internal error: no virtual interface member scope";
    return failure();
  }

  auto registerMember = [&](const slang::ast::ValueSymbol &member,
                            StringRef fieldName) {
    VirtualInterfaceMemberAccess entry;
    entry.base = &base;
    entry.fieldName = builder.getStringAttr(fieldName);

    if (auto existing = virtualIfaceMembers.lookup(&member);
        existing.base == &base && existing.fieldName == entry.fieldName)
      return;

    virtualIfaceMembers.insertIntoScope(scope, &member, entry);
  };

  if (const auto *modport = type.modport) {
    for (auto &sym : modport->members()) {
      const auto *port = sym.as_if<slang::ast::ModportPortSymbol>();
      if (!port) {
        auto d = mlir::emitError(convertLocation(sym.location))
                 << "unsupported modport member: "
                 << slang::ast::toString(sym.kind);
        if (!sym.name.empty())
          d << " `" << sym.name << "`";
        return failure();
      }
      registerMember(*port, port->name);
      if (port->internalSymbol)
        if (const auto *internal =
                port->internalSymbol->as_if<slang::ast::ValueSymbol>())
          registerMember(*internal, port->name);
    }
    return success();
  }

  const slang::ast::InstanceBodySymbol &ifaceBody = type.iface.body;

  // Register interface ports by mapping their internal symbols (where
  // applicable) to the corresponding virtual interface field.
  for (const auto *symbol : ifaceBody.getPortList()) {
    if (!symbol)
      continue;
    const auto *port = symbol->as_if<slang::ast::PortSymbol>();
    if (!port) {
      auto d = mlir::emitError(convertLocation(symbol->location))
               << "unsupported interface port symbol: "
               << slang::ast::toString(symbol->kind);
      if (!symbol->name.empty())
        d << " `" << symbol->name << "`";
      return failure();
    }
    if (!port->internalSymbol)
      continue;
    if (const auto *internal =
            port->internalSymbol->as_if<slang::ast::ValueSymbol>())
      registerMember(*internal, port->name);
  }

  // Register variables and nets declared in the interface body.
  for (auto &member : ifaceBody.members()) {
    if (const auto *var = member.as_if<slang::ast::VariableSymbol>()) {
      registerMember(*var, var->name);
      continue;
    }
    if (const auto *net = member.as_if<slang::ast::NetSymbol>()) {
      registerMember(*net, net->name);
      continue;
    }
  }

  return success();
}
