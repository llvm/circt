//===- Structure.cpp - Slang hierarchy conversion -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/ast/Compilation.h"

using namespace circt;
using namespace ImportVerilog;

//===----------------------------------------------------------------------===//
// Module Member Conversion
//===----------------------------------------------------------------------===//

static moore::ProcedureKind
convertProcedureKind(slang::ast::ProceduralBlockKind kind) {
  switch (kind) {
  case slang::ast::ProceduralBlockKind::Always:
    return moore::ProcedureKind::Always;
  case slang::ast::ProceduralBlockKind::AlwaysComb:
    return moore::ProcedureKind::AlwaysComb;
  case slang::ast::ProceduralBlockKind::AlwaysLatch:
    return moore::ProcedureKind::AlwaysLatch;
  case slang::ast::ProceduralBlockKind::AlwaysFF:
    return moore::ProcedureKind::AlwaysFF;
  case slang::ast::ProceduralBlockKind::Initial:
    return moore::ProcedureKind::Initial;
  case slang::ast::ProceduralBlockKind::Final:
    return moore::ProcedureKind::Final;
  }
  llvm_unreachable("all procedure kinds handled");
}

static moore::NetKind convertNetKind(slang::ast::NetType::NetKind kind) {
  switch (kind) {
  case slang::ast::NetType::Supply0:
    return moore::NetKind::Supply0;
  case slang::ast::NetType::Supply1:
    return moore::NetKind::Supply1;
  case slang::ast::NetType::Tri:
    return moore::NetKind::Tri;
  case slang::ast::NetType::TriAnd:
    return moore::NetKind::TriAnd;
  case slang::ast::NetType::TriOr:
    return moore::NetKind::TriOr;
  case slang::ast::NetType::TriReg:
    return moore::NetKind::TriReg;
  case slang::ast::NetType::Tri0:
    return moore::NetKind::Tri0;
  case slang::ast::NetType::Tri1:
    return moore::NetKind::Tri1;
  case slang::ast::NetType::UWire:
    return moore::NetKind::UWire;
  case slang::ast::NetType::Wire:
    return moore::NetKind::Wire;
  case slang::ast::NetType::WAnd:
    return moore::NetKind::WAnd;
  case slang::ast::NetType::WOr:
    return moore::NetKind::WOr;
  case slang::ast::NetType::Interconnect:
    return moore::NetKind::Interconnect;
  case slang::ast::NetType::UserDefined:
    return moore::NetKind::UserDefined;
  case slang::ast::NetType::Unknown:
    return moore::NetKind::Unknown;
  }
  llvm_unreachable("all net kinds handled");
}

namespace {
struct MemberVisitor {
  Context &context;
  Location loc;
  OpBuilder &builder;

  MemberVisitor(Context &context, Location loc)
      : context(context), loc(loc), builder(context.builder) {}

  // Skip empty members (stray semicolons).
  LogicalResult visit(const slang::ast::EmptyMemberSymbol &) {
    return success();
  }

  // Skip members that are implicitly imported from some other scope for the
  // sake of name resolution, such as enum variant names.
  LogicalResult visit(const slang::ast::TransparentMemberSymbol &) {
    return success();
  }

  // Skip typedefs.
  LogicalResult visit(const slang::ast::TypeAliasType &) { return success(); }

  // Handle instances.
  LogicalResult visit(const slang::ast::InstanceSymbol &instNode) {
    auto targetModule = context.convertModuleHeader(&instNode.body);
    if (!targetModule)
      return failure();

    builder.create<moore::InstanceOp>(
        loc, builder.getStringAttr(instNode.name),
        FlatSymbolRefAttr::get(targetModule.getSymNameAttr()));

    return success();
  }

  // Handle variables.
  LogicalResult visit(const slang::ast::VariableSymbol &varNode) {
    auto loweredType = context.convertType(*varNode.getDeclaredType());
    if (!loweredType)
      return failure();

    Value initial;
    if (const auto *init = varNode.getInitializer()) {
      initial = context.convertExpression(*init);
      if (!initial)
        return failure();

      if (initial.getType() != loweredType)
        initial =
            builder.create<moore::ConversionOp>(loc, loweredType, initial);
    }

    auto varOp = builder.create<moore::VariableOp>(
        loc, loweredType, builder.getStringAttr(varNode.name), initial);
    context.valueSymbols.insert(&varNode, varOp);
    return success();
  }

  // Handle nets.
  LogicalResult visit(const slang::ast::NetSymbol &netNode) {
    auto loweredType = context.convertType(*netNode.getDeclaredType());
    if (!loweredType)
      return failure();

    Value assignment;
    if (netNode.getInitializer()) {
      assignment = context.convertExpression(*netNode.getInitializer());
      if (!assignment)
        return failure();
    }

    auto netkind = convertNetKind(netNode.netType.netKind);
    if (netkind == moore::NetKind::Interconnect ||
        netkind == moore::NetKind::UserDefined ||
        netkind == moore::NetKind::Unknown)
      return mlir::emitError(loc, "unsupported net kind `")
             << netNode.netType.name << "`";

    auto netOp = builder.create<moore::NetOp>(
        loc, loweredType, builder.getStringAttr(netNode.name), netkind,
        assignment);
    context.valueSymbols.insert(&netNode, netOp);
    return success();
  }

  // Handle continuous assignments.
  LogicalResult visit(const slang::ast::ContinuousAssignSymbol &assignNode) {
    if (const auto *delay = assignNode.getDelay()) {
      auto loc = context.convertLocation(delay->sourceRange);
      return mlir::emitError(loc,
                             "delayed continuous assignments not supported");
    }

    const auto &expr =
        assignNode.getAssignment().as<slang::ast::AssignmentExpression>();

    auto lhs = context.convertExpression(expr.left());
    auto rhs = context.convertExpression(expr.right());
    if (!lhs || !rhs)
      return failure();

    if (lhs.getType() != rhs.getType())
      rhs = builder.create<moore::ConversionOp>(loc, lhs.getType(), rhs);

    builder.create<moore::ContinuousAssignOp>(loc, lhs, rhs);
    return success();
  }

  // Handle procedures.
  LogicalResult visit(const slang::ast::ProceduralBlockSymbol &procNode) {
    auto procOp = builder.create<moore::ProcedureOp>(
        loc, convertProcedureKind(procNode.procedureKind));
    procOp.getBodyRegion().emplaceBlock();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(procOp.getBody());
    Context::ValueSymbolScope scope(context.valueSymbols);
    return context.convertStatement(procNode.getBody());
  }

  // Ignore statement block symbols. These get generated by Slang for blocks
  // with variables and other declarations. For example, having an initial
  // procedure with a variable declaration, such as `initial begin int x;
  // end`, will create the procedure with a block and variable declaration as
  // expected, but will also create a `StatementBlockSymbol` with just the
  // variable layout _next to_ the initial procedure.
  LogicalResult visit(const slang::ast::StatementBlockSymbol &) {
    return success();
  }

  /// Emit an error for all other members.
  template <typename T>
  LogicalResult visit(T &&node) {
    mlir::emitError(loc, "unsupported construct: ")
        << slang::ast::toString(node.kind);
    return failure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Structure and Hierarchy Conversion
//===----------------------------------------------------------------------===//

/// Convert an entire Slang compilation to MLIR ops. This is the main entry
/// point for the conversion.
LogicalResult
Context::convertCompilation(slang::ast::Compilation &compilation) {
  const auto &root = compilation.getRoot();

  // Visit all top-level declarations in all compilation units. This does not
  // include instantiable constructs like modules, interfaces, and programs,
  // which are listed separately as top instances.
  for (auto *unit : root.compilationUnits) {
    for (const auto &member : unit->members()) {
      // Error out on all top-level declarations.
      auto loc = convertLocation(member.location);
      return mlir::emitError(loc, "unsupported construct: ")
             << slang::ast::toString(member.kind);
    }
  }

  // Prime the root definition worklist by adding all the top-level modules.
  SmallVector<const slang::ast::InstanceSymbol *> topInstances;
  for (auto *inst : root.topInstances)
    convertModuleHeader(&inst->body);

  // Convert all the root module definitions.
  while (!moduleWorklist.empty()) {
    auto *module = moduleWorklist.front();
    moduleWorklist.pop();
    if (failed(convertModuleBody(module)))
      return failure();
  }

  return success();
}

/// Convert a module and its ports to an empty module op in the IR. Also adds
/// the op to the worklist of module bodies to be lowered. This acts like a
/// module "declaration", allowing instances to already refer to a module even
/// before its body has been lowered.
moore::SVModuleOp
Context::convertModuleHeader(const slang::ast::InstanceBodySymbol *module) {
  if (auto op = moduleOps.lookup(module))
    return op;
  auto loc = convertLocation(module->location);
  OpBuilder::InsertionGuard g(builder);

  // We only support modules for now. Extension to interfaces and programs
  // should be trivial though, since they are essentially the same thing with
  // only minor differences in semantics.
  if (module->getDefinition().definitionKind !=
      slang::ast::DefinitionKind::Module) {
    mlir::emitError(loc, "unsupported construct: ")
        << module->getDefinition().getKindString();
    return {};
  }

  // Handle the port list.
  for (auto *symbol : module->getPortList()) {
    auto portLoc = convertLocation(symbol->location);
    mlir::emitError(portLoc, "unsupported module port: ")
        << slang::ast::toString(symbol->kind);
    return {};
  }

  // Pick an insertion point for this module according to the source file
  // location.
  auto it = orderedRootOps.lower_bound(module->location);
  if (it == orderedRootOps.end())
    builder.setInsertionPointToEnd(intoModuleOp.getBody());
  else
    builder.setInsertionPoint(it->second);

  // Create an empty module that corresponds to this module.
  auto moduleOp = builder.create<moore::SVModuleOp>(loc, module->name);
  orderedRootOps.insert(it, {module->location, moduleOp});
  moduleOp.getBodyRegion().emplaceBlock();

  // Add the module to the symbol table of the MLIR module, which uniquifies its
  // name as we'd expect.
  symbolTable.insert(moduleOp);

  // Schedule the body to be lowered.
  moduleWorklist.push(module);
  moduleOps.insert({module, moduleOp});
  return moduleOp;
}

/// Convert a module's body to the corresponding IR ops. The module op must have
/// already been created earlier through a `convertModuleHeader` call.
LogicalResult
Context::convertModuleBody(const slang::ast::InstanceBodySymbol *module) {
  auto moduleOp = moduleOps.lookup(module);
  assert(moduleOp);
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  ValueSymbolScope scope(valueSymbols);
  for (auto &member : module->members()) {
    auto loc = convertLocation(member.location);
    if (failed(member.visit(MemberVisitor(*this, loc))))
      return failure();
  }

  return success();
}
