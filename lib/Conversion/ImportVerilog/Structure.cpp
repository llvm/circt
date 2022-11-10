//===- Structure.cpp - Slang hierarchy conversion -------------------------===//
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

LogicalResult Context::convertCompilation() {
  auto &root = compilation.getRoot();

  // Visit all the compilation units. This will mainly cover non-instantiable
  // things like packages.
  for (auto *unit : root.compilationUnits)
    for (auto &member : unit->members())
      LLVM_DEBUG(llvm::dbgs() << "Converting symbol " << member.name << "\n");

  // Prime the root definition worklist by adding all the top-level modules to
  // it.
  for (auto *inst : root.topInstances)
    convertModuleHeader(&inst->body);

  // Convert all the root module definitions.
  while (!moduleWorklist.empty()) {
    auto module = moduleWorklist.front();
    moduleWorklist.pop();
    if (failed(convertModuleBody(module)))
      return failure();
  }

  return success();
}

Operation *
Context::convertModuleHeader(const slang::ast::InstanceBodySymbol *module) {
  if (auto *op = moduleOps.lookup(module))
    return op;
  auto loc = convertLocation(module->location);

  // We only support modules for now. Extension to interfaces and programs
  // should be trivial though, since they are essentially the same thing with
  // only minor differences in semantics.
  if (module->getDefinition().definitionKind !=
      slang::ast::DefinitionKind::Module) {
    mlir::emitError(loc, "unsupported construct: ")
        << module->getDefinition().getKindString();
    return nullptr;
  }

  // Create an empty module that corresponds to this module.
  auto moduleOp = rootBuilder.create<moore::SVModuleOp>(loc, module->name);
  moduleOp.getBody().emplaceBlock();

  // Add the module to the symbol table of the MLIR module, which uniquifies its
  // name as we'd expect.
  symbolTable.insert(moduleOp);

  // Schedule the body to be lowered.
  moduleWorklist.push(module);
  moduleOps.insert({module, moduleOp});
  return moduleOp;
}

LogicalResult
Context::convertModuleBody(const slang::ast::InstanceBodySymbol *module) {
  LLVM_DEBUG(llvm::dbgs() << "Converting body of module " << module->name
                          << "\n");
  auto moduleOp = moduleOps.lookup(module);
  assert(moduleOp);
  auto builder =
      OpBuilder::atBlockEnd(&cast<moore::SVModuleOp>(moduleOp).getBodyBlock());

  for (auto &member : module->members()) {
    LLVM_DEBUG(llvm::dbgs()
               << "- Handling " << slang::ast::toString(member.kind) << "\n");
    auto loc = convertLocation(member.location);

    // Skip parameters.
    if (member.kind == slang::ast::SymbolKind::Parameter)
      continue;

    // Handle instances.
    if (member.kind == slang::ast::SymbolKind::Instance) {
      auto &instAst = member.as<slang::ast::InstanceSymbol>();
      auto targetModule = convertModuleHeader(&instAst.body);
      if (!targetModule)
        return failure();
      builder.create<moore::InstanceOp>(
          loc, builder.getStringAttr(instAst.name),
          FlatSymbolRefAttr::get(SymbolTable::getSymbolName(targetModule)));
      continue;
    }

    // Handle variables.
    if (member.kind == slang::ast::SymbolKind::Variable) {
      auto &varAst = member.as<slang::ast::VariableSymbol>();
      auto loweredType = convertType(*varAst.getDeclaredType());
      if (!loweredType)
        return failure();
      builder.create<moore::VariableOp>(convertLocation(varAst.location),
                                        loweredType,
                                        builder.getStringAttr(varAst.name));
      continue;
    }

    mlir::emitError(loc, "unsupported module member: ")
        << slang::ast::toString(member.kind);
    return failure();
  }

  return success();
}
