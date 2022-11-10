//===- ImportVerilogInternals.h - Internal implementation details ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_IMPORTVERILOG_IMPORTVERILOGINTERNALS_H
#define CONVERSION_IMPORTVERILOG_IMPORTVERILOGINTERNALS_H

#include "circt/Conversion/ImportVerilog.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "slang/ast/Compilation.h"
#include "slang/ast/Definition.h"
#include "slang/syntax/SyntaxTree.h"
#include "slang/syntax/SyntaxVisitor.h"
#include "slang/text/SourceManager.h"
#include "llvm/Support/Debug.h"
#include <queue>

#define DEBUG_TYPE "import-verilog"

namespace circt {
namespace ImportVerilog {

struct Context {
  Context(mlir::ModuleOp intoModuleOp,
          const slang::SourceManager &sourceManager,
          std::function<StringRef(slang::BufferID)> getBufferFilePath,
          slang::ast::Compilation &compilation)
      : intoModuleOp(intoModuleOp), sourceManager(sourceManager),
        getBufferFilePath(getBufferFilePath), compilation(compilation),
        rootBuilder(OpBuilder::atBlockEnd(intoModuleOp.getBody())),
        symbolTable(intoModuleOp) {}
  Context(const Context &) = delete;

  /// Return the MLIR context.
  MLIRContext *getContext() { return intoModuleOp.getContext(); }

  /// Convert a slang `SourceLocation` into an MLIR `Location`.
  Location convertLocation(slang::SourceLocation loc);

  /// Convert a slang type into an MLIR type. Returns null on failure. Uses the
  /// provided location for error reporting, or tries to guess one from the
  /// given type. Types tend to have unreliable location information, so it's
  /// generally a good idea to pass in a location.
  Type convertType(const slang::ast::Type &type, LocationAttr loc = {});
  Type convertType(const slang::ast::DeclaredType &type);

  LogicalResult convertCompilation();
  Operation *convertModuleHeader(const slang::ast::InstanceBodySymbol *module);
  LogicalResult convertModuleBody(const slang::ast::InstanceBodySymbol *module);

  mlir::ModuleOp intoModuleOp;
  const slang::SourceManager &sourceManager;
  std::function<StringRef(slang::BufferID)> getBufferFilePath;
  slang::ast::Compilation &compilation;

  /// A builder for modules and other top-level ops.
  OpBuilder rootBuilder;
  /// A symbol table of the MLIR module we are emitting into.
  SymbolTable symbolTable;

  /// How we have lowered modules to MLIR.
  DenseMap<const slang::ast::InstanceBodySymbol *, Operation *> moduleOps;
  /// A list of modules for which the header has been created, but the body has
  /// not been converted yet.
  std::queue<const slang::ast::InstanceBodySymbol *> moduleWorklist;
};

/// Convert a slang `SourceLocation` to an MLIR `Location`.
Location convertLocation(
    MLIRContext *context, const slang::SourceManager &sourceManager,
    llvm::function_ref<StringRef(slang::BufferID)> getBufferFilePath,
    slang::SourceLocation loc);

} // namespace ImportVerilog
} // namespace circt

#endif // CONVERSION_IMPORTVERILOG_IMPORTVERILOGINTERNALS_H
