//===- ImportVerilogInternals.h - Internal implementation details ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef CONVERSION_IMPORTVERILOG_IMPORTVERILOGINTERNALS_H
#define CONVERSION_IMPORTVERILOG_IMPORTVERILOGINTERNALS_H

#include "circt/Conversion/ImportVerilog.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "slang/ast/ASTVisitor.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/Debug.h"
#include <map>
#include <queue>

#define DEBUG_TYPE "import-verilog"

namespace circt {
namespace ImportVerilog {

/// A helper class to facilitate the conversion from a Slang AST to MLIR
/// operations. Keeps track of the destination MLIR module, builders, and
/// various worklists and utilities needed for conversion.
struct Context {
  Context(mlir::ModuleOp intoModuleOp,
          const slang::SourceManager &sourceManager,
          SmallDenseMap<slang::BufferID, StringRef> &bufferFilePaths)
      : intoModuleOp(intoModuleOp), sourceManager(sourceManager),
        bufferFilePaths(bufferFilePaths),
        builder(OpBuilder::atBlockEnd(intoModuleOp.getBody())),
        symbolTable(intoModuleOp) {}
  Context(const Context &) = delete;

  /// Return the MLIR context.
  MLIRContext *getContext() { return intoModuleOp.getContext(); }

  /// Convert a slang `SourceLocation` into an MLIR `Location`.
  Location convertLocation(slang::SourceLocation loc);
  /// Convert a slang `SourceRange` into an MLIR `Location`.
  Location convertLocation(slang::SourceRange range);

  /// Convert a slang type into an MLIR type. Returns null on failure. Uses the
  /// provided location for error reporting, or tries to guess one from the
  /// given type. Types tend to have unreliable location information, so it's
  /// generally a good idea to pass in a location.
  Type convertType(const slang::ast::Type &type, LocationAttr loc = {});
  Type convertType(const slang::ast::DeclaredType &type);

  /// Convert hierarchy and structure AST nodes to MLIR ops.
  LogicalResult convertCompilation(slang::ast::Compilation &compilation);
  moore::SVModuleOp
  convertModuleHeader(const slang::ast::InstanceBodySymbol *module);
  LogicalResult convertModuleBody(const slang::ast::InstanceBodySymbol *module);

  // Convert a statement AST node to MLIR ops.
  LogicalResult convertStatement(const slang::ast::Statement &stmt);

  // Convert an expression AST node to MLIR ops.
  Value convertExpression(const slang::ast::Expression &expr);

  mlir::ModuleOp intoModuleOp;
  const slang::SourceManager &sourceManager;
  SmallDenseMap<slang::BufferID, StringRef> &bufferFilePaths;

  /// The builder used to create IR operations.
  OpBuilder builder;
  /// A symbol table of the MLIR module we are emitting into.
  SymbolTable symbolTable;

  /// The top-level operations ordered by their Slang source location. This is
  /// used to produce IR that follows the source file order.
  std::map<slang::SourceLocation, Operation *> orderedRootOps;
  /// How we have lowered modules to MLIR.
  DenseMap<const slang::ast::InstanceBodySymbol *, moore::SVModuleOp> moduleOps;
  /// A list of modules for which the header has been created, but the body has
  /// not been converted yet.
  std::queue<const slang::ast::InstanceBodySymbol *> moduleWorklist;

  /// A table of defined values, such as variables, that may be referred to by
  /// name in expressions. The expressions use this table to lookup the MLIR
  /// value that was created for a given declaration in the Slang AST node.
  using ValueSymbols =
      llvm::ScopedHashTable<const slang::ast::ValueSymbol *, Value>;
  using ValueSymbolScope = ValueSymbols::ScopeTy;
  ValueSymbols valueSymbols;
};

} // namespace ImportVerilog
} // namespace circt

#endif // CONVERSION_IMPORTVERILOG_IMPORTVERILOGINTERNALS_H
