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
#include "circt/Dialect/Debug/DebugOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "slang/ast/ASTVisitor.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/Debug.h"
#include <map>
#include <queue>

#define DEBUG_TYPE "import-verilog"

namespace circt {
namespace ImportVerilog {

using moore::Domain;

/// Port lowering information.
struct PortLowering {
  const slang::ast::PortSymbol &ast;
  Location loc;
  BlockArgument arg;
};

/// Module lowering information.
struct ModuleLowering {
  moore::SVModuleOp op;
  SmallVector<PortLowering> ports;
  DenseMap<const slang::syntax::SyntaxNode *, const slang::ast::PortSymbol *>
      portsBySyntaxNode;
};

/// Function lowering information.
struct FunctionLowering {
  mlir::func::FuncOp op;
};

/// Information about a loops continuation and exit blocks relevant while
/// lowering the loop's body statements.
struct LoopFrame {
  /// The block to jump to from a `continue` statement.
  Block *continueBlock;
  /// The block to jump to from a `break` statement.
  Block *breakBlock;
};

/// A helper class to facilitate the conversion from a Slang AST to MLIR
/// operations. Keeps track of the destination MLIR module, builders, and
/// various worklists and utilities needed for conversion.
struct Context {
  Context(const ImportVerilogOptions &options,
          slang::ast::Compilation &compilation, mlir::ModuleOp intoModuleOp,
          const slang::SourceManager &sourceManager,
          SmallDenseMap<slang::BufferID, StringRef> &bufferFilePaths)
      : options(options), compilation(compilation), intoModuleOp(intoModuleOp),
        sourceManager(sourceManager), bufferFilePaths(bufferFilePaths),
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
  LogicalResult convertCompilation();
  ModuleLowering *
  convertModuleHeader(const slang::ast::InstanceBodySymbol *module);
  LogicalResult convertModuleBody(const slang::ast::InstanceBodySymbol *module);
  LogicalResult convertPackage(const slang::ast::PackageSymbol &package);
  FunctionLowering *
  declareFunction(const slang::ast::SubroutineSymbol &subroutine);
  LogicalResult convertFunction(const slang::ast::SubroutineSymbol &subroutine);

  // Convert a statement AST node to MLIR ops.
  LogicalResult convertStatement(const slang::ast::Statement &stmt);

  // Convert an expression AST node to MLIR ops.
  Value convertRvalueExpression(const slang::ast::Expression &expr,
                                Type requiredType = {});
  Value convertLvalueExpression(const slang::ast::Expression &expr);

  // Convert a slang timing control into an MLIR timing control.
  LogicalResult convertTimingControl(const slang::ast::TimingControl &ctrl,
                                     const slang::ast::Statement &stmt);

  /// Helper function to convert a value to its "truthy" boolean value.
  Value convertToBool(Value value);

  /// Helper function to convert a value to its "truthy" boolean value and
  /// convert it to the given domain.
  Value convertToBool(Value value, Domain domain);

  /// Helper function to materialize an `SVInt` as an SSA value.
  Value materializeSVInt(const slang::SVInt &svint,
                         const slang::ast::Type &type, Location loc);

  /// Helper function to materialize a `ConstantValue` as an SSA value. Returns
  /// null if the constant cannot be materialized.
  Value materializeConstant(const slang::ConstantValue &constant,
                            const slang::ast::Type &type, Location loc);

  const ImportVerilogOptions &options;
  slang::ast::Compilation &compilation;
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
  DenseMap<const slang::ast::InstanceBodySymbol *,
           std::unique_ptr<ModuleLowering>>
      modules;
  /// A list of modules for which the header has been created, but the body has
  /// not been converted yet.
  std::queue<const slang::ast::InstanceBodySymbol *> moduleWorklist;

  /// Functions that have already been converted.
  DenseMap<const slang::ast::SubroutineSymbol *,
           std::unique_ptr<FunctionLowering>>
      functions;

  /// A table of defined values, such as variables, that may be referred to by
  /// name in expressions. The expressions use this table to lookup the MLIR
  /// value that was created for a given declaration in the Slang AST node.
  using ValueSymbols =
      llvm::ScopedHashTable<const slang::ast::ValueSymbol *, Value>;
  using ValueSymbolScope = ValueSymbols::ScopeTy;
  ValueSymbols valueSymbols;

  /// A stack of assignment left-hand side values. Each assignment will push its
  /// lowered left-hand side onto this stack before lowering its right-hand
  /// side. This allows expressions to resolve the opaque
  /// `LValueReferenceExpression`s in the AST.
  SmallVector<Value> lvalueStack;

  /// A stack of loop continuation and exit blocks. Each loop will push the
  /// relevant info onto this stack, lower its loop body statements, and pop the
  /// info off the stack again. Continue and break statements encountered as
  /// part of the loop body statements will use this information to branch to
  /// the correct block.
  SmallVector<LoopFrame> loopStack;

  /// A listener called for every variable or net being read. This can be used
  /// to collect all variables read as part of an expression or statement, for
  /// example to populate the list of observed signals in an implicit event
  /// control `@*`.
  std::function<void(moore::ReadOp)> rvalueReadCallback;
};

} // namespace ImportVerilog
} // namespace circt

#endif // CONVERSION_IMPORTVERILOG_IMPORTVERILOGINTERNALS_H
