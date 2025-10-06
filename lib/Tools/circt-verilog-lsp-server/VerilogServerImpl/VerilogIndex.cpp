//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "slang/ast/ASTVisitor.h"
#include "slang/ast/Expression.h"
#include "slang/ast/statements/MiscStatements.h"
#include "slang/ast/symbols/CompilationUnitSymbols.h"
#include "slang/ast/symbols/InstanceSymbols.h"
#include "slang/ast/symbols/MemberSymbols.h"
#include "slang/ast/symbols/VariableSymbols.h"
#include "slang/syntax/AllSyntax.h"
#include "slang/text/SourceManager.h"

#include "mlir/IR/Location.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/ADT/StringExtras.h"

#include "../Utils/LSPUtils.h"
#include "VerilogIndex.h"

using namespace circt::lsp;
using namespace llvm;

namespace {

/// Helper function to determine whether an AST node is outside of the main
/// buffer. If this can't be safely determined, return false.
template <typename T>
inline bool definitelyOutsideMainBuffer(const T &t, slang::BufferID bufferId) {
  if constexpr (requires {
                  t.location;
                }) { // AST nodes with location, e.g. Symbols
    auto r = t.location;
    if (r.valid())
      return r.buffer() != bufferId;
  }

  if constexpr (requires {
                  t.sourceRange;
                }) { // AST nodes with sourceRange, e.g. expressions
    auto r = t.sourceRange;
    if (r.start().valid() && r.end().valid())
      return r.start().buffer() != bufferId && r.end().buffer() != bufferId;
  }

  if constexpr (requires { t.getSyntax(); }) { // Fallback to syntax range
    if (auto *syn = t.getSyntax()) {
      auto r = syn->sourceRange(); // SyntaxNodes always have a source range.
      if (r.start().valid() && r.end().valid())
        return r.start().buffer() != bufferId && r.end().buffer() != bufferId;
    }
  }
  return false; // not enough info => don’t prune
}

// Index the AST to find symbol uses and definitions.
struct VerilogIndexer : slang::ast::ASTVisitor<VerilogIndexer, true, true> {
  using ASTBase = slang::ast::ASTVisitor<VerilogIndexer, true, true>;
  VerilogIndexer(VerilogIndex &index) : index(index) {}
  VerilogIndex &index;

  void insertSymbol(const slang::ast::Symbol *symbol, slang::SourceRange range,
                    bool isDefinition = true) {
    if (symbol->name.empty())
      return;
    assert(range.start().valid() && range.end().valid() &&
           "range must be valid");

    // TODO: This implementation does not handle expanded MACROs. Return
    // instead.
    if (range.start().offset() >= range.end().offset()) {
      return;
    }

    index.insertSymbol(symbol, range, isDefinition);
  }

  void insertSymbol(const slang::ast::Symbol *symbol,
                    slang::SourceLocation from, bool isDefinition = false) {
    if (symbol->name.empty())
      return;
    assert(from.valid() && "location must be valid");
    insertSymbol(symbol, slang::SourceRange(from, from + symbol->name.size()),
                 isDefinition);
  }

  // Handle named values, such as references to declared variables.
  void visitExpression(const slang::ast::Expression &expr) {
    auto *symbol = expr.getSymbolReference(true);
    if (!symbol)
      return;
    insertSymbol(symbol, expr.sourceRange, /*isDefinition=*/false);
  }

  void visitSymbol(const slang::ast::Symbol &symbol) {
    insertSymbol(&symbol, symbol.location, /*isDefinition=*/true);
  }

  void visit(const slang::ast::NetSymbol &expr) {
    insertSymbol(&expr, expr.location, /*isDefinition=*/true);
  }

  void visit(const slang::ast::VariableSymbol &expr) {
    insertSymbol(&expr, expr.location, /*isDefinition=*/true);
  }

  void visit(const slang::ast::ExplicitImportSymbol &expr) {
    auto *def = expr.package();
    if (!def)
      return;

    if (auto *syn = expr.getSyntax()) {
      if (auto *item = syn->as_if<slang::syntax::PackageImportItemSyntax>()) {
        insertSymbol(def, item->package.location(), /*isDefinition=*/false);
      }
    }
  }

  void visit(const slang::ast::WildcardImportSymbol &expr) {
    auto *def = expr.getPackage();
    if (!def)
      return;

    if (auto *syn = expr.getSyntax()) {
      if (auto *item = syn->as_if<slang::syntax::PackageImportItemSyntax>()) {
        insertSymbol(def, item->package.location(), false);
      }
    }
  }

  void visit(const slang::ast::InstanceSymbol &expr) {
    auto *def = &expr.getDefinition();
    if (!def)
      return;

    // Add the module definition
    insertSymbol(def, def->location, /*isDefinition=*/true);

    // Walk up the syntax tree until we hit the type token;
    // Link that token back to the instance declaration.
    if (auto *hierInst =
            expr.getSyntax()
                ->as_if<slang::syntax::HierarchicalInstanceSyntax>())
      if (auto *modInst =
              hierInst->parent
                  ->as_if<slang::syntax::HierarchyInstantiationSyntax>())
        if (modInst->type)
          insertSymbol(def, modInst->type.location(), false);

    // Link the module instance name back to the module definition
    insertSymbol(def, expr.location, /*isDefinition=*/false);
  }

  void visit(const slang::ast::VariableDeclStatement &expr) {
    insertSymbol(&expr.symbol, expr.sourceRange, /*isDefinition=*/true);
  }

  template <typename T>
  void visit(const T &node) {
    if constexpr (std::is_base_of_v<slang::ast::Expression, T>)
      visitExpression(node);
    if constexpr (std::is_base_of_v<slang::ast::Symbol, T>)
      visitSymbol(node);

    // Check if this node is already out of the main buffer.
    if (definitelyOutsideMainBuffer(node, index.getBufferId()))
      return;

    // Otherwise, recurse and keep indexing.
    ASTBase::template visitDefault<T>(node);
  }

  template <typename T>
  void visitInvalid(const T &t) {}
};
} // namespace

void VerilogIndex::initialize(slang::ast::Compilation &compilation) {
  const auto &root = compilation.getRoot();
  VerilogIndexer visitor(*this);
  for (auto *inst : root.topInstances) {

    if (inst->body.location.buffer() != getBufferId())
      continue;

    // Visit the body of the instance.
    inst->body.visit(visitor);

    // Insert the symbols in the port list.
    for (const auto *symbol : inst->body.getPortList())
      insertSymbolDefinition(symbol);
  }

  // Parse the source location from the main file.
  parseSourceLocation();
}

void VerilogIndex::parseSourceLocation(StringRef toParse) {
  // No multiline entries.
  if (toParse.contains('\n'))
    return;

  StringRef filePath;
  SmallVector<StringRef, 3> fileLineColStrs;

  // Parse the source location emitted by ExportVerilog, e.g.
  // @[foo.mlir:1:10, :20:30, bar.mlir:2:{30, 40}]
  for (auto chunk : llvm::split(toParse, ", ")) {
    fileLineColStrs.clear();
    chunk.split(fileLineColStrs, ':');
    if (fileLineColStrs.size() != 3)
      continue;

    auto filePathMaybeEmpty = fileLineColStrs[0].trim();
    // If the file path is empty, use the previous file path.
    if (!filePathMaybeEmpty.empty())
      filePath = filePathMaybeEmpty;

    auto line = fileLineColStrs[1].trim();
    auto column = fileLineColStrs[2].trim();

    uint32_t lineInt;
    // Line must be always valid.
    if (line.getAsInteger(10, lineInt))
      continue;

    // A pair of column and start location. Start location may include filepath
    // and line string.
    SmallVector<std::pair<StringRef, const char *>> columns;

    // Column string may contains several columns like `{col1, col2, ...}`.
    if (column.starts_with('{') && column.ends_with('}')) {
      bool first = true;
      for (auto str : llvm::split(column.drop_back().drop_front(), ',')) {
        columns.emplace_back(str,
                             first ? filePathMaybeEmpty.data() : str.data());
        first = false;
      }
    } else {
      columns.push_back({column, filePathMaybeEmpty.data()});
    }

    // Insert the interval into the interval map.
    for (auto [column, start] : columns) {
      uint32_t columnInt;
      if (column.getAsInteger(10, columnInt))
        continue;
      auto loc = mlir::FileLineColRange::get(&mlirContext, filePath,
                                             lineInt - 1, columnInt - 1,
                                             lineInt - 1, columnInt - 1);
      const char *end = column.end();
      if (!intervalMap.overlaps(start, end))
        intervalMap.insert(start, end, loc);
    }
  }
}

void VerilogIndex::parseSourceLocation() {
  auto &sourceMgr = getSlangSourceManager();
  auto getMainBuffer = sourceMgr.getSourceText(getBufferId());
  StringRef text(getMainBuffer);

  // Loop over comments starting with "@[", and parse the source location.
  // TODO: Consider supporting other location format. This is currently
  // very specific to `locationInfoStyle=WrapInAtSquareBracket`.
  while (true) {
    // Find the source location from the text.
    StringRef start = "// @[";
    auto loc = text.find(start);
    if (loc == StringRef::npos)
      break;

    text = text.drop_front(loc + start.size());
    auto endPos = text.find_first_of("]\n");
    if (endPos == StringRef::npos)
      break;
    auto toParse = text.take_front(endPos);
    circt::lsp::Logger::info(toParse);
    parseSourceLocation(toParse);
  }
}

void VerilogIndex::insertSymbol(const slang::ast::Symbol *symbol,
                                slang::SourceRange from, bool isDefinition) {
  assert(from.start().valid() && from.end().valid());

  // TODO: Currently doesn't handle expanded macros
  if (from.start().offset() >= from.end().offset())
    return;

  auto lhsBufferId = from.start().buffer();
  auto rhsBufferId = from.end().buffer();
  if (lhsBufferId.getId() != rhsBufferId.getId() || !rhsBufferId.valid() ||
      !lhsBufferId.valid())
    return;

  auto buffer = getSlangSourceManager().getSourceText(lhsBufferId);

  const auto *lhsBound = buffer.data() + from.start().offset();
  const auto *rhsBound = buffer.data() + from.end().offset();

  if (lhsBound >= rhsBound)
    return;

  if (!intervalMap.overlaps(lhsBound, rhsBound)) {
    intervalMap.insert(lhsBound, rhsBound, symbol);
    if (!isDefinition)
      references[symbol].push_back(from);
  }
}

void VerilogIndex::insertSymbolDefinition(const slang::ast::Symbol *symbol) {
  if (!symbol->location)
    return;
  auto size = symbol->name.size() ? symbol->name.size() : 1;
  auto range = slang::SourceRange(symbol->location, symbol->location + size);

  insertSymbol(symbol, range, true);
}
