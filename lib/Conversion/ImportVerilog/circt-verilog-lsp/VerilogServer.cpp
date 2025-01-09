//===- VerilogServer.cpp - Verilog Language Server
//------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VerilogServer.h"

#include "Protocol.h"
#include "circt/Tools/circt-verilog-lsp/CirctVerilogLspServerMain.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/lsp-server-support/CompilationDatabase.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Protocol.h"
#include "mlir/Tools/lsp-server-support/SourceMgrUtils.h"
#include "slang/ast/ASTVisitor.h"
#include "slang/ast/Compilation.h"
#include "slang/ast/Definition.h"
#include "slang/ast/Statements.h"
#include "slang/ast/SystemSubroutine.h"
#include "slang/ast/expressions/AssignmentExpressions.h"
#include "slang/ast/symbols/CompilationUnitSymbols.h"
#include "slang/ast/symbols/InstanceSymbols.h"
#include "slang/ast/symbols/PortSymbols.h"
#include "slang/ast/symbols/VariableSymbols.h"
#include "slang/diagnostics/DiagnosticClient.h"
#include "slang/diagnostics/Diagnostics.h"
#include "slang/driver/Driver.h"
#include "slang/syntax/AllSyntax.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"

#include "slang/parsing/Preprocessor.h"
#include "slang/syntax/SyntaxPrinter.h"
#include "slang/syntax/SyntaxTree.h"
#include "slang/text/SourceLocation.h"
#include "slang/text/SourceManager.h"
#include "slang/util/Version.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>

using namespace mlir;
using namespace mlir::lsp;

using namespace circt::lsp;
using namespace circt;

/// Returns a language server uri for the given source location. `mainFileURI`
/// corresponds to the uri for the main file of the source manager.
static mlir::lsp::URIForFile
getURIFromLoc(llvm::SourceMgr &mgr, SMRange loc,
              const mlir::lsp::URIForFile &mainFileURI) {
  int bufferId = mgr.FindBufferContainingLoc(loc.Start);
  if (bufferId == 0 || bufferId == static_cast<int>(mgr.getMainFileID()))
    return mainFileURI;
  llvm::Expected<mlir::lsp::URIForFile> fileForLoc =
      mlir::lsp::URIForFile::fromFile(
          mgr.getBufferInfo(bufferId).Buffer->getBufferIdentifier());
  if (fileForLoc)
    return *fileForLoc;
  mlir::lsp::Logger::error("Failed to create URI for include file: {0}",
                           llvm::toString(fileForLoc.takeError()));
  return mainFileURI;
}

/// Returns true if the given location is in the main file of the source
/// manager.
static bool isMainFileLoc(llvm::SourceMgr &mgr, SMRange loc) {
  return mgr.FindBufferContainingLoc(loc.Start) == mgr.getMainFileID();
}

/// Returns a language server location from the given source range.
static mlir::lsp::Location
getLocationFromLoc(llvm::SourceMgr &mgr, SMRange range,
                   const mlir::lsp::URIForFile &uri) {
  return mlir::lsp::Location(getURIFromLoc(mgr, range, uri),
                             mlir::lsp::Range(mgr, range));
}

/// Convert a slang `SourceLocation` to an MLIR `Location`.
static mlir::lsp::Location
convertLocation(const slang::SourceManager &sourceManager,
                slang::SourceLocation loc) {
  if (loc && loc.buffer() != slang::SourceLocation::NoLocation.buffer()) {
    auto fileName = sourceManager.getFileName(loc);
    auto line = sourceManager.getLineNumber(loc) - 1;
    auto column = sourceManager.getColumnNumber(loc) - 1;
    auto loc = mlir::lsp::URIForFile::fromFile(
        sourceManager.makeAbsolutePath(fileName));
    if (auto e = loc.takeError())
      return mlir::lsp::Location();
    return mlir::lsp::Location(loc.get(),
                               mlir::lsp::Range(Position(line, column)));
  }
  return mlir::lsp::Location();
}

/// Convert the given MLIR diagnostic to the LSP form.
static std::optional<mlir::lsp::Diagnostic>
getLspDiagnoticFromDiag(slang::SourceManager &sourceMgr,
                        const slang::ReportedDiagnostic &diag,
                        const mlir::lsp::URIForFile &uri) {
  mlir::lsp::Diagnostic lspDiag;
  lspDiag.source = "verilog";

  // FIXME: Right now all of the diagnostics are treated as parser issues, but
  // some are parser and some are verifier.
  lspDiag.category = "Parse Error";

  // Try to grab a file location for this diagnostic.
  mlir::lsp::Location loc = convertLocation(sourceMgr, diag.location);
  lspDiag.range = loc.range;

  // Skip diagnostics that weren't emitted within the main file.
  if (loc.uri != uri)
    return std::nullopt;

  // lspDiag.severity = getSeverity(diag.severity);
  // lspDiag.message = diag.formattedMessage;

  // Attach any notes to the main diagnostic as related information.
  // std::vector<mlir::lsp::DiagnosticRelatedInformation> relatedDiags;
  // for (const slang::Diagnostic &note : diag.getNotes()) {
  //   relatedDiags.emplace_back(
  //       getLocationFromLoc(sourceMgr, note.getLocation(), uri),
  //       note.getMessage().str());
  // }
  // if (!relatedDiags.empty())
  //   lspDiag.relatedInformation = std::move(relatedDiags);

  return lspDiag;
}

/// Get or extract the documentation for the given decl.
static std::optional<std::string> getDocumentationFor(
    llvm::SourceMgr &sourceMgr, const slang::SourceManager &slangSourceMgr,
    llvm::SmallDenseMap<slang::BufferID, StringRef> &bufferFilePaths,
    const slang::ast::Statement *stmt) {
  // If the decl already had documentation set, use it.
  // if (std::optional<StringRef> doc = decl->getDocComment())
  //   return doc->str();

  // auto loc = convertLocation(slangSourceMgr, bufferFilePaths,
  // stmt->sourceRange.start()); If the decl doesn't yet have documentation, try
  // to extract it from the source file.

  // FIXME: Is this ideal to use two different source managers?
  auto line = slangSourceMgr.getLineNumber(stmt->sourceRange.start());
  auto column = slangSourceMgr.getColumnNumber(stmt->sourceRange.start());

  auto loc = sourceMgr.FindLocForLineAndColumn(sourceMgr.getMainFileID(), line,
                                               column);

  return mlir::lsp::extractSourceDocComment(sourceMgr, loc);
}

//===----------------------------------------------------------------------===//
// VerilogIndex
//===----------------------------------------------------------------------===//

namespace {
using VerilogIndexSymbol = slang::ast::Symbol;

// struct VerilogIndexSymbol =
//   explicit VerilogIndexSymbol(const slang::ast::Statement *definition)
//       : definition(definition) {}
//
//   SMRange getDefLoc() const {
//     definition->getLoc();
//     if (const slang::ast::VariableDeclStatement *decl =
//             llvm::dyn_cast_if_present<const slang::ast::VariableDeclStatement
//             *>(definition)) {
//       const ast::Name *declName = decl->getName();
//       return declName ? declName->getLoc() : decl->getLoc();
//     }
//     return definition.get<const ods::Operation *>()->getLoc();
//   }
//
//   /// The main definition of the symbol.
//   const slang::ast::VariableDeclStatement *definition;
//   /// The set of references to the symbol.
//   std::vector<SMRange> references;
// };

/// This class provides an index for definitions/uses within a Verilog document.
/// It provides efficient lookup of a definition given an input source range.

class VerilogServerContext {
public:
  struct UserHint {
    std::unique_ptr<llvm::json::Value> json = nullptr;
  };

  VerilogServerContext() {}

  const llvm::SourceMgr &getSourceMgr() const { return sourceMgr; }
  const llvm::SmallDenseMap<uint32_t, uint32_t> &getBufferIDMap() const {
    return bufferIDMap;
  }

  llvm::SMLoc getSMLoc(slang::SourceLocation loc) {
    auto bufferID = loc.buffer().getId();
    mlir::lsp::Logger::debug("id {}", bufferID);
    mlir::lsp::Logger::debug("id 2 {}", bufferID);

    auto bufferIDMapIt = bufferIDMap.find(bufferID);
    if (bufferIDMapIt == bufferIDMap.end()) {
      auto path = getSlangSourceManager().getFullPath(loc.buffer());
      auto memBuffer = llvm::MemoryBuffer::getFile(path.string());
      if (!memBuffer) {
        mlir::lsp::Logger::error("Failed to open file: {0}",
                                 memBuffer.getError().message());
        return llvm::SMLoc();
      }

      auto id =
          sourceMgr.AddNewSourceBuffer(std::move(memBuffer.get()), SMLoc());

      bufferIDMapIt =
          bufferIDMap.insert({bufferID, static_cast<uint32_t>(id)}).first;
    }

    mlir::lsp::Logger::debug("id result {}", bufferIDMapIt->second);

    const auto *buffer = sourceMgr.getMemoryBuffer(bufferIDMapIt->second);
    assert(buffer->getBufferSize() > loc.offset());
    return llvm::SMLoc::getFromPointer(buffer->getBufferStart() + loc.offset());
  }

  mlir::lsp::Location getLspLocation(slang::SourceLocation loc) const {
    if (loc && loc.buffer() != slang::SourceLocation::NoLocation.buffer()) {
      const auto &slangSourceManager = getSlangSourceManager();
      auto fileName = slangSourceManager.getFileName(loc);
      auto line = slangSourceManager.getLineNumber(loc) - 1;
      auto column = slangSourceManager.getColumnNumber(loc) - 1;
      auto loc = mlir::lsp::URIForFile::fromFile(
          slangSourceManager.makeAbsolutePath(fileName));
      if (auto e = loc.takeError())
        return mlir::lsp::Location();
      return mlir::lsp::Location(loc.get(),
                                 mlir::lsp::Range(Position(line, column)));
    }

    return mlir::lsp::Location();
  }

  const UserHint &getUserHint() const { return userHint; }
  const slang::SourceManager &getSlangSourceManager() const {
    return driver.sourceManager;
  }

  llvm::SmallDenseMap<uint32_t, uint32_t> bufferIDMap;
  llvm::StringMap<std::pair<uint32_t, SmallString<128>>> filePathMap;
  llvm::SmallVector<SmallString<128>> originalFileRoots;
  UserHint userHint;
  slang::driver::Driver driver;
  llvm::SourceMgr sourceMgr;
};

static mlir::lsp::DiagnosticSeverity
getSeverity(slang::DiagnosticSeverity severity) {
  switch (severity) {
  case slang::DiagnosticSeverity::Fatal:
  case slang::DiagnosticSeverity::Error:
    return mlir::lsp::DiagnosticSeverity::Error;
  case slang::DiagnosticSeverity::Warning:
    return mlir::lsp::DiagnosticSeverity::Warning;
  case slang::DiagnosticSeverity::Ignored:
  case slang::DiagnosticSeverity::Note:
    return mlir::lsp::DiagnosticSeverity::Information;
  }
  llvm_unreachable("all slang diagnostic severities should be handled");
  return mlir::lsp::DiagnosticSeverity::Error;
}
/// A converter that can be plugged into a slang `DiagnosticEngine` as a client
/// that will map slang diagnostics to their MLIR counterpart and emit them.
class MlirDiagnosticClient : public slang::DiagnosticClient {
  const VerilogServerContext &context;
  std::vector<mlir::lsp::Diagnostic> &diags;

public:
  MlirDiagnosticClient(const VerilogServerContext &context,
                       std::vector<mlir::lsp::Diagnostic> &diags)
      : context(context), diags(diags) {}

  void report(const slang::ReportedDiagnostic &slangDiag) override {
    // Generate the primary MLIR diagnostic.
    // auto &diagEngine = context->getDiagEngine();
    mlir::lsp::Logger::debug("MlirDiagnosticClient::report {}",
                             slangDiag.formattedMessage);
    diags.emplace_back();
    auto &mlirDiag = diags.back();
    mlirDiag.severity = getSeverity(slangDiag.severity);
    mlirDiag.range = context.getLspLocation(slangDiag.location).range;
    mlirDiag.source = "slang";
    mlirDiag.message = slangDiag.formattedMessage;

    // // Append the name of the option that can be used to control this
    // // diagnostic.
    // auto optionName = engine->getOptionName(diag.originalDiagnostic.code);
    // if (!optionName.empty())
    //   mlirDiag << " [-W" << optionName << "]";

    // Write out macro expansions, if we have any, in reverse order.
    // for (auto it = diag.expansionLocs.rbegin(); it !=
    // diag.expansionLocs.rend();
    //      it++) {
    //   auto &note = mlirDiag.attachNote(
    //       convertLocation(sourceManager->getFullyOriginalLoc(*it)));
    //   auto macroName = sourceManager->getMacroName(*it);
    //   if (macroName.empty())
    //     note << "expanded from here";
    //   else
    //     note << "expanded from macro '" << macroName << "'";
    // }
  }
};

class VerilogIndex {
public:
  VerilogIndex(VerilogServerContext &context)
      : context(context), intervalMap(allocator), intervalMapLoc(allocator) {}

  /// Initialize the index with the given ast::Module.
  void initialize(slang::ast::Compilation *compilation);

  /// Lookup a symbol for the given location. Returns nullptr if no symbol could
  /// be found. If provided, `overlappedRange` is set to the range that the
  /// provided `loc` overlapped with.
  const VerilogIndexSymbol *lookup(SMLoc loc,
                                   SMRange *overlappedRange = nullptr) const;
  struct EmittedLoc {
    StringRef filePath;
    uint32_t line;
    uint32_t column;
    SMRange range;
  };

  const EmittedLoc *lookupLoc(SMLoc loc) const;

  size_t size() const {
    return std::distance(intervalMap.begin(), intervalMap.end());
  }

  VerilogServerContext &getContext() { return context; }
  auto &getIntervalMap() { return intervalMap; }
  auto &getReferences() { return references; }

  void parseEmittedLoc();

private:
  /// The type of interval map used to store source references. SMRange is
  /// half-open, so we also need to use a half-open interval map.
  using MapT =
      llvm::IntervalMap<const char *, const VerilogIndexSymbol *,
                        llvm::IntervalMapImpl::NodeSizer<
                            const char *, const VerilogIndexSymbol *>::LeafSize,
                        llvm::IntervalMapHalfOpenInfo<const char *>>;

  /// An allocator for the interval map.
  MapT::Allocator allocator;

  VerilogServerContext &context;

  /// An interval map containing a corresponding definition mapped to a source
  /// interval.
  MapT intervalMap;

  using MapTLoc =
      llvm::IntervalMap<const char *, const EmittedLoc *,
                        llvm::IntervalMapImpl::NodeSizer<
                            const char *, const EmittedLoc *>::LeafSize,
                        llvm::IntervalMapHalfOpenInfo<const char *>>;

  // TODO: Merge them.
  MapTLoc intervalMapLoc;

  llvm::SmallDenseMap<const VerilogIndexSymbol *,
                      SmallVector<slang::SourceLocation>>
      references;

  // TODO: Use allocator.
  SmallVector<std::unique_ptr<EmittedLoc>> emittedLocs;

  /// A mapping between definitions and their corresponding symbol.
  DenseMap<const void *, std::unique_ptr<VerilogIndexSymbol>> defToSymbol;
};
} // namespace

void VerilogIndex::parseEmittedLoc() {
  auto &sourceMgr = getContext().getSourceMgr();
  auto *getMainBuffer = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  StringRef text(getMainBuffer->getBufferStart(),
                 getMainBuffer->getBufferSize());

  // SmallVector<EmittedLoc> emittedLocs;
  // Find all the locations that are emitted by ExportVerilog.

  while (true) {
    // HACK: This is bad.
    StringRef start = "// @[";
    auto loc = text.find(start);
    if (loc == StringRef::npos)
      break;

    text = text.drop_front(loc + start.size());
    auto end = text.find_first_of(']');
    if (end == StringRef::npos)
      break;
    auto toParse = text.take_front(end);
    mlir::lsp::Logger::info("toParse: {}", toParse);
    StringRef filePath;
    StringRef line;
    StringRef column;
    bool first = true;
    for (auto cur : llvm::split(toParse, ", ")) {
      auto sep = llvm::split(cur, ":");
      if (std::distance(sep.begin(), sep.end()) != 3)
        continue;
      bool addFile = first;
      first = false;

      auto it = sep.begin();
      filePath = (*it).empty() ? filePath : *it;
      line = *(++it);
      column = *(++it);
      mlir::lsp::Logger::info("filePath: {}", filePath);
      mlir::lsp::Logger::info("line: {}", line);
      mlir::lsp::Logger::info("column: {}", column);
      uint32_t lineInt;
      if (line.getAsInteger(10, lineInt))
        continue;
      uint32_t columnInt;
      if (column.getAsInteger(10, columnInt))
        continue;
      auto *start = addFile ? filePath.data() : line.data();
      auto *end = column.end();
      EmittedLoc loc = {filePath, lineInt - 1, columnInt - 1,
                        llvm::SMRange(SMLoc::getFromPointer(start),
                                      SMLoc::getFromPointer(end + 1))};
      emittedLocs.push_back(std::make_unique<EmittedLoc>(loc));
      intervalMapLoc.insert(start, end, emittedLocs.back().get());
    }
  }
}

struct IndexVisitor : slang::ast::ASTVisitor<IndexVisitor, true, true> {
  IndexVisitor(VerilogIndex &index) : index(index) {}
  VerilogIndex &index;

  void handleSymbol(const slang::ast::Symbol *symbol, slang::SourceRange range,
                    bool isDef = false) {
    if (symbol->name.empty())
      return;
    mlir::lsp::Logger::debug("index handleSymbol: {}", symbol->name);
    assert(range.start().valid() && range.end().valid());
    mlir::lsp::Logger::debug("1");

    // auto start = getContext().getSMLoc(range.start());
    mlir::lsp::Logger::debug("2");

    // auto end = getContext().getSMLoc(range.end());
    mlir::lsp::Logger::debug("3");

    // mlir::lsp::Logger::debug("handleSymbol nam: {}",
    //                         symbol->getParentScope()
    //                             ->getContainingInstance()
    //                             ->getDefinition()
    //                             .name);

    insertDeclRef(symbol, range, true);
  }

  void visit(const slang::ast::InstanceSymbol &symbol) {
    mlir::lsp::Logger::debug("visit: {}", slang::ast::toString(symbol.kind));
    auto firstLoc = symbol.getSyntax()->getFirstToken().location();
    auto text =
        getContext().getSlangSourceManager().getSourceText(firstLoc.buffer());
    {
      auto nonWhitespaceLoc = firstLoc;
      while (firstLoc.offset() > 0 && text[firstLoc.offset() - 1] != '\n') {
        if (text[firstLoc.offset() - 1] != ' ' &&
            text[firstLoc.offset() - 1] != '\t')
          nonWhitespaceLoc = firstLoc - 1;
        firstLoc -= 1;
      }

      handleSymbol(&symbol.body.asSymbol(),
                   slang::SourceRange(nonWhitespaceLoc,
                                      symbol.location + symbol.name.size()),
                   false);
    }

    for (auto *con : symbol.getPortConnections()) {
      // mlir::lsp::Logger::info("visit: {}",
      //                         con->getExpression()->syntax->toString());

      // if (auto *port = con->port.as_if<slang::ast::PortSymbol>()) {
      //   if (auto *connect = con->getExpression()
      //                           ->as_if<slang::ast::AssignmentExpression>())
      //                           {
      //     mlir::lsp::Logger::info("connect: {}",
      //     connect->syntax->toString());

      //     mlir::lsp::Logger::info(
      //         "port: {}", con->port.as_if<slang::ast::PortSymbol>()->name);
      //   }
      // }

      // Input only
      if (auto *port = con->port.as_if<slang::ast::PortSymbol>();
          port && con->getExpression()) {
        slang::SourceLocation firstLoc;
        if (con->getExpression()->syntax) {
          firstLoc = con->getExpression()->syntax->getFirstToken().location();
        } else if (auto *connect =
                       con->getExpression()
                           ->as_if<slang::ast::AssignmentExpression>()) {
          firstLoc = connect->sourceRange.start();
        } else {
          mlir::lsp::Logger::info("no expression");
        }

        if (!firstLoc.valid())
          continue;
        auto loc = firstLoc;
        // mlir::lsp::Logger::info("firstLOc: {}",
        // text.substr(firstLoc.offset(), 100)); Skip to the first start of
        // line.
        while (loc.offset() > 0 && text[loc.offset() - 1] != '\n') {
          loc -= 1;
        }
        if (loc.offset() == 0)
          continue;

        mlir::lsp::Logger::info("RUnning: {} {}", loc.offset(),
                                firstLoc.offset());
        // find the name of the port.
        auto it = text.substr(loc.offset(), firstLoc.offset() - loc.offset())
                      .find(port->name);
        if (it == std::string::npos) {
          continue;
        }
        mlir::lsp::Logger::info("Found: {}", it);
        // mlir::lsp::Logger::info("visit: {}",
        //                          port->internalSymbol->getSyntax()->toString());

        // mlir::lsp::Logger::info("visit: {}",
        //                          port->getSyntax()->toString());
        handleSymbol(port,
                     slang::SourceRange(loc + it, loc + it + port->name.size()),
                     false);
      }
    }
  }

  // Handle references to the left-hand side of a parent assignment.
  // Handle references to the left-hand side of a parent assignment.
  void visit(const slang::ast::LValueReferenceExpression &expr) {
    mlir::lsp::Logger::debug("visit: {}", slang::ast::toString(expr.kind));
    auto *symbol = expr.getSymbolReference(true);
    if (!symbol)
      return;
    handleSymbol(symbol, expr.sourceRange);
    visitDefault(expr);
  }
  void visit(const slang::ast::NetSymbol &expr) {
    mlir::lsp::Logger::debug("visit: {}", slang::ast::toString(expr.kind));
    handleSymbol(
        &expr,
        slang::SourceRange(expr.location, expr.location + expr.name.length()),
        true);
    visitDefault(expr);
  }
  void visit(const slang::ast::VariableSymbol &expr) {
    mlir::lsp::Logger::debug("visit: {}", slang::ast::toString(expr.kind));
    handleSymbol(
        &expr,
        slang::SourceRange(expr.location, expr.location + expr.name.length()),
        true);
    visitDefault(expr);
  }

  void visit(const slang::ast::VariableDeclStatement &expr) {
    mlir::lsp::Logger::debug("visit: {}", slang::ast::toString(expr.kind));
    handleSymbol(&expr.symbol, expr.sourceRange, true);
    visitDefault(expr);
  }

  // void visit(const slang::ast::AssignmentExpression &expr) {
  //   mlir::lsp::Logger::debug("visit: {}", slang::ast::toString(expr.kind));
  //   if (auto *symbol = expr.left().getSymbolReference(true))
  //     handleSymbol(symbol, expr.sourceRange);
  // }

  // Handle named values, such as references to declared variables.
  // Handle named values, such as references to declared variables.
  void visit(const slang::ast::NamedValueExpression &expr) {
    mlir::lsp::Logger::debug("visit: {}", slang::ast::toString(expr.kind));
    auto *symbol = expr.getSymbolReference(true);
    if (!symbol)
      return;
    handleSymbol(symbol, expr.sourceRange);
    visitDefault(expr);
    // if (auto value = context.valueSymbols.lookup(&expr.symbol)) {

    //   return value;
    // }

    // // Try to materialize constant values directly.
    // auto constant = context.evaluateConstant(expr);
    // if (auto value = context.materializeConstant(constant, *expr.type, loc))
    //   return value;

    // // Otherwise some other part of ImportVerilog should have added an MLIR
    // // value for this expression's symbol to the `context.valueSymbols`
    // table. auto d = mlir::emitError(loc, "unknown name `") <<
    // expr.symbol.name << "`";
    // d.attachNote(context.convertLocation(expr.symbol.location))
    //     << "no rvalue generated for " <<
    //     slang::ast::toString(expr.symbol.kind);
    // return {};
  }

  template <typename T>
  void visit(const T &t) {
    mlir::lsp::Logger::debug("visit: {}", slang::ast::toString(t.kind));

    visitDefault(t);
  }
  // Handle hierarchical values, such as `x = Top.sub.var`.
  // Value visit(const slang::ast::HierarchicalValueExpression &expr) {
  //   auto hierLoc = context.convertLocation(expr.symbol.location);
  //   if (auto value = context.valueSymbols.lookup(&expr.symbol)) {
  //     if (isa<moore::RefType>(value.getType())) {
  //       auto readOp = builder.create<moore::ReadOp>(hierLoc, value);
  //       if (context.rvalueReadCallback)
  //         context.rvalueReadCallback(readOp);
  //       value = readOp.getResult();
  //     }
  //     return value;
  //   }

  //   // Emit an error for those hierarchical values not recorded in the
  //   // `valueSymbols`.
  //   auto d = mlir::emitError(loc, "unknown hierarchical name `")
  //            << expr.symbol.name << "`";
  //   d.attachNote(hierLoc) << "no rvalue generated for "
  //                         << slang::ast::toString(expr.symbol.kind);
  //   return {};
  // }

  // Helper function to convert an argument to a simple bit vector type, pass it
  // to a reduction op, and optionally invert the result.

  /// Handle assignment patterns.
  void visitInvalid(const slang::ast::Expression &expr) {
    mlir::lsp::Logger::debug("visitInvalid: {}",
                             slang::ast::toString(expr.kind));
  }

  void visitInvalid(const slang::ast::Statement &) {}
  void visitInvalid(const slang::ast::TimingControl &) {}
  void visitInvalid(const slang::ast::Constraint &) {}
  void visitInvalid(const slang::ast::AssertionExpr &) {}
  void visitInvalid(const slang::ast::BinsSelectExpr &) {}
  void visitInvalid(const slang::ast::Pattern &) {}

  VerilogServerContext &getContext() { return index.getContext(); }

  void insertDeclRef(const VerilogIndexSymbol *sym, slang::SourceRange refLoc,
                     bool isDef = false) {
    const char *startLoc = getContext().getSMLoc(refLoc.start()).getPointer();
    const char *endLoc = getContext().getSMLoc(refLoc.end()).getPointer();
    if (!startLoc || !endLoc)
      return;
    assert(startLoc && endLoc);
    if (startLoc != endLoc &&
        !index.getIntervalMap().overlaps(startLoc, endLoc)) {
      index.getIntervalMap().insert(startLoc, endLoc, sym);
      if (!isDef) {
        index.getReferences()[sym].push_back(refLoc.start());
      }
    }
  };
};

void VerilogIndex::initialize(slang::ast::Compilation *compilation) {
  // auto &c = driver.compilation->getRoot();
  // for (auto *unit : driver.compilation->getCompilationUnits()) {
  //   mlir::lsp::Logger::debug("VerilogIndex::initialize {}", unit->name);
  // }
  const auto &root = compilation->getRoot();
  // for (auto *unit : root.compilationUnits) {
  //   for (const auto &member : unit->members()) {
  //     auto loc = convertLocation(member.location);
  //     if (failed(member.visit(RootVisitor(*this, loc))))
  //       return failure();
  //   }
  // }
  auto insertDeclRef = [&](const VerilogIndexSymbol *sym, SMRange refLoc,
                           bool isDef = false) {
    const char *startLoc = refLoc.Start.getPointer();
    const char *endLoc = refLoc.End.getPointer();
    if (!intervalMap.overlaps(startLoc, endLoc)) {
      intervalMap.insert(startLoc, endLoc, sym);
    }
  };
  const auto *slangSourceMgr = compilation->getSourceManager();

  IndexVisitor visitor(*this);
  for (auto *inst : root.topInstances) {
    inst->body.visit(visitor);
    mlir::lsp::Logger::debug("VerilogIndex::initialize {}", inst->name);
    auto &body = inst->body;

    for (auto *symbol : body.getPortList()) {
      mlir::lsp::Logger::debug("VerilogIndex::initialize port {}",
                               symbol->name);
      auto startLoc = context.getSMLoc(symbol->location);
      auto endLoc = SMLoc::getFromPointer(
          context.getSMLoc(symbol->location).getPointer() +
          symbol->name.size());
      auto range = SMRange(startLoc, endLoc);
      const char *startLoc2 = range.Start.getPointer();
      const char *endLoc2 = range.End.getPointer();
      if (!intervalMap.overlaps(startLoc2, endLoc2)) {
        intervalMap.insert(startLoc2, endLoc2, symbol);
      }
    }
  }

  SmallString<128> path;
  path += "/home/uenoku/dev/circt-dev";
  getContext().originalFileRoots.push_back(path);

  parseEmittedLoc();

  // mlir::lsp::Logger::debug("VerilogIndex::initialize {}", c.name);
  // for (auto &member : c.members()) {

  //   mlir::lsp::Logger::debug("VerilogIndex::initialize {}",
  //                           toString(member.kind));
  //   mlir::lsp::Logger::debug("VerilogIndex::initialize::syntax {}",
  //                           member.getSyntax()->toString());
  //   if (auto *module = member.as_if<slang::ast::CompilationUnitSymbol>()) {
  //     mlir::lsp::Logger::debug("VerilogIndex::initialize compilation unit
  //     {}",
  //                             module->name);
  //   }
  // }
  // auto getOrInsertDef = [&](const auto *def) -> VerilogIndexSymbol * {
  //   auto it = defToSymbol.try_emplace(def, nullptr);
  //   if (it.second)
  //     it.first->second = std::make_unique<VerilogIndexSymbol>(def);
  //   return &*it.first->second;
  // };

  // auto insertODSOpRef = [&](StringRef opName, SMRange refLoc) {
  //   const ods::Operation *odsOp = odsContext.lookupOperation(opName);
  //   if (!odsOp)
  //     return;

  //   VerilogIndexSymbol *symbol = getOrInsertDef(odsOp);
  //   insertDeclRef(symbol, odsOp->getLoc(), /*isDef=*/true);
  //   insertDeclRef(symbol, refLoc);
  // };

  // module.walk([&](const ast::Node *node) {
  //   // Handle references to PDL decls.
  //   if (const auto *decl = dyn_cast<ast::OpNameDecl>(node)) {
  //     if (std::optional<StringRef> name = decl->getName())
  //       insertODSOpRef(*name, decl->getLoc());
  //   } else if (const ast::Decl *decl = dyn_cast<ast::Decl>(node)) {
  //     const ast::Name *name = decl->getName();
  //     if (!name)
  //       return;
  //     VerilogIndexSymbol *declSym = getOrInsertDef(decl);
  //     insertDeclRef(declSym, name->getLoc(), /*isDef=*/true);

  //     if (const auto *varDecl = dyn_cast<ast::VariableDecl>(decl)) {
  //       // Record references to any constraints.
  //       for (const auto &it : varDecl->getConstraints())
  //         insertDeclRef(getOrInsertDef(it.constraint), it.referenceLoc);
  //     }
  //   } else if (const auto *expr = dyn_cast<ast::DeclRefExpr>(node)) {
  //     insertDeclRef(getOrInsertDef(expr->getDecl()), expr->getLoc());
  //   }
  // });
}

const VerilogIndexSymbol *VerilogIndex::lookup(SMLoc loc,
                                               SMRange *overlappedRange) const {
  auto it = intervalMap.find(loc.getPointer());
  if (!it.valid() || loc.getPointer() < it.start())
    return nullptr;

  if (overlappedRange) {
    *overlappedRange = SMRange(SMLoc::getFromPointer(it.start()),
                               SMLoc::getFromPointer(it.stop()));
  }
  return it.value();
}

const VerilogIndex::EmittedLoc *VerilogIndex::lookupLoc(SMLoc loc) const {
  auto it = intervalMapLoc.find(loc.getPointer());
  if (!it.valid() || loc.getPointer() < it.start())
    return nullptr;

  return it.value();
}

//===----------------------------------------------------------------------===//
// VerilogDocument
//===----------------------------------------------------------------------===//

namespace {
/// This class represents all of the information pertaining to a specific PDL
/// document.
struct VerilogDocument {
  VerilogDocument(const mlir::lsp::URIForFile &uri, StringRef contents,
                  const std::vector<std::string> &extraDirs,
                  std::vector<mlir::lsp::Diagnostic> &diagnostics);
  VerilogDocument(const VerilogDocument &) = delete;
  VerilogDocument &operator=(const VerilogDocument &) = delete;

  //===--------------------------------------------------------------------===//
  // Definitions and References
  //===--------------------------------------------------------------------===//

  void getLocationsOf(const mlir::lsp::URIForFile &uri,
                      const mlir::lsp::Position &defPos,
                      std::vector<mlir::lsp::Location> &locations);
  void findReferencesOf(const mlir::lsp::URIForFile &uri,
                        const mlir::lsp::Position &pos,
                        std::vector<mlir::lsp::Location> &references);

  //===--------------------------------------------------------------------===//
  // Document Links
  //===--------------------------------------------------------------------===//

  void getDocumentLinks(const mlir::lsp::URIForFile &uri,
                        std::vector<mlir::lsp::DocumentLink> &links);

  //===--------------------------------------------------------------------===//
  // Hover
  //===--------------------------------------------------------------------===//

  std::optional<mlir::lsp::Hover>
  findHover(const mlir::lsp::URIForFile &uri,
            const mlir::lsp::Position &hoverPos);
  std::optional<mlir::lsp::Hover> findHover(const slang::ast::Statement *stmt,
                                            const SMRange &hoverRange);
  std::optional<mlir::lsp::Hover>
  findHover(const slang::ast::Definition *instance, const SMRange &hoverRange) {
  }
  mlir::lsp::Hover buildHoverForStatement(const slang::ast::Statement *stmt,
                                          const SMRange &hoverRange);
  mlir::lsp::Hover
  buildHoverForVariable(const slang::ast::VariableDeclStatement *varDecl,
                        const SMRange &hoverRange);
  mlir::lsp::Hover
  buildHoverForInstance(const slang::ast::InstanceSymbol *instance,
                        const SMRange &hoverRange);

  //===--------------------------------------------------------------------===//
  // Document Symbols
  //===--------------------------------------------------------------------===//

  void findDocumentSymbols(std::vector<mlir::lsp::DocumentSymbol> &symbols);

  //===--------------------------------------------------------------------===//
  // Signature Help
  //===--------------------------------------------------------------------===//

  mlir::lsp::SignatureHelp getSignatureHelp(const mlir::lsp::URIForFile &uri,
                                            const mlir::lsp::Position &helpPos);

  //===--------------------------------------------------------------------===//
  // Inlay Hints
  //===--------------------------------------------------------------------===//

  void getInlayHints(const mlir::lsp::URIForFile &uri,
                     const mlir::lsp::Range &range,
                     std::vector<mlir::lsp::InlayHint> &inlayHints);
  void getInlayHintsFor(const slang::ast::VariableDeclStatement *decl,
                        const mlir::lsp::URIForFile &uri,
                        std::vector<mlir::lsp::InlayHint> &inlayHints);
  void getInlayHintsFor(const slang::ast::InstanceSymbol *instance,
                        const mlir::lsp::URIForFile &uri,
                        std::vector<mlir::lsp::InlayHint> &inlayHints);

  //  /// Add a parameter hint for the given expression using `label`.
  //  void addParameterHintFor(std::vector<mlir::lsp::InlayHint> &inlayHints,
  //                           const ast::Expr *expr, StringRef label);

  //===--------------------------------------------------------------------===//
  // VerilogL ViewOutput
  //===--------------------------------------------------------------------===//

  void getVerilogViewOutput(raw_ostream &os,
                            circt::lsp::VerilogViewOutputKind kind);

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  /// The include directories for this file.
  std::vector<std::string> includeDirs;

  /// The source manager containing the contents of the input file.
  // llvm::SourceMgr sourceMgr;

  /// The parsed AST module, or failure if the file wasn't valid.
  FailureOr<std::unique_ptr<slang::ast::Compilation>> compilation;
  VerilogServerContext verilogContext;
  VerilogServerContext &getContext() { return verilogContext; }

  /// The index of the parsed module.
  VerilogIndex index;

  /// The set of includes of the parsed module.
  SmallVector<mlir::lsp::SourceMgrInclude> parsedIncludes;
  mlir::lsp::Location getLocationFromLoc(llvm::SourceMgr &mgr,
                                         slang::SourceLocation loc,
                                         const mlir::lsp::URIForFile &uri) {
    /// Incorrect!
    return convertLocation(verilogContext.driver.sourceManager, loc);
  }
};
} // namespace

VerilogDocument::VerilogDocument(
    const mlir::lsp::URIForFile &uri, StringRef contents,
    const std::vector<std::string> &extraDirs,
    std::vector<mlir::lsp::Diagnostic> &diagnostics)
    : index(getContext()) {
  auto memBuffer = llvm::MemoryBuffer::getMemBufferCopy(contents, uri.file());
  Logger::debug("VerilogDocument::VerilogDocument");

  if (!memBuffer) {
    mlir::lsp::Logger::error("Failed to create memory buffer for file",
                             uri.file());
    return;
  }

  Logger::debug("VerilogDocument::VerilogDocument set include dirs");
  // Build the set of include directories for this file.
  llvm::SmallString<32> uriDirectory(uri.file());
  llvm::sys::path::remove_filename(uriDirectory);
  includeDirs.push_back(uriDirectory.str().str());
  includeDirs.insert(includeDirs.end(), extraDirs.begin(), extraDirs.end());

  Logger::debug("VerilogDocument::VerilogDocument set include dirs");
  auto &sourceMgr = verilogContext.sourceMgr;
  sourceMgr.setIncludeDirs(includeDirs);
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), SMLoc());

  const llvm::MemoryBuffer *mlirBuffer = sourceMgr.getMemoryBuffer(1);
  mlir::lsp::Logger::info(
      "VerilogDocument::VerilogDocument set include dirs {}", uriDirectory);
  verilogContext.driver.options.libDirs = includeDirs;
  auto slangBuffer = verilogContext.driver.sourceManager.assignText(
      uri.file(), mlirBuffer->getBuffer());
  verilogContext.driver.buffers.push_back(slangBuffer);
  // sourceMgr.getMainFileID();
  verilogContext.bufferIDMap[slangBuffer.id.getId()] =
      sourceMgr.getMainFileID();

  // astContext.getDiagEngine().setHandlerFn([&](const slang::ast::Diagnostic
  // &diag) {
  //   // if (auto lspDiag = getLspDiagnoticFromDiag(sourceMgr, diag, uri))
  //   //   diagnostics.push_back(std::move(*lspDiag));
  // });
  // astModule =
  //     parseVerilogAST(astContext, sourceMgr, /*enableDocumentation=*/true);

  Logger::debug("VerilogDocument::VerilogDocument try parse");
  // auto parsed =
  //     slang::syntax::SyntaxTree::fromBuffer(slangBuffer, slangSourceMgr, {});

  // ======
  // Read a mapping file
  StringRef path = "/scratch/hidetou/mapping.json";
  auto open = mlir::openInputFile(path);
  if (!open) {
    mlir::lsp::Logger::error("Failed to open mapping file {}", path);
  } else {
    mlir::lsp::Logger::error("JSON {}", open->getBuffer());
    auto json = llvm::json::parse(open->getBuffer());

    if (auto err = json.takeError()) {
      mlir::lsp::Logger::error("Failed to parse mapping file {}", err);
      /*
      mapping file is like this:
      {
        "module_Foo": {
          "verilogName": "hintName",
          "_GEN_123": "chisel_var"
        }
      }
      */
    } else {
      verilogContext.userHint.json =
          std::make_unique<llvm::json::Value>(std::move(json.get()));
    }
  }

  // ======

  auto diagClient =
      std::make_shared<MlirDiagnosticClient>(getContext(), diagnostics);
  verilogContext.driver.diagEngine.addClient(diagClient);
  verilogContext.driver.processOptions();

  bool success = verilogContext.driver.parseAllSources();
  auto &driver = verilogContext.driver;

  Logger::debug("VerilogDocument::VerilogDocument try parse done");
  if (!success) {
    mlir::lsp::Logger::error("Failed to parse Verilog file", uri.file());
    return;
  }

  mlir::lsp::Logger::debug("parsed: {}",
                           driver.syntaxTrees[0]->root().toString());
  compilation = driver.createCompilation();
  for (auto &diag : (*compilation)->getAllDiagnostics())
    driver.diagEngine.issue(diag);

  mlir::lsp::Logger::debug("elaborated");

  // From buffers.
  // From buffers.
  // const SourceBuffer& buffer,
  //                                                    SourceManager&
  //                                                    sourceManager, const
  //                                                    Bag& options,
  //                                                    MacroList
  //                                                    inheritedMacros

  // Initialize the set of parsed includes.
  mlir::lsp::gatherIncludeFiles(sourceMgr, parsedIncludes);

  // If we failed to parse the module, there is nothing left to initialize.
  if (failed(compilation))
    return;

  // Prepare the AST index with the parsed module.
  index.initialize(compilation.value().get());
}

//===----------------------------------------------------------------------===//
// VerilogDocument: Definitions and References
//===----------------------------------------------------------------------===//

void VerilogDocument::getLocationsOf(
    const mlir::lsp::URIForFile &uri, const mlir::lsp::Position &defPos,
    std::vector<mlir::lsp::Location> &locations) {
  SMLoc posLoc = defPos.getAsSMLoc(verilogContext.sourceMgr);

  // CHECK INTERVAL MAP LOC
  {
    const auto *it = index.lookupLoc(posLoc);
    if (it) {
      mlir::lsp::Logger::debug(
          "VerilogDocument::getLocationsOf interval map loc");
      auto fileInfo = verilogContext.filePathMap.find(it->filePath);
      if (fileInfo == verilogContext.filePathMap.end()) {
        for (auto &lib : verilogContext.originalFileRoots) {
          auto size = lib.size();
          auto fixupSize = llvm::make_scope_exit([&]() { lib.resize(size); });
          llvm::sys::path::append(lib, it->filePath);
          if (llvm::sys::fs::exists(lib)) {
            auto memoryBuffer = llvm::MemoryBuffer::getFile(lib);
            if (!memoryBuffer) {
              continue;
            }
            auto id = verilogContext.sourceMgr.AddNewSourceBuffer(
                std::move(*memoryBuffer), SMLoc());

            fileInfo = verilogContext.filePathMap
                           .insert(std::make_pair(it->filePath,
                                                  std::make_pair(id, lib)))
                           .first;
          }
        }
      }

      const auto &[bufferId, filePath] = fileInfo->second;
      mlir::lsp::Logger::debug(
          "VerilogDocument::getLocationsOf interval map loc not found");

      verilogContext.sourceMgr.getBufferInfo(bufferId);
      auto uri = mlir::lsp::URIForFile::fromFile(filePath);
      if (auto e = uri.takeError()) {
        mlir::lsp::Logger::error(
            "VerilogDocument::getLocationsOf interval map loc error");
        return;
      }

      mlir::lsp::Location loc(uri.get(),
                              mlir::lsp::Range(Position(it->line, it->column)));
      locations.push_back(loc);
      return;

      // auto loc =
      //     getLocationFromLoc(verilogContext.sourceMgr, it-range, uri);
      // auto bufferId =
      //     getContext().getBufferIDMap().at(symbol->location.buffer().getId());
      // auto buffer = sourceMgr.pat;
      // if (loc.uri.file().empty())
      // return mlir::lsp::Logger::debug(
      //     "VerilogDocument::getLocationsOf loc is empty");
      return;
    }
  }

  mlir::lsp::Logger::debug("VerilogDocument::getLocationsOf {} {}",
                           reinterpret_cast<int64_t>(posLoc.getPointer()),
                           posLoc.getPointer());
  const VerilogIndexSymbol *symbol = index.lookup(posLoc);
  mlir::lsp::Logger::debug("VerilogDocument::getLocationsOf {}", index.size());
  if (!symbol) {
    mlir::lsp::Logger::debug("VerilogDocument::getLocationsOf not found");
    return;
  }
  mlir::lsp::Logger::debug("VerilogDocument::getLocationsOf symbol {}",
                           symbol->name);

  auto loc =
      getLocationFromLoc(verilogContext.sourceMgr, symbol->location, uri);
  // auto bufferId =
  //     getContext().getBufferIDMap().at(symbol->location.buffer().getId());
  // auto buffer = sourceMgr.pat;
  // if (loc.uri.file().empty())
  // return mlir::lsp::Logger::debug(
  //     "VerilogDocument::getLocationsOf loc is empty");
  mlir::lsp::Logger::debug("VerilogDocument::getLocationsOf loc {}",
                           loc.range.start.line);
  locations.push_back(loc);
}

void VerilogDocument::findReferencesOf(
    const mlir::lsp::URIForFile &uri, const mlir::lsp::Position &pos,
    std::vector<mlir::lsp::Location> &references) {
  mlir::lsp::Logger::info("VerilogDocument::findReferencesOf loc");
  SMLoc posLoc = pos.getAsSMLoc(verilogContext.sourceMgr);
  const VerilogIndexSymbol *symbol = index.lookup(posLoc);
  if (!symbol) {
    mlir::lsp::Logger::info(
        "VerilogDocument::findReferencesOf symbol not found");
    return;
  }

  references.push_back(
      getLocationFromLoc(verilogContext.sourceMgr, symbol->location, uri));
  for (auto refLoc : index.getReferences().lookup(symbol))
    references.push_back(getContext().getLspLocation(refLoc));
}

//===--------------------------------------------------------------------===//
// VerilogDocument: Document Links
//===--------------------------------------------------------------------===//

void VerilogDocument::getDocumentLinks(
    const mlir::lsp::URIForFile &uri,
    std::vector<mlir::lsp::DocumentLink> &links) {
  for (const mlir::lsp::SourceMgrInclude &include : parsedIncludes)
    links.emplace_back(include.range, include.uri);
}

//===----------------------------------------------------------------------===//
// VerilogDocument: Hover
//===----------------------------------------------------------------------===//

std::optional<mlir::lsp::Hover>
VerilogDocument::findHover(const mlir::lsp::URIForFile &uri,
                           const mlir::lsp::Position &hoverPos) {
  SMLoc posLoc = hoverPos.getAsSMLoc(verilogContext.sourceMgr);

  // Check for a reference to an include.
  for (const mlir::lsp::SourceMgrInclude &include : parsedIncludes)
    if (include.range.contains(hoverPos))
      return include.buildHover();

  // Find the symbol at the given location.
  SMRange hoverRange;
  const VerilogIndexSymbol *symbol = index.lookup(posLoc, &hoverRange);
  if (!symbol)
    return std::nullopt;

  // Add hover for operation names.
  // if (const auto *op =
  //        llvm::dyn_cast_if_present<const ods::Operation
  //        *>(symbol->definition))
  //  return buildHoverForOpName(op, hoverRange);
  return findHover(symbol->getDeclaringDefinition(), hoverRange);
}

std::optional<mlir::lsp::Hover>
VerilogDocument::findHover(const slang::ast::Statement *stmt,
                           const SMRange &hoverRange) {
  // Add hover for variables.
  //  if (const auto *varDecl = dyn_cast<ast::VariableDecl>(decl))
  //    return buildHoverForVariable(varDecl, hoverRange);
  //
  //  // Add hover for patterns.
  //  if (const auto *patternDecl = dyn_cast<ast::PatternDecl>(decl))
  //    return buildHoverForPattern(patternDecl, hoverRange);
  //
  //  // Add hover for core constraints.
  //  if (const auto *cst = dyn_cast<ast::CoreConstraintDecl>(decl))
  //    return buildHoverForCoreConstraint(cst, hoverRange);
  //
  //  // Add hover for user constraints.
  //  if (const auto *cst = dyn_cast<ast::UserConstraintDecl>(decl))
  //    return buildHoverForUserConstraintOrRewrite("Constraint", cst,
  //    hoverRange);
  //
  //  // Add hover for user rewrites.
  //  if (const auto *rewrite = dyn_cast<ast::UserRewriteDecl>(decl))
  //    return buildHoverForUserConstraintOrRewrite("Rewrite", rewrite,
  //    hoverRange);
  //
  return std::nullopt;
}

mlir::lsp::Hover VerilogDocument::buildHoverForVariable(
    const slang::ast::VariableDeclStatement *varDecl,
    const SMRange &hoverRange) {
  mlir::lsp::Hover hover(
      mlir::lsp::Range(verilogContext.sourceMgr, hoverRange));
  {
    llvm::raw_string_ostream hoverOS(hover.contents.value);
    hoverOS << "**Variable**: `" << varDecl->symbol.name << "`\n***\n"
            << "Type: `" << varDecl->symbol.getDeclaredType() << "`\n";
  }
  return hover;
}

//===----------------------------------------------------------------------===//
// VerilogDocument: Document Symbols
//===----------------------------------------------------------------------===//

void VerilogDocument::findDocumentSymbols(
    std::vector<mlir::lsp::DocumentSymbol> &symbols) {
  if (failed(compilation))
    return;

  // for (const slang::ast::Statement *stmt : (*astModule)->getChildren()) {
  //   if (!isMainFileLoc(sourceMgr, stmt->getLoc()))
  //     continue;

  //   if (const auto *patternDecl = dyn_cast<ast::PatternDecl>(decl)) {
  //     const ast::Name *name = patternDecl->getName();

  //     SMRange nameLoc = name ? name->getLoc() : patternDecl->getLoc();
  //     SMRange bodyLoc(nameLoc.Start, patternDecl->getBody()->getLoc().End);

  //     symbols.emplace_back(name ? name->getName() : "<pattern>",
  //                          mlir::lspSymbolKind::Class,
  //                          mlir::lspRange(sourceMgr, bodyLoc),
  //                          mlir::lspRange(sourceMgr, nameLoc));
  //   } else if (const auto *cDecl = dyn_cast<ast::UserConstraintDecl>(decl))
  //   {
  //     // TODO: Add source information for the code block body.
  //     SMRange nameLoc = cDecl->getName().getLoc();
  //     SMRange bodyLoc = nameLoc;

  //     symbols.emplace_back(cDecl->getName().getName(),
  //                          mlir::lspSymbolKind::Function,
  //                          mlir::lspRange(sourceMgr, bodyLoc),
  //                          mlir::lspRange(sourceMgr, nameLoc));
  //   } else if (const auto *cDecl = dyn_cast<ast::UserRewriteDecl>(decl)) {
  //     // TODO: Add source information for the code block body.
  //     SMRange nameLoc = cDecl->getName().getLoc();
  //     SMRange bodyLoc = nameLoc;

  //     symbols.emplace_back(cDecl->getName().getName(),
  //                          mlir::lspSymbolKind::Function,
  //                          mlir::lspRange(sourceMgr, bodyLoc),
  //                          mlir::lspRange(sourceMgr, nameLoc));
  //   }
  // }
}

//===----------------------------------------------------------------------===//
// VerilogDocument: Inlay Hints
//===----------------------------------------------------------------------===//

/// Returns true if the given name should be added as a hint for `expr`.
static bool shouldAddHintFor(const slang::ast::Expression *expr,
                             StringRef name) {
  if (name.empty())
    return false;

  // If the argument is a reference of the same name, don't add it as a hint.
  // if (auto *ref = dyn_cast<ast::DeclRefExpr>(expr)) {
  //   const ast::Name *declName = ref->getDecl()->getName();
  //   if (declName && declName->getName() == name)
  //     return false;
  // }

  return true;
}

struct RvalueExprVisitor
    : slang::ast::ASTVisitor<RvalueExprVisitor, true, true> {
  RvalueExprVisitor(VerilogServerContext &context, SMRange range,
                    std::vector<mlir::lsp::InlayHint> &inlayHints)
      : context(context), range(range), inlayHints(inlayHints) {}
  bool contains(SMRange loc) {
    return mlir::lsp::contains(range, loc.Start) ||
           mlir::lsp::contains(range, loc.End);
  }

  SmallVector<StringRef> names;
  void visit(const slang::ast::InstanceBodySymbol &body) {
    names.push_back(body.name);
    mlir::lsp::Logger::debug("visit: {}", body.name);
    visitDefault(body);
    names.pop_back();
  }

  void handleSymbol(const slang::ast::Symbol *symbol, slang::SourceRange range,
                    bool useEnd = true) {
    mlir::lsp::Logger::debug("handleSymbol: {}", symbol->name);
    if (symbol->name.empty()) {
      return;
    }
    mlir::lsp::Logger::debug("baz");

    auto start = context.getSMLoc(range.start());
    auto end = context.getSMLoc(range.end());
    if (names.empty()) {
      mlir::lsp::Logger::debug("names empty");
      return;
    }
    mlir::lsp::Logger::debug("handleSymbol foo");

    if (!contains(SMRange(start, end)))
      return;
    mlir::lsp::Logger::debug("handleSymbol nam");
    if (context.getUserHint().json) {
      auto it = context.getUserHint().json->getAsObject()->find(names.back());
      StringRef newName;
      if (it != context.getUserHint().json->getAsObject()->end()) {
        auto it2 = it->second.getAsObject()->find(symbol->name);
        if (it2 == it->second.getAsObject()->end()) {
          return;
        }
        newName = it2->second.getAsString().value();
        mlir::lsp::Logger::debug("name: {}", it2->second.getAsString());
        inlayHints.emplace_back(
            mlir::lsp::InlayHintKind::Parameter,
            context.getLspLocation(useEnd ? range.end() : range.start())
                .range.start);
        auto &hint = inlayHints.back();
        hint.label = " ";
        hint.label += newName;
      }
    }
  }

  // Handle references to the left-hand side of a parent assignment.
  void visit(const slang::ast::LValueReferenceExpression &expr) {
    mlir::lsp::Logger::debug("visit: {}", slang::ast::toString(expr.kind));
    auto *symbol = expr.getSymbolReference(true);
    if (!symbol)
      return;
    handleSymbol(symbol, expr.sourceRange);
    visitDefault(expr);
  }
  void visit(const slang::ast::NetSymbol &expr) {
    mlir::lsp::Logger::debug("visit: {}", slang::ast::toString(expr.kind));
    handleSymbol(&expr, slang::SourceRange(expr.location,
                                           expr.location + expr.name.length()));
    visitDefault(expr);
  }
  void visit(const slang::ast::VariableSymbol &expr) {
    mlir::lsp::Logger::debug("visit: {}", slang::ast::toString(expr.kind));
    handleSymbol(&expr, slang::SourceRange(expr.location,
                                           expr.location + expr.name.length()));
    visitDefault(expr);
  }

  void visit(const slang::ast::VariableDeclStatement &expr) {
    mlir::lsp::Logger::debug("visit: {}", slang::ast::toString(expr.kind));
    handleSymbol(&expr.symbol, expr.sourceRange, false);
    visitDefault(expr);
  }

  // void visit(const slang::ast::AssignmentExpression &expr) {
  //   mlir::lsp::Logger::debug("visit: {}", slang::ast::toString(expr.kind));
  //   if (auto *symbol = expr.left().getSymbolReference(true))
  //     handleSymbol(symbol, expr.sourceRange);
  // }

  // Handle named values, such as references to declared variables.
  // Handle named values, such as references to declared variables.
  void visit(const slang::ast::NamedValueExpression &expr) {
    mlir::lsp::Logger::debug("visit: {}", slang::ast::toString(expr.kind));
    auto *symbol = expr.getSymbolReference(true);
    if (!symbol)
      return;
    handleSymbol(symbol, expr.sourceRange);
    visitDefault(expr);
    // if (auto value = context.valueSymbols.lookup(&expr.symbol)) {

    //   return value;
    // }

    // // Try to materialize constant values directly.
    // auto constant = context.evaluateConstant(expr);
    // if (auto value = context.materializeConstant(constant, *expr.type,
    // loc))
    //   return value;

    // // Otherwise some other part of ImportVerilog should have added an MLIR
    // // value for this expression's symbol to the `context.valueSymbols`
    // table. auto d = mlir::emitError(loc, "unknown name `") <<
    // expr.symbol.name << "`";
    // d.attachNote(context.convertLocation(expr.symbol.location))
    //     << "no rvalue generated for " <<
    //     slang::ast::toString(expr.symbol.kind);
    // return {};
  }

  template <typename T>
  void visit(const T &t) {
    mlir::lsp::Logger::debug("visit: {}", slang::ast::toString(t.kind));

    visitDefault(t);
  }
  // Handle hierarchical values, such as `x = Top.sub.var`.
  // Value visit(const slang::ast::HierarchicalValueExpression &expr) {
  //   auto hierLoc = context.convertLocation(expr.symbol.location);
  //   if (auto value = context.valueSymbols.lookup(&expr.symbol)) {
  //     if (isa<moore::RefType>(value.getType())) {
  //       auto readOp = builder.create<moore::ReadOp>(hierLoc, value);
  //       if (context.rvalueReadCallback)
  //         context.rvalueReadCallback(readOp);
  //       value = readOp.getResult();
  //     }
  //     return value;
  //   }

  //   // Emit an error for those hierarchical values not recorded in the
  //   // `valueSymbols`.
  //   auto d = mlir::emitError(loc, "unknown hierarchical name `")
  //            << expr.symbol.name << "`";
  //   d.attachNote(hierLoc) << "no rvalue generated for "
  //                         << slang::ast::toString(expr.symbol.kind);
  //   return {};
  // }
  static StringRef directionToString(slang::ast::ArgumentDirection direction) {
    switch (direction) {
    case slang::ast::ArgumentDirection::In:
      return "in";
    case slang::ast::ArgumentDirection::Out:
      return "out";
    case slang::ast::ArgumentDirection::InOut:
      return "inout";
    case slang::ast::ArgumentDirection::Ref:
      return "ref";
    }
    return "<unknown direction>";
  }

  void visit(const slang::ast::InstanceSymbol &expr) {
    mlir::lsp::Logger::info("visit: {}", slang::ast::toString(expr.kind));
    mlir::lsp::Logger::info("size: {}", expr.getPortConnections().size());

    for (const auto &con : expr.getPortConnections()) {
      auto *portSymbol = con->port.as_if<slang::ast::PortSymbol>();
      if (portSymbol) {
        auto loc = con->getExpression()->sourceRange;
        mlir::lsp::Logger::info("portSymbol: {} {}", portSymbol->name,
                                portSymbol->getType().toString());
        inlayHints.emplace_back(
            mlir::lsp::InlayHintKind::Type,
            context.getLspLocation(loc.start()).range.start);

        auto &hint = inlayHints.back();
        hint.label = directionToString(portSymbol->direction);
        hint.label += " ";
        hint.label += portSymbol->getType().toString();
        hint.label += ": ";
      }
    }
    visitDefault(expr);
  }

  // Helper function to convert an argument to a simple bit vector type, pass
  // it to a reduction op, and optionally invert the result.

  /// Handle assignment patterns.
  void visitInvalid(const slang::ast::Expression &expr) {
    mlir::lsp::Logger::debug("visitInvalid: {}",
                             slang::ast::toString(expr.kind));
  }

  void visitInvalid(const slang::ast::Statement &) {}
  void visitInvalid(const slang::ast::TimingControl &) {}
  void visitInvalid(const slang::ast::Constraint &) {}
  void visitInvalid(const slang::ast::AssertionExpr &) {}
  void visitInvalid(const slang::ast::BinsSelectExpr &) {}
  void visitInvalid(const slang::ast::Pattern &) {}

  VerilogServerContext &context;
  SMRange range;
  std::vector<mlir::lsp::InlayHint> &inlayHints;
};

void VerilogDocument::getInlayHints(
    const mlir::lsp::URIForFile &uri, const mlir::lsp::Range &range,
    std::vector<mlir::lsp::InlayHint> &inlayHints) {
  mlir::lsp::Logger::debug("getInlayHints");
  if (failed(compilation))
    return;
  SMRange rangeLoc = range.getAsSMRange(verilogContext.sourceMgr);
  mlir::lsp::Logger::debug("trying range");

  if (!rangeLoc.isValid())
    return;
  mlir::lsp::Logger::debug("trying is valid");

  // mlir::lsp::Logger::debug("trying rangeLoc.Start.getPointer() {}",
  //                         rangeLoc.Start.getPointer());
  // mlir::lsp::Logger::debug("trying rangeLoc.Start.getLine() {}",
  //                         rangeLoc.End.getPointer());

  // const auto *symbol = index.lookup(rangeLoc.Start);
  // if (!symbol)
  //   return;
  // mlir::lsp::Logger::debug("trying symbol {}", symbol->name);

  // const auto *scope = symbol->getParentScope();
  // if (!scope) {
  //   mlir::lsp::Logger::debug("no scope");
  //   return;
  // }
  // const auto *instance = scope->getContainingInstance();
  // if (!instance) {
  //   mlir::lsp::Logger::debug("no instance");
  //   return;
  // }

  // llvm::StringRef instanceName = instance->name;

  // mlir::lsp::Logger::debug("scope: {}", instanceName);
  // auto &mapping = getContext().getUserHint().json;
  // auto it = mapping->getAsObject()->find(instanceName);
  // if (it != mapping->getAsObject()->end()) {
  //   auto it2 = it->second.getAsObject()->find(symbol->name);
  //   if (it2 != it->second.getAsObject()->end()) {
  //     mlir::lsp::Logger::debug("name: {}", it2->second.getAsString());
  //   }
  // }

  RvalueExprVisitor visitor(getContext(), rangeLoc, inlayHints);
  for (auto *inst : compilation.value()->getRoot().topInstances) {
    inst->body.visit(visitor);

    // walk->body().visitExprs(visitor);
    // walk->visitExprs(visitor);
  }
  // Walk through all instances and their bodies
  // for (auto *instance : compilation.value()->getRoot().topInstances) {
  //   // Visit expressions in instance body
  //   for (auto *stmt : instance->body) {
  //     if (auto *expr = stmt->as<slang::ast::ExpressionStatement>()) {
  //       visitor.handle(expr->expr);
  //     }
  //   }
  // }
  // getContext().getUserHint().find(symbol->name, inlayHints);

  /*
  (astModule)->walk([&](const ast::Node *node) {
    SMRange loc = node->getLoc();

    // Check that the location of this node is within the input range.
    if (!lsp::contains(rangeLoc, loc.Start) &&
        !lsp::contains(rangeLoc, loc.End))
      return;

    // Handle hints for various types of nodes.
    llvm::TypeSwitch<const ast::Node *>(node)
        .Case<ast::VariableDecl, ast::CallExpr, ast::OperationExpr>(
            [&](const auto *node) {
              this->getInlayHintsFor(node, uri, inlayHints);
            });
  });
  */
}

void VerilogDocument::getInlayHintsFor(
    const slang::ast::VariableDeclStatement *decl,
    const mlir::lsp::URIForFile &uri,
    std::vector<mlir::lsp::InlayHint> &inlayHints) {
  // Check to see if the variable has a constraint list, if it does we don't
  // provide initializer hints.
  // if (!decl->getConstraints().empty())
  //   return;

  // Check to see if the variable has an initializer.
  // if (const ast::Expr *expr = decl->getInitExpr()) {
  //   // Don't add hints for operation expression initialized variables given
  //   that
  //   // the type of the variable is easily inferred by the expression
  //   operation
  //   // name.
  //   if (isa<ast::OperationExpr>(expr))
  //     return;
  // }
  auto loc = decl->sourceRange.end();
  auto &driver = getContext().driver;
  auto pos = driver.sourceManager.getLineNumber(loc);
  auto col = driver.sourceManager.getColumnNumber(loc);
  mlir::lsp::InlayHint hint(mlir::lsp::InlayHintKind::Type,
                            mlir::lsp::Position(pos, col));
  {
    llvm::raw_string_ostream labelOS(hint.label);
    labelOS << ": " << "foo";
  }

  inlayHints.emplace_back(std::move(hint));
}

void VerilogDocument::getInlayHintsFor(
    const slang::ast::InstanceSymbol *instance,
    const mlir::lsp::URIForFile &uri,
    std::vector<mlir::lsp::InlayHint> &inlayHints) {
  // Try to extract the callable of this call.
  auto &foo = instance->getDefinition();
  // const auto *callable =
  //     callableRef ? dyn_cast<ast::CallableDecl>(callableRef->getDecl())
  //                 : nullptr;
  // if (!callable)
  //   return;

  // // Add hints for the arguments to the call.
  // for (const auto &it : llvm::zip(expr->getArguments(),
  // callable->getInputs()))
  //   addParameterHintFor(inlayHints, std::get<0>(it),
  //                       std::get<1>(it)->getName().getName());
}

// void VerilogDocument::addParameterHintFor(
//     std::vector<lsp::InlayHint> &inlayHints, const ast::Expr *expr,
//     StringRef label) {
//   if (!shouldAddHintFor(expr, label))
//     return;
//
//   mlir::lspInlayHint hint(lsp::InlayHintKind::Parameter,
//                           mlir::lspPosition(sourceMgr,
//                           expr->getLoc().Start));
//   hint.label = (label + ":").str();
//   hint.paddingRight = true;
//   inlayHints.emplace_back(std::move(hint));
// }

//===----------------------------------------------------------------------===//
// Verilog ViewOutput
//===----------------------------------------------------------------------===//

void VerilogDocument::getVerilogViewOutput(
    raw_ostream &os, circt::lsp::VerilogViewOutputKind kind) {
  if (failed(compilation))
    return;
  if (kind == circt::lsp::VerilogViewOutputKind::AST) {
    os << compilation.value()->getSyntaxTrees()[0]->root().toString();
    return;
  }

  // Generate the MLIR for the ast module. We also capture diagnostics here to
  // show to the user, which may be useful if Verilog isn't capturing
  // constraints expected by PDL.
  // MLIRContext mlirContext;
  // SourceMgrDiagnosticHandler diagHandler(sourceMgr, &mlirContext, os);
  // OwningOpRef<ModuleOp> pdlModule =
  //     codegenVerilogToMLIR(&mlirContext, astContext, sourceMgr,
  //     **astModule);
  // if (!pdlModule)
  //   return;
  // if (kind == mlir::lsp::VerilogViewOutputKind::MLIR) {
  //   pdlModule->print(os, OpPrintingFlags().enableDebugInfo());
  //   return;
  // }

  // // Otherwise, generate the output for C++.
  // assert(kind == mlir::lsp::VerilogViewOutputKind::CPP &&
  //        "unexpected VerilogViewOutputKind");
  // codegenVerilogToCPP(**astModule, *pdlModule, os);
}

//===----------------------------------------------------------------------===//
// VerilogTextFileChunk
//===----------------------------------------------------------------------===//

namespace {
/// This class represents a single chunk of an Verilog text file.
struct VerilogTextFileChunk {
  VerilogTextFileChunk(uint64_t lineOffset, const mlir::lsp::URIForFile &uri,
                       StringRef contents,
                       const std::vector<std::string> &extraDirs,
                       std::vector<mlir::lsp::Diagnostic> &diagnostics)
      : lineOffset(lineOffset),
        document(uri, contents, extraDirs, diagnostics) {}

  /// Adjust the line number of the given range to anchor at the beginning of
  /// the file, instead of the beginning of this chunk.
  void adjustLocForChunkOffset(mlir::lsp::Range &range) {
    adjustLocForChunkOffset(range.start);
    adjustLocForChunkOffset(range.end);
  }
  /// Adjust the line number of the given position to anchor at the beginning
  /// of the file, instead of the beginning of this chunk.
  void adjustLocForChunkOffset(mlir::lsp::Position &pos) {
    pos.line += lineOffset;
  }

  /// The line offset of this chunk from the beginning of the file.
  uint64_t lineOffset;
  /// The document referred to by this chunk.
  VerilogDocument document;
};
} // namespace

//===----------------------------------------------------------------------===//
// VerilogTextFile
//===----------------------------------------------------------------------===//

namespace {
/// This class represents a text file containing one or more Verilog
/// documents.
class VerilogTextFile {
public:
  VerilogTextFile(const mlir::lsp::URIForFile &uri, StringRef fileContents,
                  int64_t version, const std::vector<std::string> &extraDirs,
                  std::vector<mlir::lsp::Diagnostic> &diagnostics);

  /// Return the current version of this text file.
  int64_t getVersion() const { return version; }

  /// Update the file to the new version using the provided set of content
  /// changes. Returns failure if the update was unsuccessful.
  LogicalResult
  update(const mlir::lsp::URIForFile &uri, int64_t newVersion,
         ArrayRef<mlir::lsp::TextDocumentContentChangeEvent> changes,
         std::vector<mlir::lsp::Diagnostic> &diagnostics);

  //===--------------------------------------------------------------------===//
  // LSP Queries
  //===--------------------------------------------------------------------===//

  void getLocationsOf(const mlir::lsp::URIForFile &uri,
                      mlir::lsp::Position defPos,
                      std::vector<mlir::lsp::Location> &locations);
  void findReferencesOf(const mlir::lsp::URIForFile &uri,
                        mlir::lsp::Position pos,
                        std::vector<mlir::lsp::Location> &references);
  void getDocumentLinks(const mlir::lsp::URIForFile &uri,
                        std::vector<mlir::lsp::DocumentLink> &links);
  std::optional<mlir::lsp::Hover> findHover(const mlir::lsp::URIForFile &uri,
                                            mlir::lsp::Position hoverPos);
  void findDocumentSymbols(std::vector<mlir::lsp::DocumentSymbol> &symbols);
  mlir::lsp::CompletionList getCodeCompletion(const mlir::lsp::URIForFile &uri,
                                              mlir::lsp::Position completePos);
  mlir::lsp::SignatureHelp getSignatureHelp(const mlir::lsp::URIForFile &uri,
                                            mlir::lsp::Position helpPos);
  void getInlayHints(const mlir::lsp::URIForFile &uri, mlir::lsp::Range range,
                     std::vector<mlir::lsp::InlayHint> &inlayHints);
  circt::lsp::VerilogViewOutputResult
  getVerilogViewOutput(circt::lsp::VerilogViewOutputKind kind);

private:
  using ChunkIterator = llvm::pointee_iterator<
      std::vector<std::unique_ptr<VerilogTextFileChunk>>::iterator>;

  /// Initialize the text file from the given file contents.
  void initialize(const mlir::lsp::URIForFile &uri, int64_t newVersion,
                  std::vector<mlir::lsp::Diagnostic> &diagnostics);

  /// Find the PDL document that contains the given position, and update the
  /// position to be anchored at the start of the found chunk instead of the
  /// beginning of the file.
  ChunkIterator getChunkItFor(mlir::lsp::Position &pos);
  VerilogTextFileChunk &getChunkFor(mlir::lsp::Position &pos) {
    return *getChunkItFor(pos);
  }

  /// The full string contents of the file.
  std::string contents;

  /// The version of this file.
  int64_t version = 0;

  /// The number of lines in the file.
  int64_t totalNumLines = 0;

  /// The chunks of this file. The order of these chunks is the order in which
  /// they appear in the text file.
  std::vector<std::unique_ptr<VerilogTextFileChunk>> chunks;

  /// The extra set of include directories for this file.
  std::vector<std::string> extraIncludeDirs;
};
} // namespace

VerilogTextFile::VerilogTextFile(
    const mlir::lsp::URIForFile &uri, StringRef fileContents, int64_t version,
    const std::vector<std::string> &extraDirs,
    std::vector<mlir::lsp::Diagnostic> &diagnostics)
    : contents(fileContents.str()), extraIncludeDirs(extraDirs) {
  initialize(uri, version, diagnostics);
}

LogicalResult VerilogTextFile::update(
    const mlir::lsp::URIForFile &uri, int64_t newVersion,
    ArrayRef<mlir::lsp::TextDocumentContentChangeEvent> changes,
    std::vector<mlir::lsp::Diagnostic> &diagnostics) {
  if (failed(mlir::lsp::TextDocumentContentChangeEvent::applyTo(changes,
                                                                contents))) {
    mlir::lsp::Logger::error("Failed to update contents of {0}", uri.file());
    return failure();
  }

  // If the file contents were properly changed, reinitialize the text file.
  initialize(uri, newVersion, diagnostics);
  return success();
}

void VerilogTextFile::getLocationsOf(
    const mlir::lsp::URIForFile &uri, mlir::lsp::Position defPos,
    std::vector<mlir::lsp::Location> &locations) {
  VerilogTextFileChunk &chunk = getChunkFor(defPos);
  chunk.document.getLocationsOf(uri, defPos, locations);

  // Adjust any locations within this file for the offset of this chunk.
  if (chunk.lineOffset == 0)
    return;
  for (mlir::lsp::Location &loc : locations)
    if (loc.uri == uri)
      chunk.adjustLocForChunkOffset(loc.range);
}

void VerilogTextFile::findReferencesOf(
    const mlir::lsp::URIForFile &uri, mlir::lsp::Position pos,
    std::vector<mlir::lsp::Location> &references) {
  VerilogTextFileChunk &chunk = getChunkFor(pos);
  chunk.document.findReferencesOf(uri, pos, references);

  // Adjust any locations within this file for the offset of this chunk.
  if (chunk.lineOffset == 0)
    return;
  for (mlir::lsp::Location &loc : references)
    if (loc.uri == uri)
      chunk.adjustLocForChunkOffset(loc.range);
}

void VerilogTextFile::getDocumentLinks(
    const mlir::lsp::URIForFile &uri,
    std::vector<mlir::lsp::DocumentLink> &links) {
  chunks.front()->document.getDocumentLinks(uri, links);
  for (const auto &it : llvm::drop_begin(chunks)) {
    size_t currentNumLinks = links.size();
    it->document.getDocumentLinks(uri, links);

    // Adjust any links within this file to account for the offset of this
    // chunk.
    for (auto &link : llvm::drop_begin(links, currentNumLinks))
      it->adjustLocForChunkOffset(link.range);
  }
}

std::optional<mlir::lsp::Hover>
VerilogTextFile::findHover(const mlir::lsp::URIForFile &uri,
                           mlir::lsp::Position hoverPos) {
  VerilogTextFileChunk &chunk = getChunkFor(hoverPos);
  std::optional<mlir::lsp::Hover> hoverInfo =
      chunk.document.findHover(uri, hoverPos);

  // Adjust any locations within this file for the offset of this chunk.
  if (chunk.lineOffset != 0 && hoverInfo && hoverInfo->range)
    chunk.adjustLocForChunkOffset(*hoverInfo->range);
  return hoverInfo;
}

void VerilogTextFile::findDocumentSymbols(
    std::vector<mlir::lsp::DocumentSymbol> &symbols) {
  if (chunks.size() == 1)
    return chunks.front()->document.findDocumentSymbols(symbols);

  // If there are multiple chunks in this file, we create top-level symbols
  // for each chunk.
  for (unsigned i = 0, e = chunks.size(); i < e; ++i) {
    VerilogTextFileChunk &chunk = *chunks[i];
    mlir::lsp::Position startPos(chunk.lineOffset);
    mlir::lsp::Position endPos((i == e - 1) ? totalNumLines - 1
                                            : chunks[i + 1]->lineOffset);
    mlir::lsp::DocumentSymbol symbol(
        "<file-split-" + Twine(i) + ">", mlir::lsp::SymbolKind::Namespace,
        /*range=*/mlir::lsp::Range(startPos, endPos),
        /*selectionRange=*/mlir::lsp::Range(startPos));
    chunk.document.findDocumentSymbols(symbol.children);

    // Fixup the locations of document symbols within this chunk.
    if (i != 0) {
      SmallVector<mlir::lsp::DocumentSymbol *> symbolsToFix;
      for (mlir::lsp::DocumentSymbol &childSymbol : symbol.children)
        symbolsToFix.push_back(&childSymbol);

      while (!symbolsToFix.empty()) {
        mlir::lsp::DocumentSymbol *symbol = symbolsToFix.pop_back_val();
        chunk.adjustLocForChunkOffset(symbol->range);
        chunk.adjustLocForChunkOffset(symbol->selectionRange);

        for (mlir::lsp::DocumentSymbol &childSymbol : symbol->children)
          symbolsToFix.push_back(&childSymbol);
      }
    }

    // Push the symbol for this chunk.
    symbols.emplace_back(std::move(symbol));
  }
}

void VerilogTextFile::getInlayHints(
    const mlir::lsp::URIForFile &uri, mlir::lsp::Range range,
    std::vector<mlir::lsp::InlayHint> &inlayHints) {
  auto startIt = getChunkItFor(range.start);
  auto endIt = getChunkItFor(range.end);

  // Functor used to get the chunks for a given file, and fixup any locations
  auto getHintsForChunk = [&](ChunkIterator chunkIt, mlir::lsp::Range range) {
    size_t currentNumHints = inlayHints.size();
    chunkIt->document.getInlayHints(uri, range, inlayHints);

    // If this isn't the first chunk, update any positions to account for line
    // number differences.
    if (&*chunkIt != &*chunks.front()) {
      for (auto &hint : llvm::drop_begin(inlayHints, currentNumHints))
        chunkIt->adjustLocForChunkOffset(hint.position);
    }
  };
  // Returns the number of lines held by a given chunk.
  auto getNumLines = [](ChunkIterator chunkIt) {
    return (chunkIt + 1)->lineOffset - chunkIt->lineOffset;
  };

  // Check if the range is fully within a single chunk.
  if (startIt == endIt)
    return getHintsForChunk(startIt, range);

  // Otherwise, the range is split between multiple chunks. The first chunk
  // has the correct range start, but covers the total document.
  getHintsForChunk(startIt,
                   mlir::lsp::Range(range.start, getNumLines(startIt)));

  // Every chunk in between uses the full document.
  for (++startIt; startIt != endIt; ++startIt)
    getHintsForChunk(startIt, mlir::lsp::Range(0, getNumLines(startIt)));

  // The range for the last chunk starts at the beginning of the document, up
  // through the end of the input range.
  getHintsForChunk(startIt, mlir::lsp::Range(0, range.end));
}

circt::lsp::VerilogViewOutputResult
VerilogTextFile::getVerilogViewOutput(circt::lsp::VerilogViewOutputKind kind) {
  circt::lsp::VerilogViewOutputResult result;
  {
    llvm::raw_string_ostream outputOS(result.output);
    llvm::interleave(
        llvm::make_pointee_range(chunks),
        [&](VerilogTextFileChunk &chunk) {
          chunk.document.getVerilogViewOutput(outputOS, kind);
        },
        [&] { outputOS << "\n"
                       << kDefaultSplitMarker << "\n\n"; });
  }
  return result;
}

void VerilogTextFile::initialize(
    const mlir::lsp::URIForFile &uri, int64_t newVersion,
    std::vector<mlir::lsp::Diagnostic> &diagnostics) {
  Logger::debug("VerilogTextFile::initialize");
  version = newVersion;
  chunks.clear();

  // Split the file into separate PDL documents.
  SmallVector<StringRef, 8> subContents;
  StringRef(contents).split(subContents, kDefaultSplitMarker);
  chunks.emplace_back(std::make_unique<VerilogTextFileChunk>(
      /*lineOffset=*/0, uri, subContents.front(), extraIncludeDirs,
      diagnostics));

  uint64_t lineOffset = subContents.front().count('\n');
  for (StringRef docContents : llvm::drop_begin(subContents)) {
    unsigned currentNumDiags = diagnostics.size();
    auto chunk = std::make_unique<VerilogTextFileChunk>(
        lineOffset, uri, docContents, extraIncludeDirs, diagnostics);
    lineOffset += docContents.count('\n');

    // Adjust locations used in diagnostics to account for the offset from the
    // beginning of the file.
    for (mlir::lsp::Diagnostic &diag :
         llvm::drop_begin(diagnostics, currentNumDiags)) {
      chunk->adjustLocForChunkOffset(diag.range);

      if (!diag.relatedInformation)
        continue;
      for (auto &it : *diag.relatedInformation)
        if (it.location.uri == uri)
          chunk->adjustLocForChunkOffset(it.location.range);
    }
    chunks.emplace_back(std::move(chunk));
  }
  totalNumLines = lineOffset;
}

VerilogTextFile::ChunkIterator
VerilogTextFile::getChunkItFor(mlir::lsp::Position &pos) {
  if (chunks.size() == 1)
    return chunks.begin();

  // Search for the first chunk with a greater line offset, the previous chunk
  // is the one that contains `pos`.
  auto it = llvm::upper_bound(
      chunks, pos, [](const mlir::lsp::Position &pos, const auto &chunk) {
        return static_cast<uint64_t>(pos.line) < chunk->lineOffset;
      });
  ChunkIterator chunkIt(it == chunks.end() ? (chunks.end() - 1) : --it);
  pos.line -= chunkIt->lineOffset;
  return chunkIt;
}

//===----------------------------------------------------------------------===//
// VerilogServer::Impl
//===----------------------------------------------------------------------===//

struct circt::lsp::VerilogServer::Impl {
  explicit Impl(const VerilogServerOptions &options)
      : options(options), compilationDatabase(options.compilationDatabases) {}

  /// Verilog LSP options.
  const VerilogServerOptions &options;

  /// The compilation database containing additional information for files
  /// passed to the server.
  mlir::lsp::CompilationDatabase compilationDatabase;

  /// The files held by the server, mapped by their URI file name.
  llvm::StringMap<std::unique_ptr<VerilogTextFile>> files;
};

//===----------------------------------------------------------------------===//
// VerilogServer
//===----------------------------------------------------------------------===//

circt::lsp::VerilogServer::VerilogServer(const VerilogServerOptions &options)
    : impl(std::make_unique<Impl>(options)) {}
circt::lsp::VerilogServer::~VerilogServer() = default;

void circt::lsp::VerilogServer::addDocument(
    const URIForFile &uri, StringRef contents, int64_t version,
    std::vector<mlir::lsp::Diagnostic> &diagnostics) {
  Logger::debug("VerilogServer::addDocument");
  // Build the set of additional include directories.
  std::vector<std::string> additionalIncludeDirs = impl->options.extraDirs;
  const auto &fileInfo = impl->compilationDatabase.getFileInfo(uri.file());
  llvm::append_range(additionalIncludeDirs, fileInfo.includeDirs);

  impl->files[uri.file()] = std::make_unique<VerilogTextFile>(
      uri, contents, version, additionalIncludeDirs, diagnostics);
  Logger::debug("VerilogServer::addDocument done");
}

void circt::lsp::VerilogServer::updateDocument(
    const URIForFile &uri,
    ArrayRef<mlir::lsp::TextDocumentContentChangeEvent> changes,
    int64_t version, std::vector<mlir::lsp::Diagnostic> &diagnostics) {
  // Check that we actually have a document for this uri.
  auto it = impl->files.find(uri.file());
  if (it == impl->files.end())
    return;

  // Try to update the document. If we fail, erase the file from the server. A
  // failed updated generally means we've fallen out of sync somewhere.
  if (failed(it->second->update(uri, version, changes, diagnostics)))
    impl->files.erase(it);
}

std::optional<int64_t>
circt::lsp::VerilogServer::removeDocument(const URIForFile &uri) {
  auto it = impl->files.find(uri.file());
  if (it == impl->files.end())
    return std::nullopt;

  int64_t version = it->second->getVersion();
  impl->files.erase(it);
  return version;
}

void circt::lsp::VerilogServer::getLocationsOf(
    const URIForFile &uri, const Position &defPos,
    std::vector<mlir::lsp::Location> &locations) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    fileIt->second->getLocationsOf(uri, defPos, locations);
}

void circt::lsp::VerilogServer::findReferencesOf(
    const URIForFile &uri, const Position &pos,
    std::vector<mlir::lsp::Location> &references) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    fileIt->second->findReferencesOf(uri, pos, references);
}

void circt::lsp::VerilogServer::getDocumentLinks(
    const URIForFile &uri,
    std::vector<mlir::lsp::DocumentLink> &documentLinks) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    return fileIt->second->getDocumentLinks(uri, documentLinks);
}

std::optional<mlir::lsp::Hover>
circt::lsp::VerilogServer::findHover(const URIForFile &uri,
                                     const Position &hoverPos) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    return fileIt->second->findHover(uri, hoverPos);
  return std::nullopt;
}

void circt::lsp::VerilogServer::findDocumentSymbols(
    const URIForFile &uri, std::vector<mlir::lsp::DocumentSymbol> &symbols) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    fileIt->second->findDocumentSymbols(symbols);
}

void circt::lsp::VerilogServer::getInlayHints(
    const URIForFile &uri, const mlir::lsp::Range &range,
    std::vector<mlir::lsp::InlayHint> &inlayHints) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt == impl->files.end())
    return;

  mlir::lsp::Logger::debug("trying getInlayHints");
  fileIt->second->getInlayHints(uri, range, inlayHints);

  // Drop any duplicated hints that may have cropped up.
  llvm::sort(inlayHints);
  inlayHints.erase(llvm::unique(inlayHints), inlayHints.end());
}

std::optional<circt::lsp::VerilogViewOutputResult>
circt::lsp::VerilogServer::getVerilogViewOutput(const URIForFile &uri,
                                                VerilogViewOutputKind kind) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    return fileIt->second->getVerilogViewOutput(kind);
  return std::nullopt;
}
