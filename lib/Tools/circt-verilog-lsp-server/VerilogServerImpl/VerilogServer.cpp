//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the VerilogServer class, which is responsible for
// managing the state of the Verilog server. VerilogServer keeps track of the
// contents of all open text documents, and each document has a slang
// compilation result.
//
//===----------------------------------------------------------------------===//
#include "VerilogServer.h"
#include "../Utils/LSPUtils.h"

#include "circt/Support/LLVM.h"
#include "circt/Tools/circt-verilog-lsp-server/CirctVerilogLspServerMain.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Protocol.h"
#include "slang/ast/ASTVisitor.h"
#include "slang/ast/Compilation.h"
#include "slang/diagnostics/DiagnosticClient.h"
#include "slang/diagnostics/Diagnostics.h"
#include "slang/driver/Driver.h"
#include "slang/syntax/SyntaxTree.h"
#include "slang/text/SourceLocation.h"
#include "slang/text/SourceManager.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"

#include <memory>
#include <optional>

using namespace mlir;
using namespace mlir::lsp;

using namespace circt::lsp;
using namespace circt;

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
namespace {

// A global context carried around by the server.
struct VerilogServerContext {
  VerilogServerContext(const VerilogServerOptions &options)
      : options(options) {}
  const VerilogServerOptions &options;
};

class VerilogDocument;
using VerilogIndexSymbol =
    llvm::PointerUnion<const slang::ast::Symbol *, mlir::Attribute>;

class VerilogIndex {
public:
  VerilogIndex(VerilogDocument &document)
      : document(document), mlirContext(mlir::MLIRContext::Threading::DISABLED),
        intervalMap(allocator) {}

  /// Initialize the index with the given compilation unit.
  void initialize(slang::ast::Compilation &compilation);

  /// Register a reference to a symbol `symbol` from `from`.
  void insertSymbolUse(const slang::ast::Symbol *symbol,
                       slang::SourceRange from = slang::SourceRange());

  VerilogDocument &getDocument() { return document; }

  /// The type of interval map used to store source references. SMRange is
  /// half-open, so we also need to use a half-open interval map.
  using MapT =
      llvm::IntervalMap<const char *, VerilogIndexSymbol,
                        llvm::IntervalMapImpl::NodeSizer<
                            const char *, VerilogIndexSymbol>::LeafSize,
                        llvm::IntervalMapHalfOpenInfo<const char *>>;

private:
  /// Parse source location in the file.
  void parseSourceLocation();

  MapT intervalMap;
  /// An allocator for the interval map.
  MapT::Allocator allocator;

  VerilogDocument &document;

  // MLIR context used for generating location attr.
  mlir::MLIRContext mlirContext;
};

} // namespace

namespace {

/*
class VerilogIndex {
public:
  VerilogIndex(VerilogServerContext &context)
      : context(context), intervalMap(allocator), intervalMapLoc(allocator) {}

  /// Initialize the index with the given ast::Module.
  void initialize(slang::ast::Compilation *compilation);

  using IndexElement = llvm::PointerUnion<slang::ast::Symbol *, LocationAttr>;

  /// Lookup a symbol for the given location. Returns nullptr if no symbol
  /// could be found. If provided, `overlappedRange` is set to the range that
  /// the provided `loc` overlapped with.
  const VerilogIndexSymbol *lookup(SMLoc loc,
                                   SMRange *overlappedRange = nullptr) const;

  const EmittedLoc *lookupLoc(SMLoc loc) const;

  size_t size() const {
    return std::distance(intervalMap.begin(), intervalMap.end());
  }

  VerilogServerContext &getContext() { return context; }
  auto &getIntervalMap() { return intervalMap; }
  auto &getReferences() { return references; }

  void parseEmittedLoc();

  enum SymbolUse { AssignLValue, RValue, Unknown };

  llvm::StringMap<llvm::StringMap<slang::ast::Symbol *>> moduleVariableToSymbol;

  using ReferenceNode = llvm::PointerUnion<const slang::ast::Symbol *,
                                           const slang::ast::PortConnection *,
                                           const slang::ast::Expression *>;

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
                      SmallVector<std::tuple<slang::SourceLocation, SymbolUse,
                                             std::optional<ReferenceNode>>,
                                  8>>
      references;

  // TODO: Use allocator.
  SmallVector<std::unique_ptr<EmittedLoc>> emittedLocs;

  slang::ast::Symbol *lookupSymbolFromModuleAndName() {}

  /// A mapping between definitions and their corresponding symbol.
  // DenseMap<const void *, std::unique_ptr<VerilogIndexSymbol>> defToSymbol;
}; */

//===----------------------------------------------------------------------===//
// VerilogDocument
//===----------------------------------------------------------------------===//

/// This class represents all of the information pertaining to a specific
/// Verilog document.
class LSPDiagnosticClient;
struct VerilogDocument {
  VerilogDocument(VerilogServerContext &globalContext,
                  const mlir::lsp::URIForFile &uri, StringRef contents,
                  std::vector<mlir::lsp::Diagnostic> &diagnostics);
  VerilogDocument(const VerilogDocument &) = delete;
  VerilogDocument &operator=(const VerilogDocument &) = delete;

  const mlir::lsp::URIForFile &getURI() const { return uri; }

  llvm::SourceMgr &getSourceMgr() { return sourceMgr; }
  llvm::SmallDenseMap<uint32_t, uint32_t> &getBufferIDMap() {
    return bufferIDMap;
  }

  const slang::SourceManager &getSlangSourceManager() const {
    return driver.sourceManager;
  }

  // Return LSP location from slang location.
  mlir::lsp::Location getLspLocation(slang::SourceLocation loc) const;

  // Return SMLoc from slang location.
  llvm::SMLoc getSMLoc(slang::SourceLocation loc);

private:
  // A map from slang buffer ID to the corresponding buffer ID in the LLVM
  // source manager.
  llvm::SmallDenseMap<uint32_t, uint32_t> bufferIDMap;

  // The compilation result.
  FailureOr<std::unique_ptr<slang::ast::Compilation>> compilation;

  // The slang driver.
  slang::driver::Driver driver;

  // The LLVM source manager.
  llvm::SourceMgr sourceMgr;

  /// The index of the parsed module.
  VerilogIndex index;

  // The URI of the document.
  const mlir::lsp::URIForFile &uri;
};

} // namespace

//===----------------------------------------------------------------------===//
// LSPDiagnosticClient
//===----------------------------------------------------------------------===//

namespace {
/// A converter that can be plugged into a slang `DiagnosticEngine` as a
/// client that will map slang diagnostics to LSP diagnostics.
class LSPDiagnosticClient : public slang::DiagnosticClient {
  const VerilogDocument &document;
  std::vector<mlir::lsp::Diagnostic> &diags;

public:
  LSPDiagnosticClient(const VerilogDocument &document,
                      std::vector<mlir::lsp::Diagnostic> &diags)
      : document(document), diags(diags) {}

  void report(const slang::ReportedDiagnostic &slangDiag) override;
};
} // namespace

void LSPDiagnosticClient::report(const slang::ReportedDiagnostic &slangDiag) {
  auto loc = document.getLspLocation(slangDiag.location);
  // Show only the diagnostics in the current file.
  if (loc.uri != document.getURI())
    return;
  diags.emplace_back();
  auto &mlirDiag = diags.back();
  mlirDiag.severity = getSeverity(slangDiag.severity);
  mlirDiag.range = loc.range;
  mlirDiag.source = "slang";
  mlirDiag.message = slangDiag.formattedMessage;
}

//===----------------------------------------------------------------------===//
// VerilogDocument
//===----------------------------------------------------------------------===//

VerilogDocument::VerilogDocument(
    VerilogServerContext &context, const mlir::lsp::URIForFile &uri,
    StringRef contents, std::vector<mlir::lsp::Diagnostic> &diagnostics)
    : uri(uri), index(*this) {
  unsigned int bufferId;
  if (auto memBufferOwn =
          llvm::MemoryBuffer::getMemBufferCopy(contents, uri.file())) {

    bufferId = sourceMgr.AddNewSourceBuffer(std::move(memBufferOwn), SMLoc());
  } else {
    circt::lsp::Logger::error(
        Twine("Failed to create memory buffer for file ") + uri.file());
    return;
  }

  // Build the set of include directories for this file.
  llvm::SmallString<32> uriDirectory(uri.file());
  llvm::sys::path::remove_filename(uriDirectory);

  std::vector<std::string> libDirs;
  libDirs.push_back(uriDirectory.str().str());
  libDirs.insert(libDirs.end(), context.options.libDirs.begin(),
                 context.options.libDirs.end());

  // Populate source managers.
  const llvm::MemoryBuffer *memBuffer = sourceMgr.getMemoryBuffer(bufferId);

  driver.options.libDirs = libDirs;
  driver.sourceManager.setDisableProximatePaths(true);
  // Assign text to slang.
  auto slangBuffer =
      driver.sourceManager.assignText(uri.file(), memBuffer->getBuffer());
  driver.buffers.push_back(slangBuffer);
  bufferIDMap[slangBuffer.id.getId()] = bufferId;

  auto diagClient = std::make_shared<LSPDiagnosticClient>(*this, diagnostics);
  driver.diagEngine.addClient(diagClient);

  if (!driver.parseAllSources()) {
    circt::lsp::Logger::error(Twine("Failed to parse Verilog file ") +
                              uri.file());
    return;
  }

  compilation = driver.createCompilation();
  for (auto &diag : (*compilation)->getAllDiagnostics())
    driver.diagEngine.issue(diag);
}

mlir::lsp::Location
VerilogDocument::getLspLocation(slang::SourceLocation loc) const {
  if (loc && loc.buffer() != slang::SourceLocation::NoLocation.buffer()) {
    const auto &slangSourceManager = getSlangSourceManager();
    auto line = slangSourceManager.getLineNumber(loc) - 1;
    auto column = slangSourceManager.getColumnNumber(loc) - 1;
    auto it = bufferIDMap.find(loc.buffer().getId());
    // Check if the current buffer is the main file.
    if (it != bufferIDMap.end() && it->second == sourceMgr.getMainFileID())
      return mlir::lsp::Location(uri, mlir::lsp::Range(Position(line, column)));

    // Otherwise, construct URI from slang source manager.
    auto fileName = slangSourceManager.getFileName(loc);
    auto loc = mlir::lsp::URIForFile::fromFile(
        slangSourceManager.makeAbsolutePath(fileName));
    if (auto e = loc.takeError())
      return mlir::lsp::Location();
    return mlir::lsp::Location(loc.get(),
                               mlir::lsp::Range(Position(line, column)));
  }

  return mlir::lsp::Location();
}

llvm::SMLoc VerilogDocument::getSMLoc(slang::SourceLocation loc) {
  auto bufferID = loc.buffer().getId();
  // Check if the source is already opened by LLVM source manager.
  auto bufferIDMapIt = bufferIDMap.find(bufferID);
  if (bufferIDMapIt == bufferIDMap.end()) {
    // If not, open the source file and add it to the LLVM source manager.
    auto path = getSlangSourceManager().getFullPath(loc.buffer());
    auto memBuffer = llvm::MemoryBuffer::getFile(path.string());
    if (!memBuffer) {
      circt::lsp::Logger::error("Failed to open file: " +
                                memBuffer.getError().message());
      return llvm::SMLoc();
    }

    auto id = sourceMgr.AddNewSourceBuffer(std::move(memBuffer.get()), SMLoc());
    bufferIDMapIt =
        bufferIDMap.insert({bufferID, static_cast<uint32_t>(id)}).first;
  }

  const auto *buffer = sourceMgr.getMemoryBuffer(bufferIDMapIt->second);

  return llvm::SMLoc::getFromPointer(buffer->getBufferStart() + loc.offset());
}

namespace {} // namespace

//===----------------------------------------------------------------------===//
// VerilogTextFile
//===----------------------------------------------------------------------===//

namespace {
/// This class represents a text file containing one or more Verilog
/// documents.
class VerilogTextFile {
public:
  VerilogTextFile(VerilogServerContext &globalContext,
                  const mlir::lsp::URIForFile &uri, StringRef fileContents,
                  int64_t version,
                  std::vector<mlir::lsp::Diagnostic> &diagnostics);

  /// Return the current version of this text file.
  int64_t getVersion() const { return version; }

  /// Update the file to the new version using the provided set of content
  /// changes. Returns failure if the update was unsuccessful.
  LogicalResult
  update(const mlir::lsp::URIForFile &uri, int64_t newVersion,
         ArrayRef<mlir::lsp::TextDocumentContentChangeEvent> changes,
         std::vector<mlir::lsp::Diagnostic> &diagnostics);

private:
  /// Initialize the text file from the given file contents.
  void initialize(const mlir::lsp::URIForFile &uri, int64_t newVersion,
                  std::vector<mlir::lsp::Diagnostic> &diagnostics);

  VerilogServerContext &context;

  /// The full string contents of the file.
  std::string contents;

  /// The version of this file.
  int64_t version = 0;

  /// The chunks of this file. The order of these chunks is the order in which
  /// they appear in the text file.
  std::unique_ptr<VerilogDocument> document;
};
} // namespace

VerilogTextFile::VerilogTextFile(
    VerilogServerContext &context, const mlir::lsp::URIForFile &uri,
    StringRef fileContents, int64_t version,
    std::vector<mlir::lsp::Diagnostic> &diagnostics)
    : context(context), contents(fileContents.str()) {
  initialize(uri, version, diagnostics);
}

LogicalResult VerilogTextFile::update(
    const mlir::lsp::URIForFile &uri, int64_t newVersion,
    ArrayRef<mlir::lsp::TextDocumentContentChangeEvent> changes,
    std::vector<mlir::lsp::Diagnostic> &diagnostics) {
  if (failed(mlir::lsp::TextDocumentContentChangeEvent::applyTo(changes,
                                                                contents))) {
    circt::lsp::Logger::error(Twine("Failed to update contents of ") +
                              uri.file());
    return failure();
  }

  // If the file contents were properly changed, reinitialize the text file.
  initialize(uri, newVersion, diagnostics);
  return success();
}

void VerilogTextFile::initialize(
    const mlir::lsp::URIForFile &uri, int64_t newVersion,
    std::vector<mlir::lsp::Diagnostic> &diagnostics) {
  version = newVersion;
  document =
      std::make_unique<VerilogDocument>(context, uri, contents, diagnostics);
}

struct IndexVisitor : slang::ast::ASTVisitor<IndexVisitor, true, true> {
  IndexVisitor(VerilogIndex &index) : index(index) {}
  VerilogIndex &index;

  void handleSymbol(const slang::ast::Symbol *symbol,
                    slang::SourceRange range) {
    assert(range.start().valid() && range.end().valid());
    insertDeclRef(symbol, range);
  }

  void handleSymbol(const slang::ast::Symbol *symbol,
                    slang::SourceLocation location) {
    if (symbol->name.empty())
      return;
    handleSymbol(
        symbol, slang::SourceRange(location, location + symbol->name.length()));
  }

  // Handle references to the left-hand side of a parent assignment.
  void visit(const slang::ast::LValueReferenceExpression &expr) {
    auto *symbol = expr.getSymbolReference(true);
    if (!symbol)
      return;
    handleSymbol(symbol, expr.sourceRange);
  }

  void visit(const slang::ast::Symbol &expr) {
    handleSymbol(&expr, expr.location);
  }

  //
  //  void visit(const slang::ast::VariableSymbol &expr) {
  //    handleSymbol(&expr, expr.location));
  //    visitDefault(expr);
  //  }
  //
  //  void visit(const slang::ast::VariableDeclStatement &expr) {
  //    handleSymbol(&expr.symbol, expr.sourceRange);
  //    visitDefault(expr);
  //  }

  // Handle named values, such as references to declared variables.
  // void visit(const slang::ast::NamedValueExpression &expr) {
  //   auto *symbol = expr.getSymbolReference(true);
  //   if (!symbol)
  //     return;
  //   handleSymbol(symbol, expr.sourceRange);
  //   visitDefault(expr);
  // }

  template <typename T>
  void visit(const T &t) {

    visitDefault(t);
  }
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

  VerilogDocument &getDocument() { return index.getDocument(); }

  void insertDeclRef(const slang::ast::Symbol *sym, slang::SourceRange refLoc,
                     bool isDef = false, bool isAssign = false) {
    index.insertSymbolUse(sym, refLoc);
  };
};

void VerilogIndex::initialize(slang::ast::Compilation &compilation) {
  const auto &root = compilation.getRoot();
  IndexVisitor visitor(*this);
  for (auto *inst : root.topInstances) {
    // Visit the body of the instance.
    inst->body.visit(visitor);

    // Insert the symbols in the port list.
    for (const auto *symbol : inst->body.getPortList())
      insertSymbolUse(symbol);
  }

  parseSourceLocation();
}

void VerilogIndex::parseSourceLocation() {
  auto &sourceMgr = getDocument().getSourceMgr();
  auto *getMainBuffer = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  StringRef text(getMainBuffer->getBufferStart(),
                 getMainBuffer->getBufferSize());
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
      if (llvm::any_of(*it, [](char c) { return c != ' '; })) {
        filePath = *it;
        addFile = true;
      }
      line = *(++it);
      column = *(++it);
      uint32_t lineInt;
      if (line.getAsInteger(10, lineInt))
        continue;
      SmallVector<std::tuple<StringRef, const char *, const char *>> columns;
      if (column.starts_with('{')) {
        const char *start = addFile ? filePath.data() : line.data();
        for (auto str : llvm::split(column.drop_back().drop_front(), ',')) {
          columns.push_back(
              std::make_tuple(str, start, str.drop_front(str.size()).data()));
          start = str.drop_front(str.size()).data();
        }
      } else
        columns.push_back(std::make_tuple(
            column, addFile ? filePath.data() : line.data(), column.end()));
      for (auto [column, start, end] : columns) {
        uint32_t columnInt;
        if (column.getAsInteger(10, columnInt))
          continue;
        mlir::FileLineColRange loc =
            mlir::FileLineColRange::get(&mlirContext, filePath, lineInt - 1,
                                        columnInt - 1, lineInt - 1, columnInt);
        intervalMap.insert(start, end, loc);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// VerilogServer::Impl
//===----------------------------------------------------------------------===//

struct circt::lsp::VerilogServer::Impl {
  explicit Impl(const VerilogServerOptions &options) : context(options) {}

  /// The files held by the server, mapped by their URI file name.
  llvm::StringMap<std::unique_ptr<VerilogTextFile>> files;

  VerilogServerContext context;
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
  impl->files[uri.file()] = std::make_unique<VerilogTextFile>(
      impl->context, uri, contents, version, diagnostics);
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

void VerilogIndex::insertSymbolUse(const slang::ast::Symbol *symbol,
                                   slang::SourceRange from) {
  const char *startLoc = getDocument().getSMLoc(from.start()).getPointer();
  const char *endLoc = getDocument().getSMLoc(from.end()).getPointer();
  if (!startLoc || !endLoc)
    return;
  assert(startLoc && endLoc);

  if (startLoc != endLoc && !intervalMap.overlaps(startLoc, endLoc))
    intervalMap.insert(startLoc, endLoc, symbol);
}
