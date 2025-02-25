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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Protocol.h"
#include "slang/ast/ASTVisitor.h"
#include "slang/ast/Compilation.h"
#include "slang/ast/Expression.h"
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
#include "llvm/Support/FileSystem.h"
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
                       slang::SourceRange from, bool isDefinition = false);
  void insertSymbolUse(const slang::ast::Symbol *symbol,
                       bool isDefinition = false);

  VerilogDocument &getDocument() { return document; }

  /// The type of interval map used to store source references. SMRange is
  /// half-open, so we also need to use a half-open interval map.
  using MapT =
      llvm::IntervalMap<const char *, VerilogIndexSymbol,
                        llvm::IntervalMapImpl::NodeSizer<
                            const char *, VerilogIndexSymbol>::LeafSize,
                        llvm::IntervalMapHalfOpenInfo<const char *>>;

  const MapT &getIntervalMap() { return intervalMap; }

  using ReferenceMap = SmallDenseMap<const slang::ast::Symbol *,
                                     SmallVector<slang::SourceRange>>;
  const ReferenceMap &getReferences() const { return references; }

private:
  /// Parse source location in the file.
  void parseSourceLocation();
  void parseSourceLocationOnLine(StringRef line);

  // MLIR context used for generating location attr.
  mlir::MLIRContext mlirContext;

  /// An interval map containing a corresponding definition mapped to a source
  /// interval.
  MapT intervalMap;
  /// An allocator for the interval map.
  MapT::Allocator allocator;

  /// References of symbols.
  ReferenceMap references;

  // The parent document.
  VerilogDocument &document;
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
class VerilogDocument {
public:
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
  mlir::lsp::Location getLspLocation(slang::SourceRange range) const;

  // Return SMLoc from slang location.
  llvm::SMLoc getSMLoc(slang::SourceLocation loc);

  //===--------------------------------------------------------------------===//
  // Definitions and References
  //===--------------------------------------------------------------------===//

  void getLocationsOf(const mlir::lsp::URIForFile &uri,
                      const mlir::lsp::Position &defPos,
                      std::vector<mlir::lsp::Location> &locations);

  void findReferencesOf(const mlir::lsp::URIForFile &uri,
                        const mlir::lsp::Position &pos,
                        std::vector<mlir::lsp::Location> &references);

private:
  std::optional<std::pair<uint32_t, SmallString<128>>>
  getOrOpenFile(StringRef filePath);

  VerilogServerContext &globalContext;

  // A map from slang buffer ID to the corresponding buffer ID in the LLVM
  // source manager.
  llvm::SmallDenseMap<uint32_t, uint32_t> bufferIDMap;

  // A map from a file name to the corresponding buffer ID in the LLVM
  // source manager.
  llvm::StringMap<std::pair<uint32_t, SmallString<128>>> filePathMap;

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
    : globalContext(context), index(*this), uri(uri) {
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
  circt::lsp::Logger::error("bufferID: " +
                            std::to_string(slangBuffer.id.getId()));
  auto diagClient = std::make_shared<LSPDiagnosticClient>(*this, diagnostics);
  driver.diagEngine.addClient(diagClient);

  if (!driver.parseAllSources()) {
    circt::lsp::Logger::error(Twine("Failed to parse Verilog file ") +
                              uri.file());
    return;
  }

  compilation = driver.createCompilation();
  if (failed(compilation)) {
    return;
  }

  for (auto &diag : (*compilation)->getAllDiagnostics())
    driver.diagEngine.issue(diag);

  index.initialize(**compilation);
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

mlir::lsp::Location
VerilogDocument::getLspLocation(slang::SourceRange range) const {

  auto start = getLspLocation(range.start());
  auto end = getLspLocation(range.end());

  if (start.uri != end.uri)
    return mlir::lsp::Location();

  return mlir::lsp::Location(
      start.uri, mlir::lsp::Range(start.range.start, end.range.end));
}

llvm::SMLoc VerilogDocument::getSMLoc(slang::SourceLocation loc) {
  auto bufferID = loc.buffer().getId();
  circt::lsp::Logger::error("bufferID: " + std::to_string(bufferID));

  // Check if the source is already opened by LLVM source manager.
  auto bufferIDMapIt = bufferIDMap.find(bufferID);
  if (bufferIDMapIt == bufferIDMap.end()) {
    // If not, open the source file and add it to the LLVM source manager.
    auto path = getSlangSourceManager().getFullPath(loc.buffer());
    auto memBuffer = llvm::MemoryBuffer::getFile(path.string());
    if (!memBuffer) {
      circt::lsp::Logger::error(
          "Failed to open file: " + path.filename().string() +
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

std::optional<std::pair<uint32_t, SmallString<128>>>
VerilogDocument::getOrOpenFile(StringRef filePath) {

  auto fileInfo = filePathMap.find(filePath);
  if (fileInfo != filePathMap.end())
    return fileInfo->second;

  auto getIfExist = [&](StringRef path)
      -> std::optional<std::pair<uint32_t, SmallString<128>>> {
    if (llvm::sys::fs::exists(path)) {
      auto memoryBuffer = llvm::MemoryBuffer::getFile(path);
      if (!memoryBuffer) {
        return std::nullopt;
      }
      auto id = sourceMgr.AddNewSourceBuffer(std::move(*memoryBuffer), SMLoc());

      fileInfo =
          filePathMap.insert(std::make_pair(filePath, std::make_pair(id, path)))
              .first;

      return fileInfo->second;
    }
    return std::nullopt;
  };

  if (llvm::sys::path::is_absolute(filePath))
    return getIfExist(filePath);

  for (auto &libRoot : globalContext.options.extraSourceLocationDirs) {
    SmallString<128> lib(libRoot);
    llvm::sys::path::append(lib, filePath);
    if (auto fileInfo = getIfExist(lib))
      return fileInfo;
  }

  return std::nullopt;
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

  void getLocationsOf(const mlir::lsp::URIForFile &uri,
                      mlir::lsp::Position defPos,
                      std::vector<mlir::lsp::Location> &locations);

  void findReferencesOf(const mlir::lsp::URIForFile &uri,
                        mlir::lsp::Position pos,
                        std::vector<mlir::lsp::Location> &references);

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

void VerilogTextFile::getLocationsOf(
    const mlir::lsp::URIForFile &uri, mlir::lsp::Position defPos,
    std::vector<mlir::lsp::Location> &locations) {
  document->getLocationsOf(uri, defPos, locations);
}

void VerilogTextFile::findReferencesOf(
    const mlir::lsp::URIForFile &uri, mlir::lsp::Position pos,
    std::vector<mlir::lsp::Location> &references) {
  document->findReferencesOf(uri, pos, references);
}

struct IndexVisitor : slang::ast::ASTVisitor<IndexVisitor, true, true> {
  IndexVisitor(VerilogIndex &index) : index(index) {}
  VerilogIndex &index;

  void handleSymbol(const slang::ast::Symbol *symbol, slang::SourceRange range,
                    bool isDefinition = false) {
    assert(range.start().valid() && range.end().valid());
    insertDeclRef(symbol, range, isDefinition);
  }

  void handleSymbol(const slang::ast::Symbol *symbol,
                    slang::SourceLocation location, bool isDefinition = false) {
    if (symbol->name.empty())
      return;
    handleSymbol(symbol,
                 slang::SourceRange(location, location + symbol->name.length()),
                 isDefinition);
  }

  void visit(const slang::ast::VariableDeclStatement &expr) {
    handleSymbol(&expr.symbol, expr.sourceRange, /*isDefinition=*/true);
    visitDefault(expr);
  }

  // Handle named values, such as references to declared variables.
  void visit(const slang::ast::Expression &expr) {
    auto *symbol = expr.getSymbolReference(true);
    if (!symbol)
      return;
    handleSymbol(symbol, expr.sourceRange, false);
    visitDefault(expr);
  }

  template <typename T>
  void visit(const T &t) {

    visitDefault(t);
  }
  // Helper function to convert an argument to a simple bit vector type, pass it
  // to a reduction op, and optionally invert the result.

  /// Handle assignment patterns.
  void visitInvalid(const slang::ast::Expression &expr) {
    // mlir::lsp::Logger::debug("visitInvalid: {}",
    //                          slang::ast::toString(expr.kind));
  }

  void visitInvalid(const slang::ast::Statement &) {}
  void visitInvalid(const slang::ast::TimingControl &) {}
  void visitInvalid(const slang::ast::Constraint &) {}
  void visitInvalid(const slang::ast::AssertionExpr &) {}
  void visitInvalid(const slang::ast::BinsSelectExpr &) {}
  void visitInvalid(const slang::ast::Pattern &) {}

  VerilogDocument &getDocument() { return index.getDocument(); }

  void insertDeclRef(const slang::ast::Symbol *sym, slang::SourceRange refLoc,
                     bool isDefinition = false) {
    index.insertSymbolUse(sym, refLoc, isDefinition);
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

void VerilogIndex::parseSourceLocationOnLine(StringRef toParse) {
  StringRef filePath, line, column;
  bool first = true;
  // columns := num(,num)*
  // line := num
  // lineCols := line `:` columns | line `:` `{` columns `}`+
  // location := filePath `:` lineCols+
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

    SmallVector<std::tuple<StringRef, StringRef, StringRef>> columns;
    if (column.starts_with('{')) {
      StringRef start = addFile ? filePath : line;
      for (auto str : llvm::split(column.drop_back().drop_front(), ',')) {
        columns.push_back(
            std::make_tuple(str, start, str.drop_front(str.size())));
        start = str.drop_front(str.size());
      }
    } else {
      columns.push_back(
          std::make_tuple(column, addFile ? filePath : line, column.end()));
    }
    for (auto [column, start, end] : columns) {
      uint32_t columnInt;
      if (column.getAsInteger(10, columnInt))
        continue;
      auto loc =
          mlir::FileLineColRange::get(&mlirContext, filePath, lineInt - 1,
                                      columnInt - 1, lineInt - 1, columnInt);
      intervalMap.insert(start.data(), end.data(), loc);
    }
  }
}

void VerilogIndex::parseSourceLocation() {
  auto &sourceMgr = getDocument().getSourceMgr();
  auto *getMainBuffer = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  StringRef text(getMainBuffer->getBufferStart(),
                 getMainBuffer->getBufferSize());

  while (true) {
    // Find the source location from the text.
    StringRef start = "// @[";
    auto loc = text.find(start);
    if (loc == StringRef::npos)
      break;

    text = text.drop_front(loc + start.size());
    auto end = text.find_first_of(']');
    if (end == StringRef::npos)
      break;
    auto toParse = text.take_front(end);
    parseSourceLocationOnLine(toParse);
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

void VerilogIndex::insertSymbolUse(const slang::ast::Symbol *symbol,
                                   slang::SourceRange from, bool isDefinition) {
  assert(from.start().valid() && from.end().valid());
  const char *startLoc = getDocument().getSMLoc(from.start()).getPointer();
  const char *endLoc = getDocument().getSMLoc(from.end()).getPointer();
  if (!startLoc || !endLoc)
    return;
  assert(startLoc && endLoc);

  if (startLoc != endLoc && !intervalMap.overlaps(startLoc, endLoc)) {
    intervalMap.insert(startLoc, endLoc, symbol);
    if (!isDefinition)
      references[symbol].push_back(from);
  }
}
void VerilogIndex::insertSymbolUse(const slang::ast::Symbol *symbol,
                                   bool isDefinition) {
  if (!symbol->location)
    return;
  auto size = symbol->name.size() ? symbol->name.size() : 1;
  insertSymbolUse(
      symbol, slang::SourceRange(symbol->location, symbol->location + size));
}

//===----------------------------------------------------------------------===//
// VerilogDocument: Definitions and References
//===----------------------------------------------------------------------===//

static mlir::lsp::Range getRange(const mlir::FileLineColRange &fileLoc) {
  return mlir::lsp::Range(
      mlir::lsp::Position(fileLoc.getStartLine(), fileLoc.getStartColumn()),
      mlir::lsp::Position(fileLoc.getEndLine(), fileLoc.getEndColumn()));
}

void VerilogDocument::getLocationsOf(
    const mlir::lsp::URIForFile &uri, const mlir::lsp::Position &defPos,
    std::vector<mlir::lsp::Location> &locations) {
  SMLoc posLoc = defPos.getAsSMLoc(sourceMgr);
  const auto &intervalMap = index.getIntervalMap();
  auto it = intervalMap.find(posLoc.getPointer());
  if (it == intervalMap.end())
    return;

  auto element = it.value();
  if (auto attr = dyn_cast<Attribute>(element)) {
    // Check if the attribute is a FileLineColRange.
    if (auto fileLoc = dyn_cast<mlir::FileLineColRange>(attr)) {
      // Return URI for the file.
      auto fileInfo = getOrOpenFile(fileLoc.getFilename().getValue());
      if (!fileInfo)
        return;
      const auto &[bufferId, filePath] = *fileInfo;
      auto uri = mlir::lsp::URIForFile::fromFile(filePath);
      if (auto e = uri.takeError()) {
        circt::lsp::Logger::error("failed to open file " + filePath);
        return;
      }

      mlir::lsp::Location loc(uri.get(), getRange(fileLoc));
      locations.push_back(loc);
    }

    return;
  }

  // If the element is verilog symbol, return the definition of the symbol.
  const auto *symbol = cast<const slang::ast::Symbol *>(element);
  locations.push_back(getLspLocation(symbol->location));
}

void VerilogDocument::findReferencesOf(
    const mlir::lsp::URIForFile &uri, const mlir::lsp::Position &pos,
    std::vector<mlir::lsp::Location> &references) {
  SMLoc posLoc = pos.getAsSMLoc(sourceMgr);
  const auto &intervalMap = index.getIntervalMap();
  auto intervalIt = intervalMap.find(posLoc.getPointer());
  if (intervalIt == intervalMap.end())
    return;

  const auto *symbol = cast<const slang::ast::Symbol *>(intervalIt.value());
  if (!symbol)
    return;

  auto it = index.getReferences().find(symbol);
  if (it == index.getReferences().end())
    return;
  for (auto refLoc : it->second)
    references.push_back(getLspLocation(refLoc));
}
