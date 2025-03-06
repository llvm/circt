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
#include "../Protocol.h"
#include "../Utils/LSPUtils.h"

#include "circt/Support/LLVM.h"
#include "circt/Tools/circt-verilog-lsp-server/CirctVerilogLspServerMain.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/lsp-server-support/Protocol.h"
#include "mlir/Tools/lsp-server-support/SourceMgrUtils.h"
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
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <variant>
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

template <typename T>
static void emitHint(llvm::raw_ostream &os, T symbol, bool includeName = true) {
  if (includeName)
    os << symbol->name << ": ";
  os << directionToString(symbol->direction) << " "
     << symbol->getType().toString() << ":";
}

// ===----------------------------------------------------------------------===//
// Markup Helpers
// ===----------------------------------------------------------------------===//

static llvm::raw_ostream &addHeading(llvm::raw_ostream &os, int level,
                                     StringRef heading) {
  for (int i = 0; i < level; ++i)
    os << '#';
  os << ' ' << heading << '\n';
  return os;
}

static llvm::raw_ostream &addCodeBlock(llvm::raw_ostream &os,
                                       StringRef language, StringRef content) {
  os << "```" << language << '\n';
  circt::lsp::printReindented(os, content);
  if (!content.ends_with('\n'))
    os << '\n';
  os << "```\n";
  return os;
}

static llvm::raw_ostream &addLink(llvm::raw_ostream &os, StringRef label,
                                  StringRef uri, int line) {
  os << "[" << label << "](" << uri << "#L" << line << ")";
  return os;
}

static llvm::raw_ostream &addRuler(llvm::raw_ostream &os) {
  os << "\n***\n";
  return os;
}

namespace {

// A global context carried around by the server.
struct VerilogServerContext {
  VerilogServerContext(const VerilogServerOptions &options)
      : options(options) {}
  const VerilogServerOptions &options;

  struct ObjectInlayHint {
    std::string value;
    std::optional<std::string> group;
    ObjectInlayHint(StringRef value, std::optional<std::string> group)
        : value(value), group(std::move(group)){};
  };

  // A map from module name and symbol name to the hint.
  llvm::StringMap<llvm::StringMap<llvm::SmallVector<ObjectInlayHint>>>
      inlayHintMappings;

  // Return user provided inlay hints for the given symbol.
  ArrayRef<ObjectInlayHint> getInlayHintsForSymbol(StringRef moduleName,
                                                   StringRef symbolName) const;

  void putInlayHintsOnObjects(
      const std::vector<VerilogUserProvidedInlayHint> &hints);

private:
  void putInlayHintsOnObjects(StringRef moduleName, StringRef symbolName,
                              StringRef hint,
                              const std::optional<std::string> &group);
};

class VerilogDocument;
using VerilogIndexSymbol =
    llvm::PointerUnion<const slang::ast::Symbol *, mlir::Attribute>;

class VerilogIndex {
public:
  VerilogIndex(VerilogDocument &document)
      : mlirContext(mlir::MLIRContext::Threading::DISABLED),
        intervalMap(allocator), document(document) {}

  /// Initialize the index with the given compilation unit.
  void initialize(slang::ast::Compilation &compilation);

  /// Register a reference to a symbol `symbol` from `from`.
  void insertSymbol(const slang::ast::Symbol *symbol, slang::SourceRange from,
                    bool isDefinition = false);
  void insertSymbolDefinition(const slang::ast::Symbol *symbol);

  VerilogDocument &getDocument() { return document; }

  /// The type of interval map used to store source references. SMRange is
  /// half-open, so we also need to use a half-open interval map.
  using MapT =
      llvm::IntervalMap<const char *, VerilogIndexSymbol,
                        llvm::IntervalMapImpl::NodeSizer<
                            const char *, VerilogIndexSymbol>::LeafSize,
                        llvm::IntervalMapHalfOpenInfo<const char *>>;

  /// A mapping from a symbol to their references.
  using ReferenceMap = SmallDenseMap<const slang::ast::Symbol *,
                                     SmallVector<slang::SourceRange>>;
  const ReferenceMap &getReferences() const { return references; }

  std::optional<VerilogIndexSymbol> lookup(SMLoc loc, SMRange *range = nullptr);

private:
  /// Parse source location emitted by ExportVerilog.
  void parseSourceLocation();
  void parseSourceLocation(StringRef toParse);

  // MLIR context used for generating location attr.
  mlir::MLIRContext mlirContext;

  /// An allocator for the interval map.
  MapT::Allocator allocator;

  /// An interval map containing a corresponding definition mapped to a source
  /// interval.
  MapT intervalMap;

  /// References of symbols.
  ReferenceMap references;

  // The parent document.
  VerilogDocument &document;
};

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

  const slang::SourceManager &getSlangSourceManager() const {
    return driver.sourceManager;
  }

  const VerilogServerContext &getGlobalContext() { return globalContext; }

  // Return LSP location from slang location.
  mlir::lsp::Location getLspLocation(slang::SourceLocation loc) const;
  mlir::lsp::Location getLspLocation(slang::SourceRange range) const;

  // Return SMLoc from slang location.
  llvm::SMLoc getSMLoc(slang::SourceLocation loc);

  // Return the URI for the given location.
  std::optional<mlir::lsp::URIForFile>
  getExternalURI(mlir::FileLineColRange loc);

  // Return the source line containing the given location.
  StringRef getSourceLine(slang::SourceLocation loc);

  // Check if the given location is in the main file.
  bool isInMainFile(slang::SourceLocation loc) const;

  // Get or open a file and return the buffer ID and path.
  std::optional<std::pair<uint32_t, SmallString<128>>>
  getOrOpenFile(StringRef filePath);

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
  // Inlay Hints
  //===--------------------------------------------------------------------===//

  void getInlayHints(const mlir::lsp::URIForFile &uri,
                     const mlir::lsp::Range &range,
                     std::vector<mlir::lsp::InlayHint> &inlayHints);

  //===--------------------------------------------------------------------===//
  // Hover
  //===--------------------------------------------------------------------===//

  std::optional<mlir::lsp::Hover>
  findHover(const mlir::lsp::URIForFile &uri,
            const mlir::lsp::Position &hoverPos);
  void buildHoverForSymbol(const slang::ast::Symbol *symbol,
                           llvm::raw_ostream &os);
  void buildHoverForLoc(mlir::FileLineColRange loc, llvm::raw_ostream &os);

  // Return source text for hover from an external file.
  StringRef getExternalSourceTextForHover(mlir::FileLineColRange loc);

private:
  // ===-------------------------------------------------------------------===//
  // Fields
  // ===-------------------------------------------------------------------===//

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
  mlir::lsp::URIForFile uri;
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
  auto &mlirDiag = diags.emplace_back();
  mlirDiag.severity = getSeverity(slangDiag.severity);
  mlirDiag.range = loc.range;
  mlirDiag.source = "slang";
  mlirDiag.message = slangDiag.formattedMessage;
}

// ===----------------------------------------------------------------------===//
// VerilogServerContext
//===----------------------------------------------------------------------===//

void VerilogServerContext::putInlayHintsOnObjects(
    const std::vector<VerilogUserProvidedInlayHint> &hints) {
  SmallVector<StringRef, 4> path;
  for (const auto &hint : hints) {
    StringRef root = hint.root ? StringRef(*hint.root) : StringRef();
    path.clear();
    StringRef(hint.path).split(path, '.');
    if (path.size() == 1 && !root.empty()) {
      putInlayHintsOnObjects(root, path[0], hint.value, hint.group);
    } else {
      // Currently not supported.
      circt::lsp::Logger::error("Currently only support hints with root module "
                                "and not nested hints.");
    }
  }
}

ArrayRef<VerilogServerContext::ObjectInlayHint>
VerilogServerContext::getInlayHintsForSymbol(StringRef moduleName,
                                             StringRef symbolName) const {
  auto moduleIt = inlayHintMappings.find(moduleName);
  if (moduleIt == inlayHintMappings.end())
    return {};
  auto symbolIt = moduleIt->second.find(symbolName);
  if (symbolIt == moduleIt->second.end())
    return {};
  return symbolIt->second;
}

void VerilogServerContext::putInlayHintsOnObjects(
    StringRef moduleName, StringRef symbolName, StringRef hint,
    const std::optional<std::string> &group) {

  auto &hints = inlayHintMappings[moduleName][symbolName];
  // If the group is not provided, just append.
  if (!group) {
    hints.emplace_back(hint, group);
    return;
  }

  // If the id is provided, check if we need to update hints.
  for (auto &existingHint : hints) {
    if (existingHint.group == *group) {
      existingHint.value = hint;
      return;
    }
  }

  hints.emplace_back(hint, group);
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
  if (failed(compilation))
    return;

  for (auto &diag : (*compilation)->getAllDiagnostics())
    driver.diagEngine.issue(diag);

  // Populate the index.
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

  // Search locations.
  for (auto &libRoot : globalContext.options.extraSourceLocationDirs) {
    SmallString<128> lib(libRoot);
    llvm::sys::path::append(lib, filePath);
    if (auto fileInfo = getIfExist(lib))
      return fileInfo;
  }

  return std::nullopt;
}

bool VerilogDocument::isInMainFile(slang::SourceLocation loc) const {
  auto it = bufferIDMap.find(loc.buffer().getId());
  return it != bufferIDMap.end() && it->second == sourceMgr.getMainFileID();
}

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

  void getInlayHints(const mlir::lsp::URIForFile &uri,
                     const mlir::lsp::Range &range,
                     std::vector<mlir::lsp::InlayHint> &inlayHints);

  std::optional<mlir::lsp::Hover> findHover(const mlir::lsp::URIForFile &uri,
                                            mlir::lsp::Position pos);

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

void VerilogTextFile::getInlayHints(
    const mlir::lsp::URIForFile &uri, const mlir::lsp::Range &range,
    std::vector<mlir::lsp::InlayHint> &inlayHints) {
  document->getInlayHints(uri, range, inlayHints);
}

namespace {

// Index the AST to find symbol uses and definitions.
struct VerilogIndexer : slang::ast::ASTVisitor<VerilogIndexer, true, true> {
  VerilogIndexer(VerilogIndex &index) : index(index) {}
  VerilogIndex &index;

  void insertSymbol(const slang::ast::Symbol *symbol, slang::SourceRange range,
                    bool isDefinition = true) {
    if (symbol->name.empty())
      return;
    assert(range.start().valid() && range.end().valid() &&
           "range must be valid");
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
    visitDefault(expr);
  }

  void visitSymbol(const slang::ast::Symbol &symbol) {
    insertSymbol(&symbol, symbol.location, /*isDefinition=*/true);
    visitDefault(symbol);
  }

  void visit(const slang::ast::NetSymbol &expr) {
    insertSymbol(&expr, expr.location, /*isDefinition=*/true);
    visitDefault(expr);
  }

  void visit(const slang::ast::VariableSymbol &expr) {
    insertSymbol(&expr, expr.location, /*isDefinition=*/true);
    visitDefault(expr);
  }

  void visit(const slang::ast::VariableDeclStatement &expr) {
    insertSymbol(&expr.symbol, expr.sourceRange, /*isDefinition=*/true);
    visitDefault(expr);
  }

  template <typename T>
  void visit(const T &t) {
    if constexpr (std::is_base_of_v<slang::ast::Expression, T>)
      visitExpression(t);
    if constexpr (std::is_base_of_v<slang::ast::Symbol, T>)
      visitSymbol(t);

    visitDefault(t);
  }

  template <typename T>
  void visitInvalid(const T &t) {}
};
} // namespace

void VerilogIndex::initialize(slang::ast::Compilation &compilation) {
  const auto &root = compilation.getRoot();
  VerilogIndexer visitor(*this);
  for (auto *inst : root.topInstances) {
    // Visit the body of the instance.
    inst->body.visit(visitor);

    // Insert the symbols in the port list.
    for (const auto *symbol : inst->body.getPortList())
      insertSymbolDefinition(symbol);
  }

  // Parse the source location from the main file.
  parseSourceLocation();
}

std::optional<VerilogIndexSymbol> VerilogIndex::lookup(SMLoc loc,
                                                       SMRange *range) {
  auto it = intervalMap.find(loc.getPointer());
  if (!it.valid() || loc.getPointer() < it.start())
    return std::nullopt;

  if (range)
    *range = SMRange(SMLoc::getFromPointer(it.start()),
                     SMLoc::getFromPointer(it.stop()));
  return it.value();
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
  auto &sourceMgr = getDocument().getSourceMgr();
  auto *getMainBuffer = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  StringRef text(getMainBuffer->getBufferStart(),
                 getMainBuffer->getBufferSize());

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
    auto end = text.find_first_of(']');
    if (end == StringRef::npos)
      continue;
    auto toParse = text.take_front(end);
    parseSourceLocation(toParse);
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

void circt::lsp::VerilogServer::getInlayHints(
    const URIForFile &uri, const mlir::lsp::Range &range,
    std::vector<mlir::lsp::InlayHint> &inlayHints) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    fileIt->second->getInlayHints(uri, range, inlayHints);
}

void VerilogIndex::insertSymbol(const slang::ast::Symbol *symbol,
                                slang::SourceRange from, bool isDefinition) {
  assert(from.start().valid() && from.end().valid());
  const char *startLoc = getDocument().getSMLoc(from.start()).getPointer();
  const char *endLoc = getDocument().getSMLoc(from.end()).getPointer() + 1;
  if (!startLoc || !endLoc)
    return;
  assert(startLoc && endLoc);

  if (startLoc != endLoc && !intervalMap.overlaps(startLoc, endLoc)) {
    intervalMap.insert(startLoc, endLoc, symbol);
    if (!isDefinition)
      references[symbol].push_back(from);
  }
}

void VerilogIndex::insertSymbolDefinition(const slang::ast::Symbol *symbol) {
  if (!symbol->location)
    return;
  auto size = symbol->name.size() ? symbol->name.size() : 1;
  insertSymbol(symbol,
               slang::SourceRange(symbol->location, symbol->location + size),
               true);
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
  auto it = index.lookup(posLoc);
  if (!it)
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

      locations.emplace_back(uri.get(), getRange(fileLoc));
    }

    return;
  }

  // If the element is verilog symbol, return the definition of the symbol.
  const auto *symbol = cast<const slang::ast::Symbol *>(element);

  circt::lsp::Logger::info(std::to_string(defPos.line) + ":" +
                           std::to_string(defPos.character));

  slang::SourceRange range(symbol->location,
                           symbol->location +
                               (symbol->name.size() ? symbol->name.size() : 1));
  locations.push_back(getLspLocation(range));
}

void VerilogDocument::findReferencesOf(
    const mlir::lsp::URIForFile &uri, const mlir::lsp::Position &pos,
    std::vector<mlir::lsp::Location> &references) {
  SMLoc posLoc = pos.getAsSMLoc(sourceMgr);
  auto element = index.lookup(posLoc);
  if (!element)
    return;

  const auto *symbol = dyn_cast<const slang::ast::Symbol *>(*element);
  if (!symbol)
    return;

  auto it = index.getReferences().find(symbol);
  if (it == index.getReferences().end())
    return;
  for (auto referenceRange : it->second)
    references.push_back(getLspLocation(referenceRange));
}

//===----------------------------------------------------------------------===//
// VerilogDocument: Inlay Hints
//===----------------------------------------------------------------------===//

namespace {
struct InlayHintVisitor : slang::ast::ASTVisitor<InlayHintVisitor, true, true> {
  InlayHintVisitor(VerilogDocument &document, SMRange range,
                   std::vector<mlir::lsp::InlayHint> &inlayHints)
      : document(document), range(range), inlayHints(inlayHints) {}

  // Check if the given location is within the range.
  bool contains(slang::SourceLocation loc);

  // Inlay hint for the function call.
  void visit(const slang::ast::CallExpression &expr);
  void visitCall(const slang::ast::CallExpression &expr,
                 const slang::ast::SubroutineSymbol *subroutine);
  void visitCall(const slang::ast::CallExpression &expr,
                 const slang::ast::CallExpression::SystemCallInfo &info);

  // Inlay hint for the module instantiation.
  void visit(const slang::ast::InstanceSymbol &instance);

  // Visitors for user provided inlay hints.
  void visitExpression(const slang::ast::Expression &expr);
  void visitSymbol(const slang::ast::Symbol &symbol);
  void visitSymbolUse(const slang::ast::Symbol &symbol,
                      slang::SourceLocation loc);

  template <typename T>
  void visit(const T &t) {
    if constexpr (std::is_base_of_v<slang::ast::Expression, T>)
      visitExpression(t);
    if constexpr (std::is_base_of_v<slang::ast::Symbol, T>)
      visitSymbol(t);
    visitDefault(t);
  }

  template <typename T>
  void visitInvalid(const T &t) {}

private:
  VerilogDocument &document;
  SMRange range;

  // Result of inlay hints.
  std::vector<mlir::lsp::InlayHint> &inlayHints;

  // Visited symbols to avoid duplicate hints.
  llvm::StringSet<> visitedSymbols;
};

} // namespace

bool InlayHintVisitor::contains(slang::SourceLocation loc) {
  if (!loc.valid() || !document.isInMainFile(loc))
    return false;
  return mlir::lsp::contains(range, document.getSMLoc(loc));
}

void InlayHintVisitor::visit(const slang::ast::CallExpression &expr) {
  std::visit([&](auto &subroutine) { return visitCall(expr, subroutine); },
             expr.subroutine);
  visitDefault(expr);
}

void InlayHintVisitor::visitCall(
    const slang::ast::CallExpression &expr,
    const slang::ast::SubroutineSymbol *subroutine) {
  for (auto [operand, arg] :
       llvm::zip(expr.arguments(), subroutine->getArguments())) {
    if (!contains(operand->sourceRange.start()))
      continue;
    // Add an inlay hint for the beginning of the operand.
    auto &hint = inlayHints.emplace_back(
        mlir::lsp::InlayHintKind::Type,
        document.getLspLocation(operand->sourceRange.start()).range.start);
    llvm::raw_string_ostream os(hint.label);
    emitHint(os, arg);
    hint.paddingRight = true;
  }
}

void InlayHintVisitor::visitCall(
    const slang::ast::CallExpression &expr,
    const slang::ast::CallExpression::SystemCallInfo &info) {
  // TODO: Implement this.
}

void InlayHintVisitor::visit(const slang::ast::InstanceSymbol &instance) {
  for (const auto &connect : instance.getPortConnections()) {
    auto *portSymbol = connect->port.as_if<slang::ast::PortSymbol>();
    if (portSymbol && connect->getExpression()) {
      auto loc = connect->getExpression()->sourceRange;
      if (!contains(loc.start()))
        continue;

      // Add an inlay hint for the beginning of the instance port connections.
      auto &hint = inlayHints.emplace_back(
          mlir::lsp::InlayHintKind::Type,
          document.getLspLocation(loc.start()).range.start);
      llvm::raw_string_ostream os(hint.label);
      emitHint(os, portSymbol, /*includeName=*/false);
      hint.paddingRight = true;
    }
  }
  visitDefault(instance);
}

void InlayHintVisitor::visitSymbol(const slang::ast::Symbol &symbol) {
  if (!visitedSymbols.insert(symbol.name).second)
    return;
  // Hint is added after the symbol name.
  visitSymbolUse(symbol, symbol.location + symbol.name.size());
}

void InlayHintVisitor::visitExpression(const slang::ast::Expression &expr) {
  auto *symbol = expr.getSymbolReference(true);
  if (!symbol)
    return;

  visitSymbolUse(*symbol, expr.sourceRange.end());
}

// Attach user provided inlay hints for each symbol use.
void InlayHintVisitor::visitSymbolUse(const slang::ast::Symbol &symbol,
                                      slang::SourceLocation loc) {
  if (!contains(loc))
    return;

  // Find the enclosing module name.
  auto *instance = symbol.getParentScope()->getContainingInstance();
  if (!instance)
    return;

  // Check user provided inlay hints for the symbol.
  auto hints = document.getGlobalContext().getInlayHintsForSymbol(
      instance->name, symbol.name);
  if (hints.empty())
    return;

  // Add user provided inlay hints for the location.
  mlir::lsp::Position pos(document.getSourceMgr(), document.getSMLoc(loc));
  auto &inlayHint =
      inlayHints.emplace_back(mlir::lsp::InlayHintKind::Parameter, pos);
  inlayHint.paddingLeft = true;
  bool first = true;
  for (auto &[hint, _] : hints) {
    // Use '/' as a separator for multiple user provided inlay hints.
    if (!first)
      inlayHint.label += '/';
    first = false;
    inlayHint.label += hint;
  }
}

void VerilogDocument::getInlayHints(
    const mlir::lsp::URIForFile &uri, const mlir::lsp::Range &range,
    std::vector<mlir::lsp::InlayHint> &inlayHints) {
  if (failed(compilation))
    return;
  SMRange rangeLoc = range.getAsSMRange(sourceMgr);

  if (!rangeLoc.isValid())
    return;

  InlayHintVisitor visitor(*this, rangeLoc, inlayHints);
  for (auto *inst : compilation.value()->getRoot().topInstances)
    inst->body.visit(visitor);
}

// ===----------------------------------------------------------------------===//
// VerilogDocument: Hover
// ===----------------------------------------------------------------------===//

void VerilogDocument::buildHoverForSymbol(const slang::ast::Symbol *symbol,
                                          llvm::raw_ostream &os) {
  if (auto *type = symbol->getDeclaredType())
    addHeading(os, 3, "Type") << "`" << type->getType().toString() << "`";
  addRuler(os);
  addHeading(os, 3, "Definition");
  addCodeBlock(os, "verilog", getSourceLine(symbol->location));
  auto loc = getLspLocation(symbol->location);
  addLink(os, "Go To Definition", loc.uri.uri(),
          getSlangSourceManager().getLineNumber(symbol->location));
  addRuler(os);
}

void VerilogDocument::buildHoverForLoc(mlir::FileLineColRange loc,
                                       llvm::raw_ostream &os) {
  auto content = getExternalSourceTextForHover(loc);
  if (content.empty())
    return;
  auto filename = loc.getFilename();
  auto ext = llvm::sys::path::extension(filename.getValue());
  addHeading(os, 3, "External Source");
  addCodeBlock(os, ext.drop_front(), content);
  auto sourceURI = getExternalURI(loc);
  if (!sourceURI)
    return;
  addLink(os, "Go To External Source", sourceURI->uri(), loc.getStartLine());
}

std::optional<mlir::lsp::Hover>
VerilogDocument::findHover(const mlir::lsp::URIForFile &uri,
                           const mlir::lsp::Position &hoverPos) {
  SMLoc posLoc = hoverPos.getAsSMLoc(sourceMgr);
  SMRange smRange;
  auto it = index.lookup(posLoc, &smRange);

  // Found no element in the given index.
  if (!it)
    return {};

  mlir::lsp::Hover hover(mlir::lsp::Range(sourceMgr, smRange));
  llvm::raw_string_ostream strOs(hover.contents.value);
  if (auto *symbol = dyn_cast<const slang::ast::Symbol *>(it.value())) {
    buildHoverForSymbol(symbol, strOs);
  } else {
    auto loc = cast<Attribute>(it.value());
    if (auto fileLoc = dyn_cast<mlir::FileLineColRange>(loc))
      buildHoverForLoc(fileLoc, strOs);
  }
  return hover;
}

/// Get the URI for an external file location.
/// Returns the URI if the file can be opened and converted, otherwise returns
/// nullopt.
std::optional<mlir::lsp::URIForFile>
VerilogDocument::getExternalURI(mlir::FileLineColRange loc) {
  auto file = loc.getFilename();
  auto fileInfo = getOrOpenFile(file);
  if (!fileInfo)
    return std::nullopt;
  auto *buffer = sourceMgr.getMemoryBuffer(fileInfo->first);

  auto uri = mlir::lsp::URIForFile::fromFile(buffer->getBufferIdentifier());
  if (auto e = uri.takeError())
    return std::nullopt;
  return uri.get();
}

/// Get the source line containing the given location.
/// Returns the trimmed line of text containing the location.
StringRef VerilogDocument::getSourceLine(slang::SourceLocation loc) {
  auto text = getSlangSourceManager().getSourceText(loc.buffer());
  auto offest = loc.offset();
  // Find line boundaries around the location
  auto start = text.find_last_of('\n', offest);
  auto end = text.find_first_of('\n', offest);
  return StringRef(text.substr(start, end - start)).trim();
}

/// Get source text from an external file for hover information.
/// Returns a snippet of text around the given location based on hover context
/// settings. The snippet includes lines before and after based on
/// `hoverContextLineCount` option.
StringRef
VerilogDocument::getExternalSourceTextForHover(mlir::FileLineColRange loc) {
  auto &sgr = getSourceMgr();
  auto fileInfo = getOrOpenFile(loc.getFilename());
  if (!fileInfo)
    return StringRef();

  auto [bufferId, filePath] = *fileInfo;
  auto *buffer = sgr.getMemoryBuffer(bufferId);
  int32_t startLine = loc.getStartLine();

  // Find starting location accounting for context lines before
  auto startLoc = sgr.FindLocForLineAndColumn(
      bufferId,
      std::max(1, startLine - globalContext.options.hoverContextLineCount), 1);
  StringRef content(startLoc.getPointer(),
                    buffer->getBufferEnd() - startLoc.getPointer());

  // Extract the desired number of lines after the start location
  for (auto i = 0;
       i < globalContext.options.hoverContextLineCount && !content.empty();
       ++i) {
    auto firstLine = content.find_first_of('\n');
    content = content.drop_front(firstLine + 1);
  }

  // Return the extracted snippet
  return StringRef(startLoc.getPointer(),
                   content.data() - startLoc.getPointer());
}

std::optional<mlir::lsp::Hover>
VerilogTextFile::findHover(const mlir::lsp::URIForFile &uri,
                           mlir::lsp::Position hoverPos) {
  return document->findHover(uri, hoverPos);
}

void VerilogServer::putInlayHintsOnObjects(
    const std::vector<VerilogUserProvidedInlayHint> &params) {
  impl->context.putInlayHintsOnObjects(params);
}

std::optional<mlir::lsp::Hover>
circt::lsp::VerilogServer::findHover(const URIForFile &uri,
                                     const Position &hoverPos) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    return fileIt->second->findHover(uri, hoverPos);
  return std::nullopt;
}
