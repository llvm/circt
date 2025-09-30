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
#include "mlir/Support/FileUtilities.h"
#include "slang/ast/ASTVisitor.h"
#include "slang/ast/Compilation.h"
#include "slang/ast/Scope.h"
#include "slang/ast/symbols/CompilationUnitSymbols.h"
#include "slang/diagnostics/DiagnosticClient.h"
#include "slang/diagnostics/Diagnostics.h"
#include "slang/driver/Driver.h"
#include "slang/syntax/AllSyntax.h"
#include "slang/syntax/SyntaxTree.h"
#include "slang/text/SourceLocation.h"
#include "slang/text/SourceManager.h"
#include "slang/util/Enum.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LSP/Logging.h"
#include "llvm/Support/LSP/Protocol.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>

using namespace llvm::lsp;

using namespace mlir;

using namespace circt::lsp;
using namespace circt;

static llvm::lsp::DiagnosticSeverity
getSeverity(slang::DiagnosticSeverity severity) {
  switch (severity) {
  case slang::DiagnosticSeverity::Fatal:
  case slang::DiagnosticSeverity::Error:
    return llvm::lsp::DiagnosticSeverity::Error;
  case slang::DiagnosticSeverity::Warning:
    return llvm::lsp::DiagnosticSeverity::Warning;
  case slang::DiagnosticSeverity::Ignored:
  case slang::DiagnosticSeverity::Note:
    return llvm::lsp::DiagnosticSeverity::Information;
  }
  llvm_unreachable("all slang diagnostic severities should be handled");
  return llvm::lsp::DiagnosticSeverity::Error;
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
      : mlirContext(mlir::MLIRContext::Threading::DISABLED),
        intervalMap(allocator), document(document) {}

  /// Initialize the index with the given compilation unit.
  void initialize(slang::ast::Compilation &compilation);

  /// Register a reference to a symbol `symbol` from `from`.
  void insertSymbol(const slang::ast::Symbol *symbol, slang::SourceRange from,
                    bool isDefinition = false);
  void insertSymbolDefinition(const slang::ast::Symbol *symbol);

  VerilogDocument &getDocument() { return document; }

  // Half open interval map containing paired pointers into Slang's buffer
  using SlangBufferPointer = char const *;
  using MapT = llvm::IntervalMap<
      SlangBufferPointer, VerilogIndexSymbol,
      llvm::IntervalMapImpl::NodeSizer<SlangBufferPointer,
                                       VerilogIndexSymbol>::LeafSize,
      llvm::IntervalMapHalfOpenInfo<const SlangBufferPointer>>;

  MapT &getIntervalMap() { return intervalMap; }

  /// A mapping from a symbol to their references.
  using ReferenceMap = SmallDenseMap<const slang::ast::Symbol *,
                                     SmallVector<slang::SourceRange>>;
  const ReferenceMap &getReferences() const { return references; }

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
                  const llvm::lsp::URIForFile &uri, StringRef contents,
                  std::vector<llvm::lsp::Diagnostic> &diagnostics);
  VerilogDocument(const VerilogDocument &) = delete;
  VerilogDocument &operator=(const VerilogDocument &) = delete;

  const llvm::lsp::URIForFile &getURI() const { return uri; }

  const slang::SourceManager &getSlangSourceManager() const {
    return driver.sourceManager;
  }

  // Return LSP location from slang location.
  llvm::lsp::Location getLspLocation(slang::SourceLocation loc) const;
  llvm::lsp::Location getLspLocation(slang::SourceRange range) const;

  slang::BufferID getMainBufferID() const { return mainBufferId; }

  //===--------------------------------------------------------------------===//
  // Definitions and References
  //===--------------------------------------------------------------------===//

  void getLocationsOf(const llvm::lsp::URIForFile &uri,
                      const llvm::lsp::Position &defPos,
                      std::vector<llvm::lsp::Location> &locations);

  void findReferencesOf(const llvm::lsp::URIForFile &uri,
                        const llvm::lsp::Position &pos,
                        std::vector<llvm::lsp::Location> &references);

  std::optional<uint32_t> lspPositionToOffset(const llvm::lsp::Position &pos);
  const char *getPointerFor(const llvm::lsp::Position &pos);

private:
  std::optional<std::pair<slang::BufferID, SmallString<128>>>
  getOrOpenFile(StringRef filePath);

  VerilogServerContext &globalContext;

  slang::BufferID mainBufferId;

  // A map from a file name to the corresponding buffer ID in the LLVM
  // source manager.
  llvm::StringMap<std::pair<slang::BufferID, SmallString<128>>> filePathMap;

  // The compilation result.
  FailureOr<std::unique_ptr<slang::ast::Compilation>> compilation;

  // The slang driver.
  slang::driver::Driver driver;

  /// The index of the parsed module.
  VerilogIndex index;

  /// The precomputed line offsets for faster lookups
  std::vector<uint32_t> lineOffsets;
  void computeLineOffsets(std::string_view text);

  // The URI of the document.
  llvm::lsp::URIForFile uri;
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
  std::vector<llvm::lsp::Diagnostic> &diags;

public:
  LSPDiagnosticClient(const VerilogDocument &document,
                      std::vector<llvm::lsp::Diagnostic> &diags)
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

//===----------------------------------------------------------------------===//
// VerilogDocument
//===----------------------------------------------------------------------===//

static std::filesystem::path
canonicalizeFileName(const std::filesystem::path &file) {
  std::error_code ec;
  std::filesystem::path path = std::filesystem::weakly_canonical(file, ec);
  if (ec)
    path = std::filesystem::absolute(file).lexically_normal();
  return path;
}

// Filter out the main buffer file from the command file list, if it is in
// there.
static inline bool
mainBufferFileInCommandFileList(const std::string &cmdfileStr,
                                const std::string &targetAbsStr) {
  const std::filesystem::path targetAbs =
      canonicalizeFileName(std::filesystem::path(targetAbsStr));

  std::string error;
  auto cmdFile = mlir::openInputFile(cmdfileStr, &error);
  if (!cmdFile) {
    circt::lsp::Logger::error(Twine("Failed to open command file ") +
                              cmdfileStr + ": " + error);
    return false;
  }

  const std::filesystem::path base =
      std::filesystem::path(cmdFile->getBufferIdentifier().str()).parent_path();

  // Read line by line, ignoring empty lines and comments.
  for (llvm::line_iterator i(*cmdFile); !i.is_at_eof(); ++i) {
    llvm::StringRef line = i->trim();

    if (line.empty())
      continue;

    static constexpr llvm::StringRef commandPrefixes[] = {"+", "-"};
    auto isCommand = [&line](llvm::StringRef s) { return line.starts_with(s); };
    if (llvm::any_of(commandPrefixes, isCommand))
      continue;

    auto candRel = std::filesystem::path(line.str());
    auto candAbs = canonicalizeFileName(
        candRel.is_absolute() ? candRel : (base / candRel));

    if (candAbs == targetAbs)
      return true;
  }
  return false;
}

VerilogDocument::VerilogDocument(
    VerilogServerContext &context, const llvm::lsp::URIForFile &uri,
    StringRef contents, std::vector<llvm::lsp::Diagnostic> &diagnostics)
    : globalContext(context), index(*this), uri(uri) {
  bool skipMainBufferSlangImport = false;

  llvm::SmallString<256> canonPath(uri.file());
  if (std::error_code ec = llvm::sys::fs::real_path(uri.file(), canonPath))
    canonPath = uri.file(); // fall back, but try to keep it absolute

  // --- Apply project command files (the “-C”s) to this per-buffer driver ---
  for (const std::string &cmdFile : context.options.commandFiles) {
    if (!driver.processCommandFiles(cmdFile, false, true)) {
      circt::lsp::Logger::error(Twine("Failed to open command file ") +
                                cmdFile);
    }
    skipMainBufferSlangImport |=
        mainBufferFileInCommandFileList(cmdFile, canonPath.str().str());
  }

  // Build the set of include directories for this file.
  llvm::SmallString<32> uriDirectory(uri.file());
  llvm::sys::path::remove_filename(uriDirectory);

  std::vector<std::string> libDirs;
  libDirs.push_back(uriDirectory.str().str());
  libDirs.insert(libDirs.end(), context.options.libDirs.begin(),
                 context.options.libDirs.end());

  auto memBuffer = llvm::MemoryBuffer::getMemBufferCopy(contents, uri.file());
  if (!memBuffer) {
    circt::lsp::Logger::error(
        Twine("Failed to create memory buffer for file ") + uri.file());
    return;
  }

  // This block parses the top file to determine all definitions
  // This is used in a second pass to declare all those definitions
  // as top modules, so they are elaborated and subsequently indexed.
  {
    slang::driver::Driver topDriver;

    auto topSlangBuffer =
        topDriver.sourceManager.assignText(uri.file(), memBuffer->getBuffer());
    topDriver.sourceLoader.addBuffer(topSlangBuffer);

    topDriver.addStandardArgs();

    if (!topDriver.processOptions()) {
      return;
    }

    if (!topDriver.parseAllSources()) {
      circt::lsp::Logger::error(Twine("Failed to parse Verilog file ") +
                                uri.file());
      return;
    }

    // Extract all the top modules in the file directly from the syntax tree
    std::vector<std::string> topModules;
    for (auto &t : topDriver.syntaxTrees) {
      if (auto *compUnit =
              t->root().as_if<slang::syntax::CompilationUnitSyntax>()) {
        for (auto *member : compUnit->members) {
          // While it's called "ModuleDeclarationSyntax", it also covers
          // packages
          if (auto *moduleDecl =
                  member->as_if<slang::syntax::ModuleDeclarationSyntax>()) {
            topModules.emplace_back(moduleDecl->header->name.valueText());
          }
        }
      }
    }
    driver.options.topModules = std::move(topModules);
  }

  for (const auto &libDir : libDirs) {
    driver.sourceLoader.addSearchDirectories(libDir);
  }

  // If the main buffer is **not** present in a command file, add it into
  // slang's source manager.
  if (!skipMainBufferSlangImport) {
    auto slangBuffer =
        driver.sourceManager.assignText(uri.file(), memBuffer->getBuffer());
    driver.sourceLoader.addBuffer(slangBuffer);
    mainBufferId = slangBuffer.id;
  }

  auto diagClient = std::make_shared<LSPDiagnosticClient>(*this, diagnostics);
  driver.diagEngine.addClient(diagClient);

  driver.options.compilationFlags.emplace(
      slang::ast::CompilationFlags::LintMode, false);
  driver.options.compilationFlags.emplace(
      slang::ast::CompilationFlags::DisableInstanceCaching, false);

  if (!driver.processOptions()) {
    return;
  }

  driver.diagEngine.setIgnoreAllWarnings(false);

  if (!driver.parseAllSources()) {
    circt::lsp::Logger::error(Twine("Failed to parse Verilog file ") +
                              uri.file());
    return;
  }

  compilation = driver.createCompilation();
  if (failed(compilation))
    return;

  if (skipMainBufferSlangImport) {
    // If the main buffer is present in a command file, compile it only once
    // and import directly from the command file; then figure out which buffer
    // id it was assigned and bind to llvm source manager.
    llvm::SmallString<256> slangCanonPath;
    bool mainBufferIdSet = false;

    // Iterate through all buffers in the slang compilation and set up
    // a binding to the LLVM Source Manager.
    auto *sourceManager = (**compilation).getSourceManager();
    for (auto slangBuffer : sourceManager->getAllBuffers()) {
      std::string_view slangRawPath =
          sourceManager->getRawFileName(slangBuffer);
      if (std::error_code ec =
              llvm::sys::fs::real_path(slangRawPath, slangCanonPath))
        continue;

      if (slangCanonPath == canonPath) {
        mainBufferId = slangBuffer;
        mainBufferIdSet = true;
        break;
      }
    }

    if (!mainBufferIdSet)
      circt::lsp::Logger::error(
          Twine("Failed to set main buffer id after compilation! "));
  }

  for (auto &diag : (*compilation)->getAllDiagnostics())
    driver.diagEngine.issue(diag);

  computeLineOffsets(driver.sourceManager.getSourceText(mainBufferId));

  // Populate the index.
  index.initialize(**compilation);
}

llvm::lsp::Location
VerilogDocument::getLspLocation(slang::SourceLocation loc) const {
  if (loc && loc.buffer() != slang::SourceLocation::NoLocation.buffer()) {
    const auto &slangSourceManager = getSlangSourceManager();
    auto line = slangSourceManager.getLineNumber(loc) - 1;
    auto column = slangSourceManager.getColumnNumber(loc) - 1;
    auto it = loc.buffer();
    if (it == mainBufferId)
      return llvm::lsp::Location(uri, llvm::lsp::Range(Position(line, column)));

    llvm::StringRef fileName = slangSourceManager.getFileName(loc);
    // Ensure absolute path for LSP:
    llvm::SmallString<256> abs(fileName);
    if (!llvm::sys::path::is_absolute(abs)) {
      // Try realPath first
      if (std::error_code ec = llvm::sys::fs::real_path(fileName, abs)) {
        // Fallback: make it absolute relative to the process CWD
        llvm::sys::fs::current_path(abs); // abs = CWD
        llvm::sys::path::append(abs, fileName);
      }
    }

    if (auto uriOrErr = llvm::lsp::URIForFile::fromFile(abs)) {
      if (auto e = uriOrErr.takeError())
        return llvm::lsp::Location();
      return llvm::lsp::Location(*uriOrErr,
                                 llvm::lsp::Range(Position(line, column)));
    }
    return llvm::lsp::Location();
  }
  return llvm::lsp::Location();
}

llvm::lsp::Location
VerilogDocument::getLspLocation(slang::SourceRange range) const {

  auto start = getLspLocation(range.start());
  auto end = getLspLocation(range.end());

  if (start.uri != end.uri)
    return llvm::lsp::Location();

  return llvm::lsp::Location(
      start.uri, llvm::lsp::Range(start.range.start, end.range.end));
}

std::optional<std::pair<slang::BufferID, SmallString<128>>>
VerilogDocument::getOrOpenFile(StringRef filePath) {

  auto fileInfo = filePathMap.find(filePath);
  if (fileInfo != filePathMap.end())
    return fileInfo->second;

  auto getIfExist = [&](StringRef path)
      -> std::optional<std::pair<slang::BufferID, SmallString<128>>> {
    if (llvm::sys::fs::exists(path)) {
      auto memoryBuffer = llvm::MemoryBuffer::getFile(path);
      if (!memoryBuffer) {
        return std::nullopt;
      }

      auto newSlangBuffer = driver.sourceManager.assignText(
          path.str(), memoryBuffer.get()->getBufferStart());
      driver.sourceLoader.addBuffer(newSlangBuffer);

      fileInfo = filePathMap
                     .insert(std::make_pair(
                         filePath, std::make_pair(newSlangBuffer.id, path)))
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

//===----------------------------------------------------------------------===//
// VerilogTextFile
//===----------------------------------------------------------------------===//

namespace {
/// This class represents a text file containing one or more Verilog
/// documents.
class VerilogTextFile {
public:
  VerilogTextFile(VerilogServerContext &globalContext,
                  const llvm::lsp::URIForFile &uri, StringRef fileContents,
                  int64_t version,
                  std::vector<llvm::lsp::Diagnostic> &diagnostics);

  /// Return the current version of this text file.
  int64_t getVersion() const { return version; }

  /// Update the file to the new version using the provided set of content
  /// changes. Returns failure if the update was unsuccessful.
  LogicalResult
  update(const llvm::lsp::URIForFile &uri, int64_t newVersion,
         ArrayRef<llvm::lsp::TextDocumentContentChangeEvent> changes,
         std::vector<llvm::lsp::Diagnostic> &diagnostics);

  void getLocationsOf(const llvm::lsp::URIForFile &uri,
                      llvm::lsp::Position defPos,
                      std::vector<llvm::lsp::Location> &locations);

  void findReferencesOf(const llvm::lsp::URIForFile &uri,
                        llvm::lsp::Position pos,
                        std::vector<llvm::lsp::Location> &references);

private:
  /// Initialize the text file from the given file contents.
  void initialize(const llvm::lsp::URIForFile &uri, int64_t newVersion,
                  std::vector<llvm::lsp::Diagnostic> &diagnostics);

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
    VerilogServerContext &context, const llvm::lsp::URIForFile &uri,
    StringRef fileContents, int64_t version,
    std::vector<llvm::lsp::Diagnostic> &diagnostics)
    : context(context), contents(fileContents.str()) {
  initialize(uri, version, diagnostics);
}

LogicalResult VerilogTextFile::update(
    const llvm::lsp::URIForFile &uri, int64_t newVersion,
    ArrayRef<llvm::lsp::TextDocumentContentChangeEvent> changes,
    std::vector<llvm::lsp::Diagnostic> &diagnostics) {
  if (failed(llvm::lsp::TextDocumentContentChangeEvent::applyTo(changes,
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
    const llvm::lsp::URIForFile &uri, int64_t newVersion,
    std::vector<llvm::lsp::Diagnostic> &diagnostics) {
  version = newVersion;
  document =
      std::make_unique<VerilogDocument>(context, uri, contents, diagnostics);
}

void VerilogTextFile::getLocationsOf(
    const llvm::lsp::URIForFile &uri, llvm::lsp::Position defPos,
    std::vector<llvm::lsp::Location> &locations) {
  document->getLocationsOf(uri, defPos, locations);
}

void VerilogTextFile::findReferencesOf(
    const llvm::lsp::URIForFile &uri, llvm::lsp::Position pos,
    std::vector<llvm::lsp::Location> &references) {
  document->findReferencesOf(uri, pos, references);
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
    visitDefault(expr);
  }

  void visit(const slang::ast::VariableSymbol &expr) {
    insertSymbol(&expr, expr.location, /*isDefinition=*/true);
    visitDefault(expr);
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
    visitDefault(expr);
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
    visitDefault(expr);
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

    if (inst->body.location.buffer() != document.getMainBufferID())
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
  auto &sourceMgr = document.getSlangSourceManager();
  auto getMainBuffer = sourceMgr.getSourceText(document.getMainBufferID());
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
    std::vector<llvm::lsp::Diagnostic> &diagnostics) {

  impl->files[uri.file()] = std::make_unique<VerilogTextFile>(
      impl->context, uri, contents, version, diagnostics);
}

void circt::lsp::VerilogServer::updateDocument(
    const URIForFile &uri,
    ArrayRef<llvm::lsp::TextDocumentContentChangeEvent> changes,
    int64_t version, std::vector<llvm::lsp::Diagnostic> &diagnostics) {
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
    std::vector<llvm::lsp::Location> &locations) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    fileIt->second->getLocationsOf(uri, defPos, locations);
}

void circt::lsp::VerilogServer::findReferencesOf(
    const URIForFile &uri, const Position &pos,
    std::vector<llvm::lsp::Location> &references) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    fileIt->second->findReferencesOf(uri, pos, references);
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

  auto buffer = document.getSlangSourceManager().getSourceText(lhsBufferId);

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

//===----------------------------------------------------------------------===//
// VerilogDocument: Definitions and References
//===----------------------------------------------------------------------===//

static llvm::lsp::Range getRange(const mlir::FileLineColRange &fileLoc) {
  return llvm::lsp::Range(
      llvm::lsp::Position(fileLoc.getStartLine(), fileLoc.getStartColumn()),
      llvm::lsp::Position(fileLoc.getEndLine(), fileLoc.getEndColumn()));
}

/// Build a vector of line start offsets (0-based).
void VerilogDocument::computeLineOffsets(std::string_view text) {
  lineOffsets.clear();
  lineOffsets.reserve(1024);
  lineOffsets.push_back(0);
  for (size_t i = 0; i < text.size(); ++i) {
    if (text[i] == '\n') {
      lineOffsets.push_back(static_cast<uint32_t>(i + 1));
    }
  }
}

// LSP (0-based line, UTF-16 character) -> byte offset into UTF-8 buffer.
std::optional<uint32_t>
VerilogDocument::lspPositionToOffset(const llvm::lsp::Position &pos) {

  auto &sm = getSlangSourceManager();

  std::string_view text = sm.getSourceText(mainBufferId);

  // Clamp line index
  if ((unsigned)pos.line >= lineOffsets.size())
    return std::nullopt;

  size_t lineStart = lineOffsets[pos.line];
  size_t lineEnd = ((unsigned)(pos.line + 1) < lineOffsets.size())
                       ? lineOffsets[pos.line + 1] - 1
                       : text.size();

  const llvm::UTF8 *src =
      reinterpret_cast<const llvm::UTF8 *>(text.data() + lineStart);
  const llvm::UTF8 *srcEnd =
      reinterpret_cast<const llvm::UTF8 *>(text.data() + lineEnd);

  // Convert up to 'target' UTF-16 code units; stop early if line ends.
  const uint32_t target = pos.character;
  if (target == 0)
    return static_cast<uint32_t>(
        src - reinterpret_cast<const llvm::UTF8 *>(text.data()));

  std::vector<llvm::UTF16> sink(target);
  llvm::UTF16 *out = sink.data();
  llvm::UTF16 *outEnd = out + sink.size();

  (void)llvm::ConvertUTF8toUTF16(&src, srcEnd, &out, outEnd,
                                 llvm::lenientConversion);

  return static_cast<uint32_t>(reinterpret_cast<const char *>(src) -
                               text.data());
}

const char *VerilogDocument::getPointerFor(const llvm::lsp::Position &pos) {
  auto &sm = getSlangSourceManager();
  auto slangBufferOffset = lspPositionToOffset(pos);

  if (!slangBufferOffset.has_value())
    return nullptr;

  uint32_t offset = slangBufferOffset.value();
  return sm.getSourceText(mainBufferId).data() + offset;
}

void VerilogDocument::getLocationsOf(
    const llvm::lsp::URIForFile &uri, const llvm::lsp::Position &defPos,
    std::vector<llvm::lsp::Location> &locations) {

  const auto &slangBufferPointer = getPointerFor(defPos);

  const auto &intervalMap = index.getIntervalMap();
  auto it = intervalMap.find(slangBufferPointer);

  // Found no element in the given index.
  if (!it.valid() || slangBufferPointer < it.start())
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
      auto uri = llvm::lsp::URIForFile::fromFile(filePath);
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

  slang::SourceRange range(symbol->location,
                           symbol->location +
                               (symbol->name.size() ? symbol->name.size() : 1));
  locations.push_back(getLspLocation(range));
}

void VerilogDocument::findReferencesOf(
    const llvm::lsp::URIForFile &uri, const llvm::lsp::Position &pos,
    std::vector<llvm::lsp::Location> &references) {

  const auto &slangBufferPointer = getPointerFor(pos);
  const auto &intervalMap = index.getIntervalMap();
  auto intervalIt = intervalMap.find(slangBufferPointer);

  if (!intervalIt.valid() || slangBufferPointer < intervalIt.start())
    return;

  const auto *symbol = dyn_cast<const slang::ast::Symbol *>(intervalIt.value());
  if (!symbol)
    return;

  auto it = index.getReferences().find(symbol);
  if (it == index.getReferences().end())
    return;
  for (auto referenceRange : it->second)
    references.push_back(getLspLocation(referenceRange));
}
