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
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Protocol.h"
#include "slang/ast/Compilation.h"
#include "slang/diagnostics/DiagnosticClient.h"
#include "slang/diagnostics/Diagnostics.h"
#include "slang/driver/Driver.h"
#include "slang/syntax/SyntaxTree.h"
#include "slang/text/SourceLocation.h"
#include "slang/text/SourceManager.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
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

//===----------------------------------------------------------------------===//
// VerilogDocument
//===----------------------------------------------------------------------===//

VerilogDocument::VerilogDocument(
    VerilogServerContext &context, const mlir::lsp::URIForFile &uri,
    StringRef contents, std::vector<mlir::lsp::Diagnostic> &diagnostics)
    : uri(uri) {
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
