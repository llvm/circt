//===- PDLLServer.h - PDL General Language Server ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIB_MLIR_TOOLS_MLIRPDLLSPSERVER_SERVER_H_
#define LIB_MLIR_TOOLS_MLIRPDLLSPSERVER_SERVER_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mlir {
namespace lsp {
struct Diagnostic;
struct VerilogViewOutputResult;
enum class VerilogViewOutputKind;
struct CompletionList;
struct DocumentLink;
struct DocumentSymbol;
struct Hover;
struct InlayHint;
struct Location;
struct Position;
struct Range;
struct SignatureHelp;
struct TextDocumentContentChangeEvent;
class URIForFile;
struct Diagnostic;
} // namespace lsp
} // namespace mlir

namespace circt {
namespace lsp {
class CompilationDatabase;
struct VerilogViewOutputResult;
enum class VerilogViewOutputKind;
// struct DocumentLink;
// struct DocumentSymbol;
using TextDocumentContentChangeEvent =
    mlir::lsp::TextDocumentContentChangeEvent;
using URIForFile = mlir::lsp::URIForFile;
using Diagnostic = mlir::lsp::Diagnostic;
using SignatureHelp = mlir::lsp::SignatureHelp;
using CompletionList = mlir::lsp::CompletionList;

/// This class implements all of the PDLL related functionality necessary for a
/// language server. This class allows for keeping the PDLL specific logic
/// separate from the logic that involves LSP server/client communication.
class VerilogServer {
public:
  struct Options;
  VerilogServer(const Options &options);
  ~VerilogServer();

  /// Add the document, with the provided `version`, at the given URI. Any
  /// diagnostics emitted for this document should be added to `diagnostics`.
  void addDocument(const URIForFile &uri, llvm::StringRef contents,
                   int64_t version, std::vector<Diagnostic> &diagnostics);

  /// Update the document, with the provided `version`, at the given URI. Any
  /// diagnostics emitted for this document should be added to `diagnostics`.
  void updateDocument(const URIForFile &uri,
                      llvm::ArrayRef<TextDocumentContentChangeEvent> changes,
                      int64_t version, std::vector<Diagnostic> &diagnostics);

  /// Remove the document with the given uri. Returns the version of the removed
  /// document, or std::nullopt if the uri did not have a corresponding document
  /// within the server.
  std::optional<int64_t> removeDocument(const URIForFile &uri);

  /// Return the locations of the object pointed at by the given position.
  void getLocationsOf(const URIForFile &uri, const mlir::lsp::Position &defPos,
                      std::vector<mlir::lsp::Location> &locations);

  /// Find all references of the object pointed at by the given position.
  void findReferencesOf(const URIForFile &uri, const mlir::lsp::Position &pos,
                        std::vector<mlir::lsp::Location> &references);

  /// Return the document links referenced by the given file.
  void getDocumentLinks(const URIForFile &uri,
                        std::vector<mlir::lsp::DocumentLink> &documentLinks);

  /// Find a hover description for the given hover position, or std::nullopt if
  /// one couldn't be found.
  std::optional<mlir::lsp::Hover>
  findHover(const URIForFile &uri, const mlir::lsp::Position &hoverPos);

  /// Find all of the document symbols within the given file.
  void findDocumentSymbols(const URIForFile &uri,
                           std::vector<mlir::lsp::DocumentSymbol> &symbols);

  /// Get the code completion list for the position within the given file.
  CompletionList getCodeCompletion(const URIForFile &uri,
                                   const mlir::lsp::Position &completePos);

  /// Get the signature help for the position within the given file.
  SignatureHelp getSignatureHelp(const URIForFile &uri,
                                 const mlir::lsp::Position &helpPos);

  /// Get the inlay hints for the range within the given file.
  void getInlayHints(const URIForFile &uri, const mlir::lsp::Range &range,
                     std::vector<mlir::lsp::InlayHint> &inlayHints);

  /// Get the output of the given Verilog file, or std::nullopt if there is no
  /// valid output.
  std::optional<VerilogViewOutputResult>
  getVerilogViewOutput(const URIForFile &uri, VerilogViewOutputKind kind);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace lsp
} // namespace circt

#endif // LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_VERILOGSERVER_H_
