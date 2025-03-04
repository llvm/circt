//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for circt-verilog-lsp-server for when built as standalone
// binary.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_VERILOGSERVER_H_
#define LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_VERILOGSERVER_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <optional>
#include <vector>

namespace mlir {
namespace lsp {
struct Diagnostic;
struct Position;
struct Location;
struct TextDocumentContentChangeEvent;
struct Range;
struct InlayHint;
class URIForFile;
} // namespace lsp
} // namespace mlir

namespace circt {
namespace lsp {
struct VerilogServerOptions;
using TextDocumentContentChangeEvent =
    mlir::lsp::TextDocumentContentChangeEvent;
using URIForFile = mlir::lsp::URIForFile;
using Diagnostic = mlir::lsp::Diagnostic;
struct VerilogUserProvidedInlayHint;

/// This class implements all of the Verilog related functionality necessary for
/// a language server. This class allows for keeping the Verilog specific logic
/// separate from the logic that involves LSP server/client communication.
class VerilogServer {
public:
  VerilogServer(const circt::lsp::VerilogServerOptions &options);
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

  /// Get the inlay hints for the range within the given file.
  void getInlayHints(const URIForFile &uri, const mlir::lsp::Range &range,
                     std::vector<mlir::lsp::InlayHint> &inlayHints);

  void putInlayHintsOnObjects(
      const std::vector<circt::lsp::VerilogUserProvidedInlayHint> &params);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace lsp
} // namespace circt

#endif // LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_VERILOGSERVER_H_
