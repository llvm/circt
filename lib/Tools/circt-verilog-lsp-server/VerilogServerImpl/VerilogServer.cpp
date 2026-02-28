//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the VerilogServer class, the top-level coordinator for
// the CIRCT Verilog LSP server. It manages the lifetime and synchronization of
// all open Verilog source files and provides the high-level entry points for
// Language Server Protocol (LSP) requests.
//
// Responsibilities:
//   * Maintain a map of open documents (`VerilogTextFile` instances) indexed by
//     their URI filenames.
//   * Create, update, and remove documents in response to LSP notifications
//     such as `textDocument/didOpen`, `didChange`, and `didClose`.
//   * Route definition and reference queries to the appropriate
//     VerilogTextFile, which in turn delegates to its `VerilogDocument`.
//   * Ensure each document remains synchronized with its latest text version
//     and up-to-date semantic index.
//
// Internal structure:
//   - Uses an internal `Impl` struct (pImpl pattern) to encapsulate the
//     server’s file map and its shared `VerilogServerContext`.
//   - Each document is owned via a `std::unique_ptr<VerilogTextFile>`,
//     allowing clean removal and automatic teardown when the file is closed.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/LSP/Protocol.h"

#include "circt/Support/LLVM.h"
#include "circt/Tools/circt-verilog-lsp-server/CirctVerilogLspServerMain.h"

#include "VerilogServer.h"
#include "VerilogServerContext.h"
#include "VerilogTextFile.h"

#include <memory>
#include <optional>

using namespace llvm::lsp;

struct circt::lsp::VerilogServer::Impl {
  explicit Impl(const VerilogServerOptions &options) : context(options) {}

  /// The files held by the server, mapped by their URI file name.
  llvm::StringMap<std::unique_ptr<VerilogTextFile>> files;

  VerilogServerContext context;
};

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
