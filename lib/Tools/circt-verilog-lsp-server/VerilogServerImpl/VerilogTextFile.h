//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// VerilogTextFile.h
//
// This header declares VerilogTextFile, a lightweight LSP-facing wrapper
// around VerilogDocument that represents an open, editable Verilog source
// buffer in the CIRCT Verilog LSP server.
//
// VerilogTextFile owns the current text contents and version (as tracked by
// the LSP client) and rebuilds its VerilogDocument whenever the file is
// opened or updated. It also forwards language queries (e.g. “go to
// definition”, “find references”) to the underlying VerilogDocument, which
// performs Slang-based parsing, indexing, and location translation.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_VERILOGTEXTFILE_H_
#define LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_VERILOGTEXTFILE_H_

#include "llvm/Support/LSP/Protocol.h"

#include "VerilogDocument.h"

namespace circt {
namespace lsp {

struct VerilogServerContext;

/// This class represents a text file containing one or more Verilog
/// documents.
class VerilogTextFile {
public:
  VerilogTextFile(VerilogServerContext &globalContext,
                  const llvm::lsp::URIForFile &uri,
                  llvm::StringRef fileContents, int64_t version,
                  std::vector<llvm::lsp::Diagnostic> &diagnostics);

  /// Return the current version of this text file.
  int64_t getVersion() const { return version; }

  /// Update the file to the new version using the provided set of content
  /// changes. Returns failure if the update was unsuccessful.
  llvm::LogicalResult
  update(const llvm::lsp::URIForFile &uri, int64_t newVersion,
         llvm::ArrayRef<llvm::lsp::TextDocumentContentChangeEvent> changes,
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

  void initializeProjectDriver();

  VerilogServerContext &context;

  /// The full string contents of the file.
  std::string contents;

  /// The project-scale driver
  std::unique_ptr<slang::driver::Driver> projectDriver;
  std::vector<std::string> projectIncludeDirectories;

  /// The version of this file.
  int64_t version = 0;

  /// The chunks of this file. The order of these chunks is the order in which
  /// they appear in the text file.
  std::unique_ptr<circt::lsp::VerilogDocument> document;
};

} // namespace lsp
} // namespace circt

#endif
