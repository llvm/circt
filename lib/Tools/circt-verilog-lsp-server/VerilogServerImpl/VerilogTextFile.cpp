//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// VerilogTextFile.cpp
//
// This file implements the VerilogTextFile class, a lightweight wrapper around
// VerilogDocument that represents the open text buffer for a single source file
// managed by the CIRCT Verilog LSP server.
//
// Responsibilities:
//   * Manage the current text contents and version of an open Verilog file.
//   * Rebuild the associated VerilogDocument whenever the file is opened or
//     updated via LSP “didOpen” or “didChange” notifications.
//   * Apply incremental text changes as specified by the LSP protocol.
//   * Forward symbol-definition and reference queries to the underlying
//     VerilogDocument.
//
// The class acts as the LSP-facing façade for each active text document,
// maintaining an editable in-memory copy of its contents while keeping the
// corresponding Slang-based VerilogDocument synchronized for semantic analysis.
//
//===----------------------------------------------------------------------===//

#include "slang/util/CommandLine.h"

#include "../Utils/LSPUtils.h"
#include "VerilogDocument.h"
#include "VerilogServerContext.h"
#include "VerilogTextFile.h"
#include "circt/Tools/circt-verilog-lsp-server/CirctVerilogLspServerMain.h"

using namespace circt::lsp;
using namespace llvm;
using namespace llvm::lsp;

VerilogTextFile::VerilogTextFile(
    VerilogServerContext &context, const llvm::lsp::URIForFile &uri,
    StringRef fileContents, int64_t version,
    std::vector<llvm::lsp::Diagnostic> &diagnostics)
    : context(context), contents(fileContents.str()) {
  initializeProjectDriver();
  std::scoped_lock lk(contentMutex);
  initialize(uri, version, diagnostics);
}

LogicalResult VerilogTextFile::update(
    const llvm::lsp::URIForFile &uri, int64_t newVersion,
    ArrayRef<llvm::lsp::TextDocumentContentChangeEvent> changes,
    std::vector<llvm::lsp::Diagnostic> &diagnostics) {

  std::scoped_lock lk(contentMutex);
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

void VerilogTextFile::initializeProjectDriver() {
  if (context.options.commandFiles.empty())
    return;

  projectDriver = std::make_unique<slang::driver::Driver>();

  // --- Apply project command files (the “-C”s) to this per-buffer driver ---
  for (const std::string &cmdFile : context.options.commandFiles) {
    if (!projectDriver->processCommandFiles(cmdFile, false, true)) {
      circt::lsp::Logger::error(Twine("Failed to open command file ") +
                                cmdFile);
      return;
    }

    // Open command file and parse it ourselves to get include dirs...
    slang::CommandLine unitCmdLine;
    std::vector<std::string> includes;
    unitCmdLine.add("-I,--include-directory,+incdir", includes, "", "",
                    slang::CommandLineFlags::CommaList);
    std::vector<std::string> defines;
    unitCmdLine.add("-D,--define-macro,+define", defines, "");

    std::optional<std::string> libraryName;
    unitCmdLine.add("--library", libraryName, "");

    std::vector<std::string> files;
    unitCmdLine.setPositional(
        [&](std::string_view value) {
          files.emplace_back(value);
          return "";
        },
        "");

    slang::CommandLine::ParseOptions parseOpts;
    parseOpts.expandEnvVars = true;
    parseOpts.ignoreProgramName = true;
    parseOpts.supportComments = true;
    parseOpts.ignoreDuplicates = true;

    slang::SmallVector<char> buffer;
    if (auto readEc = slang::OS::readFile(cmdFile, buffer)) {
      continue;
    }
    std::string_view argStr(buffer.data(), buffer.size());
    if (!unitCmdLine.parse(argStr, parseOpts)) {
      continue;
    }
    projectIncludeDirectories.insert(projectIncludeDirectories.end(),
                                     includes.begin(), includes.end());
  }

  projectDriver->options.compilationFlags.emplace(
      slang::ast::CompilationFlags::LintMode, false);
  projectDriver->options.compilationFlags.emplace(
      slang::ast::CompilationFlags::DisableInstanceCaching, false);

  if (!projectDriver->processOptions()) {
    circt::lsp::Logger::error(
        Twine("Failed to apply slang options on project "));
    return;
  }

  if (!projectDriver->parseAllSources()) {
    circt::lsp::Logger::error(Twine("Failed to parse Verilog project files "));
    return;
  }
}

void VerilogTextFile::initialize(
    const llvm::lsp::URIForFile &uri, int64_t newVersion,
    std::vector<llvm::lsp::Diagnostic> &diagnostics) {
  std::shared_ptr<VerilogDocument> newDocument;
  setDocument(std::make_shared<VerilogDocument>(
      context, uri, contents, diagnostics, projectDriver.get(),
      projectIncludeDirectories));
  version = newVersion;
}

void VerilogTextFile::getLocationsOf(
    const llvm::lsp::URIForFile &uri, llvm::lsp::Position defPos,
    std::vector<llvm::lsp::Location> &locations) {
  auto doc = getDocument();
  doc->getLocationsOf(uri, defPos, locations);
}

void VerilogTextFile::findReferencesOf(
    const llvm::lsp::URIForFile &uri, llvm::lsp::Position pos,
    std::vector<llvm::lsp::Location> &references) {
  auto doc = getDocument();
  doc->findReferencesOf(uri, pos, references);
}

std::shared_ptr<VerilogDocument> VerilogTextFile::getDocument() {
  std::scoped_lock lk(docMutex);
  return document;
}

void VerilogTextFile::setDocument(std::shared_ptr<VerilogDocument> newDoc) {
  std::scoped_lock lk(docMutex);
  document = std::move(newDoc);
}
