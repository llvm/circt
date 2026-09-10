//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// VerilogDocument.cpp
//
// This file implements the VerilogDocument class, which represents a single
// open Verilog/SystemVerilog source file within the CIRCT Verilog LSP server.
// It acts as the per-buffer bridge between the Language Server Protocol (LSP)
// and the Slang front-end infrastructure.
//
// Responsibilities:
//   * Parse and elaborate a single Verilog source buffer using Slang’s driver.
//   * Integrate with project-wide command files (-C) and include/library search
//     paths supplied by the VerilogServerContext.
//   * Handle main-buffer override semantics: when the buffer is already listed
//     in a command file, it reuses the existing Slang buffer; otherwise it
//     injects an in-memory buffer directly.
//   * Collect and forward diagnostics to the LSP client via
//   LSPDiagnosticClient.
//   * Build and own a VerilogIndex for symbol and location queries.
//   * Provide translation utilities between LSP and Slang coordinates, such as
//     UTF-16 ↔ UTF-8 position mapping and conversion to llvm::lsp::Location.
//
// The class is used by VerilogServerContext to maintain open documents,
// service “go to definition” and “find references” requests, and keep file
// state synchronized with the editor.
//
//===----------------------------------------------------------------------===//

#include "slang/syntax/AllSyntax.h"
#include "slang/syntax/SyntaxTree.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/Path.h"

#include "../Utils/LSPUtils.h"
#include "LSPDiagnosticClient.h"
#include "VerilogDocument.h"
#include "VerilogServerContext.h"
#include "circt/Tools/circt-verilog-lsp-server/CirctVerilogLspServerMain.h"

using namespace circt::lsp;
using namespace llvm;
using namespace llvm::lsp;

static inline void setTopModules(slang::driver::Driver &driver) {
  // Parse the main buffer
  if (!driver.parseAllSources()) {
    circt::lsp::Logger::error(Twine("Failed to parse main buffer "));
    return;
  }
  // Extract all the top modules in the file directly from the syntax tree
  std::vector<std::string> topModules;
  for (auto &t : driver.syntaxTrees) {
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

static inline void
copyBuffers(slang::driver::Driver &driver,
            const slang::driver::Driver *const projectDriver,
            const llvm::SmallString<256> &mainBufferFileName) {
  for (auto bId : projectDriver->sourceManager.getAllBuffers()) {
    std::string_view slangRawPath =
        projectDriver->sourceManager.getRawFileName(bId);

    llvm::SmallString<256> slangCanonPath;
    if (llvm::sys::fs::real_path(slangRawPath, slangCanonPath))
      continue;

    if (slangCanonPath ==
        mainBufferFileName) // skip the file you're already compiling
      continue;

    bool alreadyLoaded = false;
    for (auto id : driver.sourceManager.getAllBuffers()) {
      if (driver.sourceManager.getFullPath(id).string() ==
          slangCanonPath.str()) {
        alreadyLoaded = true;
        break;
      }
    }
    if (alreadyLoaded)
      continue;

    auto buffer = driver.sourceManager.assignText(
        slangCanonPath.str(), projectDriver->sourceManager.getSourceText(bId));
    driver.sourceLoader.addBuffer(buffer);
  }
}

VerilogDocument::VerilogDocument(
    VerilogServerContext &context, const llvm::lsp::URIForFile &uri,
    StringRef contents, std::vector<llvm::lsp::Diagnostic> &diagnostics,
    const slang::driver::Driver *const projectDriver,
    const std::vector<std::string> &projectIncludeDirectories)
    : globalContext(context), uri(uri) {

  llvm::SmallString<256> canonPath(uri.file());
  if (std::error_code ec = llvm::sys::fs::real_path(uri.file(), canonPath))
    canonPath = uri.file(); // fall back, but try to keep it absolute

  // Build the set of include directories for this file.
  llvm::SmallString<32> uriDirectory(uri.file());
  llvm::sys::path::remove_filename(uriDirectory);

  std::vector<std::string> libDirs;
  libDirs.push_back(uriDirectory.str().str());
  libDirs.insert(libDirs.end(), context.options.libDirs.begin(),
                 context.options.libDirs.end());

  for (const auto &libDir : libDirs)
    driver.sourceLoader.addSearchDirectories(libDir);

  auto memBuffer = llvm::MemoryBuffer::getMemBufferCopy(contents, uri.file());
  if (!memBuffer) {
    circt::lsp::Logger::error(
        Twine("Failed to create memory buffer for file ") + uri.file());
    return;
  }

  auto topSlangBuffer =
      driver.sourceManager.assignText(uri.file(), memBuffer->getBuffer());
  driver.sourceLoader.addBuffer(topSlangBuffer);
  mainBufferId = topSlangBuffer.id;

  auto diagClient = std::make_shared<LSPDiagnosticClient>(*this, diagnostics);
  driver.diagEngine.addClient(diagClient);

  driver.options.compilationFlags.emplace(
      slang::ast::CompilationFlags::LintMode, false);
  driver.options.compilationFlags.emplace(
      slang::ast::CompilationFlags::DisableInstanceCaching, false);

  for (auto &dir : projectIncludeDirectories)
    (void)driver.sourceManager.addUserDirectories(dir);

  if (!driver.processOptions()) {
    circt::lsp::Logger::error(Twine("Failed to process slang driver options!"));
    return;
  }

  driver.diagEngine.setIgnoreAllWarnings(false);

  // Import dependencies from projectDriver if it exists.
  if (projectDriver) {
    // Copy options from project driver
    driver.options = projectDriver->options;
    // Set top modules according to main buffer
    setTopModules(driver);
    // Copy dependency buffers from project driver
    copyBuffers(driver, projectDriver, canonPath);
  }

  if (!driver.parseAllSources()) {
    circt::lsp::Logger::error(Twine("Failed to parse Verilog file ") +
                              uri.file());
    return;
  }

  compilation = driver.createCompilation();
  if (failed(compilation)) {
    circt::lsp::Logger::error(Twine("Failed to compile Verilog file ") +
                              uri.file());
    return;
  }

  for (auto &diag : (*compilation)->getAllDiagnostics())
    driver.diagEngine.issue(diag);

  computeLineOffsets(driver.sourceManager.getSourceText(mainBufferId));

  index = std::make_unique<VerilogIndex>(mainBufferId, driver.sourceManager);
  // Populate the index.
  index->initialize(**compilation);
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

  if (!index)
    return;

  const auto &intervalMap = index->getIntervalMap();
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

  if (!index)
    return;

  const auto &slangBufferPointer = getPointerFor(pos);
  const auto &intervalMap = index->getIntervalMap();
  auto intervalIt = intervalMap.find(slangBufferPointer);

  if (!intervalIt.valid() || slangBufferPointer < intervalIt.start())
    return;

  const auto *symbol = dyn_cast<const slang::ast::Symbol *>(intervalIt.value());
  if (!symbol)
    return;

  auto it = index->getReferences().find(symbol);
  if (it == index->getReferences().end())
    return;
  for (auto referenceRange : it->second)
    references.push_back(getLspLocation(referenceRange));
}
