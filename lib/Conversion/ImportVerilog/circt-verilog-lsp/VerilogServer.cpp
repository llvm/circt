//===- VerilogServer.cpp - Verilog Language Server
//------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VerilogServer.h"

#include "Protocol.h"
#include "mlir/IR/BuiltinOps.h"
#include "circt/Tools/circt-verilog-lsp/CirctVerilogLspServerMain.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/lsp-server-support/CompilationDatabase.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/SourceMgrUtils.h"
#include "slang/ast/ASTContext.h"
#include "slang/ast/Compilation.h"
#include "slang/ast/Definition.h"
#include "slang/ast/Statements.h"
#include "slang/ast/symbols/InstanceSymbols.h"
#include "slang/ast/symbols/VariableSymbols.h"
#include "slang/diagnostics/DiagnosticClient.h"
#include "slang/diagnostics/Diagnostics.h"
#include "slang/driver/Driver.h"
#include "slang/parsing/Preprocessor.h"
#include "slang/syntax/SyntaxPrinter.h"
#include "slang/syntax/SyntaxTree.h"
#include "slang/util/Version.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <optional>

using namespace mlir;
using namespace mlir::lsp;

using namespace circt::lsp;
using namespace circt;

/// Returns a language server uri for the given source location. `mainFileURI`
/// corresponds to the uri for the main file of the source manager.
static mlir::lsp::URIForFile
getURIFromLoc(llvm::SourceMgr &mgr, SMRange loc,
              const mlir::lsp::URIForFile &mainFileURI) {
  int bufferId = mgr.FindBufferContainingLoc(loc.Start);
  if (bufferId == 0 || bufferId == static_cast<int>(mgr.getMainFileID()))
    return mainFileURI;
  llvm::Expected<mlir::lsp::URIForFile> fileForLoc =
      mlir::lsp::URIForFile::fromFile(
          mgr.getBufferInfo(bufferId).Buffer->getBufferIdentifier());
  if (fileForLoc)
    return *fileForLoc;
  mlir::lsp::Logger::error("Failed to create URI for include file: {0}",
                           llvm::toString(fileForLoc.takeError()));
  return mainFileURI;
}

/// Returns true if the given location is in the main file of the source
/// manager.
static bool isMainFileLoc(llvm::SourceMgr &mgr, SMRange loc) {
  return mgr.FindBufferContainingLoc(loc.Start) == mgr.getMainFileID();
}

/// Returns a language server location from the given source range.
static mlir::lsp::Location
getLocationFromLoc(llvm::SourceMgr &mgr, SMRange range,
                   const mlir::lsp::URIForFile &uri) {
  return mlir::lsp::Location(getURIFromLoc(mgr, range, uri),
                             mlir::lsp::Range(mgr, range));
}

/// Convert a slang `SourceLocation` to an MLIR `Location`.
static mlir::lsp::Location
convertLocation(const slang::SourceManager &sourceManager,
                slang::SourceLocation loc) {
  if (loc && loc.buffer() != slang::SourceLocation::NoLocation.buffer()) {
    auto fileName = sourceManager.getFileName(loc);
    auto line = sourceManager.getLineNumber(loc);
    auto column = sourceManager.getColumnNumber(loc);
    auto loc = mlir::lsp::URIForFile::fromFile(fileName).get();
    return mlir::lsp::Location(loc, mlir::lsp::Range(line, column));
  }
  return mlir::lsp::Location();
}

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

/// Convert the given MLIR diagnostic to the LSP form.
static std::optional<mlir::lsp::Diagnostic>
getLspDiagnoticFromDiag(slang::SourceManager &sourceMgr,
                        const slang::ReportedDiagnostic &diag,
                        const mlir::lsp::URIForFile &uri) {
  mlir::lsp::Diagnostic lspDiag;
  lspDiag.source = "verilog";

  // FIXME: Right now all of the diagnostics are treated as parser issues, but
  // some are parser and some are verifier.
  lspDiag.category = "Parse Error";

  // Try to grab a file location for this diagnostic.
  mlir::lsp::Location loc = convertLocation(sourceMgr, diag.location);
  lspDiag.range = loc.range;

  // Skip diagnostics that weren't emitted within the main file.
  if (loc.uri != uri)
    return std::nullopt;

  lspDiag.severity = getSeverity(diag.severity);
  lspDiag.message = diag.formattedMessage;

  // Attach any notes to the main diagnostic as related information.
  // std::vector<mlir::lsp::DiagnosticRelatedInformation> relatedDiags;
  // for (const slang::Diagnostic &note : diag.getNotes()) {
  //   relatedDiags.emplace_back(
  //       getLocationFromLoc(sourceMgr, note.getLocation(), uri),
  //       note.getMessage().str());
  // }
  // if (!relatedDiags.empty())
  //   lspDiag.relatedInformation = std::move(relatedDiags);

  return lspDiag;
}

/// Get or extract the documentation for the given decl.
static std::optional<std::string> getDocumentationFor(
    llvm::SourceMgr &sourceMgr, const slang::SourceManager &slangSourceMgr,
    llvm::SmallDenseMap<slang::BufferID, StringRef> &bufferFilePaths,
    const slang::ast::Statement *stmt) {
  // If the decl already had documentation set, use it.
  // if (std::optional<StringRef> doc = decl->getDocComment())
  //   return doc->str();

  // auto loc = convertLocation(slangSourceMgr, bufferFilePaths,
  // stmt->sourceRange.start()); If the decl doesn't yet have documentation, try
  // to extract it from the source file.

  // FIXME: Is this ideal to use two different source managers?
  auto line = slangSourceMgr.getLineNumber(stmt->sourceRange.start());
  auto column = slangSourceMgr.getColumnNumber(stmt->sourceRange.start());

  auto loc = sourceMgr.FindLocForLineAndColumn(sourceMgr.getMainFileID(), line,
                                               column);

  return mlir::lsp::extractSourceDocComment(sourceMgr, loc);
}

//===----------------------------------------------------------------------===//
// VerilogIndex
//===----------------------------------------------------------------------===//

namespace {
using VerilogIndexSymbol = slang::ast::Symbol;

// struct VerilogIndexSymbol =
//   explicit VerilogIndexSymbol(const slang::ast::Statement *definition)
//       : definition(definition) {}
//
//   SMRange getDefLoc() const {
//     definition->getLoc();
//     if (const slang::ast::VariableDeclStatement *decl =
//             llvm::dyn_cast_if_present<const slang::ast::VariableDeclStatement
//             *>(definition)) {
//       const ast::Name *declName = decl->getName();
//       return declName ? declName->getLoc() : decl->getLoc();
//     }
//     return definition.get<const ods::Operation *>()->getLoc();
//   }
//
//   /// The main definition of the symbol.
//   const slang::ast::VariableDeclStatement *definition;
//   /// The set of references to the symbol.
//   std::vector<SMRange> references;
// };

/// This class provides an index for definitions/uses within a Verilog document.
/// It provides efficient lookup of a definition given an input source range.
class VerilogIndex {
public:
  VerilogIndex() : intervalMap(allocator) {}

  /// Initialize the index with the given ast::Module.
  void initialize(const slang::ast::Compilation *compilation);

  /// Lookup a symbol for the given location. Returns nullptr if no symbol could
  /// be found. If provided, `overlappedRange` is set to the range that the
  /// provided `loc` overlapped with.
  const VerilogIndexSymbol *lookup(SMLoc loc,
                                   SMRange *overlappedRange = nullptr) const;

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

  /// An interval map containing a corresponding definition mapped to a source
  /// interval.
  MapT intervalMap;

  /// A mapping between definitions and their corresponding symbol.
  DenseMap<const void *, std::unique_ptr<VerilogIndexSymbol>> defToSymbol;
};
} // namespace

void VerilogIndex::initialize(const slang::ast::Compilation *compilation) {
  // auto getOrInsertDef = [&](const auto *def) -> VerilogIndexSymbol * {
  //   auto it = defToSymbol.try_emplace(def, nullptr);
  //   if (it.second)
  //     it.first->second = std::make_unique<VerilogIndexSymbol>(def);
  //   return &*it.first->second;
  // };
  // auto insertDeclRef = [&](VerilogIndexSymbol *sym, SMRange refLoc,
  //                          bool isDef = false) {
  //   const char *startLoc = refLoc.Start.getPointer();
  //   const char *endLoc = refLoc.End.getPointer();
  //   if (!intervalMap.overlaps(startLoc, endLoc)) {
  //     intervalMap.insert(startLoc, endLoc, sym);
  //     if (!isDef)
  //       sym->references.push_back(refLoc);
  //   }
  // };
  // auto insertODSOpRef = [&](StringRef opName, SMRange refLoc) {
  //   const ods::Operation *odsOp = odsContext.lookupOperation(opName);
  //   if (!odsOp)
  //     return;

  //   VerilogIndexSymbol *symbol = getOrInsertDef(odsOp);
  //   insertDeclRef(symbol, odsOp->getLoc(), /*isDef=*/true);
  //   insertDeclRef(symbol, refLoc);
  // };

  // module.walk([&](const ast::Node *node) {
  //   // Handle references to PDL decls.
  //   if (const auto *decl = dyn_cast<ast::OpNameDecl>(node)) {
  //     if (std::optional<StringRef> name = decl->getName())
  //       insertODSOpRef(*name, decl->getLoc());
  //   } else if (const ast::Decl *decl = dyn_cast<ast::Decl>(node)) {
  //     const ast::Name *name = decl->getName();
  //     if (!name)
  //       return;
  //     VerilogIndexSymbol *declSym = getOrInsertDef(decl);
  //     insertDeclRef(declSym, name->getLoc(), /*isDef=*/true);

  //     if (const auto *varDecl = dyn_cast<ast::VariableDecl>(decl)) {
  //       // Record references to any constraints.
  //       for (const auto &it : varDecl->getConstraints())
  //         insertDeclRef(getOrInsertDef(it.constraint), it.referenceLoc);
  //     }
  //   } else if (const auto *expr = dyn_cast<ast::DeclRefExpr>(node)) {
  //     insertDeclRef(getOrInsertDef(expr->getDecl()), expr->getLoc());
  //   }
  // });
}

const VerilogIndexSymbol *VerilogIndex::lookup(SMLoc loc,
                                               SMRange *overlappedRange) const {
  auto it = intervalMap.find(loc.getPointer());
  if (!it.valid() || loc.getPointer() < it.start())
    return nullptr;

  if (overlappedRange) {
    *overlappedRange = SMRange(SMLoc::getFromPointer(it.start()),
                               SMLoc::getFromPointer(it.stop()));
  }
  return it.value();
}

//===----------------------------------------------------------------------===//
// VerilogDocument
//===----------------------------------------------------------------------===//

namespace {
/// This class represents all of the information pertaining to a specific PDL
/// document.
struct VerilogDocument {
  VerilogDocument(const mlir::lsp::URIForFile &uri, StringRef contents,
                  const std::vector<std::string> &extraDirs,
                  std::vector<mlir::lsp::Diagnostic> &diagnostics);
  VerilogDocument(const VerilogDocument &) = delete;
  VerilogDocument &operator=(const VerilogDocument &) = delete;

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
  // Document Links
  //===--------------------------------------------------------------------===//

  void getDocumentLinks(const mlir::lsp::URIForFile &uri,
                        std::vector<mlir::lsp::DocumentLink> &links);

  //===--------------------------------------------------------------------===//
  // Hover
  //===--------------------------------------------------------------------===//

  std::optional<mlir::lsp::Hover>
  findHover(const mlir::lsp::URIForFile &uri,
            const mlir::lsp::Position &hoverPos);
  std::optional<mlir::lsp::Hover> findHover(const slang::ast::Statement *stmt,
                                            const SMRange &hoverRange);
  std::optional<mlir::lsp::Hover>
  findHover(const slang::ast::Definition *instance, const SMRange &hoverRange){}
  mlir::lsp::Hover buildHoverForStatement(const slang::ast::Statement *stmt,
                                          const SMRange &hoverRange);
  mlir::lsp::Hover
  buildHoverForVariable(const slang::ast::VariableDeclStatement *varDecl,
                        const SMRange &hoverRange);
  mlir::lsp::Hover
  buildHoverForInstance(const slang::ast::InstanceSymbol *instance,
                        const SMRange &hoverRange);

  //===--------------------------------------------------------------------===//
  // Document Symbols
  //===--------------------------------------------------------------------===//

  void findDocumentSymbols(std::vector<mlir::lsp::DocumentSymbol> &symbols);

  //===--------------------------------------------------------------------===//
  // Signature Help
  //===--------------------------------------------------------------------===//

  mlir::lsp::SignatureHelp getSignatureHelp(const mlir::lsp::URIForFile &uri,
                                            const mlir::lsp::Position &helpPos);

  //===--------------------------------------------------------------------===//
  // Inlay Hints
  //===--------------------------------------------------------------------===//

  void getInlayHints(const mlir::lsp::URIForFile &uri,
                     const mlir::lsp::Range &range,
                     std::vector<mlir::lsp::InlayHint> &inlayHints);
  void getInlayHintsFor(const slang::ast::VariableDeclStatement *decl,
                        const mlir::lsp::URIForFile &uri,
                        std::vector<mlir::lsp::InlayHint> &inlayHints);
  void getInlayHintsFor(const slang::ast::InstanceSymbol *instance,
                        const mlir::lsp::URIForFile &uri,
                        std::vector<mlir::lsp::InlayHint> &inlayHints);

  //  /// Add a parameter hint for the given expression using `label`.
  //  void addParameterHintFor(std::vector<mlir::lsp::InlayHint> &inlayHints,
  //                           const ast::Expr *expr, StringRef label);

  //===--------------------------------------------------------------------===//
  // VerilogL ViewOutput
  //===--------------------------------------------------------------------===//

  void getVerilogViewOutput(raw_ostream &os,
                            circt::lsp::VerilogViewOutputKind kind);

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  /// The include directories for this file.
  std::vector<std::string> includeDirs;

  /// The source manager containing the contents of the input file.
  llvm::SourceMgr sourceMgr;
  slang::SourceManager slangSourceMgr;

  /// The parsed AST module, or failure if the file wasn't valid.
  FailureOr<std::unique_ptr<slang::ast::Compilation>> compilation;

  /// The index of the parsed module.
  VerilogIndex index;

  /// The set of includes of the parsed module.
  SmallVector<mlir::lsp::SourceMgrInclude> parsedIncludes;
  mlir::lsp::Location getLocationFromLoc(llvm::SourceMgr &mgr,
                                         slang::SourceLocation loc,
                                         const mlir::lsp::URIForFile &uri) {
    /// Incorrect!
    return convertLocation(slangSourceMgr, loc);
  }
};
} // namespace

VerilogDocument::VerilogDocument(
    const mlir::lsp::URIForFile &uri, StringRef contents,
    const std::vector<std::string> &extraDirs,

    std::vector<mlir::lsp::Diagnostic> &diagnostics) {
  auto memBuffer = llvm::MemoryBuffer::getMemBufferCopy(contents, uri.file());

  if (!memBuffer) {
    mlir::lsp::Logger::error("Failed to create memory buffer for file",
                             uri.file());
    return;
  }

  // Build the set of include directories for this file.
  llvm::SmallString<32> uriDirectory(uri.file());
  llvm::sys::path::remove_filename(uriDirectory);
  includeDirs.push_back(uriDirectory.str().str());
  includeDirs.insert(includeDirs.end(), extraDirs.begin(), extraDirs.end());

  sourceMgr.setIncludeDirs(includeDirs);
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), SMLoc());

  auto slangBuffer =
      slangSourceMgr.assignText(uri.file(), memBuffer->getBuffer());
  // astContext.getDiagEngine().setHandlerFn([&](const slang::ast::Diagnostic
  // &diag) {
  //   // if (auto lspDiag = getLspDiagnoticFromDiag(sourceMgr, diag, uri))
  //   //   diagnostics.push_back(std::move(*lspDiag));
  // });
  // astModule =
  //     parseVerilogAST(astContext, sourceMgr, /*enableDocumentation=*/true);

  auto parsed =
      slang::syntax::SyntaxTree::fromBuffer(slangBuffer, slangSourceMgr, {});
  if (!parsed) {
    mlir::lsp::Logger::error("Failed to parse Verilog file", uri.file());
    return;
  }
  mlir::lsp::Logger::info("parsed:", parsed->root().toString());
  compilation = std::make_unique<slang::ast::Compilation>();
  (*compilation)->addSyntaxTree(parsed);

  // From buffers.
  // const SourceBuffer& buffer,
  //                                                    SourceManager&
  //                                                    sourceManager, const
  //                                                    Bag& options, MacroList
  //                                                    inheritedMacros

  // Initialize the set of parsed includes.
  mlir::lsp::gatherIncludeFiles(sourceMgr, parsedIncludes);

  // If we failed to parse the module, there is nothing left to initialize.
  if (failed(compilation))
    return;

  // Prepare the AST index with the parsed module.
  index.initialize(compilation.value().get());
}

//===----------------------------------------------------------------------===//
// VerilogDocument: Definitions and References
//===----------------------------------------------------------------------===//

void VerilogDocument::getLocationsOf(
    const mlir::lsp::URIForFile &uri, const mlir::lsp::Position &defPos,
    std::vector<mlir::lsp::Location> &locations) {
  SMLoc posLoc = defPos.getAsSMLoc(sourceMgr);
  const VerilogIndexSymbol *symbol = index.lookup(posLoc);
  if (!symbol)
    return;

  locations.push_back(getLocationFromLoc(sourceMgr, symbol->location, uri));
}

void VerilogDocument::findReferencesOf(
    const mlir::lsp::URIForFile &uri, const mlir::lsp::Position &pos,
    std::vector<mlir::lsp::Location> &references) {
  SMLoc posLoc = pos.getAsSMLoc(sourceMgr);
  const VerilogIndexSymbol *symbol = index.lookup(posLoc);
  if (!symbol)
    return;

  references.push_back(
      getLocationFromLoc(sourceMgr, symbol->location, uri));
  // for (SMRange refLoc : symbol->references)
  //   references.push_back(getLocationFromLoc(sourceMgr, refLoc, uri));
}

//===--------------------------------------------------------------------===//
// VerilogDocument: Document Links
//===--------------------------------------------------------------------===//

void VerilogDocument::getDocumentLinks(
    const mlir::lsp::URIForFile &uri,
    std::vector<mlir::lsp::DocumentLink> &links) {
  for (const mlir::lsp::SourceMgrInclude &include : parsedIncludes)
    links.emplace_back(include.range, include.uri);
}

//===----------------------------------------------------------------------===//
// VerilogDocument: Hover
//===----------------------------------------------------------------------===//

std::optional<mlir::lsp::Hover>
VerilogDocument::findHover(const mlir::lsp::URIForFile &uri,
                           const mlir::lsp::Position &hoverPos) {
  SMLoc posLoc = hoverPos.getAsSMLoc(sourceMgr);

  // Check for a reference to an include.
  for (const mlir::lsp::SourceMgrInclude &include : parsedIncludes)
    if (include.range.contains(hoverPos))
      return include.buildHover();

  // Find the symbol at the given location.
  SMRange hoverRange;
  const VerilogIndexSymbol *symbol = index.lookup(posLoc, &hoverRange);
  if (!symbol)
    return std::nullopt;

  // Add hover for operation names.
  // if (const auto *op =
  //        llvm::dyn_cast_if_present<const ods::Operation
  //        *>(symbol->definition))
  //  return buildHoverForOpName(op, hoverRange);
  return findHover(symbol->getDeclaringDefinition(), hoverRange);
}

std::optional<mlir::lsp::Hover>
VerilogDocument::findHover(const slang::ast::Statement *stmt,
                           const SMRange &hoverRange) {
  // Add hover for variables.
  //  if (const auto *varDecl = dyn_cast<ast::VariableDecl>(decl))
  //    return buildHoverForVariable(varDecl, hoverRange);
  //
  //  // Add hover for patterns.
  //  if (const auto *patternDecl = dyn_cast<ast::PatternDecl>(decl))
  //    return buildHoverForPattern(patternDecl, hoverRange);
  //
  //  // Add hover for core constraints.
  //  if (const auto *cst = dyn_cast<ast::CoreConstraintDecl>(decl))
  //    return buildHoverForCoreConstraint(cst, hoverRange);
  //
  //  // Add hover for user constraints.
  //  if (const auto *cst = dyn_cast<ast::UserConstraintDecl>(decl))
  //    return buildHoverForUserConstraintOrRewrite("Constraint", cst,
  //    hoverRange);
  //
  //  // Add hover for user rewrites.
  //  if (const auto *rewrite = dyn_cast<ast::UserRewriteDecl>(decl))
  //    return buildHoverForUserConstraintOrRewrite("Rewrite", rewrite,
  //    hoverRange);
  //
  return std::nullopt;
}

mlir::lsp::Hover VerilogDocument::buildHoverForVariable(
    const slang::ast::VariableDeclStatement *varDecl,
    const SMRange &hoverRange) {
  mlir::lsp::Hover hover(mlir::lsp::Range(sourceMgr, hoverRange));
  {
    llvm::raw_string_ostream hoverOS(hover.contents.value);
    hoverOS << "**Variable**: `" << varDecl->symbol.name << "`\n***\n"
            << "Type: `" << varDecl->symbol.getDeclaredType() << "`\n";
  }
  return hover;
}

//===----------------------------------------------------------------------===//
// VerilogDocument: Document Symbols
//===----------------------------------------------------------------------===//

void VerilogDocument::findDocumentSymbols(
    std::vector<mlir::lsp::DocumentSymbol> &symbols) {
  if (failed(compilation))
    return;

  // for (const slang::ast::Statement *stmt : (*astModule)->getChildren()) {
  //   if (!isMainFileLoc(sourceMgr, stmt->getLoc()))
  //     continue;

  //   if (const auto *patternDecl = dyn_cast<ast::PatternDecl>(decl)) {
  //     const ast::Name *name = patternDecl->getName();

  //     SMRange nameLoc = name ? name->getLoc() : patternDecl->getLoc();
  //     SMRange bodyLoc(nameLoc.Start, patternDecl->getBody()->getLoc().End);

  //     symbols.emplace_back(name ? name->getName() : "<pattern>",
  //                          mlir::lspSymbolKind::Class,
  //                          mlir::lspRange(sourceMgr, bodyLoc),
  //                          mlir::lspRange(sourceMgr, nameLoc));
  //   } else if (const auto *cDecl = dyn_cast<ast::UserConstraintDecl>(decl)) {
  //     // TODO: Add source information for the code block body.
  //     SMRange nameLoc = cDecl->getName().getLoc();
  //     SMRange bodyLoc = nameLoc;

  //     symbols.emplace_back(cDecl->getName().getName(),
  //                          mlir::lspSymbolKind::Function,
  //                          mlir::lspRange(sourceMgr, bodyLoc),
  //                          mlir::lspRange(sourceMgr, nameLoc));
  //   } else if (const auto *cDecl = dyn_cast<ast::UserRewriteDecl>(decl)) {
  //     // TODO: Add source information for the code block body.
  //     SMRange nameLoc = cDecl->getName().getLoc();
  //     SMRange bodyLoc = nameLoc;

  //     symbols.emplace_back(cDecl->getName().getName(),
  //                          mlir::lspSymbolKind::Function,
  //                          mlir::lspRange(sourceMgr, bodyLoc),
  //                          mlir::lspRange(sourceMgr, nameLoc));
  //   }
  // }
}

//===----------------------------------------------------------------------===//
// VerilogDocument: Inlay Hints
//===----------------------------------------------------------------------===//

/// Returns true if the given name should be added as a hint for `expr`.
static bool shouldAddHintFor(const slang::ast::Expression *expr,
                             StringRef name) {
  if (name.empty())
    return false;

  // If the argument is a reference of the same name, don't add it as a hint.
  // if (auto *ref = dyn_cast<ast::DeclRefExpr>(expr)) {
  //   const ast::Name *declName = ref->getDecl()->getName();
  //   if (declName && declName->getName() == name)
  //     return false;
  // }

  return true;
}

void VerilogDocument::getInlayHints(
    const mlir::lsp::URIForFile &uri, const mlir::lsp::Range &range,
    std::vector<mlir::lsp::InlayHint> &inlayHints) {
  if (failed(compilation))
    return;
  SMRange rangeLoc = range.getAsSMRange(sourceMgr);
  if (!rangeLoc.isValid())
    return;
  /*
  (astModule)->walk([&](const ast::Node *node) {
    SMRange loc = node->getLoc();

    // Check that the location of this node is within the input range.
    if (!lsp::contains(rangeLoc, loc.Start) &&
        !lsp::contains(rangeLoc, loc.End))
      return;

    // Handle hints for various types of nodes.
    llvm::TypeSwitch<const ast::Node *>(node)
        .Case<ast::VariableDecl, ast::CallExpr, ast::OperationExpr>(
            [&](const auto *node) {
              this->getInlayHintsFor(node, uri, inlayHints);
            });
  });
  */
}

void VerilogDocument::getInlayHintsFor(
    const slang::ast::VariableDeclStatement *decl,
    const mlir::lsp::URIForFile &uri,
    std::vector<mlir::lsp::InlayHint> &inlayHints) {
  // Check to see if the variable has a constraint list, if it does we don't
  // provide initializer hints.
  // if (!decl->getConstraints().empty())
  //   return;

  // Check to see if the variable has an initializer.
  // if (const ast::Expr *expr = decl->getInitExpr()) {
  //   // Don't add hints for operation expression initialized variables given
  //   that
  //   // the type of the variable is easily inferred by the expression
  //   operation
  //   // name.
  //   if (isa<ast::OperationExpr>(expr))
  //     return;
  // }
  auto loc = decl->sourceRange.end();
  auto pos = slangSourceMgr.getLineNumber(loc);
  auto col = slangSourceMgr.getColumnNumber(loc);
  mlir::lsp::InlayHint hint(mlir::lsp::InlayHintKind::Type,
                            mlir::lsp::Position(pos, col));
  {
    llvm::raw_string_ostream labelOS(hint.label);
    labelOS << ": " << "foo";
  }

  inlayHints.emplace_back(std::move(hint));
}

void VerilogDocument::getInlayHintsFor(
    const slang::ast::InstanceSymbol *instance,
    const mlir::lsp::URIForFile &uri,
    std::vector<mlir::lsp::InlayHint> &inlayHints) {
  // Try to extract the callable of this call.
  auto &foo = instance->getDefinition();
  // const auto *callable =
  //     callableRef ? dyn_cast<ast::CallableDecl>(callableRef->getDecl())
  //                 : nullptr;
  // if (!callable)
  //   return;

  // // Add hints for the arguments to the call.
  // for (const auto &it : llvm::zip(expr->getArguments(),
  // callable->getInputs()))
  //   addParameterHintFor(inlayHints, std::get<0>(it),
  //                       std::get<1>(it)->getName().getName());
}

// void VerilogDocument::addParameterHintFor(
//     std::vector<lsp::InlayHint> &inlayHints, const ast::Expr *expr,
//     StringRef label) {
//   if (!shouldAddHintFor(expr, label))
//     return;
//
//   mlir::lspInlayHint hint(lsp::InlayHintKind::Parameter,
//                           mlir::lspPosition(sourceMgr,
//                           expr->getLoc().Start));
//   hint.label = (label + ":").str();
//   hint.paddingRight = true;
//   inlayHints.emplace_back(std::move(hint));
// }

//===----------------------------------------------------------------------===//
// Verilog ViewOutput
//===----------------------------------------------------------------------===//

void VerilogDocument::getVerilogViewOutput(
    raw_ostream &os, circt::lsp::VerilogViewOutputKind kind) {
  if (failed(compilation))
    return;
  if (kind == circt::lsp::VerilogViewOutputKind::AST) {
    os << compilation.value()->getSyntaxTrees()[0]->root().toString();
    return;
  }

  // Generate the MLIR for the ast module. We also capture diagnostics here to
  // show to the user, which may be useful if Verilog isn't capturing
  // constraints expected by PDL.
  // MLIRContext mlirContext;
  // SourceMgrDiagnosticHandler diagHandler(sourceMgr, &mlirContext, os);
  // OwningOpRef<ModuleOp> pdlModule =
  //     codegenVerilogToMLIR(&mlirContext, astContext, sourceMgr, **astModule);
  // if (!pdlModule)
  //   return;
  // if (kind == mlir::lsp::VerilogViewOutputKind::MLIR) {
  //   pdlModule->print(os, OpPrintingFlags().enableDebugInfo());
  //   return;
  // }

  // // Otherwise, generate the output for C++.
  // assert(kind == mlir::lsp::VerilogViewOutputKind::CPP &&
  //        "unexpected VerilogViewOutputKind");
  // codegenVerilogToCPP(**astModule, *pdlModule, os);
}

//===----------------------------------------------------------------------===//
// VerilogTextFileChunk
//===----------------------------------------------------------------------===//

namespace {
/// This class represents a single chunk of an Verilog text file.
struct VerilogTextFileChunk {
  VerilogTextFileChunk(uint64_t lineOffset, const mlir::lsp::URIForFile &uri,
                       StringRef contents,
                       const std::vector<std::string> &extraDirs,
                       std::vector<mlir::lsp::Diagnostic> &diagnostics)
      : lineOffset(lineOffset),
        document(uri, contents, extraDirs, diagnostics) {}

  /// Adjust the line number of the given range to anchor at the beginning of
  /// the file, instead of the beginning of this chunk.
  void adjustLocForChunkOffset(mlir::lsp::Range &range) {
    adjustLocForChunkOffset(range.start);
    adjustLocForChunkOffset(range.end);
  }
  /// Adjust the line number of the given position to anchor at the beginning of
  /// the file, instead of the beginning of this chunk.
  void adjustLocForChunkOffset(mlir::lsp::Position &pos) {
    pos.line += lineOffset;
  }

  /// The line offset of this chunk from the beginning of the file.
  uint64_t lineOffset;
  /// The document referred to by this chunk.
  VerilogDocument document;
};
} // namespace

//===----------------------------------------------------------------------===//
// VerilogTextFile
//===----------------------------------------------------------------------===//

namespace {
/// This class represents a text file containing one or more Verilog documents.
class VerilogTextFile {
public:
  VerilogTextFile(const mlir::lsp::URIForFile &uri, StringRef fileContents,
                  int64_t version, const std::vector<std::string> &extraDirs,
                  std::vector<mlir::lsp::Diagnostic> &diagnostics);

  /// Return the current version of this text file.
  int64_t getVersion() const { return version; }

  /// Update the file to the new version using the provided set of content
  /// changes. Returns failure if the update was unsuccessful.
  LogicalResult
  update(const mlir::lsp::URIForFile &uri, int64_t newVersion,
         ArrayRef<mlir::lsp::TextDocumentContentChangeEvent> changes,
         std::vector<mlir::lsp::Diagnostic> &diagnostics);

  //===--------------------------------------------------------------------===//
  // LSP Queries
  //===--------------------------------------------------------------------===//

  void getLocationsOf(const mlir::lsp::URIForFile &uri,
                      mlir::lsp::Position defPos,
                      std::vector<mlir::lsp::Location> &locations);
  void findReferencesOf(const mlir::lsp::URIForFile &uri,
                        mlir::lsp::Position pos,
                        std::vector<mlir::lsp::Location> &references);
  void getDocumentLinks(const mlir::lsp::URIForFile &uri,
                        std::vector<mlir::lsp::DocumentLink> &links);
  std::optional<mlir::lsp::Hover> findHover(const mlir::lsp::URIForFile &uri,
                                            mlir::lsp::Position hoverPos);
  void findDocumentSymbols(std::vector<mlir::lsp::DocumentSymbol> &symbols);
  mlir::lsp::CompletionList getCodeCompletion(const mlir::lsp::URIForFile &uri,
                                              mlir::lsp::Position completePos);
  mlir::lsp::SignatureHelp getSignatureHelp(const mlir::lsp::URIForFile &uri,
                                            mlir::lsp::Position helpPos);
  void getInlayHints(const mlir::lsp::URIForFile &uri, mlir::lsp::Range range,
                     std::vector<mlir::lsp::InlayHint> &inlayHints);
  circt::lsp::VerilogViewOutputResult
  getVerilogViewOutput(circt::lsp::VerilogViewOutputKind kind);

private:
  using ChunkIterator = llvm::pointee_iterator<
      std::vector<std::unique_ptr<VerilogTextFileChunk>>::iterator>;

  /// Initialize the text file from the given file contents.
  void initialize(const mlir::lsp::URIForFile &uri, int64_t newVersion,
                  std::vector<mlir::lsp::Diagnostic> &diagnostics);

  /// Find the PDL document that contains the given position, and update the
  /// position to be anchored at the start of the found chunk instead of the
  /// beginning of the file.
  ChunkIterator getChunkItFor(mlir::lsp::Position &pos);
  VerilogTextFileChunk &getChunkFor(mlir::lsp::Position &pos) {
    return *getChunkItFor(pos);
  }

  /// The full string contents of the file.
  std::string contents;

  /// The version of this file.
  int64_t version = 0;

  /// The number of lines in the file.
  int64_t totalNumLines = 0;

  /// The chunks of this file. The order of these chunks is the order in which
  /// they appear in the text file.
  std::vector<std::unique_ptr<VerilogTextFileChunk>> chunks;

  /// The extra set of include directories for this file.
  std::vector<std::string> extraIncludeDirs;
};
} // namespace

VerilogTextFile::VerilogTextFile(
    const mlir::lsp::URIForFile &uri, StringRef fileContents, int64_t version,
    const std::vector<std::string> &extraDirs,
    std::vector<mlir::lsp::Diagnostic> &diagnostics)
    : contents(fileContents.str()), extraIncludeDirs(extraDirs) {
  initialize(uri, version, diagnostics);
}

LogicalResult VerilogTextFile::update(
    const mlir::lsp::URIForFile &uri, int64_t newVersion,
    ArrayRef<mlir::lsp::TextDocumentContentChangeEvent> changes,
    std::vector<mlir::lsp::Diagnostic> &diagnostics) {
  if (failed(mlir::lsp::TextDocumentContentChangeEvent::applyTo(changes,
                                                                contents))) {
    mlir::lsp::Logger::error("Failed to update contents of {0}", uri.file());
    return failure();
  }

  // If the file contents were properly changed, reinitialize the text file.
  initialize(uri, newVersion, diagnostics);
  return success();
}

void VerilogTextFile::getLocationsOf(
    const mlir::lsp::URIForFile &uri, mlir::lsp::Position defPos,
    std::vector<mlir::lsp::Location> &locations) {
  VerilogTextFileChunk &chunk = getChunkFor(defPos);
  chunk.document.getLocationsOf(uri, defPos, locations);

  // Adjust any locations within this file for the offset of this chunk.
  if (chunk.lineOffset == 0)
    return;
  for (mlir::lsp::Location &loc : locations)
    if (loc.uri == uri)
      chunk.adjustLocForChunkOffset(loc.range);
}

void VerilogTextFile::findReferencesOf(
    const mlir::lsp::URIForFile &uri, mlir::lsp::Position pos,
    std::vector<mlir::lsp::Location> &references) {
  VerilogTextFileChunk &chunk = getChunkFor(pos);
  chunk.document.findReferencesOf(uri, pos, references);

  // Adjust any locations within this file for the offset of this chunk.
  if (chunk.lineOffset == 0)
    return;
  for (mlir::lsp::Location &loc : references)
    if (loc.uri == uri)
      chunk.adjustLocForChunkOffset(loc.range);
}

void VerilogTextFile::getDocumentLinks(
    const mlir::lsp::URIForFile &uri,
    std::vector<mlir::lsp::DocumentLink> &links) {
  chunks.front()->document.getDocumentLinks(uri, links);
  for (const auto &it : llvm::drop_begin(chunks)) {
    size_t currentNumLinks = links.size();
    it->document.getDocumentLinks(uri, links);

    // Adjust any links within this file to account for the offset of this
    // chunk.
    for (auto &link : llvm::drop_begin(links, currentNumLinks))
      it->adjustLocForChunkOffset(link.range);
  }
}

std::optional<mlir::lsp::Hover>
VerilogTextFile::findHover(const mlir::lsp::URIForFile &uri,
                           mlir::lsp::Position hoverPos) {
  VerilogTextFileChunk &chunk = getChunkFor(hoverPos);
  std::optional<mlir::lsp::Hover> hoverInfo =
      chunk.document.findHover(uri, hoverPos);

  // Adjust any locations within this file for the offset of this chunk.
  if (chunk.lineOffset != 0 && hoverInfo && hoverInfo->range)
    chunk.adjustLocForChunkOffset(*hoverInfo->range);
  return hoverInfo;
}

void VerilogTextFile::findDocumentSymbols(
    std::vector<mlir::lsp::DocumentSymbol> &symbols) {
  if (chunks.size() == 1)
    return chunks.front()->document.findDocumentSymbols(symbols);

  // If there are multiple chunks in this file, we create top-level symbols for
  // each chunk.
  for (unsigned i = 0, e = chunks.size(); i < e; ++i) {
    VerilogTextFileChunk &chunk = *chunks[i];
    mlir::lsp::Position startPos(chunk.lineOffset);
    mlir::lsp::Position endPos((i == e - 1) ? totalNumLines - 1
                                            : chunks[i + 1]->lineOffset);
    mlir::lsp::DocumentSymbol symbol(
        "<file-split-" + Twine(i) + ">", mlir::lsp::SymbolKind::Namespace,
        /*range=*/mlir::lsp::Range(startPos, endPos),
        /*selectionRange=*/mlir::lsp::Range(startPos));
    chunk.document.findDocumentSymbols(symbol.children);

    // Fixup the locations of document symbols within this chunk.
    if (i != 0) {
      SmallVector<mlir::lsp::DocumentSymbol *> symbolsToFix;
      for (mlir::lsp::DocumentSymbol &childSymbol : symbol.children)
        symbolsToFix.push_back(&childSymbol);

      while (!symbolsToFix.empty()) {
        mlir::lsp::DocumentSymbol *symbol = symbolsToFix.pop_back_val();
        chunk.adjustLocForChunkOffset(symbol->range);
        chunk.adjustLocForChunkOffset(symbol->selectionRange);

        for (mlir::lsp::DocumentSymbol &childSymbol : symbol->children)
          symbolsToFix.push_back(&childSymbol);
      }
    }

    // Push the symbol for this chunk.
    symbols.emplace_back(std::move(symbol));
  }
}

void VerilogTextFile::getInlayHints(
    const mlir::lsp::URIForFile &uri, mlir::lsp::Range range,
    std::vector<mlir::lsp::InlayHint> &inlayHints) {
  auto startIt = getChunkItFor(range.start);
  auto endIt = getChunkItFor(range.end);

  // Functor used to get the chunks for a given file, and fixup any locations
  auto getHintsForChunk = [&](ChunkIterator chunkIt, mlir::lsp::Range range) {
    size_t currentNumHints = inlayHints.size();
    chunkIt->document.getInlayHints(uri, range, inlayHints);

    // If this isn't the first chunk, update any positions to account for line
    // number differences.
    if (&*chunkIt != &*chunks.front()) {
      for (auto &hint : llvm::drop_begin(inlayHints, currentNumHints))
        chunkIt->adjustLocForChunkOffset(hint.position);
    }
  };
  // Returns the number of lines held by a given chunk.
  auto getNumLines = [](ChunkIterator chunkIt) {
    return (chunkIt + 1)->lineOffset - chunkIt->lineOffset;
  };

  // Check if the range is fully within a single chunk.
  if (startIt == endIt)
    return getHintsForChunk(startIt, range);

  // Otherwise, the range is split between multiple chunks. The first chunk
  // has the correct range start, but covers the total document.
  getHintsForChunk(startIt,
                   mlir::lsp::Range(range.start, getNumLines(startIt)));

  // Every chunk in between uses the full document.
  for (++startIt; startIt != endIt; ++startIt)
    getHintsForChunk(startIt, mlir::lsp::Range(0, getNumLines(startIt)));

  // The range for the last chunk starts at the beginning of the document, up
  // through the end of the input range.
  getHintsForChunk(startIt, mlir::lsp::Range(0, range.end));
}

circt::lsp::VerilogViewOutputResult
VerilogTextFile::getVerilogViewOutput(circt::lsp::VerilogViewOutputKind kind) {
  circt::lsp::VerilogViewOutputResult result;
  {
    llvm::raw_string_ostream outputOS(result.output);
    llvm::interleave(
        llvm::make_pointee_range(chunks),
        [&](VerilogTextFileChunk &chunk) {
          chunk.document.getVerilogViewOutput(outputOS, kind);
        },
        [&] { outputOS << "\n"
                       << kDefaultSplitMarker << "\n\n"; });
  }
  return result;
}

void VerilogTextFile::initialize(
    const mlir::lsp::URIForFile &uri, int64_t newVersion,
    std::vector<mlir::lsp::Diagnostic> &diagnostics) {
  version = newVersion;
  chunks.clear();

  // Split the file into separate PDL documents.
  SmallVector<StringRef, 8> subContents;
  StringRef(contents).split(subContents, kDefaultSplitMarker);
  chunks.emplace_back(std::make_unique<VerilogTextFileChunk>(
      /*lineOffset=*/0, uri, subContents.front(), extraIncludeDirs,
      diagnostics));

  uint64_t lineOffset = subContents.front().count('\n');
  for (StringRef docContents : llvm::drop_begin(subContents)) {
    unsigned currentNumDiags = diagnostics.size();
    auto chunk = std::make_unique<VerilogTextFileChunk>(
        lineOffset, uri, docContents, extraIncludeDirs, diagnostics);
    lineOffset += docContents.count('\n');

    // Adjust locations used in diagnostics to account for the offset from the
    // beginning of the file.
    for (mlir::lsp::Diagnostic &diag :
         llvm::drop_begin(diagnostics, currentNumDiags)) {
      chunk->adjustLocForChunkOffset(diag.range);

      if (!diag.relatedInformation)
        continue;
      for (auto &it : *diag.relatedInformation)
        if (it.location.uri == uri)
          chunk->adjustLocForChunkOffset(it.location.range);
    }
    chunks.emplace_back(std::move(chunk));
  }
  totalNumLines = lineOffset;
}

VerilogTextFile::ChunkIterator
VerilogTextFile::getChunkItFor(mlir::lsp::Position &pos) {
  if (chunks.size() == 1)
    return chunks.begin();

  // Search for the first chunk with a greater line offset, the previous chunk
  // is the one that contains `pos`.
  auto it = llvm::upper_bound(
      chunks, pos, [](const mlir::lsp::Position &pos, const auto &chunk) {
        return static_cast<uint64_t>(pos.line) < chunk->lineOffset;
      });
  ChunkIterator chunkIt(it == chunks.end() ? (chunks.end() - 1) : --it);
  pos.line -= chunkIt->lineOffset;
  return chunkIt;
}

//===----------------------------------------------------------------------===//
// VerilogServer::Impl
//===----------------------------------------------------------------------===//

struct circt::lsp::VerilogServer::Impl {
  explicit Impl(const VerilogServerOptions &options)
      : options(options), compilationDatabase(options.compilationDatabases) {}

  /// Verilog LSP options.
  const Options &options;

  /// The compilation database containing additional information for files
  /// passed to the server.
  mlir::lsp::CompilationDatabase compilationDatabase;

  /// The files held by the server, mapped by their URI file name.
  llvm::StringMap<std::unique_ptr<VerilogTextFile>> files;
};

//===----------------------------------------------------------------------===//
// VerilogServer
//===----------------------------------------------------------------------===//

circt::lsp::VerilogServer::VerilogServer(const Options &options)
    : impl(std::make_unique<Impl>(options)) {}
circt::lsp::VerilogServer::~VerilogServer() = default;

void circt::lsp::VerilogServer::addDocument(
    const URIForFile &uri, StringRef contents, int64_t version,
    std::vector<mlir::lsp::Diagnostic> &diagnostics) {
  // Build the set of additional include directories.
  std::vector<std::string> additionalIncludeDirs = impl->options.extraDirs;
  const auto &fileInfo = impl->compilationDatabase.getFileInfo(uri.file());
  llvm::append_range(additionalIncludeDirs, fileInfo.includeDirs);

  impl->files[uri.file()] = std::make_unique<VerilogTextFile>(
      uri, contents, version, additionalIncludeDirs, diagnostics);
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

void circt::lsp::VerilogServer::getDocumentLinks(
    const URIForFile &uri,
    std::vector<mlir::lsp::DocumentLink> &documentLinks) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    return fileIt->second->getDocumentLinks(uri, documentLinks);
}

std::optional<mlir::lsp::Hover>
circt::lsp::VerilogServer::findHover(const URIForFile &uri,
                                     const Position &hoverPos) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    return fileIt->second->findHover(uri, hoverPos);
  return std::nullopt;
}

void circt::lsp::VerilogServer::findDocumentSymbols(
    const URIForFile &uri, std::vector<mlir::lsp::DocumentSymbol> &symbols) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    fileIt->second->findDocumentSymbols(symbols);
}

void circt::lsp::VerilogServer::getInlayHints(
    const URIForFile &uri, const mlir::lsp::Range &range,
    std::vector<mlir::lsp::InlayHint> &inlayHints) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt == impl->files.end())
    return;
  fileIt->second->getInlayHints(uri, range, inlayHints);

  // Drop any duplicated hints that may have cropped up.
  llvm::sort(inlayHints);
  inlayHints.erase(llvm::unique(inlayHints), inlayHints.end());
}

std::optional<circt::lsp::VerilogViewOutputResult>
circt::lsp::VerilogServer::getVerilogViewOutput(const URIForFile &uri,
                                                VerilogViewOutputKind kind) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    return fileIt->second->getVerilogViewOutput(kind);
  return std::nullopt;
}
