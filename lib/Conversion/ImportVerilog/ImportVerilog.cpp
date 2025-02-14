//===- ImportVerilog.cpp - Slang Verilog frontend integration -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements bridging from the slang Verilog frontend to CIRCT dialects.
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/Timing.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/SourceMgr.h"

#include "slang/diagnostics/DiagnosticClient.h"
#include "slang/driver/Driver.h"
#include "slang/parsing/Preprocessor.h"
#include "slang/syntax/SyntaxPrinter.h"
#include "slang/util/Version.h"

using namespace mlir;
using namespace circt;
using namespace ImportVerilog;

using llvm::SourceMgr;

std::string circt::getSlangVersion() {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  os << "slang version ";
  os << slang::VersionInfo::getMajor() << ".";
  os << slang::VersionInfo::getMinor() << ".";
  os << slang::VersionInfo::getPatch() << "+";
  os << slang::VersionInfo::getHash();
  return buffer;
}

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

/// Convert a slang `SourceLocation` to an MLIR `Location`.
static Location
convertLocation(MLIRContext *context, const slang::SourceManager &sourceManager,
                SmallDenseMap<slang::BufferID, StringRef> &bufferFilePaths,
                slang::SourceLocation loc) {
  if (loc && loc.buffer() != slang::SourceLocation::NoLocation.buffer()) {
    auto fileName = bufferFilePaths.lookup(loc.buffer());
    auto line = sourceManager.getLineNumber(loc);
    auto column = sourceManager.getColumnNumber(loc);
    return FileLineColLoc::get(context, fileName, line, column);
  }
  return UnknownLoc::get(context);
}

Location Context::convertLocation(slang::SourceLocation loc) {
  return ::convertLocation(getContext(), sourceManager, bufferFilePaths, loc);
}

Location Context::convertLocation(slang::SourceRange range) {
  return convertLocation(range.start());
}

namespace {
/// A converter that can be plugged into a slang `DiagnosticEngine` as a client
/// that will map slang diagnostics to their MLIR counterpart and emit them.
class MlirDiagnosticClient : public slang::DiagnosticClient {
public:
  MlirDiagnosticClient(
      MLIRContext *context,
      SmallDenseMap<slang::BufferID, StringRef> &bufferFilePaths)
      : context(context), bufferFilePaths(bufferFilePaths) {}

  void report(const slang::ReportedDiagnostic &diag) override {
    // Generate the primary MLIR diagnostic.
    auto &diagEngine = context->getDiagEngine();
    auto mlirDiag = diagEngine.emit(convertLocation(diag.location),
                                    getSeverity(diag.severity));
    mlirDiag << diag.formattedMessage;

    // Append the name of the option that can be used to control this
    // diagnostic.
    auto optionName = engine->getOptionName(diag.originalDiagnostic.code);
    if (!optionName.empty())
      mlirDiag << " [-W" << optionName << "]";

    // Write out macro expansions, if we have any, in reverse order.
    for (auto it = diag.expansionLocs.rbegin(); it != diag.expansionLocs.rend();
         it++) {
      auto &note = mlirDiag.attachNote(
          convertLocation(sourceManager->getFullyOriginalLoc(*it)));
      auto macroName = sourceManager->getMacroName(*it);
      if (macroName.empty())
        note << "expanded from here";
      else
        note << "expanded from macro '" << macroName << "'";
    }
  }

  /// Convert a slang `SourceLocation` to an MLIR `Location`.
  Location convertLocation(slang::SourceLocation loc) const {
    return ::convertLocation(context, *sourceManager, bufferFilePaths, loc);
  }

  static DiagnosticSeverity getSeverity(slang::DiagnosticSeverity severity) {
    switch (severity) {
    case slang::DiagnosticSeverity::Fatal:
    case slang::DiagnosticSeverity::Error:
      return DiagnosticSeverity::Error;
    case slang::DiagnosticSeverity::Warning:
      return DiagnosticSeverity::Warning;
    case slang::DiagnosticSeverity::Ignored:
    case slang::DiagnosticSeverity::Note:
      return DiagnosticSeverity::Remark;
    }
    llvm_unreachable("all slang diagnostic severities should be handled");
    return DiagnosticSeverity::Error;
  }

private:
  MLIRContext *context;
  SmallDenseMap<slang::BufferID, StringRef> &bufferFilePaths;
};
} // namespace

// Allow for `slang::BufferID` to be used as hash map keys.
namespace llvm {
template <>
struct DenseMapInfo<slang::BufferID> {
  static slang::BufferID getEmptyKey() { return slang::BufferID(); }
  static slang::BufferID getTombstoneKey() {
    return slang::BufferID(UINT32_MAX - 1, ""sv);
    // UINT32_MAX is already used by `BufferID::getPlaceholder`.
  }
  static unsigned getHashValue(slang::BufferID id) {
    return llvm::hash_value(id.getId());
  }
  static bool isEqual(slang::BufferID a, slang::BufferID b) { return a == b; }
};
} // namespace llvm

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

namespace {
const static ImportVerilogOptions defaultOptions;

struct ImportDriver {
  ImportDriver(MLIRContext *mlirContext, TimingScope &ts,
               const ImportVerilogOptions *options)
      : mlirContext(mlirContext), ts(ts),
        options(options ? *options : defaultOptions) {}

  LogicalResult prepareDriver(SourceMgr &sourceMgr);
  LogicalResult importVerilog(ModuleOp module);
  LogicalResult preprocessVerilog(llvm::raw_ostream &os);

  MLIRContext *mlirContext;
  TimingScope &ts;
  const ImportVerilogOptions &options;

  // Use slang's driver which conveniently packages a lot of the things we
  // need for compilation.
  slang::driver::Driver driver;

  // A separate mapping from slang's buffers to the original MLIR file name
  // since slang's `SourceLocation::getFileName` returns a modified version that
  // is nice for human consumption (proximate paths, just file names, etc.), but
  // breaks MLIR's assumption that the diagnostics report the exact file paths
  // that appear in the `SourceMgr`. We use this separate map to lookup the
  // exact paths and use those for reporting.
  //
  // See: https://github.com/MikePopoloski/slang/discussions/658
  SmallDenseMap<slang::BufferID, StringRef> bufferFilePaths;
};
} // namespace

/// Populate the Slang driver with source files from the given `sourceMgr`, and
/// configure driver options based on the `ImportVerilogOptions` passed to the
/// `ImportDriver` constructor.
LogicalResult ImportDriver::prepareDriver(SourceMgr &sourceMgr) {
  // Use slang's driver which conveniently packages a lot of the things we
  // need for compilation.
  auto diagClient =
      std::make_shared<MlirDiagnosticClient>(mlirContext, bufferFilePaths);
  driver.diagEngine.addClient(diagClient);

  // Populate the source manager with the source files.
  // NOTE: This is a bit ugly since we're essentially copying the Verilog
  // source text in memory. At a later stage we'll want to extend slang's
  // SourceManager such that it can contain non-owned buffers. This will do
  // for now.
  for (unsigned i = 0, e = sourceMgr.getNumBuffers(); i < e; ++i) {
    const llvm::MemoryBuffer *mlirBuffer = sourceMgr.getMemoryBuffer(i + 1);
    auto slangBuffer = driver.sourceManager.assignText(
        mlirBuffer->getBufferIdentifier(), mlirBuffer->getBuffer());
    driver.buffers.push_back(slangBuffer);
    bufferFilePaths.insert({slangBuffer.id, mlirBuffer->getBufferIdentifier()});
  }

  // Populate the driver options.
  driver.options.includeDirs = options.includeDirs;
  driver.options.includeSystemDirs = options.includeSystemDirs;
  driver.options.libDirs = options.libDirs;
  driver.options.libExts = options.libExts;
  driver.options.excludeExts.insert(options.excludeExts.begin(),
                                    options.excludeExts.end());
  driver.options.ignoreDirectives = options.ignoreDirectives;

  driver.options.maxIncludeDepth = options.maxIncludeDepth;
  driver.options.defines = options.defines;
  driver.options.undefines = options.undefines;
  driver.options.librariesInheritMacros = options.librariesInheritMacros;

  driver.options.timeScale = options.timeScale;
  driver.options.allowUseBeforeDeclare = options.allowUseBeforeDeclare;
  driver.options.ignoreUnknownModules = options.ignoreUnknownModules;
  driver.options.onlyLint =
      options.mode == ImportVerilogOptions::Mode::OnlyLint;
  driver.options.topModules = options.topModules;
  driver.options.paramOverrides = options.paramOverrides;

  driver.options.errorLimit = options.errorLimit;
  driver.options.warningOptions = options.warningOptions;
  driver.options.suppressWarningsPaths = options.suppressWarningsPaths;

  driver.options.singleUnit = options.singleUnit;
  driver.options.libraryFiles = options.libraryFiles;

  for (auto &dir : sourceMgr.getIncludeDirs())
    driver.options.includeDirs.push_back(dir);

  return success(driver.processOptions());
}

/// Parse and elaborate the prepared source files, and populate the given MLIR
/// `module` with corresponding operations.
LogicalResult ImportDriver::importVerilog(ModuleOp module) {
  // Parse the input.
  auto parseTimer = ts.nest("Verilog parser");
  bool parseSuccess = driver.parseAllSources();
  parseTimer.stop();

  // Elaborate the input.
  auto compileTimer = ts.nest("Verilog elaboration");
  auto compilation = driver.createCompilation();
  for (auto &diag : compilation->getAllDiagnostics())
    driver.diagEngine.issue(diag);
  if (!parseSuccess || driver.diagEngine.getNumErrors() > 0)
    return failure();
  compileTimer.stop();

  // If we were only supposed to lint the input, return here. This leaves the
  // module empty, but any Slang linting messages got reported as diagnostics.
  if (options.mode == ImportVerilogOptions::Mode::OnlyLint)
    return success();

  // Traverse the parsed Verilog AST and map it to the equivalent CIRCT ops.
  mlirContext
      ->loadDialect<moore::MooreDialect, hw::HWDialect, cf::ControlFlowDialect,
                    func::FuncDialect, ltl::LTLDialect, debug::DebugDialect>();
  auto conversionTimer = ts.nest("Verilog to dialect mapping");
  Context context(options, *compilation, module, driver.sourceManager,
                  bufferFilePaths);
  if (failed(context.convertCompilation()))
    return failure();
  conversionTimer.stop();

  // Run the verifier on the constructed module to ensure it is clean.
  auto verifierTimer = ts.nest("Post-parse verification");
  return verify(module);
}

/// Preprocess the prepared source files and print them to the given output
/// stream.
LogicalResult ImportDriver::preprocessVerilog(llvm::raw_ostream &os) {
  auto parseTimer = ts.nest("Verilog preprocessing");

  // Run the preprocessor to completion across all sources previously added with
  // `pushSource`, report diagnostics, and print the output.
  auto preprocessAndPrint = [&](slang::parsing::Preprocessor &preprocessor) {
    slang::syntax::SyntaxPrinter output;
    output.setIncludeComments(false);
    while (true) {
      slang::parsing::Token token = preprocessor.next();
      output.print(token);
      if (token.kind == slang::parsing::TokenKind::EndOfFile)
        break;
    }

    for (auto &diag : preprocessor.getDiagnostics()) {
      if (diag.isError()) {
        driver.diagEngine.issue(diag);
        return failure();
      }
    }
    os << output.str();
    return success();
  };

  // Depending on whether the single-unit option is set, either add all source
  // files to a single preprocessor such that they share define macros and
  // directives, or create a separate preprocessor for each, such that each
  // source file is in its own compilation unit.
  auto optionBag = driver.createOptionBag();
  if (driver.options.singleUnit == true) {
    slang::BumpAllocator alloc;
    slang::Diagnostics diagnostics;
    slang::parsing::Preprocessor preprocessor(driver.sourceManager, alloc,
                                              diagnostics, optionBag);
    // Sources have to be pushed in reverse, as they form a stack in the
    // preprocessor. Last pushed source is processed first.
    for (auto &buffer : slang::make_reverse_range(driver.buffers))
      preprocessor.pushSource(buffer);
    if (failed(preprocessAndPrint(preprocessor)))
      return failure();
  } else {
    for (auto &buffer : driver.buffers) {
      slang::BumpAllocator alloc;
      slang::Diagnostics diagnostics;
      slang::parsing::Preprocessor preprocessor(driver.sourceManager, alloc,
                                                diagnostics, optionBag);
      preprocessor.pushSource(buffer);
      if (failed(preprocessAndPrint(preprocessor)))
        return failure();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Entry Points
//===----------------------------------------------------------------------===//

/// Execute a callback and report any thrown exceptions as "internal slang
/// error" MLIR diagnostics.
static LogicalResult
catchExceptions(llvm::function_ref<LogicalResult()> callback) {
  try {
    return callback();
  } catch (const std::exception &e) {
    return emitError(UnknownLoc(), "internal slang error: ") << e.what();
  }
}

/// Parse the specified Verilog inputs into the specified MLIR context.
LogicalResult circt::importVerilog(SourceMgr &sourceMgr,
                                   MLIRContext *mlirContext, TimingScope &ts,
                                   ModuleOp module,
                                   const ImportVerilogOptions *options) {
  return catchExceptions([&] {
    ImportDriver importDriver(mlirContext, ts, options);
    if (failed(importDriver.prepareDriver(sourceMgr)))
      return failure();
    return importDriver.importVerilog(module);
  });
}

/// Run the files in a source manager through Slang's Verilog preprocessor and
/// emit the result to the given output stream.
LogicalResult circt::preprocessVerilog(SourceMgr &sourceMgr,
                                       MLIRContext *mlirContext,
                                       TimingScope &ts, llvm::raw_ostream &os,
                                       const ImportVerilogOptions *options) {
  return catchExceptions([&] {
    ImportDriver importDriver(mlirContext, ts, options);
    if (failed(importDriver.prepareDriver(sourceMgr)))
      return failure();
    return importDriver.preprocessVerilog(os);
  });
}

/// Entry point as an MLIR translation.
void circt::registerFromVerilogTranslation() {
  static TranslateToMLIRRegistration fromVerilog(
      "import-verilog", "import Verilog or SystemVerilog",
      [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        TimingScope ts;
        OwningOpRef<ModuleOp> module(
            ModuleOp::create(UnknownLoc::get(context)));
        ImportVerilogOptions options;
        options.debugInfo = true;
        if (failed(
                importVerilog(sourceMgr, context, ts, module.get(), &options)))
          module = {};
        return module;
      });
}
