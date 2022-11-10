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
#include "circt/Dialect/Moore/MooreDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/Timing.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "slang/diagnostics/DiagnosticClient.h"
#include "slang/driver/Driver.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/SourceMgr.h"

using namespace circt;
using namespace ImportVerilog;

using llvm::SourceMgr;

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

/// Convert a slang `SourceLocation` to an MLIR `Location`.
Location ImportVerilog::convertLocation(
    MLIRContext *context, const slang::SourceManager &sourceManager,
    llvm::function_ref<StringRef(slang::BufferID)> getBufferFilePath,
    slang::SourceLocation loc) {
  if (loc && loc.buffer() != slang::SourceLocation::NoLocation.buffer()) {
    auto fileName = getBufferFilePath(loc.buffer());
    auto line = sourceManager.getLineNumber(loc);
    auto column = sourceManager.getColumnNumber(loc);
    return FileLineColLoc::get(context, fileName, line, column);
  }
  return UnknownLoc::get(context);
}

Location Context::convertLocation(slang::SourceLocation loc) {
  return ImportVerilog::convertLocation(getContext(), sourceManager,
                                        getBufferFilePath, loc);
}

namespace {
/// A converter that can be plugged into a slang `DiagnosticEngine` as a client
/// that will map slang diagnostics to their MLIR counterpart and emit them.
class MlirDiagnosticClient : public slang::DiagnosticClient {
public:
  MlirDiagnosticClient(
      MLIRContext *context,
      std::function<StringRef(slang::BufferID)> getBufferFilePath)
      : context(context), getBufferFilePath(getBufferFilePath) {}

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
      auto macro_name = sourceManager->getMacroName(*it);
      if (macro_name.empty())
        note << "expanded from here";
      else
        note << "expanded from macro '" << macro_name << "'";
    }
  }

  /// Convert a slang `SourceLocation` to an MLIR `Location`.
  Location convertLocation(slang::SourceLocation loc) const {
    return ImportVerilog::convertLocation(context, *sourceManager,
                                          getBufferFilePath, loc);
  }

  static mlir::DiagnosticSeverity
  getSeverity(slang::DiagnosticSeverity severity) {
    switch (severity) {
    case slang::DiagnosticSeverity::Fatal:
    case slang::DiagnosticSeverity::Error:
      return mlir::DiagnosticSeverity::Error;
    case slang::DiagnosticSeverity::Warning:
      return mlir::DiagnosticSeverity::Warning;
    case slang::DiagnosticSeverity::Ignored:
    case slang::DiagnosticSeverity::Note:
      return mlir::DiagnosticSeverity::Remark;
    }
    llvm_unreachable("all slang diagnostic severities should be handled");
    return mlir::DiagnosticSeverity::Error;
  }

private:
  MLIRContext *context;
  std::function<StringRef(slang::BufferID)> getBufferFilePath;
};
} // namespace

// Allow for slang::BufferID to be used as hash map keys.
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

// Parse the specified Verilog inputs into the specified MLIR context.
mlir::OwningOpRef<mlir::ModuleOp> circt::importVerilog(SourceMgr &sourceMgr,
                                                       MLIRContext *context,
                                                       mlir::TimingScope &ts) {
  // Use slang's driver which conveniently packages a lot of the things we need
  // for compilation.
  slang::driver::Driver driver;

  // We keep a separate map from slang's buffers to the original MLIR file name
  // since slang's `SourceLocation::getFileName` returns a modified version that
  // is nice for human consumption (proximate paths, just file names, etc.), but
  // breaks MLIR's assumption that the diagnostics report the exact file paths
  // that appear in the `SourceMgr`. We use this separate map to lookup the
  // exact paths and use those for reporting.
  // See: https://github.com/MikePopoloski/slang/discussions/658
  SmallDenseMap<slang::BufferID, StringRef> bufferFilePaths;
  auto getBufferFilePath = [&](slang::BufferID id) {
    return bufferFilePaths.lookup(id);
  };

  auto diagClient =
      std::make_shared<MlirDiagnosticClient>(context, getBufferFilePath);
  driver.diagEngine.addClient(diagClient);

  // Populate the source manager with the source files.
  // NOTE: This is a bit ugly since we're essentially copying the Verilog source
  // text in memory. At a later stage we might want to extend slang's
  // SourceManager such that it can contain non-owned buffers. This will do for
  // now.
  for (unsigned i = 0, e = sourceMgr.getNumBuffers(); i < e; ++i) {
    const llvm::MemoryBuffer *mlirBuffer = sourceMgr.getMemoryBuffer(i + 1);
    auto slangBuffer = driver.sourceManager.assignText(
        mlirBuffer->getBufferIdentifier(), mlirBuffer->getBuffer());
    driver.buffers.push_back(slangBuffer);
    bufferFilePaths.insert({slangBuffer.id, mlirBuffer->getBufferIdentifier()});
  }

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
    return {};
  compileTimer.stop();

  // Traverse the parsed Verilog AST and map it to the equivalent CIRCT ops.
  context->loadDialect<moore::MooreDialect>();
  mlir::OwningOpRef<ModuleOp> module(
      ModuleOp::create(UnknownLoc::get(context)));
  auto conversionTimer = ts.nest("Verilog to dialect mapping");
  Context ctx(module.get(), driver.sourceManager, getBufferFilePath,
              *compilation);
  if (failed(ctx.convertCompilation()))
    return {};
  conversionTimer.stop();

  // Run the verifier on the constructed module to ensure it is clean.
  auto verifierTimer = ts.nest("Post-parse verification");
  if (failed(verify(*module)))
    return {};
  return module;
}

void circt::registerFromVerilogTranslation() {
  static mlir::TranslateToMLIRRegistration fromVerilog(
      "import-verilog", "import Verilog or SystemVerilog",
      [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        mlir::TimingScope ts;
        return importVerilog(sourceMgr, context, ts);
      });
}
