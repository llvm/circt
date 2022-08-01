//===- ExportSystemC.cpp - SystemC Emitter --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main SystemC emitter implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Target/ExportSystemC.h"
#include "EmissionPrinter.h"
#include "RegisterAllEmitters.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SystemC/SystemCDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace circt;
using namespace circt::ExportSystemC;

#define DEBUG_TYPE "export-systemc"

/// Helper to convert a file-path to a macro name that can be used to guard a
/// header file.
static std::string pathToMacroName(StringRef path) {
  std::string macroname = path.upper();
  std::replace(macroname.begin(), macroname.end(), '.', '_');
  std::replace(macroname.begin(), macroname.end(), '/', '_');
  return macroname;
}

/// Emits the given operation to a file represented by the passed ostream and
/// file-path.
static LogicalResult emitFile(Operation *op, EmissionConfig &config,
                              StringRef filePath, raw_ostream &os) {
  mlir::raw_indented_ostream ios(os);

  OpEmissionPatternSet opPatterns;
  registerAllOpEmitters(opPatterns, op->getContext());
  TypeEmissionPatternSet typePatterns;
  registerAllTypeEmitters(typePatterns);
  EmissionPrinter printer(ios, config, opPatterns, typePatterns);

  printer << "// " << filePath << "\n";
  std::string macroname = pathToMacroName(filePath);
  printer << "#ifndef " << macroname << "\n";
  printer << "#define " << macroname << "\n\n";

  // TODO: add support for 'emitc.include' and remove this hard-coded include.
  printer << "#include <systemc>\n\n";

  printer.emitOp(op);

  printer << "\n#endif // " << macroname << "\n\n";

  return failure(printer.exitState().failed());
}

//===----------------------------------------------------------------------===//
// Unified and Split Emitter implementation
//===----------------------------------------------------------------------===//

LogicalResult ExportSystemC::exportSystemC(ModuleOp module,
                                           EmissionConfig &config,
                                           llvm::raw_ostream &os) {
  return emitFile(module, config, "stdout.h", os);
}

LogicalResult ExportSystemC::exportSplitSystemC(ModuleOp module,
                                                EmissionConfig &config,
                                                StringRef directory) {
  for (Operation &op : module.getRegion().front()) {
    if (auto symbolOp = dyn_cast<mlir::SymbolOpInterface>(op)) {
      // Create the output directory if needed.
      if (std::error_code error = llvm::sys::fs::create_directories(directory))
        return module.emitError("cannot create output directory \"")
               << directory << "\": " << error.message();

      // Open or create the output file.
      std::string fileName = symbolOp.getName().str() + ".h";
      SmallString<128> filePath(directory);
      llvm::sys::path::append(filePath, fileName);
      std::string errorMessage;
      auto output = mlir::openOutputFile(filePath, &errorMessage);
      if (!output)
        return module.emitError(errorMessage);

      // Emit the content to the file.
      if (failed(emitFile(symbolOp, config, filePath, output->os())))
        return symbolOp->emitError("failed to emit to file \"")
               << filePath << "\"";

      // Do not delete the file if emission was successful.
      output->keep();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// circt-translate registration
//===----------------------------------------------------------------------===//

void ExportSystemC::registerExportSystemCTranslation() {

  static llvm::cl::opt<std::string> directory(
      "export-dir", llvm::cl::desc("Directory path to write the files to."),
      llvm::cl::init("./"));

  static mlir::TranslateFromMLIRRegistration toSystemC(
      "export-systemc",
      [](ModuleOp module, raw_ostream &output) {
        EmissionConfig config;
        return ExportSystemC::exportSystemC(module, config, output);
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<hw::HWDialect, comb::CombDialect,
                        systemc::SystemCDialect>();
      });

  static mlir::TranslateFromMLIRRegistration toSplitSystemC(
      "export-split-systemc",
      [](ModuleOp module, raw_ostream &output) {
        EmissionConfig config;
        return ExportSystemC::exportSplitSystemC(module, config, directory);
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<hw::HWDialect, comb::CombDialect,
                        systemc::SystemCDialect>();
      });
}
