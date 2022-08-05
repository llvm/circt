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

static LogicalResult emitFile(Operation *op, StringRef filePath,
                              raw_ostream &os) {
  return op->emitError("Not yet supported!");
}

//===----------------------------------------------------------------------===//
// Unified and Split Emitter implementation
//===----------------------------------------------------------------------===//

LogicalResult ExportSystemC::exportSystemC(ModuleOp module,
                                           llvm::raw_ostream &os) {
  return emitFile(module, "stdout.h", os);
}

LogicalResult ExportSystemC::exportSplitSystemC(ModuleOp module,
                                                StringRef directory) {
  for (Operation &op : module.getRegion().front()) {
    if (auto symbolOp = dyn_cast<mlir::SymbolOpInterface>(op)) {
      // Create the output directory if needed.
      if (std::error_code error = llvm::sys::fs::create_directories(directory))
        return module.emitError("cannot create output directory \"")
               << directory << "\": " << error.message();

      SmallString<128> filePath(directory);
      llvm::sys::path::append(filePath, symbolOp.getName());

      // Open or create the output file.
      std::string errorMessage;
      auto output = mlir::openOutputFile(filePath, &errorMessage);
      if (!output)
        return module.emitError(errorMessage);

      // Emit the content to the file.
      if (failed(emitFile(symbolOp, filePath, output->os())))
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
        return ExportSystemC::exportSystemC(module, output);
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<systemc::SystemCDialect>();
      });

  static mlir::TranslateFromMLIRRegistration toSplitSystemC(
      "export-split-systemc",
      [](ModuleOp module, raw_ostream &output) {
        return ExportSystemC::exportSplitSystemC(module, directory);
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<systemc::SystemCDialect>();
      });
}
