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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/SystemC/SystemCDialect.h"
#include "circt/Target/ExportSystemC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace circt;
using namespace circt::ExportSystemC;

#define DEBUG_TYPE "export-systemc"

static LogicalResult emitFile(Operation *op, EmissionConfig &config,
                              StringRef filePath, raw_ostream &os) {
  EmissionPrinter printer(os, config);

  EmissionPatternSet patterns = printer.getEmissionPatternSet();
  registerAllEmitters(patterns);

  printer.emitFileHeader(filePath);

  if (failed(printer.emitOp(op)))
    return failure();

  printer.emitFileFooter(filePath);

  return success();
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
                                                StringRef directory,
                                                StringRef fileExt) {
  for (Operation &op : module.getRegion().front()) {
    if (auto symbolOp = dyn_cast<mlir::SymbolOpInterface>(op)) {
      // Create the output directory if needed.
      if (std::error_code error = llvm::sys::fs::create_directories(directory))
        return module.emitError("cannot create output directory \"")
               << directory << "\": " << error.message();

      std::string fileName = symbolOp.getName().str() + fileExt.str();
      SmallString<128> filePath(directory);
      llvm::sys::path::append(filePath, fileName);
      std::string errorMessage;
      auto output = mlir::openOutputFile(filePath, &errorMessage);
      if (!output)
        return module.emitError(errorMessage);

      if (failed(emitFile(symbolOp, config, filePath, output->os())))
        return symbolOp->emitError("failed to emit to file: ") << filePath;

      output->keep();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// circt-translate registration
//===----------------------------------------------------------------------===//

void ExportSystemC::registerExportSystemCTranslation() {
  static llvm::cl::opt<bool> useImplicitReadWriteOnSignals(
      "use-implicit-read-write-on-signals",
      llvm::cl::desc(
          "Reads and writes from/to signal types using the read() and write() "
          "calls instead of using the overloaded operators"),
      llvm::cl::init(false));

  static mlir::TranslateFromMLIRRegistration toSystemC(
      "export-systemc",
      [](ModuleOp module, raw_ostream &output) {
        EmissionConfig config;
        config.set<bool>(implicitReadWriteFlag.getName(),
                         useImplicitReadWriteOnSignals);
        return ExportSystemC::exportSystemC(module, config, output);
      },
      [](mlir::DialectRegistry &registry) {
        // clang-format off
        registry.insert<comb::CombDialect,
                        hw::HWDialect,
                        mlir::scf::SCFDialect,
                        systemc::SystemCDialect>();
        // clang-format on
      });

  static llvm::cl::opt<std::string> headerDirectory(
      "header-dir",
      llvm::cl::desc("Directory path to write all header files to."),
      llvm::cl::init("./"));

  static llvm::cl::opt<std::string> implementationDirectory(
      "implementation-dir",
      llvm::cl::desc("Directory path to write all implementation files to."),
      llvm::cl::init("./"));

  static llvm::cl::opt<bool> separateDeclaration(
      "separate-declaration",
      llvm::cl::desc("Separate declarations and implementations."),
      llvm::cl::init(false));

  static mlir::TranslateFromMLIRRegistration toSplitSystemC(
      "export-split-systemc",
      [](ModuleOp module, raw_ostream &output) {
        EmissionConfig config;
        config.set<bool>(implicitReadWriteFlag.getName(),
                         useImplicitReadWriteOnSignals);
        if (separateDeclaration) {
          config.set<ModuleDefinitionEmission>(
              moduleDefinitionEmissionFlag.getName(),
              ModuleDefinitionEmission::DEFINITION_ONLY);
          if (failed(ExportSystemC::exportSplitSystemC(
                  module, config, implementationDirectory, ".cpp")))
            return failure();

          config.set<ModuleDefinitionEmission>(
              moduleDefinitionEmissionFlag.getName(),
              ModuleDefinitionEmission::DECLARATION_ONLY);
        } else {
          config.set<ModuleDefinitionEmission>(
              moduleDefinitionEmissionFlag.getName(),
              ModuleDefinitionEmission::INLINE_DEFINITION);
        }
        return ExportSystemC::exportSplitSystemC(module, config,
                                                 headerDirectory, ".h");
      },
      [](mlir::DialectRegistry &registry) {
        // clang-format off
        registry.insert<comb::CombDialect,
                        hw::HWDialect,
                        mlir::scf::SCFDialect,
                        systemc::SystemCDialect>();
        // clang-format on
      });
}

// flags are passed in as string (can later also expand to reading the string
// out a file) of the format "flag: value, ...". Flags are