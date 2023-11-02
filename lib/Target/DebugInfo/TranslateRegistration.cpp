//===- TranslateRegistration.cpp - Register translation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Debug/DebugDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Target/DebugInfo.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace circt {
namespace debug {

static void registerDialects(DialectRegistry &registry) {
  registry.insert<comb::CombDialect>();
  registry.insert<debug::DebugDialect>();
  registry.insert<hw::HWDialect>();
  registry.insert<seq::SeqDialect>();
  registry.insert<sv::SVDialect>();
  registry.insert<om::OMDialect>();
}

void registerDumpTranslation() {
  TranslateFromMLIRRegistration reg(
      "dump-di", "dump debug information in human-readable form",
      [](ModuleOp op, raw_ostream &output) {
        return dumpDebugInfo(op, output);
      },
      registerDialects);
}

void registerHGLDDTranslation() {
  static llvm::cl::opt<std::string> directory(
      "hgldd-output-dir", llvm::cl::desc("Output directory for HGLDD files"),
      llvm::cl::init(""));
  static llvm::cl::opt<std::string> sourcePrefix(
      "hgldd-source-prefix", llvm::cl::desc("Prefix for source file locations"),
      llvm::cl::init(""));
  static llvm::cl::opt<std::string> outputPrefix(
      "hgldd-output-prefix", llvm::cl::desc("Prefix for output file locations"),
      llvm::cl::init(""));

  auto getOptions = [] {
    EmitHGLDDOptions opts;
    opts.sourceFilePrefix = sourcePrefix;
    opts.outputFilePrefix = outputPrefix;
    opts.outputDirectory = directory;
    return opts;
  };

  static TranslateFromMLIRRegistration reg1(
      "emit-hgldd", "emit HGLDD debug information",
      [=](ModuleOp op, raw_ostream &output) {
        return emitHGLDD(op, output, getOptions());
      },
      registerDialects);

  static TranslateFromMLIRRegistration reg2(
      "emit-split-hgldd", "emit HGLDD debug information as separate files",
      [=](ModuleOp op, raw_ostream &output) {
        return emitSplitHGLDD(op, getOptions());
      },
      registerDialects);
}

void registerTranslations() {
  registerDumpTranslation();
  registerHGLDDTranslation();
}

} // namespace debug
} // namespace circt
