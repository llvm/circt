//===- circt-vcd.cpp - The vcd driver ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
//===----------------------------------------------------------------------===//

#include "circt/InitAllDialects.h"
#include "circt/Support/Version.h"

#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/InstanceGraph.h"
#include "circt/Support/VCD.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#define DEBUG_TYPE "circt-vcd"

using namespace llvm;
using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Command-line options declaration
//===----------------------------------------------------------------------===//

static cl::opt<bool>
    allowUnregisteredDialects("allow-unregistered-dialects",
                              cl::desc("Allow unknown dialects in the input"));

static cl::OptionCategory mainCategory("circt-generate-lec-mapping Options");

static cl::opt<std::string>
    refModuleName("ref-top-name", cl::Optional,
                  cl::desc("Specify a reference top module name"),
                  cl::value_desc("module name"), cl::cat(mainCategory));

static cl::opt<std::string> inputMLIR("mlir", cl::Required,
                                      cl::desc("input mlir file"),
                                      cl::cat(mainCategory));
static cl::opt<std::string> inputVCD("vcd", cl::Required,
                                     cl::desc("input vcd file"),
                                     cl::cat(mainCategory));
static cl::opt<std::string> dutPath("dut-path", cl::Required,
                                    cl::desc("path to dut module"),
                                    cl::cat(mainCategory));
static cl::opt<std::string> dutModuleName("dut-module-name", cl::Required,
                                          cl::desc("name of dut module"),
                                          cl::cat(mainCategory));
static cl::opt<std::string> outputFilename("o", cl::desc("Output name"),
                                           cl::value_desc("name"),
                                           cl::init("-"),
                                           cl::cat(mainCategory));

//===----------------------------------------------------------------------===//
// Tool implementation
//===----------------------------------------------------------------------===//

static mlir::StringAttr getVariableName(Operation *op) {
  if (auto originalName = op->getAttrOfType<StringAttr>("hw.originalName"))
    return originalName;
  if (auto name = op->getAttrOfType<StringAttr>("name"))
    return name;
  if (auto verilogName = op->getAttrOfType<StringAttr>("hw.verilogName"))
    return verilogName;

  return StringAttr();
}

struct VCDConverter {
  LogicalResult convert();
  VCDConverter(vcd::VCDFile &file, mlir::ModuleOp module)
      : file(file), module(module) {}

private:
  vcd::VCDFile &file;
  mlir::ModuleOp module;
  mlir::MLIRContext *getContext() { return module->getContext(); }
};

static llvm::SmallVector<StringRef> split(const std::string &s,
                                          char seperator) {
  llvm::SmallVector<StringRef> output;

  std::string::size_type prev_pos = 0, pos = 0;

  StringRef ref(s);

  while ((pos = s.find(seperator, pos)) != std::string::npos) {
    output.push_back(ref.substr(prev_pos, pos - prev_pos));
    prev_pos = ++pos;
  }

  output.push_back(s.substr(prev_pos, pos - prev_pos)); // Last word

  return output;
}
/// This functions initializes the various components of the tool and
/// orchestrates the work to be done.
static LogicalResult execute(MLIRContext &context) {

  std::string errorMessage;
  auto input = mlir::openInputFile(inputVCD, &errorMessage);
  if (!input) {
    errs() << errorMessage << "\n";
    return failure();
  }

  llvm::SourceMgr vcdSourceMgr;
  vcdSourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  SourceMgrDiagnosticVerifierHandler sourceMgrHandler(vcdSourceMgr, &context);
  context.printOpOnDiagnostic(false);

  auto vcdFile = circt::vcd::importVCDFile(vcdSourceMgr, &context);
  if (!vcdFile) {
    errs() << "failed to parse input vcd file `" << inputVCD << "`\n";
    return failure();
  }
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    errs() << errorMessage;
    return failure();
  }

  llvm::SourceMgr mlirSourceMgr;
  auto inputMLIRFile = mlir::openInputFile(inputMLIR, &errorMessage);
  mlirSourceMgr.AddNewSourceBuffer(std::move(inputMLIRFile), llvm::SMLoc());
  SourceMgrDiagnosticVerifierHandler mlirMgrHandler(mlirSourceMgr, &context);
  auto module = parseSourceFile<ModuleOp>(mlirSourceMgr, &context);
  auto path = split(dutPath.getValue(), '.');
  vcd::SignalMapping mapping(module.get(), *vcdFile, path, dutModuleName);
  if (failed(mapping.run()))
  {
    module.get()->emitError() << "failed";
    return failure();
  }

  mlir::raw_indented_ostream os(output->os());
  vcdFile->printVCD(os);
  output->keep();
  return success();
}

/// The entry point for the `circt-vcd` tool:
/// configures and parses the command-line options,
/// registers all dialects within a MLIR context,
/// and calls the `execute` function to do the actual work.
int main(int argc, char **argv) {
  // Configure the relevant command-line options.
  cl::HideUnrelatedOptions(mainCategory);
  registerMLIRContextCLOptions();
  cl::AddExtraVersionPrinter(
      [](llvm::raw_ostream &os) { os << circt::getCirctVersion() << '\n'; });

  // Parse the command-line options provided by the user.
  cl::ParseCommandLineOptions(argc, argv,
                              "circt-vcd - parse vcd "
                              "waveforms\n\n");

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  llvm::setBugReportMsg(circt::circtBugReportMsg);

  // Register the supported CIRCT dialects and create a context to work with.
  MLIRContext context;
  context.loadDialect<comb::CombDialect>();
  context.loadDialect<emit::EmitDialect>();
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<om::OMDialect>();
  context.loadDialect<sv::SVDialect>();
  context.loadDialect<ltl::LTLDialect>();
  // context.loadDialect<verif::VerifDialect>();

  // Setup of diagnostic handling.
  llvm::SourceMgr sourceMgr;
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  // Avoid printing a superfluous note on diagnostic emission.
  context.printOpOnDiagnostic(false);

  exit(failed(execute(context)));
}
