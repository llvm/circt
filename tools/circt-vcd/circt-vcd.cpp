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
#include "mlir/Support/ToolUtilities.h"
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

static cl::OptionCategory mainCategory("circt-generate-lec-mapping Options");

static cl::opt<std::string> inputVCD(cl::Positional,
                                     cl::desc("path to dut module"),
                                     cl::cat(mainCategory));
static cl::opt<std::string> outputFilename("o", cl::desc("Output name"),
                                           cl::value_desc("name"),
                                           cl::init("-"),
                                           cl::cat(mainCategory));
static cl::opt<bool> emitVCDDataStructure("emit-vcd-data-structure",
                                          cl::desc("Emit VCD data structure"),
                                          cl::init(false),
                                          cl::cat(mainCategory));
static cl::opt<bool> verifyDiagnostics("verify-diagnostics",
                                       cl::desc("Verify diagnostics"),
                                       cl::init(false), cl::cat(mainCategory));
static cl::opt<bool> splitInputFile("split-input-file",
                                    cl::desc("Split input file"),
                                    cl::init(false), cl::cat(mainCategory));
//===----------------------------------------------------------------------===//
// Tool implementation
//===----------------------------------------------------------------------===//

static LogicalResult
processBuffer(MLIRContext &context, llvm::SourceMgr &sourceMgr,
              std::unique_ptr<llvm::ToolOutputFile> &outputFile) {

  auto vcdFile = circt::vcd::importVCDFile(sourceMgr, &context);
  if (!vcdFile) {
    return failure();
  }

  mlir::raw_indented_ostream os(outputFile->os());
  if (emitVCDDataStructure) {
    vcdFile->dump(os);
    outputFile->keep();
    return success();
  }

  vcdFile->printVCD(os);
  outputFile->keep();
  return success();
}

/// Process a single split of the input. This allocates a source manager and
/// creates a regular or verifying diagnostic handler, depending on whether the
/// user set the verifyDiagnostics option.
static LogicalResult
processInput(MLIRContext &context, std::unique_ptr<llvm::MemoryBuffer> buffer,
             std::unique_ptr<llvm::ToolOutputFile> &outputFile) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  if (!verifyDiagnostics) {
    SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr,
                                                &context /*, shouldShow */);
    return processBuffer(context, sourceMgr, outputFile);
  }

  SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr, &context);
  context.printOpOnDiagnostic(false);
  (void)processBuffer(context, sourceMgr, outputFile);
  return sourceMgrHandler.verify();
}

static LogicalResult
processInputSplit(MLIRContext &context,
                  std::unique_ptr<llvm::MemoryBuffer> input,
                  std::unique_ptr<llvm::ToolOutputFile> &outputFile) {

  if (!splitInputFile)
    return processInput(context, std::move(input), outputFile);

  return mlir::splitAndProcessBuffer(
      std::move(input),
      [&](std::unique_ptr<MemoryBuffer> buffer, raw_ostream &) {
        return processInput(context, std::move(buffer), outputFile);
      },
      llvm::outs(), "// -----");
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

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    errs() << errorMessage;
    return failure();
  }

  return processInputSplit(context, std::move(input), output);
}

class FileLineColLocsAsNotesDiagnosticHandler : public ScopedDiagnosticHandler {
public:
  FileLineColLocsAsNotesDiagnosticHandler(MLIRContext *ctxt)
      : ScopedDiagnosticHandler(ctxt) {
    setHandler([](Diagnostic &d) {
      SmallPtrSet<Location, 8> locs;
      // Recursively scan for FileLineColLoc locations.
      d.getLocation()->walk([&](Location loc) {
        if (isa<FileLineColLoc>(loc))
          locs.insert(loc);
        return WalkResult::advance();
      });

      // Drop top-level location the diagnostic is reported on.
      locs.erase(d.getLocation());
      // As well as the location the SourceMgrDiagnosticHandler will use.
      if (auto reportLoc = d.getLocation()->findInstanceOf<FileLineColLoc>())
        locs.erase(reportLoc);

      // Attach additional locations as notes on the diagnostic.
      for (auto l : locs)
        d.attachNote(l) << "additional location here";
      return failure();
    });
  }
};

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

  // FileLineColLocsAsNotesDiagnosticHandler addLocs(&context);
  // Avoid printing a superfluous note on diagnostic emission.
  context.printOpOnDiagnostic(false);

  exit(failed(execute(context)));
}
