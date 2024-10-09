//===- circt-test.cpp - Hardware unit test discovery and execution tool ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Discover runnable unit tests in an MLIR blob and execute them through various
// backends.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/JSON.h"
#include "circt/Support/Version.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"

#include <string>

using namespace llvm;
using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Command Line Options
//===----------------------------------------------------------------------===//

namespace {

/// The tool's command line options.
struct Options {
  cl::OptionCategory cat{"circt-test Options"};
  cl::opt<std::string> inputFilename{cl::Positional, cl::desc("<input file>"),
                                     cl::init("-"), cl::cat(cat)};
  cl::opt<std::string> outputFilename{
      "o", cl::desc("Output filename (`-` for stdout)"),
      cl::value_desc("filename"), cl::init("-"), cl::cat(cat)};
  cl::opt<bool> json{"json", cl::desc("Emit test list as JSON array"),
                     cl::init(false), cl::cat(cat)};
};
Options opts;

} // namespace

//===----------------------------------------------------------------------===//
// Tool Implementation
//===----------------------------------------------------------------------===//

/// List all the tests in a given module.
static LogicalResult listTests(ModuleOp module, llvm::raw_ostream &output) {
  // Handle JSON output.
  if (opts.json) {
    json::OStream json(output, 2);
    json.arrayBegin();
    auto result = module.walk([&](Operation *op) {
      if (auto formalOp = dyn_cast<verif::FormalOp>(op)) {
        json.objectBegin();
        auto guard = make_scope_exit([&] { json.objectEnd(); });
        json.attribute("name", formalOp.getSymName());
        json.attribute("kind", "formal");
        auto attrs = formalOp->getDiscardableAttrDictionary();
        if (!attrs.empty()) {
          json.attributeBegin("attrs");
          auto guard = make_scope_exit([&] { json.attributeEnd(); });
          if (failed(convertAttributeToJSON(json, attrs))) {
            op->emitError() << "unsupported attributes: `" << attrs
                            << "` cannot be converted to JSON";
            return WalkResult::interrupt();
          }
        }
      }
      return WalkResult::advance();
    });
    json.arrayEnd();
    return failure(result.wasInterrupted());
  }

  // Handle regular text output.
  module.walk([&](Operation *op) {
    if (auto formalOp = dyn_cast<verif::FormalOp>(op)) {
      output << formalOp.getSymName() << "  formal"
             << "  " << formalOp->getDiscardableAttrDictionary() << "\n";
    }
  });
  return success();
}

/// Entry point for the circt-test tool. At this point an MLIRContext is
/// available, all dialects have been registered, and all command line options
/// have been parsed.
static LogicalResult execute(MLIRContext *context) {
  SourceMgr srcMgr;
  SourceMgrDiagnosticHandler handler(srcMgr, context);

  // Open the output file for writing.
  std::string errorMessage;
  auto output = openOutputFile(opts.outputFilename, &errorMessage);
  if (!output)
    return emitError(UnknownLoc::get(context)) << errorMessage;

  // Parse the input file.
  auto module = parseSourceFile<ModuleOp>(opts.inputFilename, srcMgr, context);
  if (!module)
    return failure();

  // List all tests in the input.
  if (failed(listTests(*module, output->os())))
    return failure();

  output->keep();
  return success();
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  setBugReportMsg(circtBugReportMsg);

  // Print the CIRCT version when requested.
  cl::AddExtraVersionPrinter(
      [](raw_ostream &os) { os << getCirctVersion() << '\n'; });

  // Register the dialects.
  DialectRegistry registry;
  registry.insert<circt::comb::CombDialect>();
  registry.insert<circt::hw::HWDialect>();
  registry.insert<circt::om::OMDialect>();
  registry.insert<circt::seq::SeqDialect>();
  registry.insert<circt::sim::SimDialect>();
  registry.insert<circt::sv::SVDialect>();
  registry.insert<circt::verif::VerifDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::scf::SCFDialect>();

  // Hide default LLVM options, other than for this tool.
  // MLIR options are added below.
  cl::HideUnrelatedOptions({&opts.cat, &llvm::getColorCategory()});
  cl::ParseCommandLineOptions(argc, argv, "Hardware unit testing tool\n");

  MLIRContext context(registry);
  exit(failed(execute(&context)));
}
