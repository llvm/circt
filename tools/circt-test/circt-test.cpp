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

  cl::opt<bool> listTests{"l", cl::desc("List tests in the input and exit"),
                          cl::init(false), cl::cat(cat)};

  cl::opt<bool> json{"json", cl::desc("Emit test list as JSON array"),
                     cl::init(false), cl::cat(cat)};
};
Options opts;

} // namespace

//===----------------------------------------------------------------------===//
// Test Discovery
//===----------------------------------------------------------------------===//

namespace {
/// The various kinds of test that can be executed.
enum class TestKind { Formal };

/// A single discovered test.
class Test {
public:
  /// The name of the test. This is also the name of the top-level module passed
  /// to the formal or simulation tool to be run.
  StringAttr name;
  /// The kind of test, such as "formal" or "simulation".
  TestKind kind;
  /// An optional location indicating where this test was discovered. This can
  /// be the location of an MLIR op, or a line in some other source file.
  LocationAttr loc;
  /// The user-defined attributes of this test.
  DictionaryAttr attrs;
};

/// A collection of tests discovered in some MLIR input.
class TestSuite {
public:
  /// The MLIR context that is used to intern attributes and where any MLIR
  /// tests were discovered.
  MLIRContext *context;
  /// The tests discovered in the input.
  std::vector<Test> tests;

  TestSuite(MLIRContext *context) : context(context) {}
  void discoverInModule(ModuleOp module);
};
} // namespace

/// Convert a `TestKind` to a string representation.
static StringRef toString(TestKind kind) {
  switch (kind) {
  case TestKind::Formal:
    return "formal";
  }
  return "unknown";
}

/// Discover all tests in an MLIR module.
void TestSuite::discoverInModule(ModuleOp module) {
  module.walk([&](verif::FormalOp op) {
    Test test;
    test.name = op.getSymNameAttr();
    test.kind = TestKind::Formal;
    test.loc = op.getLoc();
    test.attrs = op->getDiscardableAttrDictionary();
    tests.push_back(std::move(test));
  });
}

//===----------------------------------------------------------------------===//
// Tool Implementation
//===----------------------------------------------------------------------===//

/// List all the tests in a given module.
static LogicalResult listTests(TestSuite &suite) {
  // Open the output file for writing.
  std::string errorMessage;
  auto output = openOutputFile(opts.outputFilename, &errorMessage);
  if (!output)
    return emitError(UnknownLoc::get(suite.context)) << errorMessage;

  // Handle JSON output.
  if (opts.json) {
    json::OStream json(output->os(), 2);
    json.arrayBegin();
    auto guard = make_scope_exit([&] { json.arrayEnd(); });
    for (auto &test : suite.tests) {
      json.objectBegin();
      auto guard = make_scope_exit([&] { json.objectEnd(); });
      json.attribute("name", test.name.getValue());
      json.attribute("kind", toString(test.kind));
      if (!test.attrs.empty()) {
        json.attributeBegin("attrs");
        auto guard = make_scope_exit([&] { json.attributeEnd(); });
        if (failed(convertAttributeToJSON(json, test.attrs)))
          return mlir::emitError(test.loc)
                 << "unsupported attributes: `" << test.attrs
                 << "` cannot be converted to JSON";
      }
    }
    output->keep();
    return success();
  }

  // Handle regular text output.
  for (auto &test : suite.tests)
    output->os() << test.name.getValue() << "  " << toString(test.kind) << "  "
                 << test.attrs << "\n";
  output->keep();
  return success();
}

/// Entry point for the circt-test tool. At this point an MLIRContext is
/// available, all dialects have been registered, and all command line options
/// have been parsed.
static LogicalResult execute(MLIRContext *context) {
  SourceMgr srcMgr;
  SourceMgrDiagnosticHandler handler(srcMgr, context);

  // Parse the input file.
  auto module = parseSourceFile<ModuleOp>(opts.inputFilename, srcMgr, context);
  if (!module)
    return failure();

  // Discover all tests in the input.
  TestSuite suite(context);
  suite.discoverInModule(*module);
  if (suite.tests.empty()) {
    llvm::errs() << "no tests discovered\n";
    return success();
  }

  // List all tests in the input and exit if requested.
  if (opts.listTests)
    return listTests(suite);

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
