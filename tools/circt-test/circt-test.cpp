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

#include "circt/Conversion/ExportVerilog.h"
#include "circt/Conversion/VerifToSV.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Dialect/Verif/VerifPasses.h"
#include "circt/Support/JSON.h"
#include "circt/Support/Version.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Threading.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"

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

  cl::opt<bool> listIgnored{"list-ignored", cl::desc("List ignored tests"),
                            cl::init(false), cl::cat(cat)};

  cl::opt<std::string> resultDir{
      "d", cl::desc("Result directory (default `.circt-test`)"),
      cl::value_desc("dir"), cl::init(".circt-test"), cl::cat(cat)};

  cl::opt<bool> verifyPasses{
      "verify-each",
      cl::desc("Run the verifier after each transformation pass"),
      cl::init(true), cl::cat(cat)};

  cl::opt<std::string> runner{
      "r", cl::desc("Program to run individual tests"), cl::value_desc("bin"),
      cl::init("circt-test-runner-sby.py"), cl::cat(cat)};

  cl::opt<bool> runnerReadsMLIR{
      "mlir-runner",
      cl::desc("Pass the MLIR file to the runner instead of Verilog"),
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
  /// Whether or not the test should be ignored
  bool ignore;
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

  bool listIgnored;

  TestSuite(MLIRContext *context, bool listIgnored)
      : context(context), listIgnored(listIgnored) {}
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
    test.attrs = op.getParametersAttr();
    if (auto boolAttr = test.attrs.getAs<BoolAttr>("ignore"))
      test.ignore = boolAttr.getValue();
    else
      test.ignore = false;
    tests.push_back(std::move(test));
  });
}

//===----------------------------------------------------------------------===//
// Tool Implementation
//===----------------------------------------------------------------------===//

// Check if test should be included in output listing
bool ignoreTestListing(Test &test, TestSuite &suite) {
  return !suite.listIgnored && test.ignore;
}

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
      if (ignoreTestListing(test, suite))
        continue;
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
  for (auto &test : suite.tests) {
    if (ignoreTestListing(test, suite))
      continue;
    output->os() << test.name.getValue() << "  " << toString(test.kind) << "  "
                 << test.attrs << "\n";
  }
  output->keep();
  return success();
}

void reportIgnored(unsigned numIgnored) {
  if (numIgnored > 0)
    WithColor(llvm::errs(), raw_ostream::SAVEDCOLOR, true).get()
        << ", " << numIgnored << " ignored";
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
  TestSuite suite(context, opts.listIgnored);
  suite.discoverInModule(*module);
  if (suite.tests.empty()) {
    llvm::errs() << "no tests discovered\n";
    return success();
  }

  // List all tests in the input and exit if requested.
  if (opts.listTests)
    return listTests(suite);

  // Create the output directory where we keep all the run data.
  if (auto error = llvm::sys::fs::create_directory(opts.resultDir)) {
    WithColor::error() << "cannot create result directory `" << opts.resultDir
                       << "`: " << error.message() << "\n";
    return failure();
  }

  // Open the Verilog file for writing.
  SmallString<128> verilogPath(opts.resultDir);
  llvm::sys::path::append(verilogPath, "design.sv");
  std::string errorMessage;
  auto verilogFile = openOutputFile(verilogPath, &errorMessage);
  if (!verilogFile) {
    WithColor::error() << errorMessage;
    return failure();
  }

  // Generate Verilog output.
  PassManager pm(context);
  pm.enableVerifier(opts.verifyPasses);
  pm.addPass(verif::createLowerFormalToHWPass());
  pm.addNestedPass<hw::HWModuleOp>(createLowerVerifToSVPass());
  pm.addNestedPass<hw::HWModuleOp>(sv::createHWLegalizeModulesPass());
  pm.addNestedPass<hw::HWModuleOp>(sv::createPrettifyVerilogPass());
  pm.addPass(createExportVerilogPass(verilogFile->os()));
  if (failed(pm.run(*module)))
    return failure();
  verilogFile->keep();

  // Find the runner binary in the search path. Otherwise assume it is a binary
  // we can run as is.
  auto findResult = llvm::sys::findProgramByName(opts.runner);
  if (!findResult) {
    WithColor::error() << "cannot find runner `" << opts.runner
                       << "`: " << findResult.getError().message() << "\n";
    return failure();
  }
  auto &runner = findResult.get();

  // Run the tests.
  std::atomic<unsigned> numPassed(0);
  std::atomic<unsigned> numIgnored(0);
  mlir::parallelForEach(context, suite.tests, [&](auto &test) {
    if (test.ignore) {
      ++numIgnored;
      return;
    }
    // Create the directory in which we are going to run the test.
    SmallString<128> testDir(opts.resultDir);
    llvm::sys::path::append(testDir, test.name.getValue());
    if (auto error = llvm::sys::fs::create_directory(testDir)) {
      mlir::emitError(UnknownLoc::get(context))
          << "cannot create test directory `" << testDir
          << "`: " << error.message() << "\n";
      return;
    }

    // Assemble a path for the test runner log file and truncate it.
    SmallString<128> logPath(testDir);
    llvm::sys::path::append(logPath, "run.log");
    {
      std::error_code ec;
      raw_fd_ostream trunc(logPath, ec);
    }

    // Assemble the runner arguments.
    SmallVector<StringRef> args;
    args.push_back(runner);
    if (opts.runnerReadsMLIR)
      args.push_back(opts.inputFilename);
    else
      args.push_back(verilogPath);
    args.push_back("-t");
    args.push_back(test.name.getValue());
    args.push_back("-d");
    args.push_back(testDir);

    if (auto mode = test.attrs.get("mode")) {
      args.push_back("-m");
      auto modeStr = dyn_cast<StringAttr>(mode);
      if (!modeStr) {
        mlir::emitError(test.loc) << "invalid mode for test " << test.name;
        return;
      }
      args.push_back(cast<StringAttr>(mode).getValue());
    }

    if (auto depth = test.attrs.get("depth")) {
      args.push_back("-k");
      auto depthInt = dyn_cast<IntegerAttr>(depth);
      if (!depthInt) {
        mlir::emitError(test.loc) << "invalid depth for test " << test.name;
        return;
      }
      SmallVector<char> str;
      depthInt.getValue().toStringUnsigned(str);
      args.push_back(std::string(str.begin(), str.end()));
    }

    // Execute the test runner.
    std::string errorMessage;
    auto result =
        llvm::sys::ExecuteAndWait(runner, args, /*Env=*/std::nullopt,
                                  /*Redirects=*/{"", logPath, logPath},
                                  /*SecondsToWait=*/0,
                                  /*MemoryLimit=*/0, &errorMessage);
    if (result < 0) {
      mlir::emitError(UnknownLoc::get(context))
          << "cannot execute runner: " << errorMessage;
    } else if (result > 0) {
      auto d = mlir::emitError(test.loc)
               << "test " << test.name.getValue() << " failed";
      d.attachNote() << "see `" << logPath << "`";
    } else {
      ++numPassed;
    }
  });

  // Print statistics about how many tests passed and failed.
  assert((numPassed + numIgnored) <= suite.tests.size());
  unsigned numFailed = suite.tests.size() - numPassed - numIgnored;
  if (numFailed > 0) {
    WithColor(llvm::errs(), raw_ostream::SAVEDCOLOR, true).get()
        << numFailed << " tests ";
    WithColor(llvm::errs(), raw_ostream::RED, true).get() << "FAILED";
    llvm::errs() << ", " << numPassed << " passed";
    reportIgnored(numIgnored);
    llvm::errs() << "\n";
    return failure();
  }
  WithColor(llvm::errs(), raw_ostream::SAVEDCOLOR, true).get()
      << numPassed << " tests ";
  WithColor(llvm::errs(), raw_ostream::GREEN, true).get() << "passed";
  reportIgnored(numIgnored);
  llvm::errs() << "\n";
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
