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
#include "circt/Conversion/Passes.h"
#include "circt/Conversion/VerifToSV.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Dialect/Verif/VerifPasses.h"
#include "circt/InitAllDialects.h"
#include "circt/Support/JSON.h"
#include "circt/Support/LoweringOptionsParser.h"
#include "circt/Support/Passes.h"
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
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Transforms/Passes.h"
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

  cl::opt<bool> listRunners{"list-runners", cl::desc("List test runners"),
                            cl::init(false), cl::cat(cat)};

  cl::opt<bool> json{"json", cl::desc("Emit test list as JSON array"),
                     cl::init(false), cl::cat(cat)};

  cl::opt<bool> listIgnored{"list-ignored", cl::desc("List ignored tests"),
                            cl::init(false), cl::cat(cat)};

  cl::opt<std::string> resultDir{
      "d", cl::desc("Result directory (default `.circt-test`)"),
      cl::value_desc("dir"), cl::init(".circt-test"), cl::cat(cat)};

  cl::list<std::string> runners{"r", cl::desc("Use a specific set of runners"),
                                cl::value_desc("name"),
                                cl::MiscFlags::CommaSeparated, cl::cat(cat)};

  cl::opt<bool> ignoreContracts{
      "ignore-contracts",
      cl::desc("Do not use contracts to simplify and parallelize tests"),
      cl::init(false), cl::cat(cat)};

  cl::opt<verif::SymbolicValueLowering> symbolicValueLowering{
      "symbolic-values", cl::desc("Control how symbolic values are lowered"),
      cl::init(verif::SymbolicValueLowering::ExtModule),
      verif::symbolicValueLoweringCLValues(), cl::cat(cat)};

  cl::opt<bool> emitIR{"ir", cl::desc("Emit IR after initial lowering"),
                       cl::init(false), cl::cat(cat)};

  cl::opt<bool> verifyPasses{
      "verify-each",
      cl::desc("Run the verifier after each transformation pass"),
      cl::init(true), cl::Hidden, cl::cat(cat)};

  cl::opt<bool> verifyDiagnostics{
      "verify-diagnostics",
      cl::desc("Check that emitted diagnostics match expected-* lines on the "
               "corresponding line"),
      cl::init(false), cl::Hidden, cl::cat(cat)};

  cl::opt<bool> splitInputFile{
      "split-input-file",
      cl::desc("Split the input file into pieces and process each "
               "chunk independently"),
      cl::init(false), cl::Hidden, cl::cat(cat)};

  LoweringOptionsOption loweringOptions{cat};
};
Options opts;

} // namespace

//===----------------------------------------------------------------------===//
// Runners
//===----------------------------------------------------------------------===//

namespace {
/// A program that can run tests.
class Runner {
public:
  /// The name of the runner. The user can filter runners by this name, and
  /// individual tests can indicate that they can or cannot run with runners
  /// based on this name.
  StringAttr name;
  /// The runner binary. The value of this field is resolved using
  /// `findProgramByName` and stored in `binaryPath`.
  std::string binary;
  /// The full path to the runner.
  std::string binaryPath;
  /// Whether this runner operates on Verilog or MLIR input.
  bool readsMLIR = false;
  /// Whether this runner should be ignored.
  bool ignore = false;
  /// Whether this runner is available or not. This is set to false if the
  /// runner `binary` cannot be found.
  bool available = false;
};

/// A collection of test runners.
class RunnerSuite {
public:
  /// The MLIR context that is used for multi-threading.
  MLIRContext *context;
  /// The configured runners.
  std::vector<Runner> runners;

  RunnerSuite(MLIRContext *context) : context(context) {}
  void addDefaultRunners();
  LogicalResult resolve();
};
} // namespace

/// Add the default runners to the suite. These are the runners that are defined
/// as part of CIRCT.
void RunnerSuite::addDefaultRunners() {
  {
    // SymbiYosys
    Runner runner;
    runner.name = StringAttr::get(context, "sby");
    runner.binary = "circt-test-runner-sby.py";
    runners.push_back(std::move(runner));
  }
  {
    // circt-bmc
    Runner runner;
    runner.name = StringAttr::get(context, "circt-bmc");
    runner.binary = "circt-test-runner-circt-bmc.py";
    runner.readsMLIR = true;
    runners.push_back(std::move(runner));
  }
}

/// Resolve the `binary` field of each runner to a full `binaryPath`, and set
/// the `available` field to reflect whether the runner was found.
LogicalResult RunnerSuite::resolve() {
  // If the user has provided a concrete list of runners to use, mark all other
  // runners as to be ignored.
  if (opts.runners.getNumOccurrences() > 0) {
    for (auto &runner : runners)
      if (!llvm::is_contained(opts.runners, runner.name))
        runner.ignore = true;

    // Produce errors if the user listed any runners that don't exist.
    for (auto &name : opts.runners) {
      if (!llvm::is_contained(
              llvm::map_range(runners,
                              [](auto &runner) { return runner.name; }),
              name)) {
        WithColor::error() << "unknown runner `" << name << "`\n";
        return failure();
      }
    }
  }

  mlir::parallelForEach(context, runners, [&](auto &runner) {
    if (runner.ignore)
      return;

    auto findResult = llvm::sys::findProgramByName(runner.binary);
    if (!findResult)
      return;
    runner.available = true;
    runner.binaryPath = findResult.get();
  });
  return success();
}

//===----------------------------------------------------------------------===//
// Test Discovery
//===----------------------------------------------------------------------===//

namespace {
/// The various kinds of test that can be executed.
enum class TestKind { Formal, Simulation };

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
  bool ignore = false;
  /// The user-defined attributes of this test.
  DictionaryAttr attrs;
  /// The set of runners that can execute this test, specified by the
  /// "require_runners" array attribute in `attrs`.
  SmallPtrSet<StringAttr, 1> requiredRunners;
  /// The set of runners that should be skipped for this test, specified by the
  /// "exclude_runners" array attribute in `attrs`.
  SmallPtrSet<StringAttr, 1> excludedRunners;
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
  LogicalResult discoverInModule(ModuleOp module);
  LogicalResult discoverTest(Test &&test, Operation *op);
};
} // namespace

/// Convert a `TestKind` to a string representation.
static StringRef toString(TestKind kind) {
  switch (kind) {
  case TestKind::Formal:
    return "formal";
  case TestKind::Simulation:
    return "simulation";
  }
  return "unknown";
}

static LogicalResult
collectRunnerFilters(DictionaryAttr attrs, StringRef attrName,
                     llvm::SmallPtrSetImpl<StringAttr> &names, Location loc,
                     StringAttr testName) {
  auto attr = attrs.get(attrName);
  if (!attr)
    return success();

  auto arrayAttr = dyn_cast<ArrayAttr>(attr);
  if (!arrayAttr)
    return mlir::emitError(loc) << "`" << attrName << "` attribute of test "
                                << testName << " must be an array";

  for (auto elementAttr : arrayAttr.getValue()) {
    auto stringAttr = dyn_cast<StringAttr>(elementAttr);
    if (!stringAttr)
      return mlir::emitError(loc)
             << "element of `" << attrName << "` array of test " << testName
             << " must be a string; got " << elementAttr;
    names.insert(stringAttr);
  }

  return success();
}

/// Discover all tests in an MLIR module.
LogicalResult TestSuite::discoverInModule(ModuleOp module) {
  auto result = module.walk([&](Operation *op) {
    if (auto testOp = dyn_cast<verif::FormalOp>(op)) {
      Test test;
      test.kind = TestKind::Formal;
      test.name = testOp.getSymNameAttr();
      test.attrs = testOp.getParametersAttr();
      if (failed(discoverTest(std::move(test), op)))
        return WalkResult::interrupt();
    } else if (auto testOp = dyn_cast<verif::SimulationOp>(op)) {
      Test test;
      test.kind = TestKind::Simulation;
      test.name = testOp.getSymNameAttr();
      test.attrs = testOp.getParametersAttr();
      if (failed(discoverTest(std::move(test), op)))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

/// Discover a single test in an MLIR module.
LogicalResult TestSuite::discoverTest(Test &&test, Operation *op) {
  test.loc = op->getLoc();

  // Handle the `ignore` attribute.
  if (auto attr = test.attrs.get("ignore")) {
    auto boolAttr = dyn_cast<BoolAttr>(attr);
    if (!boolAttr)
      return op->emitError() << "`ignore` attribute of test " << test.name
                             << " must be a boolean";
    test.ignore = boolAttr.getValue();
  }

  // Handle the `require_runners` and `exclude_runners` attributes.
  if (failed(collectRunnerFilters(test.attrs, "require_runners",
                                  test.requiredRunners, test.loc, test.name)) ||
      failed(collectRunnerFilters(test.attrs, "exclude_runners",
                                  test.excludedRunners, test.loc, test.name)))
    return failure();

  tests.push_back(std::move(test));
  return success();
}

//===----------------------------------------------------------------------===//
// Tool Implementation
//===----------------------------------------------------------------------===//

/// List all configured runners.
static LogicalResult listRunners(RunnerSuite &suite) {
  // Open the output file for writing.
  std::string errorMessage;
  auto output = openOutputFile(opts.outputFilename, &errorMessage);
  if (!output) {
    WithColor::error() << errorMessage << "\n";
    return failure();
  }

  for (auto &runner : suite.runners) {
    auto &os = output->os();
    os << runner.name.getValue();
    if (runner.ignore)
      os << "  ignored";
    else if (runner.available)
      os << "  " << runner.binaryPath;
    else
      os << "  unavailable";
    os << "\n";
  }
  output->keep();
  return success();
}

// Check if test should be included in output listing
bool ignoreTestListing(Test &test, TestSuite &suite) {
  return !suite.listIgnored && test.ignore;
}

/// List all the tests in a given module.
static LogicalResult listTests(TestSuite &suite) {
  // Open the output file for writing.
  std::string errorMessage;
  auto output = openOutputFile(opts.outputFilename, &errorMessage);
  if (!output) {
    WithColor::error() << errorMessage << "\n";
    return failure();
  }

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

/// Called once the suite of available runners has been determined and a module
/// has been parsed. If the `--split-input-file` option is set, this function is
/// called once for each split of the input file.
static LogicalResult executeWithHandler(MLIRContext *context,
                                        RunnerSuite &runnerSuite,
                                        SourceMgr &srcMgr) {
  std::string errorMessage;
  auto module = parseSourceFile<ModuleOp>(srcMgr, context);
  if (!module)
    return failure();

  // Load the emitter options from the command line. Command line options if
  // specified will override any module options.
  if (opts.loweringOptions.toString() != LoweringOptions().toString())
    opts.loweringOptions.setAsAttribute(module.get());

  // Preprocess the input.
  {
    PassManager pm(context);
    pm.enableVerifier(opts.verifyPasses);
    if (opts.ignoreContracts)
      pm.addPass(verif::createStripContractsPass());
    else
      pm.addPass(verif::createLowerContractsPass());
    pm.addNestedPass<hw::HWModuleOp>(verif::createSimplifyAssumeEqPass());
    pm.addPass(createCSEPass());
    pm.addPass(createSimpleCanonicalizerPass());
    if (failed(pm.run(*module)))
      return failure();
  }

  // Emit the IR and exit if requested.
  if (opts.emitIR) {
    auto output = openOutputFile(opts.outputFilename, &errorMessage);
    if (!output) {
      WithColor::error() << errorMessage << "\n";
      return failure();
    }
    module->print(output->os());
    output->keep();
    return success();
  }

  // Discover all tests in the input.
  TestSuite suite(context, opts.listIgnored);
  if (failed(suite.discoverInModule(*module)))
    return failure();
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

  // Generate the MLIR output.
  SmallString<128> mlirPath(opts.resultDir);
  llvm::sys::path::append(mlirPath, "design.mlir");
  auto mlirFile = openOutputFile(mlirPath, &errorMessage);
  if (!mlirFile) {
    WithColor::error() << errorMessage << "\n";
    return failure();
  }
  module->print(mlirFile->os());
  mlirFile->os().flush();
  mlirFile->keep();

  // Open the Verilog file for writing.
  SmallString<128> verilogPath(opts.resultDir);
  llvm::sys::path::append(verilogPath, "design.sv");
  auto verilogFile = openOutputFile(verilogPath, &errorMessage);
  if (!verilogFile) {
    WithColor::error() << errorMessage << "\n";
    return failure();
  }

  // Generate Verilog output.
  PassManager pm(context);
  pm.enableVerifier(opts.verifyPasses);
  pm.addPass(verif::createLowerFormalToHWPass());
  pm.addPass(
      verif::createLowerSymbolicValuesPass({opts.symbolicValueLowering}));
  pm.addPass(createLowerSimToSVPass());
  pm.addPass(createLowerSeqToSVPass());
  pm.addNestedPass<hw::HWModuleOp>(createLowerVerifToSVPass());
  pm.addNestedPass<hw::HWModuleOp>(sv::createHWLegalizeModulesPass());
  pm.addNestedPass<hw::HWModuleOp>(sv::createPrettifyVerilogPass());
  pm.addPass(createExportVerilogPass(verilogFile->os()));
  if (failed(pm.run(*module)))
    return failure();
  verilogFile->os().flush();
  verilogFile->keep();

  // Run the tests.
  std::atomic<unsigned> numPassed(0);
  std::atomic<unsigned> numIgnored(0);
  std::atomic<unsigned> numUnsupported(0);

  mlir::parallelForEach(context, suite.tests, [&](auto &test) {
    if (test.ignore) {
      ++numIgnored;
      return;
    }

    // Pick a runner for this test. In the future we'll want to filter this
    // based on the test's and runner's metadata, and potentially use a
    // prioritized list of runners.
    Runner *runner = nullptr;
    for (auto &candidate : runnerSuite.runners) {
      if (candidate.ignore || !candidate.available)
        continue;
      if (!test.requiredRunners.empty() &&
          !test.requiredRunners.contains(candidate.name))
        continue;
      if (test.excludedRunners.contains(candidate.name))
        continue;
      runner = &candidate;
      break;
    }
    if (!runner) {
      ++numUnsupported;
      return;
    }

    // Create the directory in which we are going to run the test.
    SmallString<128> testDir(opts.resultDir);
    llvm::sys::path::append(testDir, test.name.getValue());
    if (auto error = llvm::sys::fs::create_directory(testDir)) {
      mlir::emitError(test.loc) << "cannot create test directory `" << testDir
                                << "`: " << error.message();
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
    args.push_back(runner->binary);
    if (runner->readsMLIR)
      args.push_back(mlirPath);
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
    auto result = llvm::sys::ExecuteAndWait(
        runner->binaryPath, args, /*Env=*/std::nullopt,
        /*Redirects=*/{"", logPath, logPath},
        /*SecondsToWait=*/0,
        /*MemoryLimit=*/0, &errorMessage);
    if (result < 0) {
      mlir::emitError(test.loc) << "cannot execute runner: " << errorMessage;
    } else if (result > 0) {
      auto d = mlir::emitError(test.loc)
               << "test " << test.name.getValue() << " failed";
      d.attachNote() << "see `" << logPath << "`";
      d.attachNote() << "executed with " << runner->name.getValue();
    } else {
      ++numPassed;
    }
  });

  // Print statistics about how many tests passed and failed.
  unsigned numNonFailed = numPassed + numIgnored + numUnsupported;
  assert(numNonFailed <= suite.tests.size());
  unsigned numFailed = suite.tests.size() - numNonFailed;
  if (numFailed > 0) {
    WithColor(llvm::errs(), raw_ostream::SAVEDCOLOR, true).get()
        << numFailed << " tests ";
    WithColor(llvm::errs(), raw_ostream::RED, true).get() << "FAILED";
    llvm::errs() << ", " << numPassed << " passed";
  } else {
    WithColor(llvm::errs(), raw_ostream::SAVEDCOLOR, true).get()
        << numPassed << " tests ";
    WithColor(llvm::errs(), raw_ostream::GREEN, true).get() << "passed";
  }
  if (numIgnored > 0)
    llvm::errs() << ", " << numIgnored << " ignored";
  if (numUnsupported > 0)
    llvm::errs() << ", " << numUnsupported << " unsupported";
  llvm::errs() << "\n";
  return success(numFailed == 0);
}

/// Entry point for the circt-test tool. At this point an MLIRContext is
/// available, all dialects have been registered, and all command line options
/// have been parsed.
static LogicalResult execute(MLIRContext *context) {
  // Discover all available test runners.
  RunnerSuite runnerSuite(context);
  runnerSuite.addDefaultRunners();
  if (failed(runnerSuite.resolve()))
    return failure();

  // List all runners and exit if requested.
  if (opts.listRunners)
    return listRunners(runnerSuite);

  // Read the input file.
  auto input = llvm::MemoryBuffer::getFileOrSTDIN(opts.inputFilename);
  if (input.getError()) {
    WithColor::error() << "could not open input file " << opts.inputFilename;
    return failure();
  }

  // Process the input file. If requested by the user, split the input file and
  // process each chunk separately. This is useful for verifying diagnostics.
  auto processBuffer = [&](std::unique_ptr<llvm::MemoryBuffer> chunk,
                           raw_ostream &) {
    SourceMgr srcMgr;
    srcMgr.AddNewSourceBuffer(std::move(chunk), llvm::SMLoc());

    // Call `executeWithHandler` with either the regular diagnostic handler, or,
    // if `--verify-diagnostics` is set, with the verifying handler.
    if (opts.verifyDiagnostics) {
      SourceMgrDiagnosticVerifierHandler handler(srcMgr, context);
      context->printOpOnDiagnostic(false);
      (void)executeWithHandler(context, runnerSuite, srcMgr);
      return handler.verify();
    }

    SourceMgrDiagnosticHandler handler(srcMgr, context);
    return executeWithHandler(context, runnerSuite, srcMgr);
  };

  return mlir::splitAndProcessBuffer(
      std::move(*input), processBuffer, llvm::outs(),
      opts.splitInputFile ? mlir::kDefaultSplitMarker : "");
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
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  circt::registerAllDialects(registry);

  // Hide default LLVM options, other than for this tool.
  // MLIR options are added below.
  cl::HideUnrelatedOptions({&opts.cat, &llvm::getColorCategory()});
  cl::ParseCommandLineOptions(argc, argv, "Hardware unit testing tool\n");

  MLIRContext context(registry);
  exit(failed(execute(&context)));
}
