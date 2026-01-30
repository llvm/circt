//===----------------------------------------------------------------------===//
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
#include "circt/Firtool/Firtool.h"
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
#include "llvm/Support/Process.h"
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
/// Which test kinds to execute.
enum class KindFilter { All, Formal, Simulation };

/// The tool's command line options.
struct Options {
  cl::OptionCategory cat{"Basic Options"};
  cl::OptionCategory testCat{"Test Options"};

  cl::opt<std::string> inputFilename{cl::Positional, cl::desc("<input file>"),
                                     cl::init("-"), cl::cat(cat)};

  cl::opt<std::string> outputFilename{
      "o", cl::desc("Output filename (`-` for stdout)"),
      cl::value_desc("filename"), cl::init("-"), cl::cat(cat)};

  cl::opt<bool> listTests{"l", cl::desc("List tests in the input and exit"),
                          cl::init(false), cl::cat(testCat)};

  cl::opt<KindFilter> kindFilter{
      cl::desc("Filter which kinds of tests to run:"),
      cl::values(clEnumValN(KindFilter::Formal, "only-formal",
                            "Only run formal tests"),
                 clEnumValN(KindFilter::Simulation, "only-sim",
                            "Only run simulation tests")),
      cl::init(KindFilter::All), cl::cat(testCat)};

  cl::opt<bool> listRunners{"list-runners", cl::desc("List test runners"),
                            cl::init(false), cl::cat(cat)};

  cl::opt<bool> json{"json", cl::desc("Emit test list as JSON array"),
                     cl::init(false), cl::cat(testCat)};

  cl::opt<bool> listIgnored{"list-ignored", cl::desc("List ignored tests"),
                            cl::init(false), cl::cat(testCat)};

  cl::opt<bool> dryRun{
      "dry-run",
      cl::desc("Print command for each test to stdout instead of executing it"),
      cl::init(false), cl::cat(testCat)};

  cl::opt<unsigned> numThreads{
      "j", cl::value_desc("N"),
      cl::desc("Number of tests to run in parallel\n"
               "- specify 0 to run all tests in parallel\n"
               "- uses the available hardware concurrency by default"),
      cl::cat(testCat)};

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
// Time Utilities
//===----------------------------------------------------------------------===//

/// A reasonable clock to use to display test runtimes to the user.
using Clock = std::chrono::steady_clock;

/// Format a duration as `SS s`, `MM:SS`, or `HH:MM:SS`.
void formatDuration(raw_ostream &os, Clock::duration duration) {
  auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
  auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
  seconds -= hours;
  auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
  seconds -= minutes;

  if (hours.count() != 0) {
    os << hours.count();
    os << ":";
    llvm::write_integer(os, minutes.count(), 2, llvm::IntegerStyle::Integer);
    os << ":";
    llvm::write_integer(os, seconds.count(), 2, llvm::IntegerStyle::Integer);
  } else if (minutes.count() != 0) {
    os << minutes.count();
    os << ":";
    llvm::write_integer(os, seconds.count(), 2, llvm::IntegerStyle::Integer);
  } else {
    os << seconds.count() << " s";
  }
}

//===----------------------------------------------------------------------===//
// Runners
//===----------------------------------------------------------------------===//

namespace {
/// The various kinds of test that can be executed.
enum class TestKind { Formal, Simulation };

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
  /// The kind of test this runner can execute.
  TestKind kind;
  /// The full path to the runner.
  std::string binaryPath;
  /// Whether the full path to the runner was found by searching for `binary` in
  /// the current PATH. In that case we may chose to display `binary` back to
  /// the user instead of `binaryPath`.
  bool implicitBinaryPath = false;
  /// Whether this runner operates on Verilog or MLIR input.
  bool readsMLIR = false;
  /// Whether this runner should be ignored.
  bool ignore = false;
  /// Whether this runner is available or not. This is set to false if the
  /// runner `binary` cannot be found.
  bool available = false;

  /// Get the human-friendly binary. This is either the full binary path, or if
  /// the path was simply expanded from the binary name by looking for the
  /// program in the path, just the binary name.
  StringRef getFriendlyBinary() const {
    return implicitBinaryPath ? binary : binaryPath;
  }
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
    runner.kind = TestKind::Formal;
    runner.binary = "circt-test-runner-sby.py";
    runners.push_back(std::move(runner));
  }
  {
    // circt-bmc
    Runner runner;
    runner.name = StringAttr::get(context, "circt-bmc");
    runner.kind = TestKind::Formal;
    runner.binary = "circt-test-runner-circt-bmc.py";
    runner.readsMLIR = true;
    runners.push_back(std::move(runner));
  }
  {
    // Verilator
    Runner runner;
    runner.name = StringAttr::get(context, "verilator");
    runner.kind = TestKind::Simulation;
    runner.binary = "circt-test-runner-verilator.py";
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

    // Determine the runner's binary path if it only provides a binary name.
    if (runner.binaryPath.empty()) {
      auto findResult = llvm::sys::findProgramByName(runner.binary);
      if (!findResult)
        return;
      runner.binaryPath = findResult.get();
      runner.implicitBinaryPath = true;
    }

    // Check if the program actually exists at that path.
    runner.available = llvm::sys::fs::can_execute(runner.binaryPath);
  });
  return success();
}

//===----------------------------------------------------------------------===//
// Test Discovery
//===----------------------------------------------------------------------===//

namespace {
enum class TestStatus {
  /// The test has not been started yet.
  Pending, // this needs to be first
  /// The test is currently running.
  Running, // this needs to be second
  /// The test was ignored.
  Ignored,
  /// No runner was available to run the test.
  Unsupported,
  /// The test finished and reported a pass.
  Passed,
  /// The test finished and reported a failure.
  Failed,
  /// The test could not be run. This is distinct from the test running but
  /// reporting a failure. For example, this occurs if the result directory
  /// cannot be created, the runner cannot be started, the test metadata is
  /// malformed, or some other reason why the test itself cannot be executed.
  Aborted,
};

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

  /// Execution status.
  TestStatus status = TestStatus::Pending;
  /// An error message if status is `Aborted`.
  std::string message;
  /// When this test was started.
  Clock::time_point startTime;
  /// When this test finished.
  Clock::time_point finishTime;
  /// Path to the log file where test output is captured.
  std::string logPath;
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

  // If the user specified a test kind filter, ignore the test if the kind does
  // not match.
  if (opts.kindFilter != KindFilter::All) {
    if ((opts.kindFilter == KindFilter::Formal &&
         test.kind != TestKind::Formal) ||
        (opts.kindFilter == KindFilter::Simulation &&
         test.kind != TestKind::Simulation))
      test.ignore = true;
  }

  tests.push_back(std::move(test));
  return success();
}

//===----------------------------------------------------------------------===//
// Progress Display
//===----------------------------------------------------------------------===//

namespace {
/// A helper to print and maintain a nice interactive progress display as the
/// tests execute across multiple threads. This assumes that the tests are
/// roughly executed in order.
struct ProgressDisplay {
  ProgressDisplay(ArrayRef<Test> tests) : tests(tests) {}

  /// Update the progress display. Use a mutex to ensure that only one thread
  /// executes this function, and that none of the `status` fields of the tests
  /// change.
  void update();
  /// Print the status of the given test.
  void printTestStatus(const Test &test, bool final);

  /// Where to print the display.
  llvm::raw_ostream &os = llvm::errs();
  /// Whether to use ANSI escape codes to have a more dynamic display.
  bool isDynamic = llvm::sys::Process::StandardErrIsDisplayed();
  /// When testing started. This is used to compute overall execution time.
  Clock::time_point startTime = Clock::now();
  /// The tests being executed.
  ArrayRef<Test> tests;
  /// The index past the last finished test that was reported.
  unsigned doneIndex = 0;
  /// The index past the last running test that was considered.
  unsigned runningIndex = 0;
  /// The number of dynamic lines that we want to overwrite with updated
  /// information on the next update.
  unsigned dynamicLines = 0;
};
} // namespace

void ProgressDisplay::update() {
  // Erase the dynamic lines we have already printed.
  if (dynamicLines > 0 && isDynamic) {
    for (unsigned i = 0; i < dynamicLines; ++i)
      os << "\x1b[1A"; // move up one line
    os << "\x1b[G";    // move to beginning of line
    os << "\x1b[J";    // clear to end of screen
  }
  dynamicLines = 0;

  // Print the tests that have finished since the last update call.
  while (doneIndex < tests.size() &&
         tests[doneIndex].status > TestStatus::Running) {
    printTestStatus(tests[doneIndex], true);
    ++doneIndex;
  }

  // Find the first pending test such that we roughly know in which range of
  // tests we should look for the printing.
  while (runningIndex < tests.size() &&
         tests[runningIndex].status > TestStatus::Pending)
    ++runningIndex;

  // Only print progress bar and other dynamic content if there are any tests
  // left and we are writing to a terminal that appreciates ANSI escape codes.
  if (!isDynamic || doneIndex == tests.size())
    return;
  auto activeTests = tests.slice(doneIndex, runningIndex - doneIndex);

  // Print the status of tests currently running. This will get erased and
  // overwritten on the next call to this function.
  for (auto &test : activeTests) {
    printTestStatus(test, false);
    ++dynamicLines; // erase this line on the next update
  }

  // Count how many tests are already done.
  unsigned doneCount = doneIndex;
  unsigned runningCount = 0;
  for (auto &test : activeTests) {
    if (test.status == TestStatus::Running)
      ++runningCount;
    if (test.status > TestStatus::Running)
      ++doneCount;
  }

  // Print a final interactive progress bar.
  os << "\n";
  unsigned columnsUsed = 0;
  columnsUsed -= os.tell(); // count printed chars from here
  os << "running ";
  formatDuration(os, Clock::now() - startTime);
  os << " [";
  columnsUsed += os.tell(); // to here

  // Compute the number of blocks to fill in for the done tests, the fractional
  // block since we can use unicode blocks to draw 1/8 to 7/8 full blocks, and
  // the blocks to fill differently for the running tests.
  unsigned barLength = 10;
  auto computeBarPosition = [&](unsigned testProgress) {
    // Compute round(8 * testProgress / numTests * barLength) with integers.
    return (8 * testProgress * barLength + tests.size() / 2) / tests.size();
  };
  unsigned barDone = computeBarPosition(doneCount);
  unsigned barDonePartial = barDone % 8;
  barDone /= 8;
  unsigned barRunning = computeBarPosition(doneCount + runningCount) / 8;
  unsigned barIdx = 0;

  // Actually write out the progress bar.
  {
    WithColor bar(os, raw_ostream::CYAN, true);
    for (; barIdx < barDone; ++barIdx)
      bar << "\u2588";
    if (barDonePartial > 0) {
      // Partially full horizontal blocks are U+2589 (7/8) to U+258F (1/8).
      // UTF-8 encoding is e.g. 0xE2 0x96 0x89.
      bar << "\xE2\x96";
      bar << static_cast<char>(0x90 - barDonePartial);
      ++barIdx;
    }
    for (; barIdx < barRunning; ++barIdx)
      bar << "\u00b7";
    for (; barIdx < barLength; ++barIdx)
      bar << ' ';
  }
  columnsUsed += barLength;

  // Add a tally of the tests that have at least started running.
  columnsUsed -= os.tell();
  os << "] " << (doneCount + runningCount) << "/" << tests.size();
  columnsUsed += os.tell();

  // Print out the currently running tests. We want to truncate this in case it
  // would spill onto another line of the screen, because this is just some
  // pretty info for the user.
  unsigned columnsMax = llvm::sys::Process::StandardErrColumns();
  if (columnsMax > 0)
    columnsMax -= 1; // don't cause the cursor to wrap around
  bool isFirst = true;
  for (auto &test : activeTests) {
    if (test.status != TestStatus::Running)
      continue;
    auto name = test.name.getValue();
    columnsUsed += name.size() + 2;
    if (columnsUsed + 3 > columnsMax) {
      if (!isFirst)
        os << ", …";
      break;
    }
    os << (isFirst ? ": " : ", ");
    os << name;
    isFirst = false;
  }

  os << "\n";
  dynamicLines += 2; // erase the progress bar on the next update
}

void ProgressDisplay::printTestStatus(const Test &test, bool final) {
  // Special handling for dry runs.
  if (opts.dryRun) {
    auto &os = llvm::outs(); // dry run goes to stdout
    WithColor(os, raw_ostream::SAVEDCOLOR) << test.name.getValue();
    os << ": " << test.message << "\n";
    return;
  }

  os << "test " << test.name.getValue() << " ... ";

  bool hasColors = WithColor(os).colorsEnabled();
  const char *ansiDim = hasColors ? "\x1b[2m" : "";
  const char *ansiReset = hasColors ? "\x1b[0m" : "";
  bool shouldAddTime = false;
  auto untilTime = test.finishTime;

  switch (test.status) {
  case TestStatus::Pending:
    break;
  case TestStatus::Running:
    WithColor(os, raw_ostream::CYAN) << "running";
    shouldAddTime = true;
    untilTime = Clock::now();
    break;
  case TestStatus::Ignored:
    os << ansiDim << "ignored" << ansiReset;
    break;
  case TestStatus::Unsupported:
    os << ansiDim << "unsupported" << ansiReset;
    break;
  case TestStatus::Passed:
    WithColor(os, raw_ostream::GREEN) << "passed";
    shouldAddTime = true;
    break;
  case TestStatus::Failed:
    WithColor(os, raw_ostream::RED) << "FAILED";
    shouldAddTime = true;
    break;
  case TestStatus::Aborted:
    WithColor(os, raw_ostream::RED) << "ABORTED";
    shouldAddTime = true;
    break;
  }

  // Print the time duration if requested.
  if (shouldAddTime) {
    os << "  " << ansiDim;
    formatDuration(os, untilTime - test.startTime);
    os << ansiReset;
  }

  os << "\n";

  // If this isn't the final print of the test status, stop here.
  if (!final)
    return;

  // Print the error message if the test aborted, which means that we weren't
  // even able to run the test.
  if (test.status == TestStatus::Aborted) {
    mlir::emitError(test.loc) << test.message;
    return;
  }

  // If the test failed, print part of the failure log.
  if (test.status == TestStatus::Failed) {
    mlir::emitError(test.loc) << "failing test defined here";
    if (auto log = llvm::MemoryBuffer::getFile(test.logPath, /*isText=*/true)) {
      os << "----- 8< ----- " << test.logPath << " ----- 8< -----\n";
      auto logStr = log.get()->getBuffer();
      constexpr size_t truncAbove = 10000;  // truncate above this log size
      constexpr size_t newlineRange = 1000; // bytes to scan for a newline
      if (logStr.size() > truncAbove) {
        size_t pos = logStr.size() - truncAbove;
        size_t minPos = pos > newlineRange ? pos - newlineRange : 0;
        while (pos > minPos && logStr[pos] != '\n')
          --pos;
        if (pos > 0 && logStr[pos - 1] == '\r')
          --pos;
        os << "[" << pos << " bytes truncated]";
        logStr = logStr.substr(pos);
      }
      os << logStr;
      os << "----- 8< ----- " << test.logPath << " ----- 8< -----\n";
    } else {
      os << "see " << test.logPath << "\n";
    }
    return;
  }
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
    os << "  " << toString(runner.kind);
    if (runner.ignore)
      os << "  ignored";
    else if (runner.available)
      os << "  " << runner.getFriendlyBinary();
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
    auto guard = scope_exit([&] { json.arrayEnd(); });
    for (auto &test : suite.tests) {
      if (ignoreTestListing(test, suite))
        continue;
      json.objectBegin();
      auto guard = scope_exit([&] { json.objectEnd(); });
      json.attribute("name", test.name.getValue());
      json.attribute("kind", toString(test.kind));
      if (!test.attrs.empty()) {
        json.attributeBegin("attrs");
        auto guard = scope_exit([&] { json.attributeEnd(); });
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

/// Execute a single test. This function is called for each test in the test
/// suite, distributed across multiple threads.
///
/// CAUTION: Do not mutate `test.status` directly. Instead, return the result
/// status of the test. The caller will update `test.status` with the progress
/// display mutex locked, to ensure terminal output is consistent.
static TestStatus executeTest(Test &test, RunnerSuite &runnerSuite,
                              StringRef mlirPath, StringRef verilogPath) {
  if (test.ignore)
    return TestStatus::Ignored;

  // Pick a runner for this test. In the future we'll want to filter this
  // based on the test's and runner's metadata, and potentially use a
  // prioritized list of runners.
  Runner *runner = nullptr;
  for (auto &candidate : runnerSuite.runners) {
    if (candidate.ignore || !candidate.available)
      continue;
    if (test.kind != candidate.kind)
      continue;
    if (!test.requiredRunners.empty() &&
        !test.requiredRunners.contains(candidate.name))
      continue;
    if (test.excludedRunners.contains(candidate.name))
      continue;
    runner = &candidate;
    break;
  }
  if (!runner)
    return TestStatus::Unsupported;

  // Create the directory in which we are going to run the test.
  raw_string_ostream msg(test.message);
  SmallString<128> testDir(opts.resultDir);
  llvm::sys::path::append(testDir, test.name.getValue());
  if (auto error = llvm::sys::fs::create_directory(testDir)) {
    msg << "cannot create test directory `" << testDir
        << "`: " << error.message();
    return TestStatus::Aborted;
  }

  // Assemble a path for the test runner log file and truncate it.
  SmallString<128> logPath = testDir;
  llvm::sys::path::append(logPath, "run.log");
  test.logPath = logPath.str();
  {
    std::error_code ec;
    raw_fd_ostream trunc(logPath, ec);
  }

  // Assemble the runner arguments.
  BumpPtrAllocator argsAlloc;
  StringSaver argsSaver(argsAlloc);
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
      msg << "invalid `mode` attribute " << mode;
      return TestStatus::Aborted;
    }
    args.push_back(modeStr.getValue());
  }

  if (auto depth = test.attrs.get("depth")) {
    args.push_back("-k");
    auto depthInt = dyn_cast<IntegerAttr>(depth);
    if (!depthInt) {
      msg << "invalid `depth` attribute " << depth;
      return TestStatus::Aborted;
    }
    SmallString<8> str;
    depthInt.getValue().toStringUnsigned(str);
    args.push_back(argsSaver.save(str.str()));
  }

  // If we are doing a dry run, store the command we would run in the test's
  // `message` and skip actually executing the test runner. The progress display
  // will then print the message.
  if (opts.dryRun) {
    llvm::sys::printArg(msg, runner->getFriendlyBinary(), false);
    for (auto &arg : llvm::drop_begin(args)) {
      msg << " ";
      llvm::sys::printArg(msg, arg, false);
    }
    return TestStatus::Passed;
  }

  // Execute the test runner.
  std::string errorMessage;
  auto result =
      llvm::sys::ExecuteAndWait(runner->binaryPath, args, /*Env=*/std::nullopt,
                                /*Redirects=*/{"", test.logPath, test.logPath},
                                /*SecondsToWait=*/0,
                                /*MemoryLimit=*/0, &errorMessage);
  if (result < 0) {
    msg << "cannot execute runner: " << errorMessage;
    return TestStatus::Aborted;
  } else if (result > 0) {
    return TestStatus::Failed;
  } else {
    return TestStatus::Passed;
  }
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

  // Since some tools insist on treating empty modules as black boxes, we need
  // to fix those up explicitly.
  opts.loweringOptions.fixUpEmptyModules = true;

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
  llvm::errs() << "running " << suite.tests.size() << " tests\n";

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

  // Generate Verilog output.
  firtool::FirtoolOptions firtoolOptions;
  SmallString<128> verilogPath(opts.resultDir);
  llvm::sys::path::append(verilogPath, "design");
  firtoolOptions.setOutputFilename(verilogPath);

  // Set verification mode to All if verifyPasses is enabled.
  if (opts.verifyPasses)
    firtoolOptions.setVerificationMode(
        firtool::FirtoolOptions::VerificationMode::All);

  PassManager pm(context);
  pm.enableVerifier(opts.verifyPasses);
  if (failed(firtool::populateHWToSV(pm, firtoolOptions)))
    return failure();
  if (failed(
          firtool::populateExportSplitVerilog(pm, firtoolOptions, verilogPath)))
    return failure();
  if (failed(pm.run(*module)))
    return failure();

  // Determine how many threads to use for running tests.
  unsigned numThreads = 1;
  if (!opts.dryRun && context->isMultithreadingEnabled()) {
    // If the user specified the `-j` option, use that as the number of threads.
    // If they specified 0, create a thread for each test. If `-j` was not
    // specified, default to the hardware concurrency.
    if (opts.numThreads.getNumOccurrences()) {
      numThreads = opts.numThreads;
      if (numThreads == 0)
        numThreads = suite.tests.size();
    } else {
      numThreads = std::thread::hardware_concurrency();
    }
  }
  if (numThreads > suite.tests.size())
    numThreads = suite.tests.size();

  // Setup a helper to dynamically display progress.
  ProgressDisplay pd(suite.tests);
  std::mutex pdMutex;
  std::condition_variable pdWake;

  // Spawn a separate thread that updates the progress display every second. We
  // use a condition variable to wait for 1s or until the variables is triggered
  // at the end of the run.
  auto timerThread = std::thread([&] {
    while (true) {
      std::unique_lock<std::mutex> lock(pdMutex);
      if (pd.doneIndex == pd.tests.size())
        return;
      pd.update();
      pdWake.wait_for(lock, std::chrono::seconds(1));
    }
  });

  // Run the tests. We spin up our own threads here instead of using
  // `mlir::parallelForEach`, since we want more explicit control over the
  // number of threads and we want to not create a `ParallelDiagnosticHandler`,
  // giving us more precise control over the error messages we print.
  auto runTest = [&](auto &test) {
    {
      std::lock_guard<std::mutex> lock(pdMutex);
      test.status = TestStatus::Running;
      test.startTime = Clock::now();
      pd.update();
    }
    auto status = executeTest(test, runnerSuite, mlirPath, verilogPath);
    assert(status != TestStatus::Pending && status != TestStatus::Running);
    {
      std::lock_guard<std::mutex> lock(pdMutex);
      test.status = status;
      test.finishTime = Clock::now();
      pd.update();
    }
  };

  std::atomic<unsigned> globalIndex = 0;
  auto runTests = [&] {
    while (true) {
      unsigned threadIndex = globalIndex++;
      if (threadIndex >= suite.tests.size())
        return;
      runTest(suite.tests[threadIndex]);
    }
  };

  std::vector<std::thread> threads;
  threads.resize(numThreads);
  for (auto &thread : threads)
    thread = std::thread(runTests);
  for (auto &thread : threads)
    thread.join();

  // Signal the timer thread to stop updating the progress display.
  pdWake.notify_all();
  timerThread.join();

  // Stop here if we are doing a dry run.
  if (opts.dryRun)
    return success();

  // Print statistics about how many tests passed and failed.
  unsigned numIgnored = 0;
  unsigned numUnsupported = 0;
  unsigned numPassed = 0;
  unsigned numFailed = 0;
  for (auto &test : suite.tests) {
    switch (test.status) {
    case TestStatus::Pending:
    case TestStatus::Running:
      llvm_unreachable("all tests executed");
      break;
    case TestStatus::Ignored:
      ++numIgnored;
      break;
    case TestStatus::Unsupported:
      ++numUnsupported;
      break;
    case TestStatus::Passed:
      ++numPassed;
      break;
    case TestStatus::Failed:
    case TestStatus::Aborted:
      ++numFailed;
      break;
    }
  }

  auto &os = llvm::errs();
  os << "\n  ";
  if (numFailed > 0) {
    os << numFailed << " of " << (numPassed + numFailed) << " tests ";
    WithColor(os, raw_ostream::RED, true) << "FAILED";
    os << "; " << numPassed << " passed";
  } else {
    os << "all " << numPassed << " tests ";
    WithColor(os, raw_ostream::GREEN, true) << "passed";
  }
  if (numIgnored > 0)
    os << "; " << numIgnored << " ignored";
  if (numUnsupported > 0)
    os << "; " << numUnsupported << " unsupported";
  os << "; finished in ";
  formatDuration(os, Clock::now() - pd.startTime);
  os << "\n\n";
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
  cl::HideUnrelatedOptions(
      {&opts.cat, &opts.testCat, &llvm::getColorCategory()});
  cl::ParseCommandLineOptions(
      argc, argv, "Hardware unit test discovery and execution tool\n");

  MLIRContext context(registry);
  exit(failed(execute(&context)));
}
