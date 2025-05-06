//===- hlstool.cpp - The hlstool utility for working with .fir files ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements 'hlstool', which composes together a variety of
// CIRCT libraries that can be used to realise HLS (High Level Synthesis)
// flows.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Chrono.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "circt/Conversion/CalyxToFSM.h"
#include "circt/Conversion/ExportVerilog.h"
#include "circt/Conversion/Passes.h"
#include "circt/Conversion/SCFToCalyx.h"
#include "circt/Dialect/Calyx/CalyxDialect.h"
#include "circt/Dialect/Calyx/CalyxPasses.h"
#include "circt/Dialect/DC/DCPasses.h"
#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/ESI/ESIPasses.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Support/LoweringOptions.h"
#include "circt/Support/LoweringOptionsParser.h"
#include "circt/Support/Version.h"
#include "circt/Transforms/Passes.h"

using namespace llvm;
using namespace mlir;
using namespace circt;

// --------------------------------------------------------------------------
// Tool options
// --------------------------------------------------------------------------

static cl::OptionCategory mainCategory("hlstool Options");

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init("-"), cl::cat(mainCategory));

static cl::opt<std::string> outputFilename(
    "o", cl::desc("Output filename, or directory for split output"),
    cl::value_desc("filename"), cl::init("-"), cl::cat(mainCategory));

static cl::opt<bool>
    splitInputFile("split-input-file",
                   cl::desc("Split the input file into pieces and process each "
                            "chunk independently"),
                   cl::init(false), cl::Hidden, cl::cat(mainCategory));

static cl::opt<bool>
    verifyDiagnostics("verify-diagnostics",
                      cl::desc("Check that emitted diagnostics match "
                               "expected-* lines on the corresponding line"),
                      cl::init(false), cl::Hidden, cl::cat(mainCategory));

static cl::opt<bool>
    verbosePassExecutions("verbose-pass-executions",
                          cl::desc("Log executions of toplevel module passes"),
                          cl::init(false), cl::cat(mainCategory));

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true), cl::cat(mainCategory));

static cl::opt<bool>
    allowUnregisteredDialects("allow-unregistered-dialects",
                              cl::desc("Allow unknown dialects in the input"),
                              cl::init(false), cl::Hidden,
                              cl::cat(mainCategory));

enum HLSFlow {
  // Compilation through the dynamically scheduled handshake dialect.
  HLSFlowDynamicHW,
  // Compilation through Calyx's CIRCT lowering implementation.
  HLSFlowCalyxHW,
};

static cl::opt<HLSFlow>
    hlsFlow(cl::desc("HLS flow"),
            cl::values(clEnumValN(HLSFlowDynamicHW, "dynamic-hw",
                                  "Dynamically scheduled (HW path)"),
                       clEnumValN(HLSFlowCalyxHW, "calyx-hw",
                                  "Statically scheduled (Calyx path)")),
            cl::cat(mainCategory));

enum DynamicParallelismKind { None, Locking, Pipelining };

static cl::opt<DynamicParallelismKind> dynParallelism(
    "dynamic-parallelism", cl::desc("Specify the DHLS task parallelism kind"),
    cl::values(
        clEnumValN(None, "none",
                   "Add no protection mechanisms that could prevent data races "
                   "when a function has multiple active invocations."),
        clEnumValN(Locking, "locking",
                   "Add function locking protection mechanism which ensures "
                   "that only one function invocation is active."),
        clEnumValN(Pipelining, "pipelining",
                   "Add function pipelining mechanism that enables a "
                   "pipelined execution of multiple function invocations while "
                   "preserving correctness.")),
    cl::init(Pipelining), cl::cat(mainCategory));

enum IRLevel {
  // A high-level dialect like affine or scf
  High,
  // The IR right before the core lowering dialect
  PreCompile,
  // The IR in core dialect
  Core,
  // The lowest form of core IR (i.e. after all passes have run)
  PostCompile,
  // The IR after lowering is performed
  RTL,
  // System verilog representation
  SV
};

static cl::opt<IRLevel> irInputLevel(
    "input-level",
    cl::desc("Level at which to input IR at. It is flow-defined "
             "which value corersponds to which IR level."),
    cl::values(
        clEnumValN(High, "high", "High-level dialects like affine"),
        clEnumValN(
            PreCompile, "pre-compile",
            "The IR right before the core lowering dialect (handshake/calyx)"),
        clEnumValN(Core, "core", "The IR in core dialect (handshake/calyx)"),
        clEnumValN(
            PostCompile, "post-compile",
            "The lowest form of core IR (handshake/calyx) (i.e. after all "
            "passes have run)"),
        clEnumValN(RTL, "rtl", "The IR after lowering is performed")),
    cl::init(IRLevel::High), cl::cat(mainCategory));

static cl::opt<IRLevel> irOutputLevel(
    "output-level",
    cl::desc("Level at which to output IR at. It is flow-defined "
             "which value corersponds to which IR level."),
    cl::values(
        clEnumValN(
            PreCompile, "pre-compile",
            "The IR right before the core lowering dialect (handshake/calyx)"),
        clEnumValN(Core, "core", "The IR in core dialect (handshake/calyx)"),
        clEnumValN(
            PostCompile, "post-compile",
            "The lowest form of core IR (handshake/calyx) (i.e. after all "
            "passes have run)"),
        clEnumValN(RTL, "rtl", "The IR after lowering is performed"),
        clEnumValN(SV, "sv", "System verilog representation")),
    cl::init(IRLevel::SV), cl::cat(mainCategory));

enum OutputFormatKind { OutputIR, OutputVerilog, OutputSplitVerilog };

static cl::opt<OutputFormatKind> outputFormat(
    cl::desc("Specify output format:"),
    cl::values(clEnumValN(OutputIR, "ir", "Emit post-HLS IR"),
               clEnumValN(OutputVerilog, "verilog", "Emit Verilog"),
               clEnumValN(OutputSplitVerilog, "split-verilog",
                          "Emit Verilog (one file per module; specify "
                          "directory with -o=<dir>)")),
    cl::init(OutputVerilog), cl::cat(mainCategory));

static cl::opt<bool>
    traceIVerilog("sv-trace-iverilog",
                  cl::desc("Add tracing to an iverilog simulated module"),
                  cl::init(false), cl::cat(mainCategory));

// --------------------------------------------------------------------------
// Handshake options
// --------------------------------------------------------------------------

static cl::opt<std::string>
    bufferingStrategy("buffering-strategy",
                      cl::desc("Strategy to apply. Possible values are: "
                               "cycles, allFIFO, all (default)"),
                      cl::init("all"), cl::cat(mainCategory));

static cl::opt<unsigned> bufferSize("buffer-size",
                                    cl::desc("Number of slots in each buffer"),
                                    cl::init(2), cl::cat(mainCategory));

static cl::opt<bool> withESI("with-esi",
                             cl::desc("Create ESI compatible modules"),
                             cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> withDC("dc", cl::desc("Use the DC flow"), cl::init(false),
                            cl::cat(mainCategory));

static LoweringOptionsOption loweringOptions(mainCategory);

// --------------------------------------------------------------------------
// Calyx options
// --------------------------------------------------------------------------
static cl::opt<std::string> topLevelFunction("top-level-function",
                                             cl::desc("Top level function"),
                                             cl::init(""),
                                             cl::cat(mainCategory));

// --------------------------------------------------------------------------
// (Configurable) pass pipelines
// --------------------------------------------------------------------------

/// Create a simple canonicalizer pass.
static std::unique_ptr<Pass> createSimpleCanonicalizerPass() {
  mlir::GreedyRewriteConfig config;
  config.setUseTopDownTraversal(true);
  config.setRegionSimplificationLevel(
      mlir::GreedySimplifyRegionLevel::Disabled);
  return mlir::createCanonicalizerPass(config);
}

static void loadDHLSPipeline(OpPassManager &pm) {
  // Memref legalization.
  pm.addPass(circt::createFlattenMemRefPass());
  pm.nest<func::FuncOp>().addPass(
      circt::handshake::createHandshakeLegalizeMemrefsPass());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.nest<handshake::FuncOp>().addPass(createSimpleCanonicalizerPass());

  // DHLS conversion
  pm.addPass(circt::createCFToHandshakePass(
      /*sourceConstants=*/false,
      /*disableTaskPipelining=*/dynParallelism != Pipelining));
  pm.addPass(circt::handshake::createHandshakeLowerExtmemToHWPass(withESI));

  if (dynParallelism == Locking) {
    pm.nest<handshake::FuncOp>().addPass(
        circt::handshake::createHandshakeLockFunctionsPass());
    // The locking pass does not adapt forks, thus this additional pass is
    // required
    pm.nest<handshake::FuncOp>().addPass(
        handshake::createHandshakeMaterializeForksSinksPass());
  }
}

static void loadHandshakeTransformsPipeline(OpPassManager &pm) {
  pm.nest<handshake::FuncOp>().addPass(createSimpleCanonicalizerPass());
  pm.nest<handshake::FuncOp>().addPass(
      handshake::createHandshakeMaterializeForksSinksPass());
  pm.nest<handshake::FuncOp>().addPass(createSimpleCanonicalizerPass());
  pm.nest<handshake::FuncOp>().addPass(
      handshake::createHandshakeInsertBuffersPass(bufferingStrategy,
                                                  bufferSize));
  pm.nest<handshake::FuncOp>().addPass(createSimpleCanonicalizerPass());
}

static void loadESILoweringPipeline(OpPassManager &pm) {
  pm.addPass(circt::esi::createESIPortLoweringPass());
  pm.addPass(circt::esi::createESIPhysicalLoweringPass());
  pm.addPass(circt::esi::createESItoHWPass());
}

static void loadHWLoweringPipeline(OpPassManager &pm) {
  pm.addPass(createSimpleCanonicalizerPass());
  pm.nest<hw::HWModuleOp>().addPass(circt::seq::createLowerSeqHLMemPass());
  pm.addPass(seq::createHWMemSimImplPass());
  pm.addPass(circt::createLowerSeqToSVPass());
  pm.nest<hw::HWModuleOp>().addPass(sv::createHWCleanupPass());

  // Legalize unsupported operations within the modules.
  pm.nest<hw::HWModuleOp>().addPass(sv::createHWLegalizeModulesPass());
  pm.addPass(createSimpleCanonicalizerPass());

  // Tidy up the IR to improve verilog emission quality.
  auto &modulePM = pm.nest<hw::HWModuleOp>();
  modulePM.addPass(sv::createPrettifyVerilogPass());
}

// --------------------------------------------------------------------------
// Tool driver code
// --------------------------------------------------------------------------

static LogicalResult doHLSFlowDynamic(
    PassManager &pm, ModuleOp module,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {

  bool suppressLaterPasses = false;
  auto notSuppressed = [&]() { return !suppressLaterPasses; };
  auto addIfNeeded = [&](llvm::function_ref<bool()> predicate,
                         llvm::function_ref<void()> passAdder) {
    if (predicate())
      passAdder();
  };

  auto addIRLevel = [&](int level, llvm::function_ref<void()> passAdder) {
    addIfNeeded(notSuppressed, [&]() {
      // Add the pass if the input IR level is at least the current
      // abstraction.
      if (irInputLevel <= level)
        passAdder();
      // Suppresses later passes if we're emitting IR and the output IR level is
      // the current level.
      if (outputFormat == OutputIR && irOutputLevel == level)
        suppressLaterPasses = true;
    });
  };

  // Software lowering
  addIRLevel(IRLevel::PreCompile, [&]() {
    pm.addPass(mlir::createLowerAffinePass());
    pm.addPass(mlir::createSCFToControlFlowPass());
  });

  addIRLevel(IRLevel::Core, [&]() { loadDHLSPipeline(pm); });
  addIRLevel(IRLevel::PostCompile,
             [&]() { loadHandshakeTransformsPipeline(pm); });

  // HW path.

  addIRLevel(IRLevel::RTL, [&]() {
    pm.nest<handshake::FuncOp>().addPass(createSimpleCanonicalizerPass());
    if (withDC) {
      pm.addPass(circt::createHandshakeToDC({"clock", "reset"}));
      // This pass sometimes resolves an error in the
      pm.addPass(createSimpleCanonicalizerPass());
      pm.nest<hw::HWModuleOp>().addPass(
          circt::dc::createDCMaterializeForksSinksPass());
      // TODO: We assert without a canonicalizer pass here. Debug.
      pm.addPass(createSimpleCanonicalizerPass());
      pm.addPass(circt::createDCToHWPass());
      pm.addPass(createSimpleCanonicalizerPass());
      pm.addPass(circt::createMapArithToCombPass());
      pm.addPass(createSimpleCanonicalizerPass());
    } else {
      pm.addPass(circt::createHandshakeToHWPass());
    }
    pm.addPass(createSimpleCanonicalizerPass());
    loadESILoweringPipeline(pm);
  });

  addIRLevel(IRLevel::SV, [&]() { loadHWLoweringPipeline(pm); });

  if (traceIVerilog)
    pm.addPass(circt::sv::createSVTraceIVerilogPass());

  if (loweringOptions.getNumOccurrences())
    loweringOptions.setAsAttribute(module);
  if (outputFormat == OutputVerilog) {
    pm.addPass(createExportVerilogPass((*outputFile)->os()));
  } else if (outputFormat == OutputSplitVerilog) {
    pm.addPass(createExportSplitVerilogPass(outputFilename));
  }

  // Go execute!
  if (failed(pm.run(module)))
    return failure();

  if (outputFormat == OutputIR)
    module->print((*outputFile)->os());

  return success();
}

static LogicalResult doHLSFlowCalyx(
    PassManager &pm, ModuleOp module,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {

  // XXX(rachitnigam): Duplicated from doHLSFlowDynamic. We should probably
  // abstract this pattern. The only problem is that addIRLevel captures this
  // mutable variable.
  bool suppressLaterPasses = false;
  auto notSuppressed = [&]() { return !suppressLaterPasses; };
  auto addIfNeeded = [&](llvm::function_ref<bool()> predicate,
                         llvm::function_ref<void()> passAdder) {
    if (predicate())
      passAdder();
  };

  auto addIRLevel = [&](int level, llvm::function_ref<void()> passAdder) {
    addIfNeeded(notSuppressed, [&]() {
      // Add the pass if the input IR level is at least the current
      // abstraction.
      if (irInputLevel <= level)
        passAdder();
      // Suppresses later passes if we're emitting IR and the output IR level is
      // the current level.
      if (outputFormat == OutputIR && irOutputLevel == level)
        suppressLaterPasses = true;
    });
  };

  // Lower to only SCF abstractions
  addIRLevel(IRLevel::PreCompile, [&]() {
    /* Nothing to do */
  });

  // Lower to Calyx
  addIRLevel(IRLevel::Core, [&]() {
    pm.addPass(circt::createSCFToCalyxPass(topLevelFunction));
  });

  // Run Calyx transforms
  addIRLevel(IRLevel::PostCompile, [&]() {
    pm.addPass(createSimpleCanonicalizerPass());
    // Eliminate Calyx's comb group abstraction
    pm.addNestedPass<calyx::ComponentOp>(
        circt::calyx::createRemoveCombGroupsPass());
    pm.addPass(createSimpleCanonicalizerPass());

    // Compile to FSM
    pm.addNestedPass<calyx::ComponentOp>(circt::createCalyxToFSMPass());
    pm.addPass(createSimpleCanonicalizerPass());
    pm.addNestedPass<calyx::ComponentOp>(
        circt::createMaterializeCalyxToFSMPass());
    pm.addPass(createSimpleCanonicalizerPass());

    // Eliminate Calyx's group abstraction
    pm.addNestedPass<calyx::ComponentOp>(
        circt::createRemoveGroupsFromFSMPass());
    pm.addPass(createSimpleCanonicalizerPass());
  });

  // HW path.

  addIRLevel(IRLevel::RTL, [&]() {
    // Compile to HW
    pm.addPass(circt::createCalyxToHWPass());
    pm.addPass(createSimpleCanonicalizerPass());
  });

  addIRLevel(IRLevel::SV, [&]() {
    pm.addPass(circt::createConvertFSMToSVPass());
    loadHWLoweringPipeline(pm);
  });

  if (outputFormat == OutputVerilog) {
    pm.addPass(createExportVerilogPass((*outputFile)->os()));
  } else if (outputFormat == OutputSplitVerilog) {
    pm.addPass(createExportSplitVerilogPass(outputFilename));
  }

  if (failed(pm.run(module)))
    return failure();

  if (outputFormat == OutputIR)
    module->print((*outputFile)->os());

  return success();
}

/// Process a single buffer of the input.
static LogicalResult processBuffer(
    MLIRContext &context, TimingScope &ts, llvm::SourceMgr &sourceMgr,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  // Parse the input.
  mlir::OwningOpRef<mlir::ModuleOp> module;
  llvm::sys::TimePoint<> parseStartTime;
  if (verbosePassExecutions) {
    llvm::errs() << "[hlstool] Running MLIR parser\n";
    parseStartTime = llvm::sys::TimePoint<>::clock::now();
  }
  auto parserTimer = ts.nest("MLIR Parser");
  module = parseSourceFile<ModuleOp>(sourceMgr, &context);

  if (!module)
    return failure();

  if (verbosePassExecutions) {
    auto elpased = std::chrono::duration<double>(
                       llvm::sys::TimePoint<>::clock::now() - parseStartTime) /
                   std::chrono::seconds(1);
    llvm::errs() << "[hlstool] -- Done in " << llvm::format("%.3f", elpased)
                 << " sec\n";
  }

  // Apply any pass manager command line options.
  PassManager pm(&context);
  pm.enableVerifier(verifyPasses);
  pm.enableTiming(ts);
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();

  switch (hlsFlow) {
  case HLSFlowDynamicHW: {
    if (failed(doHLSFlowDynamic(pm, module.get(), outputFile)))
      return failure();
    break;
  }
  case HLSFlowCalyxHW: {
    if (failed(doHLSFlowCalyx(pm, module.get(), outputFile)))
      return failure();
    break;
  }
  }

  // We intentionally "leak" the Module into the MLIRContext instead of
  // deallocating it.  There is no need to deallocate it right before process
  // exit.
  (void)module.release();
  return success();
}

/// Process a single split of the input. This allocates a source manager and
/// creates a regular or verifying diagnostic handler, depending on whether
/// the user set the verifyDiagnostics option.
static LogicalResult processInputSplit(
    MLIRContext &context, TimingScope &ts,
    std::unique_ptr<llvm::MemoryBuffer> buffer,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  if (!verifyDiagnostics) {
    SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
    return processBuffer(context, ts, sourceMgr, outputFile);
  }

  SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr, &context);
  context.printOpOnDiagnostic(false);
  (void)processBuffer(context, ts, sourceMgr, outputFile);
  return sourceMgrHandler.verify();
}

/// Process the entire input provided by the user, splitting it up if the
/// corresponding option was specified.
static LogicalResult
processInput(MLIRContext &context, TimingScope &ts,
             std::unique_ptr<llvm::MemoryBuffer> input,
             std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  if (!splitInputFile)
    return processInputSplit(context, ts, std::move(input), outputFile);

  return splitAndProcessBuffer(
      std::move(input),
      [&](std::unique_ptr<MemoryBuffer> buffer, raw_ostream &) {
        return processInputSplit(context, ts, std::move(buffer), outputFile);
      },
      llvm::outs());
}

static LogicalResult executeHlstool(MLIRContext &context) {
  if (allowUnregisteredDialects)
    context.allowUnregisteredDialects();

  // Create the timing manager we use to sample execution times.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  auto ts = tm.getRootScope();

  // Set up the input file.
  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
  if (outputFormat != OutputSplitVerilog) {
    outputFile.emplace(openOutputFile(outputFilename, &errorMessage));
    if (!*outputFile) {
      llvm::errs() << errorMessage << "\n";
      return failure();
    }
  }

  // Process the input.
  if (failed(processInput(context, ts, std::move(input), outputFile)))
    return failure();

  // If the result succeeded and we're emitting a file, close it.
  if (outputFile.has_value())
    (*outputFile)->keep();

  return success();
}

/// Main driver for hlstool command.  This sets up LLVM and MLIR, and parses
/// command line options before passing off to 'executeHlstool'.  This is set
/// up so we can `exit(0)` at the end of the program to avoid teardown of the
/// MLIRContext and modules inside of it (reducing compile time).
int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  setBugReportMsg(circtBugReportMsg);

  // Hide default LLVM options, other than for this tool.
  // MLIR options are added below.
  cl::HideUnrelatedOptions(mainCategory);

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "CIRCT HLS tool\n");

  DialectRegistry registry;
  // Register MLIR dialects.
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::scf::SCFDialect>();

  // Register MLIR passes.
  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerInlinerPass();
  mlir::registerCanonicalizerPass();

  // Register CIRCT dialects.
  registry.insert<hw::HWDialect, comb::CombDialect, seq::SeqDialect,
                  sv::SVDialect, handshake::HandshakeDialect, esi::ESIDialect,
                  calyx::CalyxDialect>();

  // Do the guts of the hlstool process.
  MLIRContext context(registry);
  auto result = executeHlstool(context);

  // Use "exit" instead of return'ing to signal completion.  This avoids
  // invoking the MLIRContext destructor, which spends a bunch of time
  // deallocating memory etc which process exit will do for us.
  exit(failed(result));
}
