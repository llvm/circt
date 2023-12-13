//===- ibistool.cpp - The ibistool utility for working with the Ibis dialect =//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements 'ibistool', which composes together a variety of
// CIRCT libraries that can be used to realise an Ibis-based lowering flow.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

#include "circt/Conversion/ExportVerilog.h"
#include "circt/Conversion/Passes.h"
#include "circt/Dialect/DC/DCDialect.h"
#include "circt/Dialect/DC/DCPasses.h"
#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/ESI/ESIPasses.h"
#include "circt/Dialect/Ibis/IbisDialect.h"
#include "circt/Dialect/Ibis/IbisOps.h"
#include "circt/Dialect/Ibis/IbisPassPipelines.h"
#include "circt/Dialect/Ibis/IbisPasses.h"
#include "circt/Dialect/Pipeline/PipelineDialect.h"
#include "circt/Dialect/Pipeline/PipelinePasses.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Support/LoweringOptions.h"
#include "circt/Support/LoweringOptionsParser.h"
#include "circt/Support/Version.h"
#include "circt/Transforms/Passes.h"

#include <iostream>

using namespace llvm;
using namespace mlir;
using namespace circt;
using namespace ibis;

// --------------------------------------------------------------------------
// Tool options
// --------------------------------------------------------------------------

static cl::OptionCategory mainCategory("ibistool Options");

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

enum OutputFormatKind {
  OutputLoweredIbis,
  OutputIR,
  OutputVerilog,
  OutputSplitVerilog
};

static cl::opt<OutputFormatKind> outputFormat(
    cl::desc("Specify output format:"),
    cl::values(
        clEnumValN(OutputLoweredIbis, "post-ibis-ir",
                   "Emit IR after Ibis constructs have been lowered away"),
        clEnumValN(OutputIR, "ir", "Emit pre-emission IR"),
        clEnumValN(OutputVerilog, "verilog", "Emit Verilog"),
        clEnumValN(OutputSplitVerilog, "split-verilog",
                   "Emit Verilog (one file per module; specify "
                   "directory with -o=<dir>)")),
    cl::init(OutputVerilog), cl::cat(mainCategory));

static cl::opt<bool>
    traceIVerilog("sv-trace-iverilog",
                  cl::desc("Add tracing to an iverilog simulated module"),
                  cl::init(false), cl::cat(mainCategory));

enum FlowKind { HiIbis, LoIbis };
static cl::opt<FlowKind>
    flowKind(cl::desc("Specify flow kind:"),
             cl::values(clEnumValN(HiIbis, "hi", "High-level Ibis flow"),
                        clEnumValN(LoIbis, "lo", "Low-level Ibis flow")),
             cl::Required, cl::cat(mainCategory));

static LoweringOptionsOption loweringOptions(mainCategory);

// --------------------------------------------------------------------------
// (Configurable) pass pipelines
// --------------------------------------------------------------------------

/// Create a simple canonicalizer pass.
static std::unique_ptr<Pass> createSimpleCanonicalizerPass() {
  mlir::GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.enableRegionSimplification = false;
  return mlir::createCanonicalizerPass(config);
}

static void loadHighLevelControlflowTransformsPipeline(OpPassManager &pm) {
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(createSimpleCanonicalizerPass());
}

static void loadHandshakeTransformsPipeline(OpPassManager &pm) {
  // Make the CFG a binary tree by inserting merge blocks.
  pm.addPass(circt::createInsertMergeBlocksPass());

  // Perform dataflow conversion
  pm.nest<ibis::ClassOp>().addPass(ibis::createConvertCFToHandshakePass());
  // Canonicalize - necessary after handshake conversion to clean up a lot of
  // stuff e.g. simple branches.
  pm.addPass(createSimpleCanonicalizerPass());
  pm.nest<handshake::FuncOp>().addPass(
      handshake::createHandshakeInsertBuffersPass("all", 2));
}

static void loadDCTransformsPipeline(OpPassManager &pm) {
  pm.nest<ClassOp>().addPass(ibis::createConvertHandshakeToDCPass());
  pm.addPass(createSimpleCanonicalizerPass());
  pm.nest<ClassOp>().nest<DataflowMethodOp>().addPass(
      dc::createDCMaterializeForksSinksPass());
  // pm.nest<ClassOp>().addPass(circt::createDCToHWPass());
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
  pm.addPass(mlir::createCSEPass());

  // Legalize unsupported operations within the modules.
  pm.nest<hw::HWModuleOp>().addPass(sv::createHWLegalizeModulesPass());
  pm.addPass(createSimpleCanonicalizerPass());

  // Tidy up the IR to improve verilog emission quality.
  auto &modulePM = pm.nest<hw::HWModuleOp>();
  modulePM.addPass(sv::createPrettifyVerilogPass());
}

static void loadSchedulingPipeline(OpPassManager &pm) {
  // Inject operator library
  pm.addPass(ibis::createAddOperatorLibraryPass());

  // Map any arith operators to comb
  pm.nest<ibis::ClassOp>()
      .nest<ibis::DataflowMethodOp>()
      .nest<ibis::IsolatedStaticBlockOp>()
      .addPass(circt::createMapArithToCombPass());

  // Prepare for scheduling
  pm.nest<ibis::ClassOp>()
      .nest<ibis::DataflowMethodOp>()
      .nest<ibis::IsolatedStaticBlockOp>()
      .addPass(ibis::createPrepareSchedulingPass());

  // Schedule!
  pm.nest<ibis::ClassOp>()
      .nest<ibis::DataflowMethodOp>()
      .nest<ibis::IsolatedStaticBlockOp>()
      .addPass(pipeline::createScheduleLinearPipelinePass());
}

static void loadPipelineLoweringPipeline(OpPassManager &pm) {
  pm.addPass(pipeline::createExplicitRegsPass());
  pm.addPass(createPipelineToHWPass());
}

// --------------------------------------------------------------------------
// Tool driver code
// --------------------------------------------------------------------------

static void loadLowLevelPassPipeline(
    PassManager &pm, ModuleOp module,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  loadPipelineLoweringPipeline(pm);
  loadESILoweringPipeline(pm);
  loadHWLoweringPipeline(pm);
  if (traceIVerilog)
    pm.addPass(circt::sv::createSVTraceIVerilogPass());

  if (loweringOptions.getNumOccurrences())
    loweringOptions.setAsAttribute(module);
  if (outputFormat == OutputVerilog) {
    pm.addPass(createExportVerilogPass((*outputFile)->os()));
  } else if (outputFormat == OutputSplitVerilog) {
    pm.addPass(createExportSplitVerilogPass(outputFilename));
  }
}

static void loadIbisHiFlow(
    PassManager &pm, ModuleOp module,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  if (verbosePassExecutions)
    llvm::errs() << "[ibistool] Will run high-level Ibis flow\n";

  loadHighLevelControlflowTransformsPipeline(pm);
  loadIbisHighLevelPassPipeline(pm);
  if (outputFormat != OutputLoweredIbis) {
    loadHandshakeTransformsPipeline(pm);
    loadSchedulingPipeline(pm);
    loadDCTransformsPipeline(pm);
    if (outputFormat != OutputLoweredIbis)
      loadLowLevelPassPipeline(pm, module, outputFile);
  }
}

static void loadIbisLoFlow(
    PassManager &pm, ModuleOp module,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  if (verbosePassExecutions)
    llvm::errs() << "[ibistool] Will run low-level Ibis flow\n";
  loadIbisLowLevelPassPipeline(pm);

  if (outputFormat != OutputLoweredIbis)
    loadLowLevelPassPipeline(pm, module, outputFile);
}

/// Process a single buffer of the input.
static LogicalResult processBuffer(
    MLIRContext &context, TimingScope &ts, llvm::SourceMgr &sourceMgr,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  // Parse the input.
  mlir::OwningOpRef<mlir::ModuleOp> module;
  llvm::sys::TimePoint<> parseStartTime;
  if (verbosePassExecutions) {
    llvm::errs() << "[ibistool] Running MLIR parser\n";
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
    llvm::errs() << "[ibistool] -- Done in " << llvm::format("%.3f", elpased)
                 << " sec\n";
  }

  // Apply any pass manager command line options.
  PassManager pm(&context);
  pm.enableVerifier(verifyPasses);
  pm.enableTiming(ts);
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();

  pm.addPass(createSimpleCanonicalizerPass());
  if (flowKind == HiIbis)
    loadIbisHiFlow(pm, module.get(), outputFile);
  else if (flowKind == LoIbis)
    loadIbisLoFlow(pm, module.get(), outputFile);

  // Go execute!
  if (failed(pm.run(module.get())))
    return failure();
  if (outputFormat != OutputVerilog || outputFormat == OutputSplitVerilog)
    module->print((*outputFile)->os());

  return success();

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

static LogicalResult executeIbistool(MLIRContext &context) {
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

/// Main driver for ibistool command.  This sets up LLVM and MLIR, and parses
/// command line options before passing off to 'executeIbistool'.  This is set
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
  cl::ParseCommandLineOptions(argc, argv, "CIRCT Ibis tool\n");

  DialectRegistry registry;
  // Register MLIR dialects.
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
                  sv::SVDialect, handshake::HandshakeDialect, ibis::IbisDialect,
                  dc::DCDialect, esi::ESIDialect, pipeline::PipelineDialect>();

  // Do the guts of the ibistool process.
  MLIRContext context(registry);
  auto result = executeIbistool(context);

  // Use "exit" instead of return'ing to signal completion.  This avoids
  // invoking the MLIRContext destructor, which spends a bunch of time
  // deallocating memory etc which process exit will do for us.
  exit(failed(result));
}
