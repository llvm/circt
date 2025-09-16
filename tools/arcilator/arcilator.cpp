//===- arcilator.cpp - An experimental circuit simulator ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'arcilator' compiler, which converts HW designs into
// a corresponding LLVM-based software model.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ArcToLLVM.h"
#include "circt/Conversion/CombToArith.h"
#include "circt/Conversion/ConvertToArcs.h"
#include "circt/Conversion/Passes.h"
#include "circt/Conversion/SeqToSV.h"
#include "circt/Dialect/Arc/ArcDialect.h"
#include "circt/Dialect/Arc/ArcInterfaces.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Arc/ModelInfo.h"
#include "circt/Dialect/Arc/ModelInfoExport.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Emit/EmitDialect.h"
#include "circt/Dialect/Emit/EmitPasses.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/OM/OMPasses.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Sim/SimPasses.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Support/Passes.h"
#include "circt/Support/Version.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

#include <optional>

using namespace mlir;
using namespace circt;
using namespace arc;

//===----------------------------------------------------------------------===//
// Command Line Arguments
//===----------------------------------------------------------------------===//

static llvm::cl::OptionCategory mainCategory("arcilator Options");

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"),
                                                llvm::cl::cat(mainCategory));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"),
                   llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool>
    observePorts("observe-ports", llvm::cl::desc("Make all ports observable"),
                 llvm::cl::init(false), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool>
    observeWires("observe-wires", llvm::cl::desc("Make all wires observable"),
                 llvm::cl::init(false), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> observeNamedValues(
    "observe-named-values",
    llvm::cl::desc("Make values with `sv.namehint` observable"),
    llvm::cl::init(false), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool>
    observeRegisters("observe-registers",
                     llvm::cl::desc("Make all registers observable"),
                     llvm::cl::init(false), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool>
    observeMemories("observe-memories",
                    llvm::cl::desc("Make all memory contents observable"),
                    llvm::cl::init(false), llvm::cl::cat(mainCategory));

static llvm::cl::opt<std::string> stateFile("state-file",
                                            llvm::cl::desc("State file"),
                                            llvm::cl::value_desc("filename"),
                                            llvm::cl::init(""),
                                            llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> shouldInline("inline", llvm::cl::desc("Inline arcs"),
                                        llvm::cl::init(true),
                                        llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> shouldDedup("dedup",
                                       llvm::cl::desc("Deduplicate arcs"),
                                       llvm::cl::init(true),
                                       llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> shouldDetectEnables(
    "detect-enables",
    llvm::cl::desc("Infer enable conditions for states to avoid computation"),
    llvm::cl::init(true), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> shouldDetectResets(
    "detect-resets",
    llvm::cl::desc("Infer reset conditions for states to avoid computation"),
    llvm::cl::init(false), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool>
    shouldMakeLUTs("lookup-tables",
                   llvm::cl::desc("Optimize arcs into lookup tables"),
                   llvm::cl::init(true), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool>
    printDebugInfo("print-debug-info",
                   llvm::cl::desc("Print debug information"),
                   llvm::cl::init(false), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> verifyPasses(
    "verify-each",
    llvm::cl::desc("Run the verifier after each transformation pass"),
    llvm::cl::init(true), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> verifyDiagnostics(
    "verify-diagnostics",
    llvm::cl::desc("Check that emitted diagnostics match "
                   "expected-* lines on the corresponding line"),
    llvm::cl::init(false), llvm::cl::Hidden, llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> verbosePassExecutions(
    "verbose-pass-executions",
    llvm::cl::desc("Log executions of toplevel module passes"),
    llvm::cl::init(false), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> splitInputFile(
    "split-input-file",
    llvm::cl::desc("Split the input file into pieces and process each "
                   "chunk independently"),
    llvm::cl::init(false), llvm::cl::Hidden, llvm::cl::cat(mainCategory));

static llvm::cl::opt<unsigned> splitFuncsThreshold(
    "split-funcs-threshold",
    llvm::cl::desc(
        "Split large MLIR functions that occur above the given size threshold"),
    llvm::cl::ValueOptional, llvm::cl::cat(mainCategory));

// Options to control early-out from pipeline.
enum Until {
  UntilPreprocessing,
  UntilArcConversion,
  UntilArcOpt,
  UntilStateLowering,
  UntilStateAlloc,
  UntilLLVMLowering,
  UntilEnd
};
static auto runUntilValues = llvm::cl::values(
    clEnumValN(UntilPreprocessing, "preproc", "Input preprocessing"),
    clEnumValN(UntilArcConversion, "arc-conv", "Conversion of modules to arcs"),
    clEnumValN(UntilArcOpt, "arc-opt", "Arc optimizations"),
    clEnumValN(UntilStateLowering, "state-lowering", "Stateful arc lowering"),
    clEnumValN(UntilStateAlloc, "state-alloc", "State allocation"),
    clEnumValN(UntilLLVMLowering, "llvm-lowering", "Lowering to LLVM"),
    clEnumValN(UntilEnd, "all", "Run entire pipeline (default)"));
static llvm::cl::opt<Until> runUntilBefore(
    "until-before", llvm::cl::desc("Stop pipeline before a specified point"),
    runUntilValues, llvm::cl::init(UntilEnd), llvm::cl::cat(mainCategory));
static llvm::cl::opt<Until> runUntilAfter(
    "until-after", llvm::cl::desc("Stop pipeline after a specified point"),
    runUntilValues, llvm::cl::init(UntilEnd), llvm::cl::cat(mainCategory));

// Options to control the output format.
enum OutputFormat { OutputMLIR, OutputLLVM, OutputRunJIT, OutputDisabled };
static llvm::cl::opt<OutputFormat> outputFormat(
    llvm::cl::desc("Specify output format"),
    llvm::cl::values(clEnumValN(OutputMLIR, "emit-mlir", "Emit MLIR dialects"),
                     clEnumValN(OutputLLVM, "emit-llvm", "Emit LLVM"),
                     clEnumValN(OutputRunJIT, "run",
                                "Run the simulation and emit its output"),
                     clEnumValN(OutputDisabled, "disable-output",
                                "Do not output anything")),
    llvm::cl::init(OutputLLVM), llvm::cl::cat(mainCategory));

static llvm::cl::opt<std::string>
    jitEntryPoint("jit-entry",
                  llvm::cl::desc("Name of the function containing the "
                                 "simulation to run when output is set to run"),
                  llvm::cl::init("entry"), llvm::cl::cat(mainCategory));

static llvm::cl::list<std::string> sharedLibs{
    "shared-libs", llvm::cl::desc("Libraries to link dynamically"),
    llvm::cl::MiscFlags::CommaSeparated, llvm::cl::cat(mainCategory)};

//===----------------------------------------------------------------------===//
// Main Tool Logic
//===----------------------------------------------------------------------===//

static bool untilReached(Until until) {
  return until >= runUntilBefore || until > runUntilAfter;
}

/// Populate a pass manager with the arc simulator pipeline for the given
/// command line options. This pipeline lowers modules to the Arc dialect.
static void populateHwModuleToArcPipeline(PassManager &pm) {
  if (verbosePassExecutions)
    pm.addInstrumentation(
        std::make_unique<VerbosePassInstrumentation<mlir::ModuleOp>>(
            "arcilator"));

  // Pre-process the input such that it no longer contains any SV dialect ops
  // and external modules that are relevant to the arc transformation are
  // represented as intrinsic ops.
  if (untilReached(UntilPreprocessing))
    return;
  pm.addPass(om::createStripOMPass());
  pm.addPass(emit::createStripEmitPass());
  pm.addPass(createLowerFirMemPass());
  pm.addPass(createLowerVerifSimulationsPass());
  {
    arc::AddTapsOptions opts;
    opts.tapPorts = observePorts;
    opts.tapWires = observeWires;
    opts.tapNamedValues = observeNamedValues;
    pm.addPass(arc::createAddTapsPass(opts));
  }
  pm.addPass(arc::createStripSVPass());
  {
    arc::InferMemoriesOptions opts;
    opts.tapPorts = observePorts;
    opts.tapMemories = observeMemories;
    pm.addPass(arc::createInferMemoriesPass(opts));
  }
  pm.addPass(sim::createLowerDPIFunc());
  pm.addPass(createCSEPass());
  pm.addPass(arc::createArcCanonicalizerPass());

  // Restructure the input from a `hw.module` hierarchy to a collection of arcs.
  if (untilReached(UntilArcConversion))
    return;
  {
    ConvertToArcsPassOptions opts;
    opts.tapRegisters = observeRegisters;
    pm.addPass(createConvertToArcsPass(opts));
  }
  if (shouldDedup)
    pm.addPass(arc::createDedupPass());
  pm.addPass(hw::createFlattenModules());
  pm.addPass(createCSEPass());
  pm.addPass(arc::createArcCanonicalizerPass());

  // Perform arc-level optimizations that are not specific to software
  // simulation.
  if (untilReached(UntilArcOpt))
    return;
  pm.addPass(arc::createSplitLoopsPass());
  if (shouldDedup)
    pm.addPass(arc::createDedupPass());
  {
    arc::InferStatePropertiesOptions opts;
    opts.detectEnables = shouldDetectEnables;
    opts.detectResets = shouldDetectResets;
    pm.addPass(arc::createInferStateProperties(opts));
  }
  pm.addPass(createCSEPass());
  pm.addPass(arc::createArcCanonicalizerPass());
  if (shouldMakeLUTs)
    pm.addPass(arc::createMakeTablesPass());
  pm.addPass(createCSEPass());
  pm.addPass(arc::createArcCanonicalizerPass());

  // Now some arguments may be unused because reset conditions are not passed as
  // inputs anymore pm.addPass(arc::createRemoveUnusedArcArgumentsPass());
  // Because we replace a lot of StateOp inputs with constants in the enable
  // patterns we may be able to sink a lot of them
  // TODO: maybe merge RemoveUnusedArcArguments with SinkInputs?
  // pm.addPass(arc::createSinkInputsPass());
  // pm.addPass(createCSEPass());
  // pm.addPass(createSimpleCanonicalizerPass());
  // Removing some muxes etc. may lead to additional dedup opportunities
  // if (shouldDedup)
  // pm.addPass(arc::createDedupPass());

  // Lower stateful arcs into explicit state reads and writes.
  if (untilReached(UntilStateLowering))
    return;
  pm.addPass(arc::createLowerStatePass());

  // TODO: LowerClocksToFuncsPass might not properly consider scf.if operations
  // (or nested regions in general) and thus errors out when muxes are also
  // converted in the hw.module or arc.model
  // TODO: InlineArcs seems to not properly handle scf.if operations, thus the
  // following is commented out
  // pm.addPass(arc::createMuxToControlFlowPass());
  if (shouldInline)
    pm.addPass(arc::createInlineArcsPass());

  pm.addPass(arc::createMergeIfsPass());
  pm.addPass(createCSEPass());
  pm.addPass(arc::createArcCanonicalizerPass());

  // Allocate states.
  if (untilReached(UntilStateAlloc))
    return;
  pm.addPass(arc::createLowerArcsToFuncsPass());
  pm.nest<arc::ModelOp>().addPass(arc::createAllocateStatePass());
  pm.addPass(arc::createLowerClocksToFuncsPass()); // no CSE between state alloc
                                                   // and clock func lowering
  if (splitFuncsThreshold.getNumOccurrences()) {
    pm.addPass(arc::createSplitFuncs({splitFuncsThreshold}));
  }
  pm.addPass(createCSEPass());
  pm.addPass(arc::createArcCanonicalizerPass());
}

/// Populate a pass manager with the Arc to LLVM pipeline for the given
/// command line options. This pipeline lowers modules to LLVM IR.
static void populateArcToLLVMPipeline(PassManager &pm) {
  // Lower the arcs and update functions to LLVM.
  if (untilReached(UntilLLVMLowering))
    return;
  pm.addPass(createLowerArcToLLVMPass());
  pm.addPass(createCSEPass());
  pm.addPass(arc::createArcCanonicalizerPass());
}

static LogicalResult processBuffer(
    MLIRContext &context, TimingScope &ts, llvm::SourceMgr &sourceMgr,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  {
    auto parserTimer = ts.nest("Parse MLIR input");
    module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  }
  if (!module)
    return failure();

  // Lower HwModule to Arc model.
  PassManager pmArc(&context);
  pmArc.enableVerifier(verifyPasses);
  pmArc.enableTiming(ts);
  if (failed(applyPassManagerCLOptions(pmArc)))
    return failure();
  populateHwModuleToArcPipeline(pmArc);

  if (failed(pmArc.run(module.get())))
    return failure();

  // Output state info as JSON if requested.
  if (!stateFile.empty() && !untilReached(UntilStateLowering)) {
    std::error_code ec;
    llvm::ToolOutputFile outputFile(stateFile, ec,
                                    llvm::sys::fs::OpenFlags::OF_None);
    if (ec) {
      llvm::errs() << "unable to open state file: " << ec.message() << '\n';
      return failure();
    }
    if (failed(collectAndExportModelInfo(module.get(), outputFile.os()))) {
      llvm::errs() << "failed to collect model info\n";
      return failure();
    }

    outputFile.keep();
  }

  // Lower Arc model to LLVM IR.
  PassManager pmLlvm(&context);
  pmLlvm.enableVerifier(verifyPasses);
  pmLlvm.enableTiming(ts);
  if (failed(applyPassManagerCLOptions(pmLlvm)))
    return failure();
  if (verbosePassExecutions)
    pmLlvm.addInstrumentation(
        std::make_unique<VerbosePassInstrumentation<mlir::ModuleOp>>(
            "arcilator"));
  populateArcToLLVMPipeline(pmLlvm);

  if (printDebugInfo && outputFormat == OutputLLVM)
    pmLlvm.addPass(LLVM::createDIScopeForLLVMFuncOpPass());

  if (failed(pmLlvm.run(module.get())))
    return failure();

#ifdef ARCILATOR_ENABLE_JIT
  // Handle JIT execution.
  if (outputFormat == OutputRunJIT) {
    if (runUntilBefore != UntilEnd || runUntilAfter != UntilEnd) {
      llvm::errs() << "full pipeline must be run for JIT execution\n";
      return failure();
    }

    Operation *toCall = module->lookupSymbol(jitEntryPoint);
    if (!toCall) {
      llvm::errs() << "entry point not found: '" << jitEntryPoint << "'\n";
      return failure();
    }

    auto toCallFunc = llvm::dyn_cast<LLVM::LLVMFuncOp>(toCall);
    if (!toCallFunc) {
      llvm::errs() << "entry point '" << jitEntryPoint
                   << "' was found but on an operation of type '"
                   << toCall->getName()
                   << "' while an LLVM function was expected\n";
      return failure();
    }

    if (toCallFunc.getNumArguments() != 0) {
      llvm::errs() << "entry point '" << jitEntryPoint
                   << "' must have no arguments\n";
      return failure();
    }

    SmallVector<StringRef, 4> sharedLibraries(sharedLibs.begin(),
                                              sharedLibs.end());

    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
    std::function<llvm::Error(llvm::Module *)> transformer =
        mlir::makeOptimizingTransformer(
            /*optLevel=*/3, /*sizeLevel=*/0,
            /*targetMachine=*/nullptr);
    engineOptions.transformer = transformer;
    engineOptions.sharedLibPaths = sharedLibraries;

    auto executionEngine =
        mlir::ExecutionEngine::create(module.get(), engineOptions);
    if (!executionEngine) {
      llvm::handleAllErrors(
          executionEngine.takeError(), [](const llvm::ErrorInfoBase &info) {
            llvm::errs() << "failed to create execution engine: "
                         << info.message() << "\n";
          });
      return failure();
    }

    auto expectedFunc = (*executionEngine)->lookupPacked(jitEntryPoint);
    if (!expectedFunc) {
      llvm::handleAllErrors(
          expectedFunc.takeError(), [](const llvm::ErrorInfoBase &info) {
            llvm::errs() << "failed to run simulation: " << info.message()
                         << "\n";
          });
      return failure();
    }

    void (*simulationFunc)(void **) = *expectedFunc;
    (*simulationFunc)(nullptr);

    return success();
  }
#endif // ARCILATOR_ENABLE_JIT

  // Handle MLIR output.
  if (runUntilBefore != UntilEnd || runUntilAfter != UntilEnd ||
      outputFormat == OutputMLIR) {
    OpPrintingFlags printingFlags;
    // Only set the debug info flag to true in order to not overwrite MLIR
    // printer CLI flags when the custom debug info option is not set.
    if (printDebugInfo)
      printingFlags.enableDebugInfo(printDebugInfo);
    auto outputTimer = ts.nest("Print MLIR output");
    module->print(outputFile.value()->os(), printingFlags);
    return success();
  }

  // Handle LLVM output.
  if (outputFormat == OutputLLVM) {
    auto outputTimer = ts.nest("Print LLVM output");
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module.get(), llvmContext);
    if (!llvmModule)
      return failure();
    llvmModule->print(outputFile.value()->os(), nullptr);
    return success();
  }

  return success();
}

/// Process a single split of the input. This allocates a source manager and
/// creates a regular or verifying diagnostic handler, depending on whether the
/// user set the verifyDiagnostics option.
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
      [&](std::unique_ptr<llvm::MemoryBuffer> buffer, raw_ostream &) {
        return processInputSplit(context, ts, std::move(buffer), outputFile);
      },
      llvm::outs());
}

static LogicalResult executeArcilator(MLIRContext &context) {
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

  // Create the output directory or output file depending on our mode.
  std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
  // Create an output file.
  outputFile.emplace(openOutputFile(outputFilename, &errorMessage));
  if (!outputFile.value()) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  // Register our dialects.
  DialectRegistry registry;
  // clang-format off
  registry.insert<
    arc::ArcDialect,
    comb::CombDialect,
    emit::EmitDialect,
    hw::HWDialect,
    llhd::LLHDDialect,
    mlir::arith::ArithDialect,
    mlir::cf::ControlFlowDialect,
    mlir::DLTIDialect,
    mlir::func::FuncDialect,
    mlir::index::IndexDialect,
    mlir::LLVM::LLVMDialect,
    mlir::scf::SCFDialect,
    om::OMDialect,
    seq::SeqDialect,
    sim::SimDialect,
    sv::SVDialect,
    verif::VerifDialect
  >();
  // clang-format on

  arc::initAllExternalInterfaces(registry);

  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);

  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  context.appendDialectRegistry(registry);

  // Process the input.
  if (failed(processInput(context, ts, std::move(input), outputFile)))
    return failure();

  // If the result succeeded and we're emitting a file, close it.
  if (outputFile.has_value())
    outputFile.value()->keep();

  return success();
}

/// Main driver for the command. This sets up LLVM and MLIR, and parses command
/// line options before passing off to 'executeArcilator'. This is set up so we
/// can `exit(0)` at the end of the program to avoid teardown of the MLIRContext
/// and modules inside of it (reducing compile time).
int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Hide default LLVM options, other than for this tool.
  // MLIR options are added below.
  llvm::cl::HideUnrelatedOptions(mainCategory);

  // Register passes before parsing command-line options, so that they are
  // available for use with options like `--mlir-print-ir-before`.
  {
    // MLIR transforms:
    // Don't use registerTransformsPasses, pulls in too much.
    registerCSEPass();
    registerCanonicalizerPass();
    registerStripDebugInfoPass();

    // Dialect passes:
    arc::registerPasses();
    registerConvertToArcsPass();
    registerLowerArcToLLVMPass();
  }

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();
  llvm::cl::AddExtraVersionPrinter(
      [](raw_ostream &os) { os << getCirctVersion() << '\n'; });

  // Parse pass names in main to ensure static initialization completed.
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MLIR-based circuit simulator\n");

  if (outputFormat == OutputRunJIT) {
#ifdef ARCILATOR_ENABLE_JIT
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
#else
    llvm::errs() << "This arcilator binary was not built with JIT support.\n";
    llvm::errs() << "To enable JIT features, build arcilator with MLIR's "
                    "execution engine.\n";
    llvm::errs() << "This can be achieved by building arcilator with the "
                    "host's LLVM target enabled.\n";
    exit(1);
#endif // ARCILATOR_ENABLE_JIT
  }

  MLIRContext context;
  auto result = executeArcilator(context);

  // Use "exit" instead of returning to signal completion. This avoids
  // invoking the MLIRContext destructor, which spends a bunch of time
  // deallocating memory etc which process exit will do for us.
  exit(failed(result));
}
