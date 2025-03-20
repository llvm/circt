//===- circt-bmc.cpp - The circt-bmc bounded model checker ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'circt-bmc' tool
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CombToSMT.h"
#include "circt/Conversion/HWToSMT.h"
#include "circt/Conversion/SMTToZ3LLVM.h"
#include "circt/Conversion/VerifToSMT.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Emit/EmitDialect.h"
#include "circt/Dialect/Emit/EmitPasses.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/OM/OMPasses.h"
#include "circt/Dialect/SMT/SMTDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Support/Passes.h"
#include "circt/Support/Version.h"
#include "circt/Tools/circt-bmc/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#ifdef CIRCT_BMC_ENABLE_JIT
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "llvm/Support/TargetSelect.h"
#endif

namespace cl = llvm::cl;

using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Command-line options declaration
//===----------------------------------------------------------------------===//

static cl::OptionCategory mainCategory("circt-bmc Options");

static cl::opt<std::string>
    moduleName("module",
               cl::desc("Specify a named module to verify properties over."),
               cl::value_desc("module name"), cl::cat(mainCategory));

static cl::opt<int> clockBound(
    "b", cl::Required,
    cl::desc("Specify a number of clock cycles to model check up to."),
    cl::value_desc("clock cycle count"), cl::cat(mainCategory));

static cl::opt<std::string> inputFilename(cl::Positional, cl::Required,
                                          cl::desc("<input file>"),
                                          cl::cat(mainCategory));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"),
                                           cl::cat(mainCategory));

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true), cl::cat(mainCategory));

static cl::opt<bool> printSolverOutput(
    "print-solver-output",
    cl::desc("Print the output (counterexample or proof) produced by the "
             "solver on each invocation and the assertion set that they "
             "prove/disprove."),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<bool>
    verbosePassExecutions("verbose-pass-executions",
                          cl::desc("Log executions of toplevel module passes"),
                          cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> risingClocksOnly(
    "rising-clocks-only",
    cl::desc("Only consider the circuit and property on rising clock edges"),
    cl::init(false), cl::cat(mainCategory));

#ifdef CIRCT_BMC_ENABLE_JIT

enum OutputFormat { OutputMLIR, OutputLLVM, OutputSMTLIB, OutputRunJIT };
static cl::opt<OutputFormat> outputFormat(
    cl::desc("Specify output format"),
    cl::values(clEnumValN(OutputMLIR, "emit-mlir", "Emit LLVM MLIR dialect"),
               clEnumValN(OutputLLVM, "emit-llvm", "Emit LLVM"),
               clEnumValN(OutputSMTLIB, "emit-smtlib", "Emit SMT-LIB file"),
               clEnumValN(OutputRunJIT, "run",
                          "Perform BMC and output result")),
    cl::init(OutputRunJIT), cl::cat(mainCategory));

static cl::list<std::string> sharedLibs{
    "shared-libs", llvm::cl::desc("Libraries to link dynamically"),
    cl::MiscFlags::CommaSeparated, llvm::cl::cat(mainCategory)};

#else

enum OutputFormat { OutputMLIR, OutputLLVM, OutputSMTLIB };
static cl::opt<OutputFormat> outputFormat(
    cl::desc("Specify output format"),
    cl::values(clEnumValN(OutputMLIR, "emit-mlir", "Emit LLVM MLIR dialect"),
               clEnumValN(OutputLLVM, "emit-llvm", "Emit LLVM"),
               clEnumValN(OutputSMTLIB, "emit-smtlib", "Emit SMT-LIB file")),
    cl::init(OutputLLVM), cl::cat(mainCategory));

#endif

//===----------------------------------------------------------------------===//
// Tool implementation
//===----------------------------------------------------------------------===//

/// This function initializes the various components of the tool and
/// orchestrates the work to be done.
static LogicalResult executeBMC(MLIRContext &context) {
  // Create the timing manager we use to sample execution times.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  auto ts = tm.getRootScope();

  OwningOpRef<ModuleOp> module;
  {
    auto parserTimer = ts.nest("Parse MLIR input");
    // Parse the provided input files.
    module = parseSourceFile<ModuleOp>(inputFilename, &context);
  }
  if (!module)
    return failure();

  // Create the output directory or output file depending on our mode.
  std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
  std::string errorMessage;
  // Create an output file.
  outputFile.emplace(openOutputFile(outputFilename, &errorMessage));
  if (!outputFile.value()) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  PassManager pm(&context);
  pm.enableVerifier(verifyPasses);
  pm.enableTiming(ts);
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();

  if (verbosePassExecutions)
    pm.addInstrumentation(
        std::make_unique<VerbosePassInstrumentation<mlir::ModuleOp>>(
            "circt-bmc"));

  pm.addPass(om::createStripOMPass());
  pm.addPass(emit::createStripEmitPass());
  pm.addPass(createExternalizeRegisters());
  LowerToBMCOptions lowerToBMCOptions;
  lowerToBMCOptions.bound = clockBound;
  lowerToBMCOptions.topModule = moduleName;
  lowerToBMCOptions.risingClocksOnly = risingClocksOnly;
  pm.addPass(createLowerToBMC(lowerToBMCOptions));
  pm.addPass(createConvertHWToSMT());
  pm.addPass(createConvertCombToSMT());
  ConvertVerifToSMTOptions convertVerifToSMTOptions;
  convertVerifToSMTOptions.risingClocksOnly = risingClocksOnly;
  pm.addPass(createConvertVerifToSMT(convertVerifToSMTOptions));
  pm.addPass(createSimpleCanonicalizerPass());

  if (outputFormat != OutputMLIR && outputFormat != OutputSMTLIB) {
    LowerSMTToZ3LLVMOptions options;
    options.debug = printSolverOutput;
    pm.addPass(createLowerSMTToZ3LLVM(options));
    pm.addPass(createCSEPass());
    pm.addPass(createSimpleCanonicalizerPass());
    pm.addPass(LLVM::createDIScopeForLLVMFuncOpPass());
  }

  if (failed(pm.run(module.get())))
    return failure();

  if (outputFormat == OutputMLIR) {
    auto timer = ts.nest("Print MLIR output");
    OpPrintingFlags printingFlags;
    module->print(outputFile.value()->os(), printingFlags);
    outputFile.value()->keep();
    return success();
  }

  if (outputFormat == OutputSMTLIB) {
    auto timer = ts.nest("Print SMT-LIB output");
    llvm::errs() << "Printing SMT-LIB not yet supported!\n";
    return failure();
  }

  if (outputFormat == OutputLLVM) {
    auto timer = ts.nest("Translate to and print LLVM output");
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module.get(), llvmContext);
    if (!llvmModule)
      return failure();
    llvmModule->print(outputFile.value()->os(), nullptr);
    outputFile.value()->keep();
    return success();
  }

#ifdef CIRCT_BMC_ENABLE_JIT

  auto handleErr = [](llvm::Error error) -> LogicalResult {
    llvm::handleAllErrors(std::move(error),
                          [](const llvm::ErrorInfoBase &info) {
                            llvm::errs() << "Error: ";
                            info.log(llvm::errs());
                            llvm::errs() << '\n';
                          });
    return failure();
  };

  std::unique_ptr<mlir::ExecutionEngine> engine;
  {
    auto timer = ts.nest("Setting up the JIT");
    auto entryPoint =
        dyn_cast_or_null<LLVM::LLVMFuncOp>(module->lookupSymbol(moduleName));
    if (!entryPoint || entryPoint.empty()) {
      llvm::errs() << "no valid entry point found, expected 'llvm.func' named '"
                   << moduleName << "'\n";
      return failure();
    }

    if (entryPoint.getNumArguments() != 0) {
      llvm::errs() << "entry point '" << moduleName
                   << "' must have no arguments";
      return failure();
    }

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    SmallVector<StringRef, 4> sharedLibraries(sharedLibs.begin(),
                                              sharedLibs.end());
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = mlir::makeOptimizingTransformer(
        /*optLevel*/ 3, /*sizeLevel=*/0, /*targetMachine=*/nullptr);
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
    engineOptions.sharedLibPaths = sharedLibraries;
    engineOptions.enableObjectDump = true;

    auto expectedEngine =
        mlir::ExecutionEngine::create(module.get(), engineOptions);
    if (!expectedEngine)
      return handleErr(expectedEngine.takeError());

    engine = std::move(*expectedEngine);
  }

  auto timer = ts.nest("JIT Execution");
  if (auto err = engine->invokePacked(moduleName))
    return handleErr(std::move(err));

  return success();
#else
  return failure();
#endif
}

/// The entry point for the `circt-bmc` tool:
/// configures and parses the command-line options,
/// registers all dialects within a MLIR context,
/// and calls the `executeBMC` function to do the actual work.
int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Hide default LLVM options, other than for this tool.
  // MLIR options are added below.
  cl::HideUnrelatedOptions(mainCategory);

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();
  cl::AddExtraVersionPrinter(
      [](llvm::raw_ostream &os) { os << circt::getCirctVersion() << '\n'; });

  // Parse the command-line options provided by the user.
  cl::ParseCommandLineOptions(
      argc, argv,
      "circt-bmc - bounded model checker\n\n"
      "\tThis tool checks all possible executions of a hardware module up to a "
      "given time bound to check whether any asserted properties can be "
      "violated.\n");

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  llvm::setBugReportMsg(circt::circtBugReportMsg);

  // Register the supported CIRCT dialects and create a context to work with.
  DialectRegistry registry;
  // clang-format off
  registry.insert<
    circt::comb::CombDialect,
    circt::emit::EmitDialect,
    circt::hw::HWDialect,
    circt::om::OMDialect,
    circt::seq::SeqDialect,
    circt::smt::SMTDialect,
    circt::verif::VerifDialect,
    mlir::arith::ArithDialect,
    mlir::BuiltinDialect,
    mlir::func::FuncDialect,
    mlir::LLVM::LLVMDialect
  >();
  // clang-format on
  mlir::func::registerInlinerExtension(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  MLIRContext context(registry);

  // Setup of diagnostic handling.
  llvm::SourceMgr sourceMgr;
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  // Avoid printing a superfluous note on diagnostic emission.
  context.printOpOnDiagnostic(false);

  // Perform the logical equivalence checking; using `exit` to avoid the slow
  // teardown of the MLIR context.
  exit(failed(executeBMC(context)));
}
