//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file initiliazes the 'circt-lec' tool, which interfaces with a logical
/// engine to allow its user to check whether two input circuit descriptions
/// are equivalent, and when not provides a counterexample as for why.
///
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CombToSMT.h"
#include "circt/Conversion/DatapathToSMT.h"
#include "circt/Conversion/HWToSMT.h"
#include "circt/Conversion/SMTToZ3LLVM.h"
#include "circt/Conversion/SynthToComb.h"
#include "circt/Conversion/VerifToSMT.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Datapath/DatapathDialect.h"
#include "circt/Dialect/Emit/EmitDialect.h"
#include "circt/Dialect/Emit/EmitPasses.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/OM/OMPasses.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Support/Passes.h"
#include "circt/Support/Version.h"
#include "circt/Tools/circt-lec/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/SMT/IR/SMTDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/SMTLIB/ExportSMTLIB.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#ifdef CIRCT_LEC_ENABLE_JIT
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#endif

namespace cl = llvm::cl;

using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Command-line options declaration
//===----------------------------------------------------------------------===//

static cl::OptionCategory mainCategory("circt-lec Options");

static cl::opt<std::string> firstModuleName(
    "c1", cl::Required,
    cl::desc("Specify a named module for the first circuit of the comparison"),
    cl::value_desc("module name"), cl::cat(mainCategory));

static cl::opt<std::string> secondModuleName(
    "c2", cl::Required,
    cl::desc("Specify a named module for the second circuit of the comparison"),
    cl::value_desc("module name"), cl::cat(mainCategory));

static cl::list<std::string> inputFilenames(cl::Positional, cl::OneOrMore,
                                            cl::desc("<input files>"),
                                            cl::cat(mainCategory));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"),
                                           cl::cat(mainCategory));

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true), cl::cat(mainCategory));

static cl::opt<bool>
    verbosePassExecutions("verbose-pass-executions",
                          cl::desc("Log executions of toplevel module passes"),
                          cl::init(false), cl::cat(mainCategory));

#ifdef CIRCT_LEC_ENABLE_JIT

enum OutputFormat { OutputMLIR, OutputLLVM, OutputSMTLIB, OutputRunJIT };
static cl::opt<OutputFormat> outputFormat(
    cl::desc("Specify output format"),
    cl::values(clEnumValN(OutputMLIR, "emit-mlir", "Emit LLVM MLIR dialect"),
               clEnumValN(OutputLLVM, "emit-llvm", "Emit LLVM"),
               clEnumValN(OutputSMTLIB, "emit-smtlib", "Emit object file"),
               clEnumValN(OutputRunJIT, "run",
                          "Perform LEC and output result")),
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
               clEnumValN(OutputSMTLIB, "emit-smtlib", "Emit object file")),
    cl::init(OutputLLVM), cl::cat(mainCategory));

#endif

//===----------------------------------------------------------------------===//
// Tool implementation
//===----------------------------------------------------------------------===//

// Move all operations in `src` to `dest`. Rename all symbols in `src` to avoid
// conflict.
static FailureOr<StringAttr> mergeModules(ModuleOp dest, ModuleOp src,
                                          StringAttr name) {

  SymbolTable destTable(dest), srcTable(src);
  StringAttr newName = {};
  for (auto &op : src.getOps()) {
    if (SymbolOpInterface symbol = dyn_cast<SymbolOpInterface>(op)) {
      auto oldSymbol = symbol.getNameAttr();
      auto result = srcTable.renameToUnique(&op, {&destTable});
      if (failed(result))
        return src->emitError() << "failed to rename symbol " << oldSymbol;

      if (oldSymbol == name) {
        assert(!newName && "symbol must be unique");
        newName = *result;
      }
    }
  }

  if (!newName)
    return src->emitError()
           << "module " << name << " was not found in the second module";

  dest.getBody()->getOperations().splice(dest.getBody()->begin(),
                                         src.getBody()->getOperations());
  return newName;
}

// Parse one or two MLIR modules and merge it into a single module.
static FailureOr<OwningOpRef<ModuleOp>>
parseAndMergeModules(MLIRContext &context, TimingScope &ts) {
  auto parserTimer = ts.nest("Parse and merge MLIR input(s)");

  if (inputFilenames.size() > 2) {
    llvm::errs() << "more than 2 files are provided!\n";
    return failure();
  }

  auto module = parseSourceFile<ModuleOp>(inputFilenames[0], &context);
  if (!module)
    return failure();

  if (inputFilenames.size() == 2) {
    auto moduleOpt = parseSourceFile<ModuleOp>(inputFilenames[1], &context);
    if (!moduleOpt)
      return failure();
    auto result = mergeModules(module.get(), moduleOpt.get(),
                               StringAttr::get(&context, secondModuleName));
    if (failed(result))
      return failure();

    secondModuleName.setValue(result->getValue().str());
  }

  return module;
}

/// This functions initializes the various components of the tool and
/// orchestrates the work to be done.
static LogicalResult executeLEC(MLIRContext &context) {
  // Create the timing manager we use to sample execution times.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  auto ts = tm.getRootScope();

  auto parsedModule = parseAndMergeModules(context, ts);
  if (failed(parsedModule))
    return failure();

  OwningOpRef<ModuleOp> module = std::move(parsedModule.value());

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
            "circt-lec"));

  pm.addPass(om::createStripOMPass());
  pm.addPass(emit::createStripEmitPass());
  {
    ConstructLECOptions opts;
    opts.firstModule = firstModuleName;
    opts.secondModule = secondModuleName;
    if (outputFormat == OutputSMTLIB)
      opts.insertMode = lec::InsertAdditionalModeEnum::None;
    pm.addPass(createConstructLEC(opts));
  }
  pm.addPass(createConvertSynthToComb());
  pm.addPass(createConvertHWToSMT());
  pm.addPass(createConvertDatapathToSMT());
  pm.addPass(createConvertCombToSMT());
  pm.addPass(createConvertVerifToSMT());
  pm.addPass(createSimpleCanonicalizerPass());

  if (outputFormat != OutputMLIR && outputFormat != OutputSMTLIB) {
    pm.addPass(createLowerSMTToZ3LLVM());
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
    if (failed(smt::exportSMTLIB(module.get(), outputFile.value()->os())))
      return failure();
    outputFile.value()->keep();
    return success();
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

#ifdef CIRCT_LEC_ENABLE_JIT

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
  std::function<llvm::Error(llvm::Module *)> transformer =
      mlir::makeOptimizingTransformer(
          /*optLevel*/ 3, /*sizeLevel=*/0, /*targetMachine=*/nullptr);
  {
    auto timer = ts.nest("Setting up the JIT");
    auto entryPoint = dyn_cast_or_null<LLVM::LLVMFuncOp>(
        module->lookupSymbol(firstModuleName));
    if (!entryPoint || entryPoint.empty()) {
      llvm::errs() << "no valid entry point found, expected 'llvm.func' named '"
                   << firstModuleName << "'\n";
      return failure();
    }

    if (entryPoint.getNumArguments() != 0) {
      llvm::errs() << "entry point '" << firstModuleName
                   << "' must have no arguments";
      return failure();
    }

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    SmallVector<StringRef, 4> sharedLibraries(sharedLibs.begin(),
                                              sharedLibs.end());
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = transformer;
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
  if (auto err = engine->invokePacked(firstModuleName))
    return handleErr(std::move(err));

  return success();
#else
  return failure();
#endif
}

/// The entry point for the `circt-lec` tool:
/// configures and parses the command-line options,
/// registers all dialects within a MLIR context,
/// and calls the `executeLEC` function to do the actual work.
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
      "circt-lec - logical equivalence checker\n\n"
      "\tThis tool compares two input circuit descriptions to determine whether"
      " they are logically equivalent.\n");

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  llvm::setBugReportMsg(circt::circtBugReportMsg);

  // Register the supported CIRCT dialects and create a context to work with.
  DialectRegistry registry;
  // clang-format off
  registry.insert<
    circt::comb::CombDialect,
    circt::datapath::DatapathDialect,
    circt::emit::EmitDialect,
    circt::hw::HWDialect,
    circt::om::OMDialect,
    circt::synth::SynthDialect,
    mlir::smt::SMTDialect,
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
  exit(failed(executeLEC(context)));
}
