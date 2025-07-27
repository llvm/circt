//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that runs external logic solvers
// (ABC/Yosys/mockturtle) on AIGER files by exporting the current module to
// AIGER format, running the solver, and importing the results back.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ExportAIGER.h"
#include "circt/Conversion/ImportAIGER.h"
#include "circt/Dialect/AIG/AIGPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/Timing.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace circt::aig;

#define DEBUG_TYPE "aig-runner"

namespace circt {
namespace aig {
#define GEN_PASS_DEF_AIGERRUNNER
#define GEN_PASS_DEF_ABCRUNNER
#include "circt/Dialect/AIG/AIGPasses.h.inc"
} // namespace aig
} // namespace circt

//===----------------------------------------------------------------------===//
// Converter
//===----------------------------------------------------------------------===//

namespace {
class Converter : public circt::aiger::ExportAIGERHandler {
public:
  void cleanup(hw::HWModuleOp module);
  void integrateOptimizedModule(hw::HWModuleOp originalModule,
                                hw::HWModuleOp optimizedModule);

private:
  Value clock;
  // Map from operand to AIGER outputs.
  llvm::MapVector<std::pair<Operation *, size_t>, SmallVector<int>> operandMap;

  // Map from result to AIGER inputs
  llvm::MapVector<Value, SmallVector<int>> valueMap;

  llvm::SetVector<Operation *> willBeErased;

  bool operandCallback(OpOperand &op, size_t bitPos,
                       size_t outputIndex) override;
  bool valueCallback(Value value, size_t bitPos, size_t inputIndex) override;
  void notifyEmitted(Operation *op) override;
  void notifyClock(Value value) override;
};

} // namespace

/// Callback invoked during AIGER export for each operand bit.
/// Maps each bit position of an operand to its corresponding AIGER output
/// index.
bool Converter::operandCallback(OpOperand &op, size_t bitPos,
                                size_t outputIndex) {
  // Create a unique key for this operand (owner operation + operand number)
  auto operandKey = std::make_pair(op.getOwner(), op.getOperandNumber());
  assert(op.get().getType().isInteger() && "operand is not an integer");

  // Find or create entry in the operand map
  auto *mapIterator = operandMap.find(operandKey);
  if (mapIterator == operandMap.end()) {
    // Initialize with -1 for all bit positions (indicating unmapped)
    auto bitWidth = hw::getBitWidth(op.get().getType());
    mapIterator =
        operandMap.insert({operandKey, SmallVector<int>(bitWidth, -1)}).first;
  }

  // Map this specific bit position to the AIGER output index
  mapIterator->second[bitPos] = outputIndex;
  return true;
}

/// Callback invoked during AIGER export for each value bit.
/// Maps each bit position of a value to its corresponding AIGER input index.
bool Converter::valueCallback(Value value, size_t bitPos, size_t inputIndex) {
  assert(value.getType().isInteger() && "value is not an integer");

  // Find or create entry in the value map
  auto *mapIterator = valueMap.find(value);
  if (mapIterator == valueMap.end()) {
    // Initialize with -1 for all bit positions (indicating unmapped)
    auto bitWidth = hw::getBitWidth(value.getType());
    mapIterator =
        valueMap.insert({value, SmallVector<int>(bitWidth, -1)}).first;
  }

  LLVM_DEBUG(llvm::dbgs() << "Mapping value: " << value << " bitPos: " << bitPos
                          << " inputIndex: " << inputIndex << "\n");

  // Map this specific bit position to the AIGER input index
  mapIterator->second[bitPos] = inputIndex;
  return true;
}

/// Clean up operations marked for erasure during the conversion process.
/// This method replaces marked operations with unrealized conversion casts
/// and then runs dead code elimination to remove unused operations.
void Converter::cleanup(hw::HWModuleOp module) {
  SetVector<Operation *> operationsToErase;

  // Replace all operations marked for erasure with unrealized conversion casts
  // This allows DCE to determine if they're actually unused
  mlir::IRRewriter rewriter(module);
  while (!willBeErased.empty()) {
    auto *operationToReplace = willBeErased.pop_back_val();
    rewriter.setInsertionPoint(operationToReplace);

    // Create an unrealized conversion cast as a placeholder
    auto conversionCast =
        rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
            operationToReplace, operationToReplace->getResultTypes(),
            ValueRange{});
    (void)conversionCast;

#ifdef DEBUG
    // Mark the cast for verification that it gets eliminated
    conversionCast->setAttr("aig.runner.must_be_dead", rewriter.getUnitAttr());
#endif
  }

  // Run dead code elimination to remove unused operations
  (void)mlir::runRegionDCE(rewriter, module->getRegions());

#ifdef DEBUG
  // Verify that all marked operations were actually eliminated
  module.walk([&](mlir::UnrealizedConversionCastOp castOp) {
    assert(!castOp->hasAttr("aig.runner.must_be_dead") &&
           "Operation marked for deletion was not eliminated by DCE");
  });
#endif
}

/// Integrate the optimized module from AIGER import into the original module.
/// This method handles the complex process of reconnecting multi-bit values
/// that were decomposed during AIGER export/import.
void Converter::integrateOptimizedModule(hw::HWModuleOp originalModule,
                                         hw::HWModuleOp optimizedModule) {
  mlir::IRRewriter builder(originalModule->getContext());
  builder.setInsertionPointToStart(originalModule.getBodyBlock());

  auto *optimizedTerminator = optimizedModule.getBodyBlock()->getTerminator();
  auto *originalTerminator = originalModule.getBodyBlock()->getTerminator();

  // Reconnect multi-bit operands by concatenating their individual bits
  for (const auto &[operandKey, bitOutputIndices] : operandMap) {
    auto [operationOwner, operandIndex] = operandKey;
    SmallVector<Value> bitsToConcat;
    builder.setInsertionPoint(operationOwner);
    bitsToConcat.reserve(bitOutputIndices.size());

    // Collect the optimized values for each bit (in reverse order for concat)
    for (auto outputIndex : llvm::reverse(bitOutputIndices)) {
      assert(outputIndex != -1 && "Unmapped output index found");
      bitsToConcat.push_back(optimizedTerminator->getOperand(outputIndex));
    }

    // Create single value or concatenate multiple bits
    if (bitsToConcat.size() == 1) {
      operationOwner->setOperand(operandIndex, bitsToConcat.front());
    } else {
      auto concatenatedValue = builder.createOrFold<comb::ConcatOp>(
          operationOwner->getLoc(), bitsToConcat);
      operationOwner->setOperand(operandIndex, concatenatedValue);
    }
  }

  // Prepare arguments for the optimized module by extracting bits from values
  SmallVector<Value> moduleArguments(
      optimizedModule.getBodyBlock()->getNumArguments());
  for (const auto &[originalValue, bitInputIndices] : valueMap) {
    builder.setInsertionPointAfterValue(originalValue);

    // Extract each bit from the multi-bit value
    for (auto [bitPosition, argumentIndex] : llvm::enumerate(bitInputIndices)) {
      // TODO: Consider caching extract operations for efficiency
      auto extractedBit = builder.createOrFold<comb::ExtractOp>(
          originalValue.getLoc(), originalValue, bitPosition, 1);
      moduleArguments[argumentIndex] = extractedBit;
    }
  }

  // Handle clock signal if present (always the last argument)
  auto arguments = optimizedModule.getBodyBlock()->getArguments();
  if (arguments.size() > 0 && isa<seq::ClockType>(arguments.back().getType())) {
    assert(clock && "Clock signal not found");
    moduleArguments.back() = clock;
  }

  // Verify all arguments are properly mapped
  assert(llvm::all_of(moduleArguments, [](Value v) { return v; }) &&
         "Some module arguments were not properly mapped");

  // Inline the optimized module into the original module
  builder.inlineBlockBefore(optimizedModule.getBodyBlock(),
                            originalModule.getBodyBlock(),
                            originalTerminator->getIterator(), moduleArguments);
  optimizedTerminator->erase();
}

/// Callback invoked when an operation is emitted during AIGER export.
/// Marks the operation for potential cleanup since it may become unused
/// after the optimization process replaces it with optimized equivalents.
void Converter::notifyEmitted(Operation *op) { willBeErased.insert(op); }

/// Callback to notify about the clock signal during AIGER export.
/// Stores the clock value for later use when reconnecting the optimized module.
void Converter::notifyClock(Value value) { clock = value; }

//===----------------------------------------------------------------------===//
// AIGERRunner
//===----------------------------------------------------------------------===//

namespace {
class AIGERRunner {
public:
  AIGERRunner(llvm::StringRef solverPath, SmallVector<std::string> solverArgs,
              bool continueOnFailure)
      : solverPath(solverPath), solverArgs(std::move(solverArgs)),
        continueOnFailure(continueOnFailure) {}

  LogicalResult run(hw::HWModuleOp module);

private:
  // Helper methods
  LogicalResult runSolver(hw::HWModuleOp module, StringRef inputPath,
                          StringRef outputPath);
  LogicalResult exportToAIGER(Converter &converter, hw::HWModuleOp module,
                              StringRef outputPath);
  LogicalResult importFromAIGER(Converter &converter, StringRef inputPath,
                                hw::HWModuleOp module);
  llvm::StringRef solverPath;
  SmallVector<std::string> solverArgs;
  bool continueOnFailure;
};

} // namespace

LogicalResult AIGERRunner::run(hw::HWModuleOp module) {

  // Create temporary files for AIGER input/output
  SmallString<128> tempDir;
  if (auto error =
          llvm::sys::fs::createUniqueDirectory("aiger-runner", tempDir))
    return emitError(module.getLoc(), "failed to create temporary directory: ")
           << error.message();

  SmallString<128> inputPath(tempDir);
  llvm::sys::path::append(inputPath, "input.aig");
  SmallString<128> outputPath(tempDir);
  llvm::sys::path::append(outputPath, "output.aig");

  Converter converter;

  auto reportWarningOrError = [&](const Twine &message) -> LogicalResult {
    (continueOnFailure ? mlir::emitWarning(module.getLoc())
                       : mlir::emitError(module.getLoc()))
        << message << " on module " << module.getModuleNameAttr();
    return success(continueOnFailure);
  };

  // Export current module to AIGER format
  if (failed(exportToAIGER(converter, cast<hw::HWModuleOp>(module), inputPath)))
    return reportWarningOrError("failed to export module to AIGER format");

  // Run the external solver
  if (failed(runSolver(module, inputPath, outputPath)))
    return reportWarningOrError("failed to run external solver");

  // Import the results back
  if (failed(
          importFromAIGER(converter, outputPath, cast<hw::HWModuleOp>(module))))
    return reportWarningOrError("failed to import results from AIGER format");

  // If we get here, we succeeded. Clean up temporary files.
  if (llvm::sys::fs::remove(inputPath))
    return emitError(module.getLoc(), "failed to remove input file: ")
           << inputPath.str();
  if (llvm::sys::fs::remove(outputPath))
    return emitError(module.getLoc(), "failed to remove output file: ")
           << outputPath.str();
  if (llvm::sys::fs::remove(tempDir))
    return emitError(module.getLoc(), "failed to remove temporary directory: ")
           << tempDir.str();

  return llvm::success();
}

/// Execute the external solver (ABC, Yosys, etc.) on the AIGER file.
/// Replaces placeholder tokens in solver arguments with actual file paths.
LogicalResult AIGERRunner::runSolver(hw::HWModuleOp module, StringRef inputPath,
                                     StringRef outputPath) {
  // Prepare command line arguments for the solver
  SmallVector<StringRef> commandArgs;
  std::vector<std::string> processedSolverArgs;

  // Helper function to replace all occurrences of a substring
  auto replaceAll = [](std::string str, StringRef from, StringRef to) {
    size_t pos = 0;
    while ((pos = str.find(from.str(), pos)) != std::string::npos) {
      str.replace(pos, from.size(), to.str());
      pos += to.size();
    }
    return str;
  };

  // Process all solver arguments, replacing placeholders with actual paths
  for (const auto &solverArg : solverArgs) {
    std::string processedArg =
        replaceAll(replaceAll(solverArg, "<inputFile>", inputPath),
                   "<outputFile>", outputPath);
    processedSolverArgs.push_back(std::move(processedArg));
  }

  // Find the solver program in the system PATH
  std::string executionError;
  auto solverProgram = llvm::sys::findProgramByName(solverPath);
  if (auto e = solverProgram.getError())
    return emitError(module.getLoc(), "failed to find solver program: ")
           << solverPath.str() << ": " << e.message();

  // Build complete command line with program name and arguments
  commandArgs.push_back(*solverProgram);
  for (auto &processedArg : processedSolverArgs)
    commandArgs.push_back(processedArg);

  // Execute the solver
  int executionResult = llvm::sys::ExecuteAndWait(
      solverProgram.get(), commandArgs,
      /*Env=*/std::nullopt, /*Redirects=*/{},
      /*SecondsToWait=*/0, /*MemoryLimit=*/0, &executionError);

  // Check for execution failure
  if (executionResult != 0)
    return emitError(module.getLoc(), "solver execution failed for module ")
           << module.getModuleNameAttr() << " with error: " << executionError;

  return success();
}

/// Export the hardware module to AIGER format for external solver processing.
LogicalResult AIGERRunner::exportToAIGER(Converter &converter,
                                         hw::HWModuleOp module,
                                         StringRef outputPath) {
  // Open output file for writing AIGER data
  auto outputFile = mlir::openOutputFile(outputPath);
  if (!outputFile)
    return emitError(module.getLoc(), "failed to open AIGER output file: ")
           << outputPath.str();

  LLVM_DEBUG(llvm::dbgs() << "Exporting module " << module.getModuleNameAttr()
                          << " to AIGER format\n");

  circt::aiger::ExportAIGEROptions exportOptions;
  exportOptions.binaryFormat = true;       // Use binary format for efficiency
  exportOptions.includeSymbolTable = true; // Include names for debugging

  // Perform the actual AIGER export
  auto exportResult = circt::aiger::exportAIGER(module, outputFile->os(),
                                                &exportOptions, &converter);

  // Ensure the file is properly written to disk
  outputFile->keep();
  return exportResult;
}

/// Import the optimized AIGER file back into MLIR format.
/// Creates a new module from the AIGER data and replaces the original module.
LogicalResult AIGERRunner::importFromAIGER(Converter &converter,
                                           StringRef inputPath,
                                           hw::HWModuleOp originalModule) {
  // Set up source manager for file parsing
  llvm::SourceMgr sourceMgr;
  auto inputFile = mlir::openInputFile(inputPath);
  if (!inputFile)
    return emitError(originalModule.getLoc(),
                     "failed to open AIGER input file: ")
           << inputPath.str();

  // Add the file buffer to the source manager
  sourceMgr.AddNewSourceBuffer(std::move(inputFile), llvm::SMLoc());

  // Create a temporary module to hold the imported AIGER content
  mlir::TimingScope timingScope;
  mlir::Block temporaryBlock;
  mlir::OpBuilder builder(originalModule->getContext());
  builder.setInsertionPointToStart(&temporaryBlock);
  auto temporaryModule =
      mlir::ModuleOp::create(builder, builder.getUnknownLoc());

  // Import the AIGER file into the temporary module
  if (failed(circt::aiger::importAIGER(sourceMgr, originalModule->getContext(),
                                       timingScope, temporaryModule)))
    return emitError(originalModule.getLoc(),
                     "failed to import optimized AIGER file");

  // Extract the hardware module from the imported content
  auto optimizedModule =
      cast<hw::HWModuleOp>(temporaryModule.getBody()->front());

  // Integrate the optimized module into the original module structure
  converter.integrateOptimizedModule(originalModule, optimizedModule);

  // Clean up any operations that became unused during the replacement
  converter.cleanup(originalModule);

  return llvm::success();
}

//===----------------------------------------------------------------------===//
// AIGERRunnerPass
//===----------------------------------------------------------------------===//

namespace {
class AIGERRunnerPass : public impl::AIGERRunnerBase<AIGERRunnerPass> {
public:
  using AIGERRunnerBase<AIGERRunnerPass>::AIGERRunnerBase;
  void runOnOperation() override;
};
} // namespace

void AIGERRunnerPass::runOnOperation() {
  auto module = getOperation();

  // Convert pass options to the format expected by AIGERRunner
  SmallVector<std::string> solverArgsRef;
  for (const auto &arg : solverArgs)
    solverArgsRef.push_back(arg);

  AIGERRunner runner(solverPath, std::move(solverArgsRef), continueOnFailure);
  if (failed(runner.run(module)))
    signalPassFailure();
}

//===----------------------------------------------------------------------===//
// ABCRunnerPass
//===----------------------------------------------------------------------===//

namespace {
class ABCRunnerPass : public impl::ABCRunnerBase<ABCRunnerPass> {
public:
  using ABCRunnerBase<ABCRunnerPass>::ABCRunnerBase;
  void runOnOperation() override;
};
} // namespace

/// Run ABC optimization commands on the current hardware module.
/// Builds a command sequence to read AIGER, run optimizations, and write back.
void ABCRunnerPass::runOnOperation() {
  auto module = getOperation();

  SmallVector<std::string> abcArguments;

  // Helper to add ABC commands with quiet flag
  auto addABCCommand = [&](const std::string &command) {
    abcArguments.push_back("-q"); // Run ABC in quiet mode
    abcArguments.push_back(command);
  };

  // Start with reading the input AIGER file
  addABCCommand("read <inputFile>");

  // Add all user-specified optimization commands
  for (const auto &optimizationCmd : abcCommands)
    addABCCommand(optimizationCmd);

  // Finish by writing the optimized result
  addABCCommand("write <outputFile>");

  // Execute the ABC optimization sequence
  AIGERRunner abcRunner(abcPath, std::move(abcArguments), continueOnFailure);
  if (failed(abcRunner.run(module)))
    signalPassFailure();
}
