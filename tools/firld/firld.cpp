//===- firld.cpp - Link multiple circuits together ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Link FIRRTL circuits (.mlir or .mlirbc files) into one.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRParser.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/InitAllDialects.h"
#include "circt/Support/Version.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Transform/IR/Utils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace llvm;
using namespace mlir;
using namespace circt;

static constexpr const char toolName[] = "firld";
static cl::OptionCategory mainCategory("firld Options");

static cl::list<std::string> inputFilenames(cl::Positional,
                                            cl::desc("<input files>"),
                                            cl::cat(mainCategory));

static cl::opt<std::string> baseCircuitName("base-circuit", cl::Required,
                                            cl::desc("Base circuit name"),
                                            cl::value_desc("circuit name"),
                                            cl::cat(mainCategory));

static cl::opt<std::string> outputFilename("o",
                                           cl::desc("Override output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"),
                                           cl::cat(mainCategory));

static cl::opt<bool>
    emitBytecode("emit-bytecode",
                 cl::desc("Emit bytecode when generating MLIR output"),
                 cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> force("f", cl::desc("Enable binary output on terminals"),
                           cl::init(false), cl::cat(mainCategory));

static cl::opt<bool>
    noMangle("no-mangle", cl::desc("Do not perform private symbol mangling"),
             cl::init(false), cl::cat(mainCategory));

static LogicalResult emitError(const Twine &err) {
  WithColor::error(errs(), toolName) << err << "\n";
  return failure();
}

static bool checkBytecodeOutputToConsole(raw_ostream &os) {
  if (os.is_displayed()) {
    WithColor::warning(errs(), toolName)
        << "you're attempting to print out a bytecode file."
        << " This is inadvisable as it may cause display problems\n";
    WithColor::remark(errs(), toolName)
        << "if you really want to do this, force output with the `-f` option\n";
    return true;
  }
  return false;
}

static LogicalResult printOp(Operation *op, raw_ostream &os) {
  if (emitBytecode && (force || !checkBytecodeOutputToConsole(os)))
    return writeBytecodeToFile(op, os, BytecodeWriterConfig(getCirctVersion()));
  op->print(os);
  return success();
}

static LogicalResult mergeModules(MLIRContext &context,
                                  SmallVector<OwningOpRef<ModuleOp>> &&modules,
                                  ModuleOp mergedModule) {
  // destroy original modules after merge
  auto modulesToMerge = std::move(modules);

  // firrtl.circuit doesn't have sym_name hence we have no need to process
  // SymbolTable
  SmallVector<Operation *> opsToMove;
  std::set<StringRef> circuitNames;
  for (auto &module : modulesToMerge) {
    for (auto &op : module->getBody()->getOperations()) {
      if (auto circuit = dyn_cast<firrtl::CircuitOp>(op)) {
        opsToMove.push_back(&op);
        circuitNames.insert(
            cast<StringAttr>(circuit->getAttr("name")).getValue());
      }
    }
  }

  if (circuitNames.size() != opsToMove.size())
    return emitError(
        "Duplicate circuit names detected, all circuit names must be unique");
  if (circuitNames.find(baseCircuitName) == circuitNames.end())
    return emitError("Base circuit '" + baseCircuitName +
                     "' not found in circuit names");

  for (auto *op : opsToMove)
    op->moveBefore(mergedModule.getBody(), mergedModule.getBody()->end());

  return success();
}

static LogicalResult execute(MLIRContext &context) {
  std::string errorMessage;
  SmallVector<OwningOpRef<ModuleOp>> modules;
  for (const auto &inputFile : inputFilenames) {
    auto input = openInputFile(inputFile, &errorMessage);
    if (!input)
      return emitError(errorMessage);

    SourceMgr srcMgr;
    srcMgr.AddNewSourceBuffer(std::move(input), SMLoc());
    if (auto module = parseSourceFile<ModuleOp>(srcMgr, &context)) {
      modules.push_back(std::move(module));
    } else
      return emitError("Failed to parse input file:" + inputFile);
  }

  auto mergedModule =
      OwningOpRef<ModuleOp>(ModuleOp::create(UnknownLoc::get(&context)));
  if (failed(mergeModules(context, std::move(modules), mergedModule.get())))
    return emitError("Failed to merge MLIR modules");

  PassManager pm(&context);
  pm.addPass(firrtl::createLinkCircuitsPass(baseCircuitName, noMangle));
  if (failed(pm.run(mergedModule.get())))
    return emitError("Failed to link circuits together");

  std::optional<std::unique_ptr<ToolOutputFile>> outputFile;
  outputFile.emplace(openOutputFile(outputFilename, &errorMessage));
  if (!(*outputFile))
    return emitError(errorMessage);

  if (failed(printOp(*mergedModule, (*outputFile)->os())))
    return emitError("Failed to print output to" + outputFilename);
  if (outputFile.has_value())
    (*outputFile)->keep();

  return success();
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);
  setBugReportMsg(circtBugReportMsg);

  DialectRegistry registry;
  circt::registerAllDialects(registry);

  cl::HideUnrelatedOptions({&mainCategory, &getColorCategory()});
  cl::ParseCommandLineOptions(argc, argv, "FIRRTL Circuits Linker\n");

  MLIRContext context(registry);
  exit(failed(execute(context)));
}
