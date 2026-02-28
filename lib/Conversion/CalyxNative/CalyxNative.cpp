//===- CalyxNative.cpp - Invoke the native Calyx compiler
//----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// Calls out to the native, Rust-based Calyx compiler using the `calyx` binary
// to run passes.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"

#include "circt/Conversion/CalyxNative.h"
#include "circt/Dialect/Calyx/CalyxEmitter.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/ToolOutputFile.h"

namespace circt {
#define GEN_PASS_DEF_CALYXNATIVE
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;

/// ConversionPatterns.

/// Pass entrypoint.

namespace {
class CalyxNativePass : public circt::impl::CalyxNativeBase<CalyxNativePass> {
public:
  void runOnOperation() override;

private:
  LogicalResult runOnModule(ModuleOp root);
};
} // end anonymous namespace

void CalyxNativePass::runOnOperation() {
  ModuleOp mod = getOperation();
  if (failed(runOnModule(mod)))
    return signalPassFailure();
}

LogicalResult CalyxNativePass::runOnModule(ModuleOp root) {
  SmallString<32> execName = llvm::sys::path::filename("calyx");
  llvm::ErrorOr<std::string> exeMb = llvm::sys::findProgramByName(execName);

  // If cannot find the executable, then nothing to do, return.
  if (!exeMb) {
    root.emitError() << "cannot find the `calyx` executable in PATH. "
                     << "Consider installing `calyx` using `cargo install "
                        "calyx` or reading the instructions here: "
                        "https://docs.calyxir.org/#compiler-installation";
    return failure();
  }
  StringRef calyxExe = exeMb.get();

  std::string errMsg;
  SmallString<32> nativeInputFileName;
  std::error_code errCode = llvm::sys::fs::getPotentiallyUniqueTempFileName(
      "calyxNativeTemp", /*suffix=*/"", nativeInputFileName);

  if (std::error_code ok; errCode != ok) {
    root.emitError(
        "cannot generate a unique temporary file for input to Calyx compiler");
    return failure();
  }

  // Emit the current program into a file so the native compiler can operate
  // over it.
  std::unique_ptr<llvm::ToolOutputFile> inputFile =
      mlir::openOutputFile(nativeInputFileName, &errMsg);
  if (inputFile == nullptr) {
    root.emitError(errMsg);
    return failure();
  }

  auto res = circt::calyx::exportCalyx(root, inputFile->os());
  if (failed(res))
    return failure();
  inputFile->os().flush();

  // Create a file for the native compiler to write the results into
  SmallString<32> nativeOutputFileName;
  errCode = llvm::sys::fs::getPotentiallyUniqueTempFileName(
      "calyxNativeOutTemp", /*suffix=*/"", nativeOutputFileName);
  if (std::error_code ok; errCode != ok) {
    root.emitError(
        "cannot generate a unique temporary file name to store output");
    return failure();
  }

  llvm::SmallVector<StringRef> calyxArgs = {
      calyxExe, nativeInputFileName, "-o", nativeOutputFileName, "-b", "mlir"};

  // Configure the native pass pipeline. If passPipeline is provided, we expect
  // it to be a comma separated list of passes to run.
  if (!passPipeline.empty()) {
    llvm::SmallVector<StringRef> passArgs;
    llvm::StringRef ppRef = passPipeline;
    ppRef.split(passArgs, ",");
    for (auto pass : passArgs) {
      if (pass.empty())
        continue;
      calyxArgs.push_back("-p");
      calyxArgs.push_back(pass);
    }
  } else {
    // If no arguments are specified, use the default pipeline.
    calyxArgs.push_back("-p");
    calyxArgs.push_back("all");
  }

  // We always need to run the lower-guards pass in the native compiler to emit
  // valid MLIR
  calyxArgs.push_back("-p");
  calyxArgs.push_back("lower-guards");

  std::optional<StringRef> redirects[] = {/*stdin=*/std::nullopt,
                                          /*stdout=*/nativeOutputFileName,
                                          /*stderr=*/std::nullopt};

  int result = llvm::sys::ExecuteAndWait(
      calyxExe, calyxArgs, /*Env=*/std::nullopt,
      /*Redirects=*/redirects,
      /*SecondsToWait=*/0, /*MemoryLimit=*/0, &errMsg);

  if (result != 0) {
    root.emitError() << errMsg;
    return failure();
  }

  // Parse the output buffer into a Calyx operation so that we can insert it
  // back into the program.
  auto bufferRead = llvm::MemoryBuffer::getFile(nativeInputFileName);
  if (!bufferRead || !*bufferRead) {
    root.emitError("execution of '" + calyxExe +
                   "' did not produce any output file named '" +
                   nativeInputFileName + "'");
    return failure();
  }

  // Load the output from the native compiler as a ModuleOp
  auto loadedMod =
      parseSourceFile<ModuleOp>(nativeOutputFileName.str(), root.getContext());
  auto *loadedBlock = loadedMod->getBody();

  // XXX(rachitnigam): This is quite baroque. We insert the new block before the
  // previous one and then remove the old block. A better thing to do would be
  // to replace the moduleOp completely but I couldn't figure out how to do
  // that.
  auto *oldBlock = root.getBody();
  loadedBlock->moveBefore(oldBlock);
  oldBlock->erase();
  return success();
}

std::unique_ptr<mlir::Pass> circt::createCalyxNativePass() {
  return std::make_unique<CalyxNativePass>();
}
