//===- CalyxNative.cpp - Invoke the native Calyx compiler
//----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This is the main Calyx to HW Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"

#include "circt/Conversion/CalyxNative.h"
#include "circt/Dialect/Calyx/CalyxEmitter.h"
#include "circt/Dialect/Calyx/CalyxOps.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace circt;
using namespace circt::calyx;
using namespace circt::comb;
using namespace circt::seq;
using namespace circt::sv;

/// ConversionPatterns.

/// Pass entrypoint.

namespace {
class CalyxNativePass : public CalyxNativeBase<CalyxNativePass> {
public:
  void runOnOperation() override;
};
} // end anonymous namespace

void CalyxNativePass::runOnOperation() {
  ModuleOp root = getOperation();

  SmallString<32> execName = llvm::sys::path::filename("calyx");

  auto exeMb = llvm::sys::findProgramByName(execName);
  // If cannot find the executable, then nothing to do, return.
  if (!exeMb) {
    root.emitError("cannot find executable: '" + execName + "'");
    return;
  }
  StringRef generatorExe = exeMb.get();

  SmallVector<std::string> generatorArgs;
  // First argument should be the executable name.
  generatorArgs.push_back(generatorExe.str());

  SmallVector<StringRef> generatorArgStrRef;
  for (const std::string &a : generatorArgs)
    generatorArgStrRef.push_back(a);

  std::string errMsg;
  SmallString<32> nativeInputFileName;
  auto errCode = llvm::sys::fs::getPotentiallyUniqueTempFileName(
      "calyxNativeTemp", StringRef(""), nativeInputFileName);

  // Default error code is 0.
  std::error_code ok;
  if (errCode != ok) {
    root.emitError("cannot generate a unique temporary file name");
    return;
  }

  // Create a new emitter with the output file
  // Open the file
  auto fd = mlir::openOutputFile(nativeInputFileName, &errMsg);
  if (!fd) {
    root.emitError(errMsg);
    return;
  }

  auto emitter = circt::calyx::exportCalyx(root, fd->os());

  std::optional<StringRef> redirects[] = {
      /*stdin=*/std::nullopt,
      /*stdout=*/StringRef(nativeInputFileName),
      /*stderr=*/std::nullopt};

  auto args = llvm::ArrayRef<StringRef>();
  int result = llvm::sys::ExecuteAndWait(
      generatorExe, args, /*Env=*/std::nullopt,
      /*Redirects=*/redirects,
      /*SecondsToWait=*/0, /*MemoryLimit=*/0, &errMsg);

  if (result != 0) {
    root.emitError("execution of '" + generatorExe + "' failed");
    return;
  }

  auto bufferRead = llvm::MemoryBuffer::getFile(nativeInputFileName);
  if (!bufferRead || !*bufferRead) {
    root.emitError("execution of '" + generatorExe +
                   "' did not produce any output file named '" +
                   nativeInputFileName + "'");
    return;
  }
}

std::unique_ptr<mlir::Pass> circt::createCalyxNativePass() {
  return std::make_unique<CalyxNativePass>();
}
