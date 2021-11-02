//===- BaseWrapper.h - Simulation wrapper base class ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines BaseWrapper, the base class for all HLT wrappers.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_HLT_WRAPGEN_BASEWRAPPER_H
#define CIRCT_TOOLS_HLT_WRAPGEN_BASEWRAPPER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace mlir;

namespace circt {
namespace hlt {

class BaseWrapper {
public:
  BaseWrapper(StringRef inputFile, StringRef outDir)
      : inputFile(inputFile), outDir(outDir) {}
  virtual ~BaseWrapper(){};

  /// The wrap function takes a source operation and a kernel operation. The
  /// kernel operation is wrapper-specific, whereas the source operation
  /// represents the "software" interface that a kernel is expected to adhere
  /// to.
  LogicalResult wrap(Operation *srcOp, Operation *kernelOp);

  std::string getOutputFileName() { return outputFilename; }

protected:
  virtual SmallVector<std::string> getNamespaces() = 0;
  virtual SmallVector<std::string> getIncludes() = 0;

  virtual void emitAsyncCall(Operation *kernelOp);
  virtual void emitAsyncAwait(Operation *kernelOp);

  virtual LogicalResult emitPreamble(Operation * /*kernelOp*/) {
    return success();
  };
  StringRef funcName() { return funcOp.getName(); }
  using TypeEmitter =
      std::function<LogicalResult(llvm::raw_ostream &, Location, Type)>;
  LogicalResult emitIOTypes(const TypeEmitter &emitter);

  raw_fd_ostream &os() { return *outputFile; }
  raw_indented_ostream &osi() { return *outputFileIndented; }

  std::unique_ptr<raw_fd_ostream> outputFile;
  std::unique_ptr<raw_indented_ostream> outputFileIndented;
  std::string outputFilename;
  StringRef inputFile;
  StringRef outDir;
  FuncOp funcOp;

private:
  /// Creates an output file with filename fn, and associates os() and osi()
  /// with this file.
  LogicalResult createFile(Location loc, Twine fn);
};

} // namespace hlt
} // namespace circt

#endif // CIRCT_TOOLS_HLT_WRAPGEN_BASEWRAPPER_H
