//===- WrapGen.cpp - HLT wrapper generation tool --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the hlt-wrapgen tool. Based on an input reference
// function, a target kernel operation is wrapped to generate a .cpp file
// suitable for interaction with the HLT simulation library.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"

#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "circt/Tools/hlt/WrapGen/BaseWrapper.h"
#include "circt/Tools/hlt/WrapGen/handshake/HandshakeVerilatorWrapper.h"
#include "circt/Tools/hlt/WrapGen/std/StdWrapper.h"

using namespace llvm;
using namespace mlir;
using namespace circt;

// Used for the software interface.
static cl::opt<std::string>
    inputFunctionFilename("func", cl::Required,
                          cl::desc("<builtin function filename>"),
                          cl::init("-"));

// Used for the transactional interface.
static cl::opt<std::string>
    inputReferenceFilename("ref", cl::Optional,
                           cl::desc("<Reference function filename>"),
                           cl::init("-"));

// Used for the hardware interface.
static cl::opt<std::string> inputKernelFilename("kernel", cl::Optional,
                                                cl::desc("<Kernel filename>"),
                                                cl::init("-"));

static cl::opt<std::string> outputDirectory("o", cl::Required,
                                            cl::desc("<output directory>"),
                                            cl::init("-"));

static cl::opt<std::string>
    functionName("name", cl::Required,
                 cl::desc("The name of the function to wrap"), cl::init("-"));

enum class KernelType { HandshakeFIRRTL, Standard };

static cl::opt<KernelType> kernelType(
    "type", cl::Required,
    cl::desc("The type of the kernel to wrap. This will guide "
             "which wrapper is used, and what operation type the "
             "source kernel operation should be."),
    cl::init(KernelType::HandshakeFIRRTL),
    llvm::cl::values(clEnumValN(KernelType::HandshakeFIRRTL, "handshakeFIRRTL",
                                "Use the Handshake wrapper"),
                     clEnumValN(KernelType::Standard, "std",
                                "Use the standard wrapper")));
namespace circt {
namespace hlt {

/// Instantiates a wrapper based on the type of the kernel operation.
static std::unique_ptr<BaseWrapper> getWrapper() {
  switch (kernelType) {
  case KernelType::HandshakeFIRRTL:
    return std::make_unique<HandshakeVerilatorWrapper>(outputDirectory);
  case KernelType::Standard:
    return std::make_unique<StdWrapper>(outputDirectory);
  }
}

/// Container for the current set of loaded modules.
static SmallVector<OwningModuleRef> modules;

/// Load a module from the argument file fn into the modules vector.
static ModuleOp getModule(MLIRContext *ctx, const std::string &fn) {
  auto file_or_err = MemoryBuffer::getFileOrSTDIN(fn.c_str());
  if (std::error_code error = file_or_err.getError()) {
    errs() << "Error: Could not open input file '" << fn
           << "': " << error.message() << "\n";
    return nullptr;
  }

  // Load the MLIR module.
  SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(*file_or_err), SMLoc());
  modules.emplace_back(mlir::parseSourceFile(source_mgr, ctx));
  if (!modules.back()) {
    errs() << "Error: Found no modules in input file '" << fn << "'\n";
    return nullptr;
  }
  return modules.back().get();
}

/// Locates the operation defining the provided symbol within the set of loaded
/// modules.
static mlir::Operation *getOpToWrap(mlir::MLIRContext *ctx,
                                    const std::string &fn, StringRef symbol) {
  auto file_or_err = MemoryBuffer::getFileOrSTDIN(fn.c_str());
  if (std::error_code error = file_or_err.getError()) {
    errs() << ": could not open input file '" << fn << "': " << error.message()
           << "\n";
    return nullptr;
  }

  auto mod = getModule(ctx, fn);
  if (!mod) {
    errs() << "No module in file: " << fn << "\n";
    return nullptr;
  }

  Operation *targetOp = nullptr;
  SymbolTable::walkSymbolTables(
      mod, /*unused*/ false, [&](Operation *symOp, bool) {
        Operation *op = SymbolTable::lookupSymbolIn(symOp, symbol);
        if (op) {
          assert(targetOp == nullptr);
          targetOp = op;
        }
      });

  if (!targetOp) {
    errs() << "Found no definitions of symbol '" << symbol << "' in '" << fn
           << "'\n";
    return nullptr;
  }
  return targetOp;
}

static void registerDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::StandardOpsDialect>();
  registry.insert<arith::ArithmeticDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<handshake::HandshakeDialect>();
  registry.insert<firrtl::FIRRTLDialect>();
}

} // namespace hlt
} // namespace circt

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "hlt test wrapper generator\n\n");

  if (!sys::fs::exists(outputDirectory)) {
    errs() << "Output directory '" << outputDirectory << "' does not exist!\n";
    return 1;
  }

  mlir::DialectRegistry registry;
  circt::hlt::registerDialects(registry);
  mlir::MLIRContext context(registry);

  mlir::Operation *funcOpPtr =
      circt::hlt::getOpToWrap(&context, inputFunctionFilename, functionName);
  auto funcOp = dyn_cast<mlir::FuncOp>(funcOpPtr);
  if (!funcOp) {
    errs() << "Expected --func to be a builtin.func\n";
    return 1;
  }

  Operation *refOp = nullptr;
  Operation *kernelOp = nullptr;

  if (!inputReferenceFilename.empty()) {
    refOp =
        circt::hlt::getOpToWrap(&context, inputReferenceFilename, functionName);
    if (!refOp) {
      errs() << "No symbol named '" << functionName << "' found in '"
             << inputReferenceFilename << "\n";
      return 1;
    }
  }
  if (!inputKernelFilename.empty()) {
    kernelOp =
        circt::hlt::getOpToWrap(&context, inputKernelFilename, functionName);
    if (!kernelOp) {
      errs() << "No symbol named '" << functionName << "' found in '"
             << inputReferenceFilename << "\n";
      return 1;
    }
  }

  /// Locate wrapping handler for the operation.
  auto wrapper = circt::hlt::getWrapper();
  if (!wrapper)
    return 1;

  /// Go wrap!
  if (wrapper->wrap(funcOp, refOp, kernelOp).failed())
    return 1;
}
