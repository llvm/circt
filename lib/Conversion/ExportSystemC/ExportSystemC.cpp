//===- ExportSystemC.cpp - SystemC Emitter --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main SystemC emitter implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ExportSystemC.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SystemC/SystemCDialect.h"
#include "mlir/IR/BuiltinOps.h"

using namespace circt;

#define DEBUG_TYPE "export-systemc"

static LogicalResult emitModule(ModuleOp module, raw_ostream &os) {
  os << "Not implemented yet.\n";

  return success();
}

static LogicalResult emitModule(ModuleOp module, StringRef directory) {
  return emitModule(module, llvm::outs());
}

//===----------------------------------------------------------------------===//
// Unified Emitter
//===----------------------------------------------------------------------===//

namespace {

struct ExportSystemCPass : public ExportSystemCBase<ExportSystemCPass> {
  ExportSystemCPass(raw_ostream &os) : os(os) {}
  void runOnOperation() override {
    if (failed(emitModule(getOperation(), os)))
      signalPassFailure();
  }

private:
  raw_ostream &os;
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
circt::createExportSystemCPass(llvm::raw_ostream &os) {
  return std::make_unique<ExportSystemCPass>(os);
}

std::unique_ptr<mlir::Pass> circt::createExportSystemCPass() {
  return createExportSystemCPass(llvm::outs());
}

//===----------------------------------------------------------------------===//
// Split Emitter
//===----------------------------------------------------------------------===//

namespace {

struct ExportSplitSystemCPass
    : public ExportSplitSystemCBase<ExportSplitSystemCPass> {
  ExportSplitSystemCPass(StringRef directory) {
    directoryName = directory.str();
  }
  void runOnOperation() override {
    if (failed(emitModule(getOperation(), directoryName)))
      signalPassFailure();
  }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
circt::createExportSplitSystemCPass(StringRef directory) {
  return std::make_unique<ExportSplitSystemCPass>(directory);
}
