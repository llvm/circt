//===- Passes.cpp - Helpers for pipeline instrumentation --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/Passes.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Format.h"

using namespace mlir;
using namespace circt;

void VerbosePassInstrumentation::runBeforePass(Pass *pass, Operation *op) {
  // This assumes that it is safe to log messages to stderr if the operation
  // is circuit or module op.
  if (isa<firrtl::CircuitOp, mlir::ModuleOp>(op)) {
    timePoints.push_back(TimePoint::clock::now());
    auto &os = llvm::errs();
    os << "[" << toolName << "] ";
    os.indent(2 * level++);
    os << "Running \"";
    pass->printAsTextualPipeline(llvm::errs());
    os << "\"\n";
  }
}

void VerbosePassInstrumentation::runAfterPass(Pass *pass, Operation *op) {
  using namespace std::chrono;
  // This assumes that it is safe to log messages to stderr if the operation
  // is circuit or module op.
  if (isa<firrtl::CircuitOp, mlir::ModuleOp>(op)) {
    auto &os = llvm::errs();
    auto elapsed =
        duration<double>(TimePoint::clock::now() - timePoints.pop_back_val()) /
        seconds(1);
    os << "[" << toolName << "] ";
    os.indent(2 * --level);
    os << "-- Done in " << llvm::format("%.3f", elapsed) << " sec\n";
  }
}
