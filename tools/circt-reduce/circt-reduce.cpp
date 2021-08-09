//===- circt-reduce.cpp - The circt-reduce driver -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'circt-reduce' tool, which is the circt analog of
// mlir-reduce, used to drive test case reduction.
//
//===----------------------------------------------------------------------===//

#include "circt/InitAllDialects.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-reduce/MlirReduceMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  circt::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  return failed(mlirReduceMain(argc, argv, context));
}
