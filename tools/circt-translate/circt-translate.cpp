//===- circt-translate.cpp - CIRCT Translation Driver ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that translates a file from/to MLIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"

int main(int argc, char **argv) {
  circt::registerAllTranslations();
  return mlir::failed(
      mlir::mlirTranslateMain(argc, argv, "CIRCT Translation Testing Tool"));
}
