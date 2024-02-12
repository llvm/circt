//===- ModelInfoExport.cpp - Exports model info to JSON format ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Register the MLIR translation to export model info to JSON format.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ModelInfoExport.h"
#include "circt/Dialect/Arc/ArcDialect.h"
#include "circt/Dialect/Arc/ModelInfo.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace llvm;
using namespace mlir;
using namespace circt;
using namespace arc;

LogicalResult circt::arc::collectAndExportModelInfo(ModuleOp module,
                                                    llvm::raw_ostream &os) {
  SmallVector<ModelInfo> models;
  if (failed(collectModels(module, models)))
    return failure();
  serializeModelInfoToJson(os, models);
  return success();
}

void circt::arc::registerArcModelInfoTranslation() {
  static mlir::TranslateFromMLIRRegistration modelInfoToJson(
      "export-arc-model-info", "export Arc model info in JSON format",
      [](ModuleOp module, llvm::raw_ostream &os) {
        return arc::collectAndExportModelInfo(module, os);
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<arc::ArcDialect>();
      });
}
