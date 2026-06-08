//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRCTStandalone/CIRCTStandaloneDialect.h"
#include "CIRCTStandalone/CIRCTStandalonePasses.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Compiler.h"

extern "C" LLVM_ATTRIBUTE_WEAK mlir::DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "CIRCTStandalone", LLVM_VERSION_STRING,
          [](mlir::DialectRegistry *registry) {
            registry->insert<circt::standalone::CIRCTStandaloneDialect>();
            circt::standalone::registerPasses();
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "CIRCTStandalonePasses", LLVM_VERSION_STRING,
          []() { circt::standalone::registerPasses(); }};
}
