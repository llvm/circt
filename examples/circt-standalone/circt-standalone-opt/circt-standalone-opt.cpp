//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRCTStandalone/CIRCTStandaloneDialect.h"
#include "CIRCTStandalone/CIRCTStandalonePasses.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Support/Version.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/PrettyStackTrace.h"

int main(int argc, char **argv) {
  llvm::setBugReportMsg(circt::circtBugReportMsg);

  mlir::DialectRegistry registry;
  registry.insert<circt::hw::HWDialect>();
  registry.insert<circt::standalone::CIRCTStandaloneDialect>();

  circt::standalone::registerPasses();
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  llvm::cl::AddExtraVersionPrinter(
      [](llvm::raw_ostream &os) { os << circt::getCirctVersion() << '\n'; });

  return mlir::failed(mlir::MlirOptMain(
      argc, argv, "CIRCT standalone optimizer driver", registry));
}
