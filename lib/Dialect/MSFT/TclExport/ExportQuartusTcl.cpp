//===- ExportQuartusTcl.cpp - Emit Quartus-flavored TCL -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Write out TCL with the appropriate API calls for Quartus.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/MSFT/ExportTcl.h"
#include "circt/Dialect/MSFT/MSFTAttributes.h"
#include "circt/Dialect/RTL/RTLOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Translation/ExportVerilog.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Translation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace rtl;
using namespace msft;

LogicalResult circt::msft::exportQuartusTCL(ModuleOp module,
                                            llvm::raw_ostream &os) {
  return failure();
}

void circt::msft::registerMSFTTclTranslation() {
  mlir::TranslateFromMLIRRegistration toQuartusTcl(
      "export-quartus-tcl", exportQuartusTCL,
      [](mlir::DialectRegistry &registry) {
        registry.insert<MSFTDialect, RTLDialect>();
      });
}
