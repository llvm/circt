//===- MSFTPasses.td - MSFT dialect passes -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

def LowerToHW: Pass<"lower-msft-to-hw", "mlir::ModuleOp"> {
  let summary = "Lower MSFT ops to hw ops";
  let constructor = "circt::msft::createLowerToHWPass()";
  let dependentDialects = ["circt::sv::SVDialect", "circt::hw::HWDialect"];
}

def ExportQuartusTcl: Pass<"export-quartus-tcl", "mlir::ModuleOp"> {
  let summary = "Export Quartus-flavored Tcl into sv.verbatim node(s)";
  let constructor = "circt::msft::createExportQuartusTclPass()";
  let dependentDialects = ["circt::msft::MSFTDialect", "circt::hw::HWDialect",
                           "circt::sv::SVDialect"];
}
