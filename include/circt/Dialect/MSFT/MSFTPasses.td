//===- MSFTPasses.td - MSFT dialect passes -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def ExportTcl: Pass<"msft-export-tcl", "mlir::ModuleOp"> {
  let summary = "Create tcl ops";
  let constructor = "circt::msft::createExportTclPass()";
  let dependentDialects = [
    "circt::emit::EmitDialect",
    "circt::hw::HWDialect",
    "circt::sv::SVDialect",
  ];
  let options = [
    ListOption<"tops", "tops", "std::string",
               "List of top modules to export Tcl for",
               "llvm::cl::ZeroOrMore,">,
    Option<"tclFile", "tcl-file", "std::string",
           "", "File to output Tcl into">
  ];
}

def LowerInstances: Pass<"msft-lower-instances", "mlir::ModuleOp"> {
  let summary = "Lower dynamic instances";
  let constructor = "circt::msft::createLowerInstancesPass()";
  let dependentDialects = ["circt::hw::HWDialect"];
}

def LowerConstructs: Pass<"msft-lower-constructs", "mlir::ModuleOp"> {
  let summary = "Lower high-level constructs";
  let constructor = "circt::msft::createLowerConstructsPass()";
  let dependentDialects = ["circt::hw::HWDialect"];
}
