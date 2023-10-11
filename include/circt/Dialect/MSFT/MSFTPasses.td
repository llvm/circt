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
  let dependentDialects = ["circt::sv::SVDialect", "circt::hw::HWDialect"];
  let options = [
    ListOption<"topsOpt", "tops", "std::string",
               "List of top modules to export Tcl for. If empty, exports all tops.",
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

def CreateGenericPDHierarchy : Pass<"msft-create-generic-pd-hierarchy", "mlir::ModuleOp"> {
  let summary = "Create generic PD hierarchy from module-local PD constraints and the instance graph";
  let constructor = "circt::msft::createCreateGenericPDHierarchyPass()";
}
