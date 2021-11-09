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
  let options = [
    Option<"verilogFile", "verilog-file", "std::string",
           "", "File to output Verilog into">,
    Option<"tclFile", "tcl-file", "std::string",
           "", "File to output Tcl into">
  ];
}
