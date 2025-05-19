//===-- Passes.td - LEC Transforms pass definition file ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for LEC transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_CIRCT_LEC_PASSES_TD
#define CIRCT_TOOLS_CIRCT_LEC_PASSES_TD

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def ConstructLEC : Pass<"construct-lec", "::mlir::ModuleOp"> {
  let summary = "Lower CIRCTs core dialects to a LEC problem statement";
  let description = [{
    Takes two `hw.module` operations and lowers them to a `verif.lec` operation
    inside a `func.func` matching the name of the first module.

    This pass can also (optionally) insert a 'main' function that calls the
    function with the `verif.lec` and adheres to the typical main function
    signature. This is useful when compiling to a standalone binary rather than
    using the JIT compiler.

    This pass also inserts a few LLVM dialect operation to print the result of
    the `verif.lec` using the `printf` function of the standard library.
  }];

  let options = [
    Option<"firstModule", "first-module", "std::string",
           /*default=*/"",
           "Name of the first of the two modules to compare.">,
    Option<"secondModule", "second-module", "std::string",
           /*default=*/"",
           "Name of the second of the two modules to compare.">,
    Option<"insertMainFunc", "insert-main", "bool",
           /*default=*/"false",
           "Whether a main function should be inserted for AOT compilation.">,
    Option<"insertReporting", "insert-report", "bool",
           /*default=*/"true",
           "Whether LLVM code should be inserted to report the result.">,
  ];

  let dependentDialects = [
    "verif::VerifDialect", "mlir::func::FuncDialect", "mlir::LLVM::LLVMDialect"
  ];
}

#endif // CIRCT_TOOLS_CIRCT_LEC_PASSES_TD

