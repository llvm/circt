//===- SystemCTypesImpl.td - SystemC type implementations --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Type implementations for the SystemC dialect.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/EnumAttr.td"

class SystemCTypeDef<string name> : TypeDef<SystemCDialect, name> { }

// A handle to a systemc::SCFuncOp. Declares the systemc::FuncHandleType in C++.
def FuncHandleTypeImpl : SystemCTypeDef<"FuncHandle"> {
  let summary = "A function handle type";
  let description = [{
    Represents a handle to a SystemC module's member function that
    can be used in places like SC_METHOD, SC_THREAD, etc.
  }];

  let mnemonic = "func_handle";
}
