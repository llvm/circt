//===- LLHDMemoryOps.td - LLHD memory operations -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for LLHD memory allocation and access.
//
//===----------------------------------------------------------------------===//

include "circt/Dialect/LLHD/IR/LLHDDialect.td"

def VarOp : LLHDOp<"var", [
    TypesMatchWith<
      "type of 'init' and underlying type of 'result' have to match.",
      "init", "result", "PtrType::get($_self)">
  ]> {
  let summary = "Stack allocation.";
  let description = [{
    The `llhd.var` operation allocates a memory region on the stack with the
    initial value given by `init`, and returns a pointer to the allocated
    region.

    Example:

    ```
    %int = hw.constant 0 : i32
    %iPtr = llhd.var %int : i32
    ```
  }];

  let arguments = (ins HWValueType: $init);
  let results = (outs Res<LLHDPtrType, "the newly allocated reference",
                          [MemAlloc]>: $result);

  let assemblyFormat = "$init attr-dict `:` qualified(type($init))";
}

def LoadOp : LLHDOp<"load", [
    TypesMatchWith<
      "type of 'result' and underlying type of 'pointer' have to match.",
      "pointer", "result", "llvm::cast<PtrType>($_self).getElementType()">
  ]> {
  let summary = "Load a value.";
  let description = [{
    The `llhd.load` operation loads a value from a memory region given by
    `pointer`.

    Example:

    ```
    %int = hw.constant 0 : i32
    %iPtr = llhd.var %int : i32

    %iLd = llhd.load %iPtr : !llhd.ptr<i32>
    ```
  }];

  let arguments = (ins Arg<LLHDPtrType, "the reference to load from",
                           [MemRead]>: $pointer);
  let results = (outs HWValueType: $result);

  let assemblyFormat = "$pointer attr-dict `:` qualified(type($pointer))";
}

def StoreOp : LLHDOp<"store", [
    TypesMatchWith<
      "type of 'value' and underlying type of 'pointer' have to match.",
      "pointer", "value", "llvm::cast<PtrType>($_self).getElementType()">
  ]> {
  let summary = "Store a value.";
  let description = [{
    The `llhd.store` operation stores the value `value` to the memory region
    given by `pointer`.

    Example:

    ```
    %int = hw.constant 0 : i32
    %iPtr = llhd.var %int : i32

    llhd.store %iPtr, %int : !llhd.ptr<i32>
    ```
  }];

  let arguments = (ins Arg<LLHDPtrType, "the reference to store to",
                           [MemWrite]>: $pointer,
                       HWValueType: $value);

  let assemblyFormat = "$pointer `,` $value attr-dict `:` qualified(type($pointer))";
}
