//===- Seq.td - Seq dialect definition ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Seq dialect. It contains the one op and
// pass. Once we add more than one, we should break it out like the other
// dialects.
//
//===----------------------------------------------------------------------===//

#ifndef SEQ_TD
#define SEQ_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Pass/PassBase.td"

def SeqDialect : Dialect {
  let name = "seq";

  let summary = "Types and operations for seq dialect";
  let description = [{
    The `seq` dialect is intended to model digital sequential logic.
  }];

  let hasConstantMaterializer = 1;
  let cppNamespace = "::circt::seq";
}

// Base class for the operation in this dialect.
class SeqOp<string mnemonic, list<Trait> traits = []> :
    Op<SeqDialect, mnemonic, traits>;

def CompRegOp : SeqOp<"compreg",
    [NoSideEffect, AllTypesMatch<["input", "data"/*, "resetValue"*/]>,
     SameVariadicOperandSize,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]> ]> {
       // AllTypesMatch doesn't work with Optional types yet.

  let summary = "Register a value, storing it for one cycle";
  let description = "See the Seq dialect rationale for a longer description";

  let arguments = (ins AnyType:$input, I1:$clk, StrAttr:$name,
    Optional<I1>:$reset, Optional<AnyType>:$resetValue,
    OptionalAttr<SymbolNameAttr>:$sym_name);
  let results = (outs AnyType:$data);
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$clk, "StringRef":$sym_name), [{
      return build($_builder, $_state, input.getType(),
                   input, clk, sym_name, Value(), Value(),
                   StringAttr::get($_builder.getContext(), sym_name));
    }]>,
  ];
}

def FirRegOp : SeqOp<"firreg",
    [NoSideEffect, AllTypesMatch<["next", "data"/*, "resetValue"*/]>,
     SameVariadicOperandSize,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
      // AllTypesMatch doesn't work with Optional types yet.

  let summary = "Register with preset and sync or async reset";
  let description = [{
    `firreg` represents registers originating from FIRRTL after the lowering
    of the IR to HW.  The register is used as an intermediary in the process
    of lowering to SystemVerilog to facilitate optimisation at the HW level,
    compactly representing a register with a single operation instead of
    composing it from register definitions, always blocks and if statements.

    The `data` output of the register accesses the value it stores.  On the
    rising edge of the `clk` input, the register takes a new value provided
    by the `next` signal.  Optionally, the register can also be provided with
    a synchronous or an asynchronous `reset` signal and `resetValue`, as shown
    in the example below.

    ```
    %name = seq.firreg %next clock %clk [ sym @sym ]
        [ reset (sync|async) %reset, %value ] : type
    ```

    Implicitly, all registers are pre-set to a randomized value.

    A register implementing a counter starting at 0 from reset can be defined
    as follows:

    ```
    %zero = hw.constant 0 : i32
    %reg = seq.firreg %next clock %clk reset sync %reset, %zero : i32
    %one = hw.constant 1 : i32
    %next = comb.add %reg, %one : i32
    ```
  }];

  let arguments = (ins AnyType:$next, I1:$clk, StrAttr:$name,
                       OptionalAttr<SymbolNameAttr>:$inner_sym,
                       Optional<I1>:$reset, Optional<AnyType>:$resetValue,
                       UnitAttr:$isAsync);
  let results = (outs AnyType:$data);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$next, "Value":$clk,
                   "StringAttr":$name,
                   CArg<"StringAttr", "{}">:$inner_sym)>,
    OpBuilder<(ins "Value":$next, "Value":$clk,
                   "StringAttr":$name,
                   "Value":$reset, "Value":$resetValue,
                   CArg<"StringAttr", "{}">:$inner_sym,
                   CArg<"bool", "false">:$isAsync)>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Check whether the register has a reset value.
    bool hasReset() { return !!getReset(); }
  }];
}

#endif // SEQ_TD
