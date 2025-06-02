//===- LLHDSignalOps.td - LLHD signal operations -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for LLHD signal creation and manipulation.
//
//===----------------------------------------------------------------------===//

include "circt/Dialect/LLHD/IR/LLHDDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"

def SignalOp : LLHDOp<"sig", [
  DeclareOpInterfaceMethods<DestructurableAllocationOpInterface>,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  TypesMatchWith<
    "type of 'init' and underlying type of 'signal' have to match.",
    "init", "result", "hw::InOutType::get($_self)">
]> {
  let summary = "Create a signal.";
  let description = [{
    The `llhd.sig` instruction introduces a new signal in the IR. The input
    operand determines the initial value carried by the signal, while the
    result type will always be a signal carrying the type of the init operand.
    A signal defines a unique name within the entity it resides in.

    Example:

    ```mlir
    %c123_i64 = hw.constant 123 : i64
    %foo = llhd.sig %c123_i64 : i64
    %0 = llhd.sig name "foo" %c123_i64 : i64
    ```

    This example creates a new signal named "foo", carrying an `i64` type with
    initial value of 123.
  }];
  let arguments = (ins
    OptionalAttr<StrAttr>:$name,
    HWValueType:$init
  );
  let results = (outs Res<InOutType, "", [MemAlloc]>:$result);
  let assemblyFormat = [{
    `` custom<ImplicitSSAName>($name) $init attr-dict
    `:` type($init)
  }];
}

def PrbOp : LLHDOp<"prb", [
    DeclareOpInterfaceMethods<DestructurableAccessorOpInterface>,
    DeclareOpInterfaceMethods<SafeMemorySlotAccessOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    TypesMatchWith<
      "type of 'result' and underlying type of 'signal' have to match.",
      "signal", "result", "llvm::cast<hw::InOutType>($_self).getElementType()">
  ]> {
  let summary = "Probe a signal.";
  let description = [{
    This operation probes a signal and returns the value it
    currently carries as a new SSA operand. The result type is always
    the type carried by the signal. In SSACFG regions, the operation has a read
    side effect on the signal operand. In graph regions, the operation is
    memory-effect free.

    Example:

    ```mlir
    %true = hw.constant true
    %sig_i1 = llhd.sig %true : i1
    %prbd = llhd.prb %sig_i1 : !hw.inout<i1>
    ```
  }];

  let arguments = (ins InOutType:$signal);
  let results = (outs HWValueType:$result);

  let assemblyFormat = "$signal attr-dict `:` qualified(type($signal))";
}

def OutputOp : LLHDOp<"output", [
    TypesMatchWith<
      "type of 'value' and underlying type of 'result' have to match.",
      "value", "result", "hw::InOutType::get($_self)">
  ]> {
  let summary = "Introduce a new signal and drive a value onto it.";
  let description = [{
    The `llhd.output` operation introduces a new signal and continuously
    drives a the given value onto it after a given time-delay. The same
    value is used to initialize the signal in the same way as the 'init'
    value in `llhd.sig`. An optional name can be given to the created signal.
    This shows up, e.g., in the simulation trace.

    Example:

    ```mlir
    %value = hw.constant true
    %time = llhd.constant_time <1ns, 0d, 0e>
    %sig = llhd.output "sigName" %value after %time : i1

    // is equivalent to

    %value = hw.constant true
    %time = llhd.constant_time <1ns, 0d, 0e>
    %sig = llhd.sig "sigName" %value : i1
    llhd.drv %sig, %value after %time : !hw.inout<i1>
    ```
  }];

  let arguments = (ins OptionalAttr<StrAttr>: $name,
                       HWValueType: $value,
                       LLHDTimeType: $time);

  let results = (outs InOutType: $result);

  let assemblyFormat = [{
    ( $name^ )? $value `after` $time attr-dict `:` qualified(type($value))
  }];
}

def DrvOp : LLHDOp<"drv", [
    DeclareOpInterfaceMethods<DestructurableAccessorOpInterface>,
    DeclareOpInterfaceMethods<SafeMemorySlotAccessOpInterface>,
    TypesMatchWith<
      "type of 'value' and underlying type of 'signal' have to match.",
      "signal", "value", "llvm::cast<hw::InOutType>($_self).getElementType()">
  ]> {
  let summary = "Drive a value into a signal.";
  let description = [{
    The `llhd.drv` operation drives a new value onto a signal. A time
    operand also has to be passed, which specifies the frequency at which
    the drive will be performed. An optional enable value can be passed as
    last argument. In this case the drive will only be performed if the
    value is 1. In case no enable signal is passed the drive will always be
    performed. This operation does not define any new SSA operands.

    Example:

    ```mlir
    %true = hw.constant true
    %false = hw.constant false
    %time = llhd.constant_time <1ns, 0d, 0e>
    %sig = llhd.sig %true : i1

    llhd.drv %sig, %false after %time : !hw.inout<i1>
    llhd.drv %sig, %false after %time if %true : !hw.inout<i1>
    ```
  }];

  let arguments = (ins Arg<InOutType, "the signal to drive to",
                           [MemWrite]>: $signal,
                       HWValueType: $value,
                       LLHDTimeType: $time,
                       Optional<I1>: $enable);

  let assemblyFormat = [{
    $signal `,` $value `after` $time ( `if` $enable^ )? attr-dict `:`
    qualified(type($signal))
  }];

  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

def DelayOp : LLHDOp<"delay", [Pure, SameOperandsAndResultType]> {
  let summary = "specifies value propagation delay";
  let description = [{
    This operation propagates all value changes of the input to the output after
    the specified time delay.
    Reference values are not supported (e.g., pointers, inout, etc.)
    since the store-like operation used for those types should encode a delayed
    store.
  }];

  let arguments = (ins HWNonInOutType:$input, LLHDTimeAttr:$delay);
  let results = (outs HWNonInOutType:$result);

  let assemblyFormat = "$input `by` $delay attr-dict `:` type($result)";
}
