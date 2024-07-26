//===- SignalOps.td - LLHD signal operations ---------------*- tablegen -*-===//
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

include "mlir/IR/EnumAttr.td"

def LLHD_SigOp : LLHD_Op<"sig", [
    ParentOneOf<["hw::HWModuleOp", "llhd::ProcessOp"]>,
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

    Syntax:

    ```
    sig-op ::= ssa-id `=` `llhd.sig` sig-name ssa-init attr-dict `:` init-type
    ```

    Example:

    ```mlir
    %init_i64 = llhd.const 123 : i64
    %sig_i64 = llhd.sig "foo" %init_64 : i64

    %init_i1 = llhd.const 1 : i1
    %sig_i1 = llhd.sig "bar" %init_i1 : i1
    ```

    The first `llhd.sig` instruction creates a new signal named "foo", carrying
    an `i64` type with initial value of 123, while the second one creates a new
    signal named "bar", carrying an `i1` type with initial value of 1.
  }];

  let arguments = (ins StrAttr: $name, LLHD_AnyElementType: $init);
  let results = (outs InOutType: $result);

  let assemblyFormat = "$name $init attr-dict `:` qualified(type($init))";
}

def LLHD_PrbOp : LLHD_Op<"prb", [
    TypesMatchWith<
      "type of 'result' and underlying type of 'signal' have to match.",
      "signal", "result", "llvm::cast<hw::InOutType>($_self).getElementType()">
  ]> {
  let summary = "Probe a signal.";
  let description = [{
    The `llhd.prb` instruction probes a signal and returns the value it
    currently carries as a new SSA operand. The result type is always
    the type carried by the signal.

    Syntax:

    ```
    prb-op ::= ssa-id `=` `llhd.prb` ssa-sig attr-dict `:` !hw.inout<type>
    ```

    Example:

    ```mlir
    %const_i1 = llhd.const 1 : i1
    %sig_i1 = llhd.sig %const_i1 : i1
    %prbd = llhd.prb %sig_i1 : !hw.inout<i1>
    ```
  }];

  let arguments = (ins Arg<InOutType, "the signal to probe from",
                           [MemRead]>: $signal);
  let results = (outs LLHD_AnyElementType: $result);

  let assemblyFormat = "$signal attr-dict `:` qualified(type($signal))";
}

def LLHD_OutputOp : LLHD_Op<"output", [
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
    %value = llhd.const 1 : i1
    %time = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
    %sig = llhd.output "sigName" %value after %time : i1

    // is equivalent to

    %value = llhd.const 1 : i1
    %time = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
    %sig = llhd.sig "sigName" %value : i1
    llhd.drv %sig, %value after %time : !hw.inout<i1>
    ```
  }];

  let arguments = (ins OptionalAttr<StrAttr>: $name,
                       LLHD_AnyElementType: $value,
                       LLHD_TimeType: $time);

  let results = (outs InOutType: $result);

  let assemblyFormat = [{
    ( $name^ )? $value `after` $time attr-dict `:` qualified(type($value))
  }];
}

def LLHD_DrvOp : LLHD_Op<"drv", [
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

    Syntax:

    ```
    drv-op ::= `llhd.drv` ssa-signal `,` ssa-const `after` ssa-time
      (`if` ssa-enable)? `:` !hw.inout<const-type>
    ```

    Example:

    ```mlir
    %init = llhd.const 1 : i1
    %en = llhd.const 0 : i1
    %time = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
    %sig = llhd.sig %init : i1
    %new = llhd.not %init : i1

    llhd.drv %sig, %new after %time : !hw.inout<i1>
    llhd.drv %sig, %new after %time if %en : !hw.inout<i1>
    ```
  }];

  let arguments = (ins Arg<InOutType, "the signal to drive to",
                           [MemWrite]>: $signal,
                       LLHD_AnyElementType: $value,
                       LLHD_TimeType: $time,
                       Optional<I1>: $enable);

  let assemblyFormat = [{
    $signal `,` $value `after` $time ( `if` $enable^ )? attr-dict `:`
    qualified(type($signal))
  }];

  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

def DelayOp : LLHD_Op<"delay", [Pure, SameOperandsAndResultType]> {
  let summary = "specifies value propagation delay";
  let description = [{
    This operation propagates all value changes of the input to the output after
    the specified time delay.
    Reference values are not supported (e.g., pointers, inout, etc.)
    since the store-like operation used for those types should encode a delayed
    store.
  }];

  let arguments = (ins HWNonInOutType:$input, LLHD_TimeAttr:$delay);
  let results = (outs HWNonInOutType:$result);

  let assemblyFormat = "$input `by` $delay attr-dict `:` type($result)";
}
