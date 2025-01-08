//===- LLHDStructureOps.td - Process and Entity defs -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the LLHD Process, Entity and control flow MLIR ops.
//
//===----------------------------------------------------------------------===//

def ProcessOp : LLHDOp<"process", [
  NoRegionArguments,
  RecursiveMemoryEffects,
  HasParent<"hw::HWModuleOp">
]> {
  let summary = "create a process";
  let description = [{
    An `llhd.process` represents control-flow within a module. It may contain
    arbitrarily many basic blocks terminated with `cf.br`, `cf.cond_br`, and
    `llhd.yield`.

    The body is executed whenever one of the SSA values defined outside the
    region and used inside of it changes. It may also be executed when no values
    have changed and it is the bodies responsibility to detect and do the
    appropriate thing (e.g., not execute some side-effecting operation). This
    can be achieved by adding the signals of interest to the process outputs and
    comparing the current with the previous value of that signal.

    Execution order between multiple processes is undefined. To enforce a
    certain order, additional control signals can be added and passed between
    them.

    The outputs of the process are initially undefined (TODO: add initial args).
    A process holds the current outputs until the next time the body is
    executed. Then they are set to the newly yielded values.

    The body must not contain any operations that delay time (e.g.,
    `llhd.delay`).


    Example:

    ```mlir
    %0:2 = llhd.process {
      // Check for clock posedge
      %true = hw.constant true
      %1 = comb.xor %0#1, %true : i1
      %2 = comb.icmp eq %clk, %1 : i1
      cf.cond_br %2, ^bb1, ^bb2
    ^bb1:
      // If not posedge, do nothing
      llhd.yield %0#0, %0#1 : i32, i1
    ^bb2:
      // If posedge, do the thing
      %3 = hw.constant 1 : i32
      %4 = comb.add %0#0, %3 : i32
      func.call @display(%4) : (i32) -> () // print current val
      llhd.yield %4, %clk : i32, i1
    }
    ```
  }];

  let regions = (region MinSizedRegion<1>: $body);
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = "(`->` type($results)^)? attr-dict-with-keyword $body";

  let hasRegionVerifier = 1;
}

def FinalOp : LLHDOp<"final", [
  NoRegionArguments,
  RecursiveMemoryEffects,
  HasParent<"hw::HWModuleOp">,
]> {
  let summary = "A process that runs at the end of simulation";
  let description = [{
    An `llhd.final` op encapsulates a region of IR that is to be executed after
    the last time step of a simulation has completed. This can be used to
    implement various forms of state cleanup and tear-down. Some verifications
    ops may also want to check that certain final conditions hold at the end of
    a simulation run.

    Control flow must eventually end in an `llhd.yield` terminator.

    Execution order between multiple `llhd.final` ops is undefined.

    Example:
    ```mlir
    hw.module @Foo() {
      llhd.final {
        func.call @printSimulationStatistics() : () -> ()
        llhd.yield
      }
    }
    ```
  }];
  let regions = (region MinSizedRegion<1>: $body);
  let assemblyFormat = "attr-dict-with-keyword $body";

  let hasRegionVerifier = 1;
}

def ConnectOp : LLHDOp<"con", [
    SameTypeOperands,
    HasParent<"hw::HWModuleOp">
  ]> {
  let summary = "Connect two signals.";
  let description = [{
    The `con` instruction connects two signals such that they essentially become
    one signal. All driven values on one signal will be reflected on the other.
  }];

  let arguments = (ins InOutType:$lhs,
                       InOutType:$rhs);

  let assemblyFormat = [{
    operands attr-dict `:` qualified(type($lhs))
  }];

  let hasCanonicalizeMethod = 1;
}

def YieldOp : LLHDOp<"yield", [
  Terminator,
  ParentOneOf<["ProcessOp", "FinalOp"]>
]> {
  let summary = "generic LLHD dialect terminator";

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}
