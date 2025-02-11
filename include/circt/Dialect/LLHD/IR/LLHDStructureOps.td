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
    A `llhd.process` represents control-flow in a timed fashion. It allows a
    procedural description of how a circuit's output signals change in
    reaction to changing input signals. It has a region with arbitrarily
    many basic blocks. The first block is the entry block and cannot be
    targeted by the terminators. It uses `llhd.wait` as a terminator to add
    timed control-flow. Immediate control-flow with `br` or `cond_br` is
    also possible. Every process must either contain an infinite loop or
    terminate with the `llhd.halt` terminator.

    How does a process compare to functions and entities?

    | Unit     | Paradigm     | Timing    | Models                         |
    |----------|--------------|-----------|--------------------------------|
    | Function | control-flow | immediate | Computation in zero time       |
    | Process  | control-flow | timed     | Behavioral circuit description |
    | Entity   | data-flow    | timed     | Structural circuit description |

    Examples:

    ```mlir
    hw.module @top() {
      llhd.process {
        br ^bb1
      ^bb1:
        llhd.halt
      }
    }
    ```
  }];

  let regions = (region MinSizedRegion<1>: $body);

  let assemblyFormat = "attr-dict-with-keyword $body";

  let hasCanonicalizeMethod = 1;
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

    The `llhd.wait` terminator is not allowed in `llhd.final` processes since
    there is no later time slot for the execution to resume. Control flow must
    eventually end in an `llhd.halt` terminator.

    Execution order between multiple `llhd.final` ops is undefined.

    Example:
    ```mlir
    hw.module @Foo() {
      llhd.final {
        func.call @printSimulationStatistics() : () -> ()
        llhd.halt
      }
    }
    ```
  }];
  let regions = (region MinSizedRegion<1>: $body);
  let assemblyFormat = "attr-dict-with-keyword $body";
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

def WaitOp : LLHDOp<"wait", [
    Terminator,
    AttrSizedOperandSegments,
    HasParent<"ProcessOp">,
    DeclareOpInterfaceMethods<BranchOpInterface>
  ]> {
  let summary = "Suspends execution of a process.";
  let description = [{
    The `wait` instruction suspends execution of a process until any of the
    observed signals change or a fixed time interval has passed. Execution
    resumes at the specified basic block with the passed arguments.
    * This is a terminator instruction.
    * This instruction is only allowed in processes (`llhd.process`).

    Example:

    ```mlir
    llhd.wait ^bb1
    llhd.wait for %time, ^bb1(%time : !llhd.time)
    llhd.wait (%0, %1 : !hw.inout<i64>, !hw.inout<i1>), ^bb1(%1 : !hw.inout<i1>)
    llhd.wait for %time, (%0, %1 : !hw.inout<i64>, !hw.inout<i1>),
      ^bb1(%1, %0 : !hw.inout<i1>, !hw.inout<i64>)
    ```
  }];

  let arguments = (ins Variadic<HWValueType>:$observed,
                       Optional<LLHDTimeType>:$time,
                       Variadic<AnyType>:$destOps);

  let successors = (successor AnySuccessor:$dest);

  let assemblyFormat = [{
    (`for` $time^ `,`)? (`(`$observed^ `:` qualified(type($observed))`)` `,`)?
    $dest (`(` $destOps^ `:` qualified(type($destOps)) `)`)? attr-dict
  }];
}

def HaltOp : LLHDOp<"halt", [
  Terminator,
  ParentOneOf<["ProcessOp", "FinalOp"]>
]> {
  let summary = "Terminates execution of a process.";
  let description = [{
    The `halt` instruction terminates execution of a process. All processes
    must halt eventually or consist of an infinite loop.
  }];
  let assemblyFormat = "attr-dict";
}
