//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "circt/Dialect/LLHD/IR/LLHDDialect.td"

//===----------------------------------------------------------------------===//
// ProcessOp
//===----------------------------------------------------------------------===//

def ProcessOp : LLHDOp<"process", [
  NoRegionArguments,
  RecursiveMemoryEffects,
  HasParent<"hw::HWModuleOp">
]> {
  let summary = "A process that runs concurrently during simulation";
  let description = [{
    An `llhd.process` op encapsulates a region of IR that executes concurrently
    during simulation. Execution can be suspended using the `llhd.wait`
    terminator, which also includes a list of values that will cause the process
    execution to resume whenever they change. The `llhd.halt` terminator can be
    used to suspend execution forever. The process may have results, in which
    case any `llhd.wait` or `llhd.halt` terminators must provide a list of
    values to yield for the process results whenever execution is suspended. The
    process holds these result values until it is resumed and new result values
    are yielded.

    Example:
    ```mlir
    hw.module @top() {
      %0, %1 = llhd.process -> i42, i9001 {
        llhd.wait yield (%2, %3 : i42, i9001), ^bb1
      ^bb1:
        llhd.halt %4, %5 : i42, i9001
      }
    }
    ```
  }];
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region MinSizedRegion<1>: $body);
  let assemblyFormat = [{
    (`->` type($results)^)?
    attr-dict-with-keyword $body
  }];
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// FinalOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// WaitOp
//===----------------------------------------------------------------------===//

def WaitOp : LLHDOp<"wait", [
  AttrSizedOperandSegments,
  HasParent<"ProcessOp">,
  Terminator,
]> {
  let summary = "Suspend execution of a process";
  let description = [{
    The `llhd.wait` terminator suspends execution of the parent process until
    any of the `observed` values change or a fixed `delay` has passed. Execution
    resumes at the `dest` block with the `destOperands` arguments. The
    `yieldOperands` are yielded as the result values of the parent process.

    Example:
    ```mlir
    llhd.wait ^bb1(%0, %1 : i42, i9001)
    llhd.wait yield (%0, %1 : i42, i9001), ^bb1
    llhd.wait delay %time, ^bb1
    llhd.wait (%0, %1 : i42, i9001), ^bb1
    ```
  }];
  let arguments = (ins
    Variadic<AnyType>:$yieldOperands,
    Optional<LLHDTimeType>:$delay,
    Variadic<HWValueType>:$observed,
    Variadic<AnyType>:$destOperands
  );
  let successors = (successor AnySuccessor:$dest);
  let assemblyFormat = [{
    (`yield` ` ` `(` $yieldOperands^ `:` type($yieldOperands) `)` `,`)?
    (`delay` $delay^ `,`)?
    (`(`$observed^ `:` qualified(type($observed))`)` `,`)?
    $dest (`(` $destOperands^ `:` qualified(type($destOperands)) `)`)?
    attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// HaltOp
//===----------------------------------------------------------------------===//

def HaltOp : LLHDOp<"halt", [
  ParentOneOf<["ProcessOp", "FinalOp"]>,
  Terminator,
]> {
  let summary = "Terminate execution of a process";
  let description = [{
    The `llhd.halt` terminator suspends execution of the parent process forever,
    effectively terminating it. The `yieldOperands` are yielded as the result
    values of the parent process.

    Example:
    ```mlir
    llhd.halt
    llhd.halt %0, %1 : i42, i9001
    ```
  }];
  let arguments = (ins
    Variadic<AnyType>:$yieldOperands
  );
  let assemblyFormat = [{
    ($yieldOperands^ `:` type($yieldOperands))?
    attr-dict
  }];
  let hasVerifier = 1;
}
