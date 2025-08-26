//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "circt/Dialect/LLHD/IR/LLHDDialect.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

def ProceduralRegion : NativeOpTrait<"ProceduralRegion"> {
  let cppNamespace = "::circt::llhd";
}

//===----------------------------------------------------------------------===//
// ProcessOp
//===----------------------------------------------------------------------===//

def ProcessOp : LLHDOp<"process", [
  HasParent<"hw::HWModuleOp">,
  NoRegionArguments,
  ProceduralRegion,
  RecursiveMemoryEffects,
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
  HasParent<"hw::HWModuleOp">,
  NoRegionArguments,
  ProceduralRegion,
  RecursiveMemoryEffects,
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
// CombinationalOp
//===----------------------------------------------------------------------===//

def CombinationalOp : LLHDOp<"combinational", [
  HasParent<"hw::HWModuleOp">,
  NoRegionArguments,
  ProceduralRegion,
  RecursiveMemoryEffects,
]> {
  let summary = "A process that runs when any of its operand values change";
  let description = [{
    An `llhd.combinational` op encapsulates a region of IR that executes once at
    the beginning of the simulation, and subsequently whenever any of the values
    used in its body change. Control flow must eventually end in an `llhd.yield`
    terminator. The process may have results, in which case the `llhd.yield`
    terminators must provide a list of values to yield for the process results.
    Whenever any of the values used in the body change, the process reexecutes
    in order to compute updated results.

    This op is commonly used to embed a control-flow description of some
    combinational logic inside the surrounding module's graph region.

    Example:
    ```mlir
    hw.module @Foo() {
      %0, %1 = llhd.combinational -> i42, i9001 {
        cf.cond_br %2, ^bb1(%3, %4 : i42, i9001), ^bb1(%5, %6 : i42, i9001)
      ^bb1(%7: i42, %8: i9001):
        llhd.yield %7, %8 : i42, i9001
      }
    }
    ```
  }];
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region MinSizedRegion<1>:$body);
  let assemblyFormat = [{
    (`->` type($results)^)?
    attr-dict-with-keyword $body
  }];
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// WaitOp
//===----------------------------------------------------------------------===//

def WaitOp : LLHDOp<"wait", [
  AttrSizedOperandSegments,
  HasParent<"ProcessOp">,
  Terminator,
  DeclareOpInterfaceMethods<BranchOpInterface, ["getSuccessorForOperands"]>
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

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def YieldOp : LLHDOp<"yield", [
  HasParent<"CombinationalOp">,
  Pure,
  ReturnLike,
  Terminator,
]> {
  let summary = "Yield results back from a combinational process";
  let description = [{
    The `llhd.yield` terminator terminates control flow in the parent process
    and yields the `yieldOperands` as the result values of the process.

    Example:
    ```mlir
    llhd.combinational {
      llhd.yield
    }
    %2:2 = llhd.combinational -> i42, i9001 {
      llhd.yield %0, %1 : i42, i9001
    }
    ```
  }];
  let arguments = (ins Variadic<AnyType>:$yieldOperands);
  let assemblyFormat = [{
    ($yieldOperands^ `:` type($yieldOperands))?
    attr-dict
  }];
  let hasVerifier = 1;
}
