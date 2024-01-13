//===- SMTOps.td - SMT dialect operations ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SMT_SMTOPS_TD
#define CIRCT_DIALECT_SMT_SMTOPS_TD

include "circt/Dialect/SMT/SMTDialect.td"
include "circt/Dialect/SMT/SMTAttributes.td"
include "circt/Dialect/SMT/SMTTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

class SMTOp<string mnemonic, list<Trait> traits = []> :
  Op<SMTDialect, mnemonic, traits>;

def DeclareConstOp : SMTOp<"declare_const", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
]> {
  let summary = "declare a symbolic value of a given sort";
  let description = [{
    This operation declares a symbolic value just as the `declare-const`
    statement in SMT-LIB 2.6. The result type determines the SMT sort of the
    symbolic value. The returned value can then be used to refer to the symbolic
    value instead of using the identifier like in SMT-LIB.

    The optionally provided string will be used as a prefix for the newly
    generated identifier (useful for easier readability when exporting to
    SMT-LIB). Each `declare_const` will always provide a unique new symbolic
    value even if the identifier strings are the same.

    Note that this operation cannot be marked as Pure since two operations (even
    with the same identifier string) could then be CSEd, leading to incorrect
    behavior.
  }];

  let arguments = (ins OptionalAttr<StrAttr>:$namePrefix);
  let results = (outs Res<AnySMTType, "a symbolic value", [MemAlloc]>:$result);

  let assemblyFormat = [{
    ($namePrefix^)? attr-dict `:` qualified(type($result))
  }];
}

def SolverOp : SMTOp<"solver", [
  IsolatedFromAbove,
  SingleBlockImplicitTerminator<"smt::YieldOp">,
]> {
  let summary = "create a solver instance within a lifespan";
  let description = [{
    This operation defines an SMT context with a solver instance. SMT operations
    are only valid when being executed between the start and end of the region
    of this operation. Any invocation outside is undefined. However, they do not
    have to be direct children of this operation. For example, it is allowed to
    have SMT operations in a `func.func` which is only called from within this
    region. No SMT value may enter or exit the lifespan of this region (such
    that no value created from another SMT context can be used in this scope and
    the solver can deallocate all state required to keep track of SMT values at
    the end).

    As a result, the region is comparable to an entire SMT-LIB script, but
    allows for concrete operations and control-flow. Concrete values may be
    passed in and returned to influence the computations after the `smt.solver`
    operation.

    Example:
    ```mlir
    %0:2 = smt.solver (%in) {smt.some_attr} : (i8) -> (i8, i32) {
    ^bb0(%arg0: i8):
      %c = smt.declare_const "c" : !smt.bool
      smt.assert %c
      %1 = smt.check sat {
        %c1_i32 = arith.constant 1 : i32
        smt.yield %c1_i32 : i32
      } unknown {
        %c0_i32 = arith.constant 0 : i32
        smt.yield %c0_i32 : i32
      } unsat {
        %c-1_i32 = arith.constant -1 : i32
        smt.yield %c-1_i32 : i32
      } -> i32
      smt.yield %arg0, %1 : i8, i32
    }
    ```

    TODO: solver configuration attributes
  }];

  let arguments = (ins Variadic<AnyNonSMTType>:$inputs);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let results = (outs Variadic<AnyNonSMTType>:$results);

  let assemblyFormat = [{
    `(` $inputs `)` attr-dict `:` functional-type($inputs, $results) $bodyRegion
  }];

  let hasRegionVerifier = true;
}

def AssertOp : SMTOp<"assert", []> {
  let summary = "assert that a boolean expression holds";
  let arguments = (ins BoolType:$input);
  let assemblyFormat = "$input attr-dict";
}

def CheckOp : SMTOp<"check", [
  NoRegionArguments,
  SingleBlockImplicitTerminator<"smt::YieldOp">,
]> {
  let summary = "check if the current set of assertions is satisfiable";
  let description = [{
    This operation checks if all the assertions in the solver defined by the
    nearest ancestor operation of type `smt.solver` are consistent. The outcome
    an be 'satisfiable', 'unknown', or 'unsatisfiable' and the corresponding
    region will be executed. It is the corresponding construct to the
    `check-sat` in SMT-LIB.

    Example:
    ```mlir
    %0 = smt.check sat {
      %c1_i32 = arith.constant 1 : i32
      smt.yield %c1_i32 : i32
    } unknown {
      %c0_i32 = arith.constant 0 : i32
      smt.yield %c0_i32 : i32
    } unsat {
      %c-1_i32 = arith.constant -1 : i32
      smt.yield %c-1_i32 : i32
    } -> i32
    ```
  }];

  let regions = (region SizedRegion<1>:$satRegion,
                        SizedRegion<1>:$unknownRegion,
                        SizedRegion<1>:$unsatRegion);
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    attr-dict `sat` $satRegion `unknown` $unknownRegion `unsat` $unsatRegion
    (`->` qualified(type($results))^ )?
  }];

  let hasRegionVerifier = true;
}

def YieldOp : SMTOp<"yield", [
  Pure,
  Terminator,
  ReturnLike,
  ParentOneOf<["smt::SolverOp", "smt::CheckOp"]>,
]> {
  let summary = "terminator operation for various regions of SMT operations";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "($values^ `:` qualified(type($values)))? attr-dict";
  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];
}

#endif // CIRCT_DIALECT_SMT_SMTOPS_TD
