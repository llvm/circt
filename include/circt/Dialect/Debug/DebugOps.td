//===- DebugOps.td - Debug dialect operations --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DEBUG_DEBUGOPS_TD
#define CIRCT_DIALECT_DEBUG_DEBUGOPS_TD

include "circt/Dialect/Debug/DebugDialect.td"
include "circt/Dialect/Debug/DebugTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

class DebugOp<string mnemonic, list<Trait> traits = []> :
  Op<DebugDialect, mnemonic, traits>;


def VariableOp : DebugOp<"variable"> {
  let summary = "A named value to be captured in debug info";
  let description = [{
    Marks a value to be tracked in DI under the given name. The `dbg.variable`
    operation is useful to represent named values in a source language. For
    example, ports, constants, parameters, variables, nodes, or name aliases can
    all be represented as a variable. In combination with `dbg.array` and
    `dbg.struct`, complex aggregate source language values can be described and
    reconstituted from individual IR values. The `dbg.variable` operation acts
    as a tracker that follows the evolution of its assigned value throughout the
    compiler's pass pipelines. The debug info analysis uses this op to populate
    a module's scope with named source language values, and to establish how
    these source language values can be reconstituted from the actual IR values
    present at the end of compilation.

    See the rationale for examples and details.
  }];
  let arguments = (ins StrAttr:$name, AnyType:$value);
  let assemblyFormat = [{ $name `,` $value attr-dict `:` type($value) }];
}


def StructOp : DebugOp<"struct", [
  Pure,
  PredOpTrait<"number of fields and names match",
    CPred<"$fields.size() == $names.size()">>
]> {
  let summary = "Aggregate values into a struct";
  let description = [{
    Creates a struct aggregate from a list of names and values. The `dbg.struct`
    operation allows for struct-like source language values to be captured in
    the debug info. This includes structs, unions, bidirectional bundles,
    interfaces, classes, and other similar structures.

    See the rationale for examples and details.
  }];
  let arguments = (ins Variadic<AnyType>:$fields, StrArrayAttr:$names);
  let results = (outs StructType:$result);
  let hasCustomAssemblyFormat = 1;
}


def ArrayOp : DebugOp<"array", [Pure, SameTypeOperands]> {
  let summary = "Aggregate values into an array";
  let description = [{
    Creates an array aggregate from a list of values. The first operand is
    placed at array index 0. The last operand is placed at the highest array
    index. The `dbg.array` operation allows for array-like source language
    values to be captured in the debug info. This includes arrays, or in the
    case of SystemVerilog, packed and unpacked arrays, lists, sequences, queues,
    FIFOs, channels, and vectors.

    See the rationale for examples and details.
  }];
  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs ArrayType:$result);
  let hasCustomAssemblyFormat = 1;
}

#endif // CIRCT_DIALECT_DEBUG_DEBUGOPS_TD
