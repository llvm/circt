//===- SystemCExpressions.td - SystemC expression ops ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the ops for SystemC expressions.
//
//===----------------------------------------------------------------------===//

def SignalReadOp : SystemCOp<"signal.read", [
    TypesMatchWith<"result has to be the base type of the input",
                   "input", "result", "systemc::getBaseType($_self)">]> {
  let summary = "Declare two SSA values equivalent.";
  let description = [{
    This operation does not model a specific construct in the SystemC spec, but
    is required to 'connect' an SSA value to a module output SSA value. It can,
    however, also be used to declare any two SSA values to be aliases which
    means that ExportSystemC will generate at most one variable name for all
    SSA values in the alias set.
  }];

  let arguments = (ins AnyTypeOf<[InputType, InOutType, SignalType]>:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";

  let builders = [
    OpBuilder<(ins "mlir::Value":$input), [{
      build($_builder, $_state, systemc::getBaseType(input.getType()), input);
    }]>
  ];
}
