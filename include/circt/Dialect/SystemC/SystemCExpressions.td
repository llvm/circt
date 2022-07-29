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
    NoSideEffect,
    TypesMatchWith<"result has to be the base type of the input",
                   "input", "result", "systemc::getSignalBaseType($_self)">]> {
  let summary = "Returns the current value of a signal or port.";
  let description = [{
    Represents the member function `const T& read() const;` and operator
    `operator const T& () const;` of class `sc_signal` as described in
    IEEE 1666-2011 §6.4.7., of class `sc_in` as described in §6.8.3., and
    of class `sc_inout` as decribed in §6.10.3. It shall return the current
    value of the signal/port.
  }];

  let arguments = (ins AnyTypeOf<[InputType, InOutType, SignalType]>:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";

  let builders = [
    OpBuilder<(ins "mlir::Value":$input), [{
      build($_builder, $_state,
            systemc::getSignalBaseType(input.getType()), input);
    }]>
  ];
}
