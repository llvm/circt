//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LLHD_LLHDINTERFACES_TD
#define CIRCT_DIALECT_LLHD_LLHDINTERFACES_TD

include "mlir/IR/OpBase.td"

def DriveLikeOp : OpInterface<"DriveLikeOp"> {
  let cppNamespace = "circt::llhd";
  let description = [{
    An operation that updates a signal with a new value, optionally guarded by
    an enable condition.
  }];

  let methods = [
    InterfaceMethod<"Get the value being updated.",
      "::mlir::Value", "getDriven">,
    InterfaceMethod<"Get the operand holding the value being updated.",
      "::mlir::MutableOperandRange", "getDrivenMutable">,
    InterfaceMethod<"Get the value written to the signal.",
      "::mlir::Value", "getValue">,
    InterfaceMethod<"Get the operand holding the value written to the signal.",
      "::mlir::MutableOperandRange", "getValueMutable">,
    InterfaceMethod<"Get the enable condition, or null if the update is "
      "unconditional.",
      "::mlir::Value", "getEnable">,
    InterfaceMethod<"Get the operand holding the enable condition.",
      "::mlir::MutableOperandRange", "getEnableMutable">,
    InterfaceMethod<"Get the time after which the update takes effect, or "
      "null if the update is immediate.",
      "::mlir::Value", "getTime", (ins), [{}], [{ return {}; }]>,
  ];
}

def ProbeLikeOp : OpInterface<"ProbeLikeOp"> {
  let cppNamespace = "circt::llhd";
  let description = [{
    An operation that reads the current value of a signal or net.
  }];

  let methods = [
    InterfaceMethod<"Get the value being probed.",
      "::mlir::Value", "getProbed">,
  ];
}

#endif // CIRCT_DIALECT_LLHD_LLHDINTERFACES_TD
