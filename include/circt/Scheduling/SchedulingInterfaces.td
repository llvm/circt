//===- SchedulingInterfaces.td - Scheduling Interfaces --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the interfaces in the Scheduling dialect.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

def SchedulableOpInterface : OpInterface<"SchedulableOpInterface"> {
  let cppNamespace = "::circt::scheduling";

  let description = [{
    This is an op interface for ops to provide information to the Scheduling
    infrastructure.
  }];

  let methods = [
    InterfaceMethod<
      "Returns the latency of the operation.",
      "circt::scheduling::Problem::OperatorType",
      "getOperatorType">
  ];
}
