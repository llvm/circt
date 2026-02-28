//===- SimOpInterfaces.td - Operation Interfaces -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the Sim operation interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_SIMOPINTERFACES
#define CIRCT_DIALECT_SIM_SIMOPINTERFACES

include "mlir/IR/OpBase.td"

def SimValueFormatter: OpInterface<"ValueFormatter", []> {
    let cppNamespace = "circt::sim";
    let description = "An operation converting a value to a format fragment";
    let methods = [
        InterfaceMethod<
            "Attempt to format an attribute as specified by this operation.",
            "::mlir::StringAttr", "formatConstant",
            (ins "::mlir::Attribute":$constVal)>
    ];
}

#endif
