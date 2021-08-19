//===- HWInterfaces.td - HW structure ops ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

def InstanceLike : OpInterface<"InstanceLike"> {
  let cppNamespace = "circt::hw";
  let methods = [
    InterfaceMethod<[{
        Get the instance name
      }],
      "StringRef", "instanceName"
    >,
    InterfaceMethod<[{
        Get the list of parameters
      }],
      "Optional<DictionaryAttr>", "parameters"
    >,
    InterfaceMethod<[{
        Get the list of inputs
      }],
      "::mlir::Operation::operand_range", "inputs"
    >,

    InterfaceMethod<[{
        Return the name of the specified result or empty string if it cannot be
        determined.
      }],
      "StringAttr", "getResultName", (ins "size_t":$i)
    >,

    InterfaceMethod< [{
        Lookup the module or extmodule for the symbol. This returns null on
        invalid IR.
      }],
      "Operation *", "getReferencedModule"
    >,

    InterfaceMethod< [{
        Get the instances's name as StringAttr.
      }],
      "StringAttr", "getNameAttr"
    >,

    InterfaceMethod< [{
        Get the instances's name.
      }],
      "StringRef", "getName", (ins),
      /*methodBody=*/[{
        return $_op.getNameAttr().getValue();
      }]
    >
  ];
}
