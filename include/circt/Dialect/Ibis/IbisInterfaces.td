//===- IbisInterfaces.td - Ibis Interfaces ---------------*- tablegen -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the interfaces in the Ibis dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_IBIS_INTERFACES_TD
#define CIRCT_DIALECT_IBIS_INTERFACES_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

def PortOpInterface : OpInterface<"PortOpInterface"> {
  let cppNamespace = "circt::ibis";
  let description =
      "An interface for operations which describe ports.";

  let methods = [
    InterfaceMethod<
      "Returns the data type of the port.",
      "mlir::Type", "getPortType">
  ];
}

def ScopeOpInterface : OpInterface<"ScopeOpInterface", [Symbol]> {
  let cppNamespace = "circt::ibis";
  let description = [{
    An interface for operations which define Ibis scopes, that can be referenced
    by an `ibis.this` operation.
  }];

  let verify = "return detail::verifyScopeOpInterface($_op);";

  let methods = [
    InterfaceMethod<
      "Returns the body of the scope",
      "mlir::Block*", "getBodyBlock",
      (ins), [{
        return $_op.getBodyBlock();
      }]
    >,
    InterfaceMethod<
      "Returns `%this` of the scope",
      "mlir::TypedValue<ScopeRefType>", "getThis",
      (ins), [{
        return *detail::getThisFromScope($_op);
      }]
    >,
    InterfaceMethod<
      "Returns the symbol name of the scope",
      "mlir::StringAttr", "getScopeName",
      (ins), [{
        return $_op.getNameAttr();
      }]
    >
  ];
}

#endif // CIRCT_DIALECT_IBIS_INTERFACES_TD
