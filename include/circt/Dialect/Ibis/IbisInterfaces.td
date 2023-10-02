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
      "mlir::Type", "getPortType">,
    InterfaceMethod<
      "Returns the name of the port",
      "mlir::StringAttr", "getPortName">,
    InterfaceMethod<
      "Returns the `!ibis.portref` value defined by the op",
      "mlir::TypedValue<PortRefType>", "getPort"
    >
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
      "Returns the `ThisOp` of this scope",
      "Operation*", "getThisOp",
      (ins), [{
        return cast<ScopeOpInterface>($_op.getOperation()).getThis().getDefiningOp();
      }]
    >,
    InterfaceMethod<
      "Returns the symbol name of the scope",
      "mlir::StringAttr", "getScopeName",
      (ins), [{
        return $_op.getNameAttr();
      }]
    >,
    InterfaceMethod<
      "Lookup a port in the scope",
      "ibis::PortOpInterface", "lookupPort",
      (ins "llvm::StringRef":$portName), [{
        // @teqdruid TODO: make this more efficient using
        // innersymtablecollection when that's available to non-firrtl dialects.
        return dyn_cast_or_null<ibis::PortOpInterface>(SymbolTable::lookupSymbolIn($_op, portName));
      }]
    >
  ];
}

def BlockOpInterface : OpInterface<"BlockOpInterface"> {
  let cppNamespace = "circt::ibis";
  let description = [{
    An interface for Ibis block-like operations.
  }];

  let methods = [
    InterfaceMethod<
      "Returns the result types of this block from the point of view of inside the block",
      "llvm::SmallVector<Type>", "getInternalResultTypes",
      (ins)>
  ];
}

#endif // CIRCT_DIALECT_IBIS_INTERFACES_TD
