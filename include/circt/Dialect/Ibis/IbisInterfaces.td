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

// TODO: @mortbopet: should inherit from InnerSymTable once we have a way to
// support nested symbol tables.
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
      "Lookup an inner symbol in the scope",
      "Operation*", "lookupInnerSym",
      (ins "llvm::StringRef":$symName),
      "",
      [{
        // TODO: @mortbopet: fix once we have a way to do nested symbol
        // tables, and by extension, do nested symbol table lookup inside this
        // scope.
        // Until then, brute-force scan the inner symbol ops for a match.
        for (auto op : $_op.getBodyBlock()->template getOps<hw::InnerSymbolOpInterface>()) {
          if (*op.getInnerName() == symName)
            return op;
        }
        return nullptr;
      }]
    >,
    InterfaceMethod<
      "Lookup a port in the scope",
      "ibis::PortOpInterface", "lookupPort",
      (ins "llvm::StringRef":$portName),
      [{}],
      [{
        return dyn_cast_or_null<ibis::PortOpInterface>(this->lookupInnerSym(portName));
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

def MethodLikeOpInterface : OpInterface<"MethodLikeOpInterface"> {
  let cppNamespace = "circt::ibis";
  let description = [{
    An interface for Ibis operations that act like callable methods.
    This partially implements similar functionality to the FunctionLike interface
    which cannot be used for Ibis Methods due to Ibis methods defining InnerSym's
    whereas the FunctionLike interface is built on the assumption of the function
    defining a Symbol (i.e. inherits from the Symbol interface).
  }];

  let methods = [
    InterfaceMethod<[{
      Returns the type of the method.
    }],
    "::mlir::FunctionType", "getFunctionType">,
    InterfaceMethod<[{
      Returns the name of the method.
    }],
    "::mlir::StringAttr", "getMethodName">
  ];

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let extraSharedClassDeclaration = [{
    /// Returns the entry block of the function.
    Block* getBodyBlock() {
      auto& region = getFunctionBody();
      assert(!region.empty() && "expected non-empty function body");
      return &region.front();
    }

    using BlockArgListType = Region::BlockArgListType;
    BlockArgListType getArguments() { return getFunctionBody().getArguments(); }
    Region &getFunctionBody() {
      assert($_op->getNumRegions() == 1 && "expected one region");
      return $_op->getRegion(0);
    }
  }];
}

#endif // CIRCT_DIALECT_IBIS_INTERFACES_TD
