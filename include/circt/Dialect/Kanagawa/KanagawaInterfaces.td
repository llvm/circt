//===- KanagawaInterfaces.td - Kanagawa Interfaces -------*- tablegen -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the interfaces in the Kanagawa dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_KANAGAWA_INTERFACES_TD
#define CIRCT_DIALECT_KANAGAWA_INTERFACES_TD

include "circt/Dialect/HW/HWOpInterfaces.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

def NamedInnerSymbol : OpInterface<"NamedInnerSymbolInterface", [InnerSymbol]> {
  let cppNamespace = "circt::kanagawa";
  let description = [{
    An interface for operations which define inner symbols with an optional
    namehint.
  }];

  let methods = [
    InterfaceMethod<
      "Returns the namehint attribute of this op",
      "mlir::StringAttr", "getNameHintAttr",
      (ins),
      "",
      [{
        if(auto nameHint = $_op.getNameAttr())
          return nameHint;
        return $_op.getInnerSymAttr().getSymName();
      }]
    >,
    InterfaceMethod<
      "Returns the namehint of this op",
      "llvm::StringRef", "getNameHint",
      (ins),
      "",
      [{
        return getNameHintAttr().strref();
      }]
    >
  ];
}

def PortOpInterface : OpInterface<"PortOpInterface", [NamedInnerSymbol]> {
  let cppNamespace = "circt::kanagawa";
  let description =
      "An interface for operations which describe ports.";

  let methods = [
    InterfaceMethod<
      "Returns the data type of the port.",
      "mlir::Type", "getPortType">,
    InterfaceMethod<
      "Returns the `!kanagawa.portref` value defined by the op",
      "mlir::TypedValue<PortRefType>", "getPort"
    >
  ];
}

def ScopeOpInterface : OpInterface<"ScopeOpInterface", [NamedInnerSymbol]> {
  let cppNamespace = "circt::kanagawa";
  let description = [{
    An interface for operations which define Kanagawa scopes.
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
      "Returns the symbol name of the scope",
      "mlir::StringAttr", "getScopeName",
      (ins), [{
        return $_op.getInnerNameAttr();
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
          auto innerName = op.getInnerName();
          if(!innerName)
            continue;
          if (*innerName == symName)
            return op;
        }
        return nullptr;
      }]
    >,
    InterfaceMethod<
      "Lookup a port in the scope",
      "kanagawa::PortOpInterface", "lookupPort",
      (ins "llvm::StringRef":$portName),
      [{}],
      [{
        return dyn_cast_or_null<kanagawa::PortOpInterface>(this->lookupInnerSym(portName));
      }]
    >
  ];
}

def BlockOpInterface : OpInterface<"BlockOpInterface"> {
  let cppNamespace = "circt::kanagawa";
  let description = [{
    An interface for Kanagawa block-like operations.
  }];

  let methods = [
    InterfaceMethod<
      "Returns the result types of this block from the point of view of inside the block",
      "llvm::SmallVector<Type>", "getInternalResultTypes",
      (ins)>
  ];
}

def MethodLikeOpInterface : OpInterface<"MethodLikeOpInterface"> {
  let cppNamespace = "circt::kanagawa";
  let description = [{
    An interface for Kanagawa operations that act like callable methods.
    This partially implements similar functionality to the FunctionLike interface
    which cannot be used for Kanagawa Methods due to Kanagawa methods defining InnerSym's
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

#endif // CIRCT_DIALECT_KANAGAWA_INTERFACES_TD
