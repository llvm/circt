//===- IbisTypes.td - Definition of Ibis dialect types --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_IBIS_IBISTYPES_TD
#define CIRCT_DIALECT_IBIS_IBISTYPES_TD

include "circt/Dialect/Ibis/IbisDialect.td"
include "mlir/IR/AttrTypeBase.td"

class IbisTypeDef<string name> : TypeDef<IbisDialect, name> { }

def ClassRefType : IbisTypeDef<"ClassRef"> {
  let mnemonic = "classref";
  let parameters = (ins OptionalParameter<"mlir::FlatSymbolRefAttr">:$classRef);
  let assemblyFormat = "(`<` $classRef^  `>`)?";
  let description = [{
    A reference to an Ibis class. May be either a reference to a specific
    class (given a `$classRef` argument) or an opaque reference.
  }];

  let extraClassDeclaration = [{
    bool isOpaque() {
      return !getClassRef();
    }
  }];

  let builders = [
    // Builds an opaque (unresolved) class reference.
    TypeBuilder<(ins), [{
      return $_get($_ctxt, nullptr);
    }]>,
    TypeBuilder<(ins "StringAttr":$classRef), [{
      return $_get($_ctxt, FlatSymbolRefAttr::get(classRef));
    }]>
  ];
}

def AnyClassRefType : Type<
  CPred<"$_self.isa<ibis::ClassRefType>()">,
  "must be a !dc.classref<T?> type",
  "ibis::ClassRefType">{
}

def OpaqueClassRefType : Type<
  CPred<"ibis::isOpaqueClassRefType($_self)">,
  "must be a !dc.classref<> type">,
  BuildableType<"$_builder.getType<ibis::ClassRefType>()"> {
}

def PortRefType : IbisTypeDef<"PortRef"> {
  let mnemonic = "portref";
  let parameters = (ins "TypeAttr":$portTypeAttr);
  let assemblyFormat = "`<` $portTypeAttr  `>`";
  let description = [{
    A reference to an Ibis port.
  }];

  let extraClassDeclaration = [{
    Type getPortType() {
      return getPortTypeAttr().getValue();
    }
  }];

  let builders = [
    TypeBuilder<(ins "Type":$t), [{
      return $_get($_ctxt, TypeAttr::get(t));
    }]>
  ];
}

#endif // CIRCT_DIALECT_IBIS_IBISTYPES_TD
