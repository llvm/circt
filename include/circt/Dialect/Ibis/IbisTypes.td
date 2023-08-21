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
include "mlir/IR/EnumAttr.td"

class IbisTypeDef<string name> : TypeDef<IbisDialect, name> { }

def ScopeRefType : IbisTypeDef<"ScopeRef"> {
  let mnemonic = "scoperef";
  let description = [{
    A reference to an Ibis scope. May be either a reference to a specific
    scope (given a `$scopeName` argument) or an opaque reference.
  }];

  let parameters = (ins OptionalParameter<"mlir::FlatSymbolRefAttr">:$scopeRef);
  let assemblyFormat = "(`<` $scopeRef^  `>`)?";
  let extraClassDeclaration = [{
    bool isOpaque() {
      return !getScopeRef();
    }
  }];

  let builders = [
    // Builds an opaque (unresolved) class reference.
    TypeBuilder<(ins), [{
      return $_get($_ctxt, nullptr);
    }]>,
    TypeBuilder<(ins "StringAttr":$scopeRef), [{
      return $_get($_ctxt, FlatSymbolRefAttr::get(scopeRef));
    }]>
  ];
}

def AnyScopeRefType : Type<
  CPred<"$_self.isa<ibis::ScopeRefType>()">,
  "must be a !dc.classref<T?> type",
  "ibis::ScopeRefType">{
}

def OpaqueScopeRefType : Type<
  CPred<"ibis::isOpaqueScopeRefType($_self)">,
  "must be a !dc.classref<> type">,
  BuildableType<"$_builder.getType<ibis::ScopeRefType>()"> {
}

def Input : I32EnumAttrCase<"Input", 0, "in">;
def Output : I32EnumAttrCase<"Output", 1, "out">;
def Direction : I32EnumAttr<"Direction", "Ibis port direction",
  [Input, Output]> {
  let cppNamespace = "::circt::ibis";
}

def PortRefType : IbisTypeDef<"PortRef"> {
  let mnemonic = "portref";
  let parameters = (ins "TypeAttr":$portTypeAttr, "ibis::Direction":$direction);
  let assemblyFormat = "`<` $direction $portTypeAttr  `>`";
  let description = [{
    A reference to an Ibis port.
  }];

  let extraClassDeclaration = [{
    Type getPortType() {
      return getPortTypeAttr().getValue();
    }
  }];

  let builders = [
    TypeBuilder<(ins "Type":$t, "ibis::Direction":$d), [{
      return $_get($_ctxt, TypeAttr::get(t), d);
    }]>
  ];
}

#endif // CIRCT_DIALECT_IBIS_IBISTYPES_TD
