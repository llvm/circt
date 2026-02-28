//===- KanagawaTypes.td - Definition of Kanagawa dialect types ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_KANAGAWA_KANAGAWATYPES_TD
#define CIRCT_DIALECT_KANAGAWA_KANAGAWATYPES_TD

include "circt/Dialect/Kanagawa/KanagawaDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class KanagawaTypeDef<string name> : TypeDef<KanagawaDialect, name> { }

def ScopeRefType : KanagawaTypeDef<"ScopeRef"> {
  let mnemonic = "scoperef";
  let description = [{
    A reference to an Kanagawa scope. May be either a reference to a specific
    scope (given a `$scopeName` argument) or an opaque reference.
  }];

  let parameters = (ins OptionalParameter<"::circt::hw::InnerRefAttr">:$scopeRef);
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    bool isOpaque() {
      return !getScopeRef();
    }
  }];

  let builders = [
    // Builds an opaque (unresolved) class reference.
    TypeBuilder<(ins), [{
      return $_get($_ctxt, nullptr);
    }]>
  ];
}

def AnyScopeRefType : Type<
  CPred<"llvm::isa<kanagawa::ScopeRefType>($_self)">,
  "must be a !dc.classref<T?> type",
  "kanagawa::ScopeRefType">{
}

def OpaqueScopeRefType : Type<
  CPred<"kanagawa::isOpaqueScopeRefType($_self)">,
  "must be a !dc.classref<> type">,
  BuildableType<"$_builder.getType<kanagawa::ScopeRefType>()"> {
}

def Input : I32EnumAttrCase<"Input", 0, "in">;
def Output : I32EnumAttrCase<"Output", 1, "out">;
def Direction : I32EnumAttr<"Direction", "Kanagawa port direction",
  [Input, Output]> {
  let cppNamespace = "::circt::kanagawa";
}

def PortRefType : KanagawaTypeDef<"PortRef"> {
  let mnemonic = "portref";
  let parameters = (ins "TypeAttr":$portTypeAttr, "kanagawa::Direction":$direction);
  let assemblyFormat = "`<` $direction $portTypeAttr  `>`";
  let description = [{
    A reference to an Kanagawa port.
  }];

  let extraClassDeclaration = [{
    Type getPortType() {
      return getPortTypeAttr().getValue();
    }
  }];

  let builders = [
    TypeBuilder<(ins "Type":$t, "kanagawa::Direction":$d), [{
      return $_get($_ctxt, TypeAttr::get(t), d);
    }]>
  ];
}

#endif // CIRCT_DIALECT_KANAGAWA_KANAGAWATYPES_TD
