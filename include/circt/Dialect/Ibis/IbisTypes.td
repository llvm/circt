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

def NestedSymProperties : AttrDef<IbisDialect, "NestedSymProperties"> {
  let mnemonic = "nestedSymProps";
  let parameters = (ins
    "::mlir::StringAttr":$symName,
    DefaultValuedParameter<"::mlir::StringAttr", "public">:$sym_visibility
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$sym),[{
      return get(sym.getContext(), sym,
                        mlir::StringAttr::get(sym.getContext(), "public") );
    }]>
  ];
  let hasCustomAssemblyFormat = 1;
  // The assembly format is as follows:
  // "`<` `@` $name `,`$sym_visibility `>`";
  let genVerifyDecl = 1;
}

def NestedSymAttr : AttrDef<IbisDialect, "NestedSym"> {
  let summary = "Nested symbol definition";
  let description = [{
    Defines the properties of a nested_sym attriute. It specifies the symbol
    name and symbol visibility.
  }];

  let mnemonic = "nestedSym";
  let parameters = (ins
    "::mlir::StringAttr":$symName,
    DefaultValuedParameter<"::mlir::StringAttr", "public">:$sym_visibility
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$sym),[{
      return get(sym.getContext(), sym,
                        mlir::StringAttr::get(sym.getContext(), "public") );
    }]>
  ];
  let hasCustomAssemblyFormat = 1;
  // The assembly format is as follows:
  // "`<` `@` $name `,`$sym_visibility `>`";
  let genVerifyDecl = 1;
}

def NestedRefAttr : AttrDef<IbisDialect, "NestedRef"> {
  let summary = "Refer to a name inside a module";
  let description = [{
    This works like a symbol reference, but to a name inside a module.
  }];
  let mnemonic = "nestedNameRef";
  let parameters = (ins ArrayRefParameter<"::mlir::StringAttr">:$path,
                        "::mlir::StringAttr":$name);

  let hasCustomAssemblyFormat = 1;
}

#endif // CIRCT_DIALECT_IBIS_IBISTYPES_TD
