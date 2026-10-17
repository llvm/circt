//===- SVTypeDecl.td - SV type declaration ops -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the ops for SystemVerilog type declartions.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SV_SVTYPEDECL_TD
#define CIRCT_DIALECT_SV_SVTYPEDECL_TD

include "circt/Dialect/HW/HWAttributes.td"
include "circt/Dialect/HW/HWOpInterfaces.td"
include "circt/Dialect/HW/HWTypes.td"
include "circt/Dialect/SV/SVDialect.td"
include "circt/Dialect/SV/SVTypes.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// Package operations
//===----------------------------------------------------------------------===//

class SVPackageOp<string mnemonic, list<Trait> traits = []> :
    SVOp<mnemonic, traits # [
    SymbolName, SymbolVisibility, Symbol, SymbolTable, TypeScopeLike,
    SingleBlock, NoTerminator, NoRegionArguments, IsolatedFromAbove,
    HasParent<"mlir::ModuleOp">
]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name $body attr-dict";

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    Block *getBodyBlock() { return &getBody().front(); }
  }];
}

def PackageOp : SVPackageOp<"package", [Emittable]> {
  let summary = "SystemVerilog package containing type declarations";
  let description = [{
    A named, top-level SystemVerilog package containing only `hw.typedecl`
    operations. Its declarations are referenced using `hw.typealias` with a
    symbol reference of the form `@package::@type`. Empty packages are allowed.

    Use `sv.package.extern` when the package's SystemVerilog definition is
    provided externally and should not be emitted.

    The body is currently restricted to type declarations because the lookup,
    legalization, and emission infrastructure for other declarations is not yet
    prepared to handle them in a package's nested symbol table. This restriction
    can be relaxed to support declarations such as functions and parameters as
    that infrastructure is extended; it is not a limitation of SystemVerilog
    packages.
  }];
}

def PackageExternOp : SVPackageOp<"package.extern"> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<StrAttr>:$verilogName
  );
  let summary = "Externally defined SystemVerilog package";
  let description = [{
    A named, top-level SystemVerilog package whose implementation is provided
    externally. Like `sv.package`, its body contains only `hw.typedecl`
    operations, which are referenced using `hw.typealias` with a symbol
    reference of the form `@package::@type`. Empty packages are allowed.

    The declarations remain available for type resolution, but ExportVerilog
    does not emit the package or create an output file for it. Like
    `hw.module.extern`, it is not emittable and cannot be an `emit.ref` target.

    The `verilogName` string attribute specifies the externally defined
    SystemVerilog package name, independently of the MLIR symbol name. It is
    required for non-public packages so that renaming their MLIR symbols does
    not change references to the external package. For public packages it is
    optional and defaults to the symbol name. The Verilog name must be valid
    SystemVerilog and is not renamed.

    Typedef and enum member names use the same legalization as `sv.package`,
    including keyword avoidance, reserved names, and member-name collisions.
    This allows a CIRCT-generated package to be represented by changing
    `sv.package` to `sv.package.extern`, transferring its `hw.verilogName` to
    `verilogName`, and using the same lowering options and reserved names.
    Only the external package's own Verilog name remains fixed.

    Example:

    ```mlir
    sv.package.extern @ExtTypes {
      hw.typedecl @word : i8
    }
    hw.module.extern @ExtMod(in %arg: !hw.typealias<@ExtTypes::@word, i8>)
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Verbatim source operation
//===----------------------------------------------------------------------===//

def SVVerbatimSourceOp
    : SVOp<"verbatim.source", [SymbolName, SymbolVisibility, Symbol,
                                HasParent<"mlir::ModuleOp">,
                               DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Verbatim verilog source";
  let description = [{
    The "sv.verbatim.source" operation represents verbatim verilog definition for
    a module which may have parameters. Concrete parametrizations and their ports
    are represented by a `hw.module.extern`.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    StrAttr:$content,
    OutputFileAttr:$output_file,
    ParamDeclArrayAttr:$parameters,
    OptionalAttr<SymbolRefArrayAttr>:$additional_files,
    StrAttr:$verilogName);

  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Verbatim module operation
//===----------------------------------------------------------------------===//

def SVVerbatimModuleOp
    : SVOp<"verbatim.module", [DeclareOpInterfaceMethods<PortList>,
                               DeclareOpInterfaceMethods<HWModuleLike>,
                               DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                               SymbolName, SymbolVisibility, Symbol,
                               HasParent<"mlir::ModuleOp">]> {
  let summary = "Verbatim verilog module";
  let description = [{
    The "sv.verbatim.module" operation represents a the module interface of a
    verbatim verilog module. It references an "sv.verbatim.source" operation
    that contains the actual verilog source code.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    TypeAttrOf<ModuleType>:$module_type,
    OptionalAttr<DictArrayAttr>:$per_port_attrs,
    OptionalAttr<LocationArrayAttr>:$port_locs,
    ParamDeclArrayAttr:$parameters,
    FlatSymbolRefAttr:$source,
    OptionalAttr<StrAttr>:$verilogName);

  let results = (outs);

  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "StringAttr":$name,
                   "ArrayRef<hw::PortInfo>":$ports,
                   "FlatSymbolRefAttr":$source,
                   CArg<"ArrayAttr", "{}">:$parameters,
                   CArg<"StringAttr", "{}">:$verilogName), [{
      SmallVector<hw::ModulePort> modulePorts;
      SmallVector<Attribute> portAttrs, portLocs;
      for (const auto &port : ports) {
        modulePorts.push_back({port.name, port.type, port.dir});
        portAttrs.push_back(port.attrs ? port.attrs : $_builder.getDictionaryAttr({}));
        portLocs.push_back(port.loc ? static_cast<Location>(port.loc) : $_builder.getUnknownLoc());
      }
      auto moduleType = hw::ModuleType::get($_builder.getContext(), modulePorts);
      build($_builder, $_state, name, /*sym_visibility=*/{},
            TypeAttr::get(moduleType),
            $_builder.getArrayAttr(portAttrs), $_builder.getArrayAttr(portLocs),
            parameters, source, verilogName);
    }]>
  ];

  let hasVerifier = 1;
}

#endif // CIRCT_DIALECT_SV_SVTYPEDECL_TD
