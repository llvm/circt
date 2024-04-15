//===- FIRRTLStructure.td - Circuit and Module Ops ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the FIRRTL Circuit and Module MLIR ops.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLSTRUCTURE_TD
#define CIRCT_DIALECT_FIRRTL_FIRRTLSTRUCTURE_TD

include "FIRRTLAttributes.td"
include "FIRRTLDialect.td"
include "FIRRTLEnums.td"
include "FIRRTLOpInterfaces.td"
include "circt/Dialect/HW/HWOpInterfaces.td"
include "circt/Dialect/HW/HWTypes.td"
include "circt/Types.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def CircuitOp : FIRRTLOp<"circuit",
      [IsolatedFromAbove, SymbolTable, SingleBlock, NoTerminator,
       NoRegionArguments, InnerRefNamespace]> {
  let summary = "FIRRTL Circuit";
  let description = [{
    The "firrtl.circuit" operation represents an overall Verilog circuit,
    containing a list of modules.
  }];
  let arguments = (ins StrAttr:$name,
                   DefaultValuedAttr<AnnotationArrayAttr, "{}">:$annotations);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name,
               CArg<"ArrayAttr","ArrayAttr()">:$annotations)>
  ];

  let extraClassDeclaration = [{
    OpBuilder getBodyBuilder() {
      assert(!getBody().empty() && "Unexpected empty 'body' region.");
      Block &bodyBlock = getBody().front();
      return OpBuilder::atBlockEnd(&bodyBlock);
    }

    /// Return body of this circuit.
    Block *getBodyBlock();
  }];

  let assemblyFormat = "$name `` custom<CircuitOpAttrs>(attr-dict) $body";
  let hasRegionVerifier = 1;
}

class FIRRTLModuleLike<string mnemonic, list<Trait> traits = []> :
  FIRRTLOp<mnemonic, traits # [
    IsolatedFromAbove, Symbol, HasParent<"CircuitOp">,
    DeclareOpInterfaceMethods<FModuleLike>,
    DeclareOpInterfaceMethods<PortList>,
    DeclareOpInterfaceMethods<InstanceGraphModuleOpInterface>,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    OpAsmOpInterface,
    InnerSymbolTable
  ]> {

  /// Additional class definitions inside the module op.
  code extraModuleClassDefinition = [{}];
  code extraModuleClassDeclaration = [{}];

  let extraClassDeclaration = extraModuleClassDeclaration # [{
  }];

  let extraClassDefinition = extraModuleClassDefinition # [{

    size_t $cppClass::getNumPorts() {
      return getPortTypesAttr().size();
    }

    size_t $cppClass::getNumInputPorts() {
      size_t count = 0;
      for (size_t i = 0, e = getNumPorts(); i < e; ++i)
        if (getPortDirection(i) == Direction::In)
          ++count;
      return count;
    }

    size_t $cppClass::getNumOutputPorts() {
      size_t count = 0;
      for (size_t i = 0, e = getNumPorts(); i < e; ++i)
        if (getPortDirection(i) == Direction::Out)
          ++count;
      return count;
    }

    size_t $cppClass::getPortIdForInputId(size_t idx) {
      for (size_t i = 0, e = getNumPorts(); i < e; ++i)
        if (getPortDirection(i) == Direction::In) {
          if (!idx)
            return i;
          --idx;
        }
      assert(0 && "Out of bounds input port id");
      return ~0ULL;
    }

    size_t $cppClass::getPortIdForOutputId(size_t idx) {
      for (size_t i = 0, e = getNumPorts(); i < e; ++i)
        if (getPortDirection(i) == Direction::Out) {
          if (!idx)
            return i;
          --idx;
        }
      assert(0 && "Out of bounds input port id");
      return ~0ULL;
    }
  }];

}


def FModuleOp : FIRRTLModuleLike<"module", [SingleBlock, NoTerminator]> {
  let summary = "FIRRTL Module";
  let description = [{
    The "firrtl.module" operation represents a Verilog module, including a given
    name, a list of ports, and a body that represents the connections within
    the module.
  }];
  let arguments =
      (ins ConventionAttr:$convention,
           DenseBoolArrayAttr:$portDirections,
           ArrayRefAttr:$portLocations,
           ArrayRefAttr:$portAnnotations,
           ArrayRefAttr:$portSyms,
           ArrayRefAttr:$portNames,
           ArrayRefAttr:$portTypes,
           DefaultValuedAttr<AnnotationArrayAttr, "{}">:$annotations,
           DefaultValuedAttr<LayerArrayAttr, "{}">:$layers
      );

  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name, "ConventionAttr":$convention,
               "ArrayRef<PortInfo>":$ports,
               CArg<"ArrayAttr", "ArrayAttr()">:$annotations,
               CArg<"ArrayAttr", "ArrayAttr()">:$layers)>
  ];

  let extraModuleClassDeclaration = [{
    Block *getBodyBlock() { return &getBody().front(); }

    using iterator = Block::iterator;
    iterator begin() { return getBodyBlock()->begin(); }
    iterator end() { return getBodyBlock()->end(); }

    Block::BlockArgListType getArguments() {
      return getBodyBlock()->getArguments();
    }

    // Return the block argument for the port with the specified index.
    BlockArgument getArgument(size_t portNumber);

    OpBuilder getBodyBuilder() {
      assert(!getBody().empty() && "Unexpected empty 'body' region.");
      Block &bodyBlock = getBody().front();
      return OpBuilder::atBlockEnd(&bodyBlock);
    }

    void getAsmBlockArgumentNames(mlir::Region &region,
                                  mlir::OpAsmSetValueNameFn setNameFn);
    ArrayAttr getParameters();
  }];

}

def FExtModuleOp : FIRRTLModuleLike<"extmodule"> {
  let summary = "FIRRTL external module";
  let description = [{
    The "firrtl.extmodule" operation represents an external reference to a
    Verilog module, including a given name and a list of ports.
    LowerAnnotations can add RefType ports to it. Each RefType port
    must have a corresponding entry in the internalPaths attribute.
    The internalPaths attribute is used to represent opaque internal paths
    into the external module, to be used for generating XMRs. Each RefType
    port must be removed by LowerXMR pass.
  }];
  let arguments = 
      (ins OptionalAttr<StrAttr>:$defname,
           ParamDeclArrayAttr:$parameters,
           ConventionAttr:$convention,
           DenseBoolArrayAttr:$portDirections,
           ArrayRefAttr:$portLocations,
           ArrayRefAttr:$portAnnotations,
           ArrayRefAttr:$portSyms,
           ArrayRefAttr:$portNames,
           ArrayRefAttr:$portTypes,
           DefaultValuedAttr<AnnotationArrayAttr, "{}">:$annotations,
           DefaultValuedAttr<LayerArrayAttr, "{}">:$layers,
           OptionalAttr<InternalPathArrayAttr>:$internalPaths
      );
  let results = (outs);
  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name,
                      "ConventionAttr":$convention,
                      "ArrayRef<PortInfo>":$ports,
                      CArg<"StringRef", "StringRef()">:$defnamAttr,
                      CArg<"ArrayAttr", "ArrayAttr()">:$annotations,
                      CArg<"ArrayAttr", "ArrayAttr()">:$parameters,
                      CArg<"ArrayAttr", "ArrayAttr()">:$internalPaths,
                      CArg<"ArrayAttr", "ArrayAttr()">:$layers)>
  ];

  let extraModuleClassDeclaration = [{
    void getAsmBlockArgumentNames(mlir::Region &region,
                                  mlir::OpAsmSetValueNameFn setNameFn);
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def FIntModuleOp : FIRRTLModuleLike<"intmodule"> {
  let summary = "FIRRTL intrinsic module";
  let description = [{
    The "firrtl.intmodule" operation represents a compiler intrinsic.
  }];
  let arguments = 
      (ins StrAttr:$intrinsic,
           ParamDeclArrayAttr:$parameters,
           DenseBoolArrayAttr:$portDirections,
           ArrayRefAttr:$portLocations,
           ArrayRefAttr:$portAnnotations,
           ArrayRefAttr:$portSyms,
           ArrayRefAttr:$portNames,
           ArrayRefAttr:$portTypes,
           DefaultValuedAttr<AnnotationArrayAttr, "ArrayAttr()">:$annotations,
           DefaultValuedAttr<LayerArrayAttr, "{}">:$layers,
           OptionalAttr<InternalPathArrayAttr>:$internalPaths
      );
  let results = (outs);
  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name,
                      "ArrayRef<PortInfo>":$ports,
                      "StringRef":$intrinsicNameStr,
                      CArg<"ArrayAttr", "ArrayAttr()">:$annotations,
                      CArg<"ArrayAttr", "ArrayAttr()">:$parameters,
                      CArg<"ArrayAttr", "ArrayAttr()">:$internalPaths,
                      CArg<"ArrayAttr", "ArrayAttr()">:$layers)>
  ];

  let extraModuleClassDeclaration = [{
    void getAsmBlockArgumentNames(mlir::Region &region,
                                  mlir::OpAsmSetValueNameFn setNameFn);
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def FMemModuleOp : FIRRTLModuleLike<"memmodule"> {
  let summary = "FIRRTL Generated Memory Module";
  let description = [{
    The "firrtl.memmodule" operation represents an external reference to a
    memory module. See the "firrtl.mem" op for a deeper explantation of the
    parameters.

    A "firrtl.mem" operation is typically lowered to this operation when they
    are not directly lowered to registers by the compiler.
  }];
  let arguments =
      (ins UI32Attr:$numReadPorts, UI32Attr:$numWritePorts,
           UI32Attr:$numReadWritePorts, UI32Attr:$dataWidth, UI32Attr:$maskBits,
           UI32Attr:$readLatency, UI32Attr:$writeLatency, UI64Attr:$depth,
           ArrayAttr:$extraPorts,
           DenseBoolArrayAttr:$portDirections,
           ArrayRefAttr:$portLocations,
           ArrayRefAttr:$portAnnotations,
           ArrayRefAttr:$portSyms,
           ArrayRefAttr:$portNames,
           ArrayRefAttr:$portTypes,
           AnnotationArrayAttr:$annotations,
           DefaultValuedAttr<LayerArrayAttr, "{}">:$layers
      );
      
  let results = (outs);
  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name, "ArrayRef<PortInfo>":$ports,
                   "uint32_t":$numReadPorts,  "uint32_t":$numWritePorts,
                   "uint32_t":$numReadWritePorts, "uint32_t":$dataWidth,
                   "uint32_t":$maskBits, "uint32_t":$readLatency,
                   "uint32_t":$writeLatency, "uint64_t":$depth,
                   CArg<"ArrayAttr", "ArrayAttr()">:$annotations,
                   CArg<"ArrayAttr", "ArrayAttr()">:$layers)>
  ];

  let extraModuleClassDeclaration = [{
    /// Return true if this memory has a mask.
    bool isMasked() { return getMaskBits() > 1; }
    void getAsmBlockArgumentNames(mlir::Region &region,
                                  mlir::OpAsmSetValueNameFn setNameFn);
    ArrayAttr getParameters();
  }];

  let hasCustomAssemblyFormat = 1;
}

def ClassOp : FIRRTLModuleLike<"class", [
  SingleBlock, NoTerminator,
  DeclareOpInterfaceMethods<FModuleLike, [
    // Class Ops do not support port annotations or enabled layers.
    // Override these methods to return an empty array attr.
    "getPortAnnotationsAttr",
    "getLayersAttr"]>,
  DeclareOpInterfaceMethods<ClassLike>,
  DeclareOpInterfaceMethods<Symbol, ["canDiscardOnUseEmpty"]>]> {
  let summary = "FIRRTL Class";
  let description = [{
    The "firrtl.class" operation defines a class of property-only objects,
    including a given name, a list of ports, and a body that represents the
    connections within the class.

    A class may only have property ports, and its body may only be ops that act
    on properties, such as propassign ops.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       DenseBoolArrayAttr:$portDirections,
                       ArrayRefAttr:$portNames, ArrayRefAttr:$portTypes,
                       ArrayRefAttr:$portSyms, ArrayRefAttr:$portLocations);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins
      "StringAttr":$name,
      "ArrayRef<PortInfo>":$ports)>];

  let extraModuleClassDeclaration = [{
    Block *getBodyBlock() { return &getBody().front(); }

    using iterator = Block::iterator;
    iterator begin() { return getBodyBlock()->begin(); }
    iterator end() { return getBodyBlock()->end(); }

    Block::BlockArgListType getArguments() {
      return getBodyBlock()->getArguments();
    }

    // Return the block argument for the port with the specified index.
    BlockArgument getArgument(size_t portNumber);

    OpBuilder getBodyBuilder() {
      assert(!getBody().empty() && "Unexpected empty 'body' region.");
      Block &bodyBlock = getBody().front();
      return OpBuilder::atBlockEnd(&bodyBlock);
    }

    void getAsmBlockArgumentNames(mlir::Region &region,
                                  mlir::OpAsmSetValueNameFn setNameFn);
    ArrayAttr getParameters();
  }];
}

def ExtClassOp : FIRRTLModuleLike<"extclass", [
  DeclareOpInterfaceMethods<FModuleLike, [
    // ExtClassOps do not support port annotations or enabled layers.
    // Override these methods to return an empty array attr.
    "getPortAnnotationsAttr",
    "getLayersAttr"]>,
  DeclareOpInterfaceMethods<ClassLike>,
  DeclareOpInterfaceMethods<Symbol, ["canDiscardOnUseEmpty"]>]> {
  let summary = "FIRRTL external class";
  let description = [{
    The "firrtl.extclass" operation represents a reference to an external
    firrtl class, and includes a given name, as well as a list of ports.
    Just as usual firrtl.class definitions, An ext.class may only have property
    ports.

    example:
    ```mlir
    firrtl.extclass @MyImportedClass(in in_str: !firrtl.string, out out_str: !firrtl.string)
    ```
  }];
  let arguments = (ins SymbolNameAttr:$sym_name, 
                       DenseBoolArrayAttr:$portDirections,
                       ArrayRefAttr:$portNames, ArrayRefAttr:$portTypes,
                       ArrayRefAttr:$portSyms, ArrayRefAttr:$portLocations);
  let results = (outs);
  let regions = (region AnyRegion:$body);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "StringAttr":$name,
      "ArrayRef<PortInfo>":$ports
    )>
  ];
  let extraModuleClassDeclaration = [{
    void getAsmBlockArgumentNames(mlir::Region &region,
                                  mlir::OpAsmSetValueNameFn setNameFn);

    ArrayAttr getParameters();
  }];
  let hasCustomAssemblyFormat = 1;
}

def LayerOp : FIRRTLOp<
  "layer",
  [IsolatedFromAbove, Symbol, SymbolTable, SingleBlock, NoTerminator,
   ParentOneOf<["firrtl::CircuitOp", "firrtl::LayerOp"]>]
> {
  let summary = "A layer definition";
  let description = [{
    The `firrtl.layer` operation defines a layer and a lowering convention for
    that layer.  Layers are a feature of FIRRTL that add verification or
    debugging code to an existing module at runtime.

    A `firrtl.layer` operation only defines the layer and any layers nested
    under it.  Functionality is added to modules using the `firrtl.group`
    operation.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name, LayerConventionAttr:$convention);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $sym_name $convention attr-dict-with-keyword $body
  }];
}

def OptionOp : FIRRTLOp<"option", [
    IsolatedFromAbove,
    Symbol,
    SymbolTable,
    NoTerminator,
    HasParent<"firrtl::CircuitOp">
]> {
  let summary = "An option group definition";
  let description = [{
    The `firrtl.option` operation defines a specializable parameter with a
    known set of values, represented by the `firrtl.option_case` operations
    nested underneath.

    Operations which support specialization reference the option and its
    cases to define the specializations they support.

    Example:
    ```mlir

    firrtl.circuit {
      firrtl.option @Target {
        firrtl.option_case @FPGA
        firrtl.option_case @ASIC
      }
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $sym_name attr-dict-with-keyword $body
  }];
}

def OptionCaseOp : FIRRTLOp<
  "option_case", [Symbol, HasParent<"firrtl::OptionOp">]
> {
  let summary = "A configuration option value definition";
  let description = [{
    `firrtl.option_case` defines an acceptable value to be provided for an
    option. Ops reference it to define their behavior when this case is active.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let results = (outs);
  let assemblyFormat = [{
    $sym_name attr-dict
  }];
}

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLSTRUCTURE_TD
