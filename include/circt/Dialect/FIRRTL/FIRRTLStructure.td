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

include "circt/Dialect/HW/HWTypes.td"
include "circt/Types.td"

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

    // Return the main module that is the entry point of the circuit.  This may
    // be either an FModuleOp or an FExtModuleOp.  A SymbolTable is optional.
    FModuleLike getMainModule(mlir::SymbolTable* symtbl = nullptr);
  }];

  let assemblyFormat = "$name custom<CircuitOpAttrs>(attr-dict) $body";
  let hasRegionVerifier = 1;
}

def FModuleOp : FIRRTLOp<"module", [IsolatedFromAbove, Symbol, SingleBlock,
                                    NoTerminator, HasParent<"CircuitOp">,
                                    DeclareOpInterfaceMethods<FModuleLike>,
                                    DeclareOpInterfaceMethods<HWModuleLike>,
                                    OpAsmOpInterface, InnerSymbolTable]> {
  let summary = "FIRRTL Module";
  let description = [{
    The "firrtl.module" operation represents a Verilog module, including a given
    name, a list of ports, and a body that represents the connections within
    the module.
  }];
  let arguments =
            (ins DefaultValuedAttr<AnnotationArrayAttr, "{}">:$annotations);

  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name, "ArrayRef<PortInfo>":$ports,
               CArg<"ArrayAttr","ArrayAttr()">:$annotations)>
  ];

  let extraClassDeclaration = [{
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
  }];

}

def FExtModuleOp : FIRRTLOp<"extmodule",
      [IsolatedFromAbove, Symbol, HasParent<"CircuitOp">, OpAsmOpInterface,
       DeclareOpInterfaceMethods<FModuleLike>, InnerSymbolTable,
       DeclareOpInterfaceMethods<HWModuleLike>]> {
  let summary = "FIRRTL extmodule";
  let description = [{
    The "firrtl.extmodule" operation represents an external reference to a
    Verilog module, including a given name and a list of ports.
    LowerAnnotations can add RefType ports to it. Each RefType port
    must have a corresponding entry in the internalPaths attribute.
    The internalPaths attribute is used to represent opaque internal paths
    into the external module, to be used for generating XMRs. Each RefType
    port must be removed by LowerXMR pass.
  }];
  let arguments = (ins
                   OptionalAttr<StrAttr>:$defname,
                   ParamDeclArrayAttr:$parameters,
                   DefaultValuedAttr<AnnotationArrayAttr,
                   "ArrayAttr()">:$annotations,
                   DefaultValuedAttr<ArrayAttr, "{}">:$internalPaths
                  );
  let results = (outs);
  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name,
                      "ArrayRef<PortInfo>":$ports,
                      CArg<"StringRef", "StringRef()">:$defnamAttr,
                      CArg<"ArrayAttr", "ArrayAttr()">:$annotations,
                      CArg<"ArrayAttr", "ArrayAttr()">:$parameters)>
  ];

  let extraClassDeclaration = [{
    void getAsmBlockArgumentNames(mlir::Region &region,
                                  mlir::OpAsmSetValueNameFn setNameFn);
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def FMemModuleOp : FIRRTLOp<"memmodule",
      [IsolatedFromAbove, Symbol, HasParent<"CircuitOp">, OpAsmOpInterface,
       DeclareOpInterfaceMethods<FModuleLike>, InnerSymbolTable,
       DeclareOpInterfaceMethods<HWModuleLike>]> {
  let summary = "FIRRTL Generated Module";
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
         ArrayAttr:$extraPorts, AnnotationArrayAttr:$annotations);
  let results = (outs);
  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name, "ArrayRef<PortInfo>":$ports,
                   "uint32_t":$numReadPorts,  "uint32_t":$numWritePorts,
                   "uint32_t":$numReadWritePorts, "uint32_t":$dataWidth,
                   "uint32_t":$maskBits, "uint32_t":$readLatency,
                   "uint32_t":$writeLatency, "uint64_t":$depth,
                   CArg<"ArrayAttr", "ArrayAttr()">:$annotations)>
  ];

  let extraClassDeclaration = [{
    /// Return true if this memory has a mask.
    bool isMasked() { return getMaskBits() > 1; }
    void getAsmBlockArgumentNames(mlir::Region &region,
                                  mlir::OpAsmSetValueNameFn setNameFn);
  }];

  let hasCustomAssemblyFormat = 1;
}

def HierPathOp : FIRRTLOp<"hierpath",
      [IsolatedFromAbove, Symbol,
       DeclareOpInterfaceMethods<InnerRefUserOpInterface>,
       HasParent<"CircuitOp">]> {
  let summary = "Hierarchical path specification";
  let description = [{
    The "firrtl.hierpath" operation represents a path through the hierarchy.
    This is used to specify namable things for use in other operations, for
    example in verbatim substitution. Non-local annotations also use these.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name, NameRefArrayAttr:$namepath);
  let results = (outs);
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    /// Drop the module from the namepath. If its a InnerNameRef, then drop
    /// the Module-Instance pair, else drop the final module from the namepath.
    /// Return true if any update is made.
    bool dropModule(StringAttr moduleToDrop);

    /// Inline the module in the namepath.
    /// Update the symbol name for the inlined module instance, by prepending
    /// the symbol name of the instance at which the inling was done.
    /// Return true if any update is made.
    bool inlineModule(StringAttr moduleToDrop);

    /// Replace the oldMod module with newMod module in the namepath of the NLA.
    /// Return true if any update is made.
    bool updateModule(StringAttr oldMod, StringAttr newMod);

    /// Replace the oldMod module with newMod module in the namepath of the NLA.
    /// Since the module is being updated, the symbols inside the module should
    /// also be renamed. Use the rename Map to update the corresponding
    /// inner_sym names in the namepath. Return true if any update is made.
    bool updateModuleAndInnerRef(StringAttr oldMod, StringAttr newMod,
         const llvm::DenseMap<StringAttr, StringAttr> &innerSymRenameMap);

    /// Truncate the namepath for this NLA, at atMod module.
    /// If includeMod is false, drop atMod and beyond, else include it and drop
    /// everything after it.
    /// Return true if any update is made.
    bool truncateAtModule(StringAttr atMod, bool includeMod = true);

    /// Return just the module part of the namepath at a specific index.
    StringAttr modPart(unsigned i);

    /// Return the root module.
    StringAttr root();

    /// Return just the reference part of the namepath at a specific index.
    /// This will return an empty attribute if this is the leaf and the leaf is
    /// a module.
    StringAttr refPart(unsigned i);

    /// Return the leaf reference.  This returns an empty attribute if the leaf
    /// reference is a module.
    StringAttr ref();

    /// Return the leaf Module.
    StringAttr leafMod();

    /// Returns true, if the NLA path contains the module.
    bool hasModule(StringAttr modName);

    /// Returns true, if the NLA path contains the InnerSym {modName, symName}.
    bool hasInnerSym(StringAttr modName, StringAttr symName) const;

    /// Returns true if this NLA targets a module or instance of a module (as
    /// opposed to an instance's port or something inside an instance).
    bool isModule();

    /// Returns true if this NLA targets something inside a module (as opposed
    /// to a module or an instance of a module);
    bool isComponent();
    }];
}
