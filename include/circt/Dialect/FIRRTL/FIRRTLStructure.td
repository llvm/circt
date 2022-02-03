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
       NoRegionArguments]> {
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
      assert(!body().empty() && "Unexpected empty 'body' region.");
      Block &bodyBlock = body().front();
      return OpBuilder::atBlockEnd(&bodyBlock);
    }

    /// Return body of this circuit.
    Region &getBodyRegion();
    Block *getBody();

    // Return the main module that is the entry point of the circuit.  This may
    // be either an FModuleOp or an FExtModuleOp.
    Operation *getMainModule();
  }];

  let assemblyFormat = "$name custom<CircuitOpAttrs>(attr-dict) $body";
  let verifier = "return ::verifyCircuitOp(*this);";
}

def FModuleOp : FIRRTLOp<"module", [IsolatedFromAbove, Symbol, SingleBlock,
                                    NoTerminator, HasParent<"CircuitOp">,
                                    DeclareOpInterfaceMethods<FModuleLike>,
                                    OpAsmOpInterface]> {
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
  let builders = [
    OpBuilder<(ins "StringAttr":$name, "ArrayRef<PortInfo>":$ports,
               CArg<"ArrayAttr","ArrayAttr()">:$annotations)>
  ];

  let extraClassDeclaration = [{
    Block *getBody() { return &body().front(); }

    using iterator = Block::iterator;
    iterator begin() { return getBody()->begin(); }
    iterator end() { return getBody()->end(); }

    Block::BlockArgListType getArguments() {
      return getBody()->getArguments();
    }

    // Return the block argument for the port with the specified index.
    BlockArgument getArgument(size_t portNumber);

    OpBuilder getBodyBuilder() {
      assert(!body().empty() && "Unexpected empty 'body' region.");
      Block &bodyBlock = body().front();
      return OpBuilder::atBlockEnd(&bodyBlock);
    }

    /// Inserts the given ports.
    void insertPorts(ArrayRef<std::pair<unsigned, PortInfo>> ports);

    /// Erases the ports listed in `portIndices`.  `portIndices` is expected to
    /// be in order and unique.
    void erasePorts(ArrayRef<unsigned> portIndices);

    void getAsmBlockArgumentNames(mlir::Region &region,
                                  mlir::OpAsmSetValueNameFn setNameFn);
  }];

  let printer = "return ::print$cppClass(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";
}

def FExtModuleOp : FIRRTLOp<"extmodule",
      [IsolatedFromAbove, Symbol, HasParent<"CircuitOp">,
       DeclareOpInterfaceMethods<FModuleLike>, OpAsmOpInterface]> {
  let summary = "FIRRTL extmodule";
  let description = [{
    The "firrtl.extmodule" operation represents an external reference to a
    Verilog module, including a given name and a list of ports.
  }];
  let arguments = (ins
                   OptionalAttr<StrAttr>:$defname,
                   ParamDeclArrayAttr:$parameters,
                   DefaultValuedAttr<AnnotationArrayAttr, "ArrayAttr()">:$annotations
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

  let printer = "return ::print$cppClass(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";
  let verifier = "return ::verifyFExtModuleOp(*this);";
}

def NonLocalAnchor : FIRRTLOp<"nla",
      [IsolatedFromAbove, Symbol, HasParent<"CircuitOp">]> {
  let summary = "Anchor for non-local annotations";
  let description = [{
    The "firrtl.nla" operation represents a common point for non-local (path)
    annotations to anchor to in the global scope.  This lets components of the
    path point to a common entity.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name, NameRefArrayAttr:$namepath);
  let results = (outs);
  let assemblyFormat = [{ $sym_name $namepath attr-dict}];
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

    /// Returns true if this NLA targets a module or instance of a module (as
    /// opposed to an instance's port or something inside an instance).
    bool isModule();

    /// Returns true if this NLA targets something inside a module (as opposed
    /// to a module or an instance of a module);
    bool isComponent();
    }];
}
