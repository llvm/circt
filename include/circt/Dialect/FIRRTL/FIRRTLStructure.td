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

def FModuleOp : FIRRTLOp<"module", [IsolatedFromAbove, FunctionLike, Symbol, 
                                    SingleBlock, NoTerminator, 
                                    DeclareOpInterfaceMethods<FModuleLike>,
                                    HasParent<"CircuitOp">]> {
  let summary = "FIRRTL Module";
  let description = [{
    The "firrtl.module" operation represents a Verilog module, including a given
    name, a list of ports, and a body that represents the connections within
    the module.
  }];
  let arguments =
            (ins StrArrayAttr:$portNames,
                 DefaultValuedAttr<AnnotationArrayAttr, "{}">:$annotations,
                 DefaultValuedAttr<PortAnnotationsAttr, "{}">:$portAnnotations);

  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name, "ArrayRef<PortInfo>":$ports,
               CArg<"ArrayAttr","ArrayAttr()">:$annotations)>
  ];

  let extraClassDeclaration = [{
    using FunctionLike::end;
    using FunctionLike::front;
    using FunctionLike::getBody;

    // Return the block argument for the port with the specified index.
    BlockArgument getPortArgument(size_t portNumber);

    // TODO(mlir): FunctionLike shouldn't produce a getBody() helper, it is
    // squatting on the name.
    Block *getBodyBlock() { return &body().front(); }

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

  private:
    // This trait needs access to the hooks defined below.
    friend class OpTrait::FunctionLike<FModuleOp>;

    /// Returns the number of arguments, implementing OpTrait::FunctionLike.
    unsigned getNumFuncArguments() { return getBodyBlock()->getNumArguments(); }
    /// Returns the number of results, implementing OpTrait::FunctionLike.
    unsigned getNumFuncResults() { return getType().getResults().size(); }

    /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
    /// attribute is present and checks if it holds a function type.  Ensures
    /// getType, getNumFuncArguments, and getNumFuncResults can be called
    ///  safely.
    LogicalResult verifyType() {
      auto type = getTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError("requires '" + getTypeAttrName() +
                           "' attribute of function type");
      return success();
    }
  public:
  }];

  let printer = "return ::print$cppClass(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";
  let verifier = "return ::verifyFModuleOp(*this);";
}

def FExtModuleOp : FIRRTLOp<"extmodule", 
      [IsolatedFromAbove, FunctionLike, Symbol, 
       DeclareOpInterfaceMethods<FModuleLike>, HasParent<"CircuitOp">]> {
  let summary = "FIRRTL extmodule";
  let description = [{
    The "firrtl.extmodule" operation represents an external reference to a
    Verilog module, including a given name and a list of ports.
  }];
  let arguments = (ins StrArrayAttr:$portNames,
                   OptionalAttr<StrAttr>:$defname,
                   OptionalAttr<DictionaryAttr>:$parameters,
                   DefaultValuedAttr<AnnotationArrayAttr, "{}">:$annotations,
                   DefaultValuedAttr<PortAnnotationsAttr, "{}">:$portAnnotations
                  );
  let results = (outs);
  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name,
                      "ArrayRef<PortInfo>":$ports,
                      CArg<"StringRef", "StringRef()">:$defnamAttr,
                      CArg<"ArrayAttr", "ArrayAttr()">:$annotations)>
  ];

  let extraClassDeclaration = [{
  private:
    // This trait needs access to the hooks defined below.
    friend class OpTrait::FunctionLike<FExtModuleOp>;

    /// Returns the number of arguments, implementing OpTrait::FunctionLike.
    unsigned getNumFuncArguments() { return getType().getInputs().size(); }

    /// Returns the number of results, implementing OpTrait::FunctionLike.
    unsigned getNumFuncResults() { return getType().getResults().size(); }

    /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
    /// attribute is present and checks if it holds a function type.  Ensures
    /// getType, getNumFuncArguments, and getNumFuncResults can be called
    ///  safely.
    LogicalResult verifyType() {
      auto type = getTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError("requires '" + getTypeAttrName() +
                           "' attribute of function type");
      return success();
    }
  public:
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
  let arguments = (ins SymbolNameAttr:$sym_name, SymbolArrayAttr:$modpath, 
                      StringArrayAttr:$namepath);
  let results = (outs);
  let assemblyFormat = [{ $sym_name $modpath $namepath attr-dict}];
}

