//===- HWStructure.td - HW structure ops -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for structure.
//
//===----------------------------------------------------------------------===//

/// Ensure symbol is one of the hw module.* types.  Passes on failed lookup, so
/// more basic symbol validation can provide more useful error messages.
def isModuleSymbol : AttrConstraint<
    CPred<"!::mlir::SymbolTable::lookupNearestSymbolFrom("
            "&$_op, $_self.cast<::mlir::FlatSymbolRefAttr>()) || "
          "hw::isAnyModule(::mlir::SymbolTable::lookupNearestSymbolFrom("
            "&$_op, $_self.cast<::mlir::FlatSymbolRefAttr>()))"
    >, "is module like">;

def HWModuleOp : HWOp<"module",
      [IsolatedFromAbove, FunctionLike, Symbol, RegionKindInterface,
       SingleBlockImplicitTerminator<"OutputOp">, HasParent<"mlir::ModuleOp">]>{
  let summary = "HW Module";
  let description = [{
    The "hw.module" operation represents a Verilog module, including a given
    name, a list of ports, and a body that represents the connections within
    the module.
  }];
  let arguments = (ins StrArrayAttr:$argNames, StrArrayAttr:$resultNames);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name, "ArrayRef<PortInfo>":$ports,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "StringAttr":$name, "const ModulePortInfo &":$ports,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
  ];

  let extraClassDeclaration = [{
    using FunctionLike::front;
    using FunctionLike::getBody;

    // Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph;}

    /// Decode information about the input and output ports on this module.
    ModulePortInfo getPorts() {
      return getModulePortInfo(*this);
    }

    /// Return all the module ports merged into one list.
    SmallVector<PortInfo> getAllPorts() {
      return getAllModulePortInfos(*this);
    }

    // TODO(mlir): FunctionLike shouldn't produce a getBody() helper, it is
    // squatting on the name.
    Block *getBodyBlock() { return &body().front(); }

    // Get the module's symbolic name as StringAttr.
    StringAttr getNameAttr() {
      return (*this)->getAttrOfType<StringAttr>(
        ::mlir::SymbolTable::getSymbolAttrName());
    }

    // Get the module's symbolic name.
    StringRef getName() {
      return getNameAttr().getValue();
    }

  private:
    // This trait needs access to the hooks defined below.
    friend class OpTrait::FunctionLike<HWModuleOp>;

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
  let verifier = "return ::verify$cppClass(*this);";
}

def HWModuleExternOp : HWOp<"module.extern",
      [FunctionLike, Symbol, HasParent<"mlir::ModuleOp">]> {
  let summary = "HW external Module";
  let description = [{
    The "hw.module.extern" operation represents an external reference to a
    Verilog module, including a given name and a list of ports.

    The 'verilogName' attribute (when present) specifies the spelling of the
    module name in Verilog we can use.  TODO: This is a hack because we don't
    have proper parameterization in the hw.dialect.  We need a way to represent
    parameterized types instead of just concrete types.
  }];
  let arguments = (ins StrArrayAttr:$argNames, StrArrayAttr:$resultNames,
                       OptionalAttr<StrAttr>:$verilogName);
  let results = (outs);
  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name, "ArrayRef<PortInfo>":$ports,
                   CArg<"StringRef", "StringRef()">:$verilogName,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "StringAttr":$name, "const ModulePortInfo &":$ports,
                   CArg<"StringRef", "StringRef()">:$verilogName,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
  ];

  let extraClassDeclaration = [{
    /// Decode information about the input and output ports on this module.
    ModulePortInfo getPorts() {
      return getModulePortInfo(*this);
    }

    /// Return all the module ports merged into one list.
    SmallVector<PortInfo> getAllPorts() {
      return getAllModulePortInfos(*this);
    }

    /// Return the name to use for the Verilog module that we're referencing
    /// here.  This is typically the symbol, but can be overridden with the
    /// verilogName attribute.
    StringRef getVerilogModuleName();

    /// Return the name to use for the Verilog module that we're referencing
    /// here.  This is typically the symbol, but can be overridden with the
    /// verilogName attribute.
    StringAttr getVerilogModuleNameAttr();

    // Get the module's symbolic name as StringAttr.
    StringAttr getNameAttr() {
      return (*this)->getAttrOfType<StringAttr>(
        ::mlir::SymbolTable::getSymbolAttrName());
    }

    // Get the module's symbolic name.
    StringRef getName() {
      return getNameAttr().getValue();
    }

  private:
    // This trait needs access to the hooks defined below.
    friend class OpTrait::FunctionLike<HWModuleExternOp>;

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
  let verifier = "return ::verify$cppClass(*this);";
}

def HWGeneratorSchemaOp : HWOp<"generator.schema",
      [Symbol, HasParent<"mlir::ModuleOp">]> {
  let summary = "HW Generator Schema declaration";
  let description = [{
    The "hw.generator.schema" operation declares a kind of generated module by
    declaring the schema of meta-data required.
    A generated module instance of a schema is independent of the external
    method of producing it.  It is assumed that for well known schema instances,
    multiple external tools might exist which can process it.  Generator nodes
    list attributes required by hw.module.generated instances.

    For example:
    generator.schema @MEMORY, "Simple-Memory", ["ports", "write_latency", "read_latency"]
    module.generated @mymem, @MEMORY(ports)
      -> (ports) {write_latency=1, read_latency=1, ports=["read","write"]}
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$descriptor,
                       StrArrayAttr:$requiredAttrs);
  let results = (outs);
  let assemblyFormat = [{
      $sym_name `,` $descriptor `,` $requiredAttrs attr-dict
    }];
}

def HWModuleGeneratedOp : HWOp<"module.generated",
      [IsolatedFromAbove, FunctionLike, Symbol, HasParent<"mlir::ModuleOp">]> {
  let summary = "HW Generated Module";
  let description = [{
    The "hw.module.generated" operation represents a reference to an external
    module that will be produced by some external process.
    This represents the name and list of ports to be generated.

    The 'verilogName' attribute (when present) specifies the spelling of the
    module name in Verilog we can use.  See hw.module for an explanation.
  }];
  let arguments = (ins FlatSymbolRefAttr:$generatorKind,
                       OptionalAttr<StrAttr>:$verilogName,
                       StrArrayAttr:$argNames, StrArrayAttr:$resultNames);
  let results = (outs);
  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "FlatSymbolRefAttr":$genKind, "StringAttr":$name,
                   "ArrayRef<PortInfo>":$ports,
                   CArg<"StringRef", "StringRef()">:$verilogName,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "FlatSymbolRefAttr":$genKind, "StringAttr":$name,
                   "const ModulePortInfo &":$ports,
                   CArg<"StringRef", "StringRef()">:$verilogName,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
  ];

  let extraClassDeclaration = [{
    /// Decode information about the input and output ports on this module.
    ModulePortInfo getPorts() {
      return getModulePortInfo(*this);
    }

    /// Return all the module ports merged into one list.
    SmallVector<PortInfo> getAllPorts() {
      return getAllModulePortInfos(*this);
    }

    /// Return the name to use for the Verilog module that we're referencing
    /// here.  This is typically the symbol, but can be overridden with the
    /// verilogName attribute.
    StringRef getVerilogModuleName();

    /// Return the name to use for the Verilog module that we're referencing
    /// here.  This is typically the symbol, but can be overridden with the
    /// verilogName attribute.
    StringAttr getVerilogModuleNameAttr() {
      if (auto vName = verilogNameAttr()){
        return vName;
      }
      return (*this)->getAttrOfType<StringAttr>(
          ::mlir::SymbolTable::getSymbolAttrName());
    }

    /// Lookup the generator kind for the symbol.  This returns null on
    /// invalid IR.
    Operation *getGeneratorKindOp();

  private:
    // This trait needs access to the hooks defined below.
    friend class OpTrait::FunctionLike<HWModuleGeneratedOp>;

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
  let verifier = "return ::verify$cppClass(*this);";
}

def InstanceOp : HWOp<"instance", [HasParent<"HWModuleOp">, Symbol,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>,
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Create an instance of a module";
  let description = [{
    This represents an instance of a module. The inputs and results are
    the referenced module's inputs and outputs.  The `argNames` and
    `resultNames` attributes must match the referenced module.  Any parameters
    are stored in the `parameters` dictionary.
  }];

  let arguments = (ins StrAttr:$instanceName,
                       Confined<FlatSymbolRefAttr, [isModuleSymbol]>:$moduleName,
                       Variadic<AnyType>:$inputs,
                       StrArrayAttr:$argNames, StrArrayAttr:$resultNames,
                       OptionalAttr<DictionaryAttr>:$parameters,
                       OptionalAttr<SymbolNameAttr>:$sym_name);
  let results = (outs Variadic<AnyType>);

  let builders = [
    /// Create a instance that refers to a known module.
    OpBuilder<(ins "Operation*":$module, "StringAttr":$name,
                   "ArrayRef<Value>":$inputs,
                   CArg<"DictionaryAttr", "{}">:$parameters,
                   CArg<"StringAttr", "{}">:$sym_name)>,
    /// Create a instance that refers to a known module.
    OpBuilder<(ins "Operation*":$module, "StringRef":$name,
                   "ArrayRef<Value>":$inputs,
                   CArg<"DictionaryAttr", "{}">:$parameters,
                   CArg<"StringAttr", "{}">:$sym_name), [{
      build($_builder, $_state, module, $_builder.getStringAttr(name), inputs,
            parameters, sym_name);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return the name of the specified input port or null if it cannot be
    /// determined.
    StringAttr getArgumentName(size_t i);

    /// Return the name of the specified result or null if it cannot be
    /// determined.
    StringAttr getResultName(size_t i);

    /// Change the name of the specified input port.
    void setArgumentName(size_t i, StringAttr name);

    /// Change the name of the specified output port.
    void setResultName(size_t i, StringAttr name);

    /// Change the names of all the input ports.
    void setArgumentNames(ArrayAttr names) {
      getOperation()->setAttr("argNames", names);
    }

    /// Change the names of all the result ports.
    void setResultNames(ArrayAttr names) {
      getOperation()->setAttr("resultNames", names);
    }

    /// Lookup the module or extmodule for the symbol.  This returns null on
    /// invalid IR.
    Operation *getReferencedModule(const SymbolCache *cache = nullptr);

    /// Get the instances's name.
    StringAttr getName() {
      return (*this)->getAttrOfType<StringAttr>("instanceName");
    }

    /// Set the instance's name.
    void setName(StringAttr name) {
      (*this)->setAttr("instanceName", name);
    }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    /// An InstanceOp may optionally define a symbol.
    bool isOptionalSymbol() { return true; }
  }];

  let parser = "return ::parse$cppClass(parser, result);";
  let printer = "return ::print$cppClass(p, *this);";
}

def OutputOp : HWOp<"output", [Terminator, HasParent<"HWModuleOp">,
                                NoSideEffect, ReturnLike]> {
  let summary = "HW termination operation";
  let description = [{
    "hw.output" marks the end of a region in the HW dialect and the values
    to put on the output ports.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), "build($_builder, $_state, llvm::None);">
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";

  let verifier = "return ::verifyOutputOp(this);";
}
