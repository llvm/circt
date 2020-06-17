//===- OpStructure.td - Circuit and Module Definitions -----*- tablegen -*-===//
//
// This describes the FIRRTL Circuit and Module MLIR ops.
//
//===----------------------------------------------------------------------===//

def CircuitOp : FIRRTLOp<"circuit",
      [IsolatedFromAbove, SymbolTable, 
       SingleBlockImplicitTerminator<"DoneOp">]> {
  let summary = "FIRRTL Circuit";
  let description = [{
    The "firrtl.circuit" operation represents an overall Verilog circuit,
    containing a list of modules.
  }];
  let arguments = (ins StrAttr:$name);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, StringAttr name">
  ];

  let extraClassDeclaration = [{
    OpBuilder getBodyBuilder() {
      assert(!body().empty() && "Unexpected empty 'body' region.");
      Block &bodyBlock = body().front();
      return OpBuilder(&bodyBlock, std::prev(bodyBlock.end()));
    }

    /// Return body of this circuit.
    Region &getBodyRegion();
    Block *getBody();
  }];
  
  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def FModuleOp : FIRRTLOp<"module",
      [IsolatedFromAbove, FunctionLike, Symbol,
       SingleBlockImplicitTerminator<"DoneOp">]> {
  let summary = "FIRRTL Module";
  let description = [{
    The "firrtl.module" operation represents a Verilog module, including a given
    name, a list of ports, and a body that represents the connections within
    the module.
  }];
  let arguments = (ins);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, StringAttr name, "
              "ArrayRef<std::pair<StringAttr, FIRRTLType>> ports">
  ];

  let extraClassDeclaration = [{
    using FunctionLike::getBody;

    // Decode information about the input and output ports on this module.
    void getPortInfo(SmallVectorImpl<ModulePortInfo> &results) {
      getModulePortInfo(*this, results);
    }

    // TODO(mlir): FunctionLike shouldn't produce a getBody() helper, it is
    // squatting on the name.
    Block *getBodyBlock() { return &body().front(); }

    OpBuilder getBodyBuilder() {
      assert(!body().empty() && "Unexpected empty 'body' region.");
      Block &bodyBlock = body().front();
      return OpBuilder(&bodyBlock, std::prev(bodyBlock.end()));
    }

  private:
    // This trait needs access to the hooks defined below.
    friend class OpTrait::FunctionLike<FModuleOp>;

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

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def FExtModuleOp : FIRRTLOp<"extmodule",
      [IsolatedFromAbove, FunctionLike, Symbol]> {
  let summary = "FIRRTL extmodule";
  let description = [{
    The "firrtl.extmodule" operation represents an external reference to a
    Verilog module, including a given name, a list of ports, and a body that
    represents the connections within the module.
  }];
  let arguments = (ins OptionalAttr<StrAttr>:$defname,
                       OptionalAttr<DictionaryAttr>:$parameters);
  let results = (outs);
  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, StringAttr name, "
              "ArrayRef<std::pair<StringAttr, FIRRTLType>> ports, "
              "StringRef defnameAttr = StringRef()">
  ];

  let extraClassDeclaration = [{
     // Decode information about the input and output ports on this module.
     void getPortInfo(SmallVectorImpl<ModulePortInfo> &results) {
       getModulePortInfo(*this, results);
     }

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

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def DoneOp : FIRRTLOp<"done", [Terminator]> {
  let summary = "FIRRTL termination operation";
  let description = [{
    "firrtl.done" marks the end of a region in the FIRRTL dialect.
  }];

  let arguments = (ins);
}
