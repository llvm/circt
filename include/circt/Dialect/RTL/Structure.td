def RTLModuleOp : RTLOp<"module",
      [IsolatedFromAbove, FunctionLike, Symbol, RegionKindInterface,
       SingleBlockImplicitTerminator<"DoneOp">]> {
  let summary = "RTL Module";
  let description = [{
    The "rtl.module" operation represents a Verilog module, including a given
    name, a list of ports, and a body that represents the connections within
    the module.
  }];
  let arguments = (ins);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, StringAttr name, "
              "ArrayRef<RTLModulePortInfo> ports">
  ];

  let extraClassDeclaration = [{
    using FunctionLike::getBody;

    // Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }

    void getRTLModulePortInfo(Operation *op, SmallVectorImpl<RTLModulePortInfo> &results);
    
    // Decode information about the input and output ports on this module.
    void getRTLPortInfo(SmallVectorImpl<RTLModulePortInfo> &results) {
      getRTLModulePortInfo(*this, results);
    }

    // TODO(mlir): FunctionLike shouldn't produce a getBody() helper, it is
    // squatting on the name.
    Block *getBodyBlock() { return &body().front(); }

  private:
    // This trait needs access to the hooks defined below.
    friend class OpTrait::FunctionLike<RTLModuleOp>;

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

def RTLInstanceOp : RTLOp<"instance", []> {
  let summary = "Create an instance of a module";
  let description = [{
    This represents an instance of a module. The inputs and results are 
    the referenced module's inputs and outputs. 
  }];

  let arguments = (ins StrAttr:$instanceName,
                       FlatSymbolRefAttr:$moduleName,
                       Variadic<AnySignlessInteger>:$inputs);
  let results = (outs);
  
  let assemblyFormat = [{
     $instanceName $moduleName `(` $inputs `)` attr-dict `:` type($inputs)
  }];

  let verifier = [{ return ::verifyFModuleOp(*this); }];
}

def DoneOp : RTLOp<"done", [Terminator, HasParent<"RTLModuleOp">]> {
  let summary = "RTL termination operation";
  let description = [{
    "rtl.done" marks the end of a region in the RTL dialect.
  }];

  let arguments = (ins);
}