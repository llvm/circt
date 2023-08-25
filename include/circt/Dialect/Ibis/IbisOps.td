//===- IbisOps.td - Definition of Ibis dialect operations -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_IBIS_IBISOPS_TD
#define CIRCT_DIALECT_IBIS_IBISOPS_TD

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

include "circt/Dialect/HW/HWOpInterfaces.td"
include "circt/Dialect/Ibis/IbisInterfaces.td"
include "circt/Dialect/Ibis/IbisTypes.td"

include "circt/Support/InstanceGraphInterface.td"

class IbisOp<string mnemonic, list<Trait> traits = []> :
    Op<IbisDialect, mnemonic, traits>;

def HasCustomSSAName :
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>;

def ClassOp : IbisOp<"class", [
    Symbol,
    IsolatedFromAbove, RegionKindInterface,
    SymbolTable, SingleBlock,
    NoTerminator, ScopeOpInterface,
    InstanceGraphModuleOpInterface,
    HasParent<"mlir::ModuleOp">]> {

  let summary = "Ibis class";
  let description = [{
    Ibis has the notion of a class which can contain methods and member
    variables.

    In the low-level Ibis representation, the ClassOp becomes a container for
    `ibis.port`s, `ibis.container`s, and contain logic for member variables.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $sym_name attr-dict-with-keyword $body
  }];

  let extraClassDeclaration = [{
    // Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }
  
    Block* getBodyBlock() { return &getBody().front(); }

    StringAttr getModuleNameAttr() {
      return getSymNameAttr();
    }

    llvm::StringRef getModuleName() {
      return getSymName();
    }
  }];
}

class InstanceOpBase<string mnemonic> : IbisOp<mnemonic, [
    Symbol,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    InstanceGraphInstanceOpInterface
]> {

  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$targetName);
  let results = (outs ScopeRefType:$scopeRef);
  let assemblyFormat = [{
    $sym_name `,` $targetName attr-dict
    custom<ScopeRefFromName>(type($scopeRef), ref($targetName))
  }];

  let builders = [
    OpBuilder<(ins "StringAttr":$instanceName, "StringAttr":$targetName), [{
      build($_builder, $_state, $_builder.getType<ScopeRefType>(targetName),
        instanceName, targetName);
    }]>
  ];

  code extraInstanceClassDeclaration = ?;

  let extraClassDeclaration = extraInstanceClassDeclaration # [{
    llvm::StringRef getInstanceName() {
      return getSymName();
    }

    mlir::StringAttr getInstanceNameAttr() {
      return getSymNameAttr();
    }

    llvm::StringRef getModuleName() {
      return getTargetName();
    }

    mlir::FlatSymbolRefAttr getModuleNameAttr() {
      return getTargetNameAttr();
    }
  }];
}

def InstanceOp : InstanceOpBase<"instance"> {
  let summary = "Ibis class instance";
  let description = [{
    Instantiates an Ibis class.
  }];

  let extraInstanceClassDeclaration = [{
    // Return the class this instance is instantiating.
    ClassOp getClass(SymbolTable *symbolTable = nullptr);

    Operation* getReferencedModuleSlow();
    Operation* getReferencedModule(SymbolTable&);
  }];

  let extraClassDefinition = [{
    Operation* InstanceOp::getReferencedModuleSlow() {
      return getClass();
    }

    Operation* InstanceOp::getReferencedModule(SymbolTable& symbolTable) {
      return getClass(&symbolTable);
    }
  }];
}

def MethodOp : IbisOp<"method", [
      IsolatedFromAbove, RegionKindInterface,
      Symbol, FunctionOpInterface,
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
      SingleBlockImplicitTerminator<"ReturnOp">,
      HasParent<"ClassOp">]> {

  let summary = "Ibis method";
  let description = [{
    Ibis functions are a lot like software functions: a list of named arguments
    and one unnamed return value.

    Can only live inside of classes.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       ArrayAttr:$argNames,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph;}

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];
}

def ReturnOp : IbisOp<"return", [
      Pure, ReturnLike, Terminator, HasParent<"MethodOp">]> {
  let summary = "Ibis function terminator";

  let arguments = (ins Optional<AnyType>:$retValue);
  let assemblyFormat = "($retValue^)? attr-dict (`:` type($retValue)^)?";
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins)>,
  ];
}

def CallOp : IbisOp<"call", [CallOpInterface]> {
  let summary = "Ibis method call";
  let description = [{
    Dispatch a call to an Ibis method.
  }];

  let arguments = (ins SymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let extraClassDeclaration = [{
    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

    /// Get the argument operands to the called method.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr(getCalleeAttrName(), callee.get<mlir::SymbolRefAttr>());
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def PathDirection : I32EnumAttr<"PathDirection", "path direction", [
  I32EnumAttrCase<"Parent", 0, "parent">,
  I32EnumAttrCase<"Child", 1, "child">
]>;

def PathStepAttr : AttrDef<IbisDialect, "PathStep", [TypedAttrInterface]> {
  let description = "Used to describe a single step in a path";
  let parameters = (ins
    "PathDirection":$direction,
    AttributeSelfTypeParameter<"">:$type,
    OptionalParameter<"mlir::FlatSymbolRefAttr">:$child
  );
  let mnemonic = "step";
  // Would like to either have no prefix before child, or `:`, but both are
  // currently invalid due to 
  // https://github.com/llvm/llvm-project/issues/64724
  let assemblyFormat = "`<` $direction (`,` $child^)? `:` $type  `>`";
  let genVerifyDecl = 1;
}

def PathStepArrayAttr :
  TypedArrayAttrBase<PathStepAttr, "Path step array attribute">;

def PathOp : IbisOp<"path", [Pure, InferTypeOpInterface]> {
  let summary = "Ibis path";
  let description = [{
    The `ibis.path` operation describes an instance hierarchy path relative to
    the current scope. The path is specified by a list of either parent or
    child identifiers (navigating up or down the hierarchy, respectively).
    
    Scopes along the path are optionally typed, however, An `ibis.path` must
    lways terminate in a fully typed specifier, i.e. never an `!ibis.scoperef<>`.

    The operation returns a single `!ibis.scoperef`-typed value representing
    the scope at the end of the path.
  }];

  let arguments = (ins
    PathStepArrayAttr:$path
  );

  let results = (outs ScopeRefType:$instance);
  let assemblyFormat = [{
    $path attr-dict
  }];

  let extraClassDeclaration = [{
    /// Infer the return types of this operation.
    static LogicalResult inferReturnTypes(MLIRContext *context,
                                          std::optional<Location> loc,
                                          ValueRange operands,
                                          DictionaryAttr attrs,
                                          mlir::OpaqueProperties properties,
                                          mlir::RegionRange regions,
                                          SmallVectorImpl<Type> &results);
    
    auto getPathAsRange() {
      return getPath().getAsRange<PathStepAttr>();
    }
  }];
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def ThisOp : IbisOp<"this", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Return a handle to the current scope `!ibis.scoperef`";
  let arguments = (ins FlatSymbolRefAttr:$scopeName);
  let results = (outs ScopeRefType:$thisRef);

  let assemblyFormat = [{
    $scopeName attr-dict custom<ScopeRefFromName>(type($thisRef), ref($scopeName))
  }];

  let builders = [
    OpBuilder<(ins "StringAttr":$name), [{
      build($_builder, $_state, $_builder.getType<ScopeRefType>(name), name);
    }]>
  ];
}

// ===---------------------------------------------------------------------===//
// Low-level Ibis operations
// ===---------------------------------------------------------------------===//

def ContainerOp : IbisOp<"container", [
  Symbol,
  SymbolTable, SingleBlock,
  NoTerminator, NoRegionArguments,
  ScopeOpInterface, IsolatedFromAbove,
  InstanceGraphModuleOpInterface,
  RegionKindInterface
]> {
  let summary = "Ibis container";
  let description = [{
    An ibis container describes a collection of logic nested within an Ibis class.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $sym_name attr-dict-with-keyword $body
  }];

  let extraClassDeclaration = [{
    // Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }

    Block* getBodyBlock() { return &getBody().front(); }

    StringAttr getModuleNameAttr() {
      return getSymNameAttr();
    }

    llvm::StringRef getModuleName() {
      return getSymName();
    }
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name), [{
      auto region = $_state.addRegion();
      region->push_back(new Block());
      $_state.addAttribute(getSymNameAttrName($_state.name), name);
    }]>
  ];
}

def ContainerInstanceOp : InstanceOpBase<"container.instance"> {
  let summary = "Ibis container instance";
  let description = [{
    Instantiates an Ibis container.
  }];

  let extraInstanceClassDeclaration = [{
    // Return the container this instance is instantiating.
    ContainerOp getContainer(SymbolTable* symbolTable = nullptr);

    Operation* getReferencedModuleSlow();
    Operation* getReferencedModule(SymbolTable&);
  }];

  let extraClassDefinition = [{
    Operation* ContainerInstanceOp::getReferencedModuleSlow() {
      return getContainer();
    }

    Operation* ContainerInstanceOp::getReferencedModule(SymbolTable& symbolTable) {
      return getContainer(&symbolTable);
    }
  }];
}

class InnerTypeToPortRefTypeConstraint<string lhs, string rhs, string dir>
  : TypesMatchWith<"the rhs type dictates the inner type of the rhs portref type",
    lhs, rhs, "PortRefType::get($_ctxt, $_self, ibis::Direction::" # dir # ")">;

class PortLikeOp<string mnemonic, list<Trait> traits = []> :
    IbisOp<mnemonic, !listconcat(traits, [
      PortOpInterface,
      ParentOneOf<["ClassOp", "ContainerOp"]>,
      InferTypeOpInterface
  ])> {
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttrOf<AnyType>:$type);
  let results = (outs PortRefType:$port);
  let assemblyFormat = [{
    $sym_name `:` $type attr-dict
  }];

  let extraClassDeclaration = [{
    Type getPortType() {
      return getTypeAttr().getValue();
    }

    mlir::StringAttr getPortName() {
      return getSymNameAttr();
    }

    static ibis::Direction getPortDirection();

    /// Infer the return types of this operation.
    static LogicalResult inferReturnTypes(MLIRContext *context,
                                          std::optional<Location> loc,
                                          ValueRange operands,
                                          DictionaryAttr attrs,
                                          mlir::OpaqueProperties properties,
                                          mlir::RegionRange regions,
                                          SmallVectorImpl<Type> &results) {
      results.push_back(PortRefType::get(context, attrs.get("type").cast<TypeAttr>().getValue(), getPortDirection()));
      return success();
    }
  }];

  code extraPortClassDefinition = ?;
  let extraClassDefinition =  extraPortClassDefinition;
}

def GetPortOp : IbisOp<"get_port", [
      Pure,
      DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Ibis get port";
  let description = [{
    Given an Ibis class reference, returns a port of said class. The port
    is specified by the symbol name of the port in the referenced class.
  }];

  let arguments = (ins
    ScopeRefType:$instance,
    FlatSymbolRefAttr:$portSymbol);
  let results = (outs PortRefType:$port);
  let assemblyFormat = [{
    $instance `,` $portSymbol `:` qualified(type($instance)) `->` qualified(type($port)) attr-dict
  }];

  let builders = [
    OpBuilder<(ins "Value":$instance, "StringAttr":$portName, "Type":$innerPortType, "ibis::Direction":$direction), [{
      build($_builder, $_state, $_builder.getType<PortRefType>(innerPortType, direction), instance, portName);
    }]>
  ];

  let hasCanonicalizeMethod = 1;
}

def InputPortOp : PortLikeOp<"port.input"> {
  let summary = "Ibis input port";

  let extraPortClassDefinition = [{
    ibis::Direction InputPortOp::getPortDirection() {
      return ibis::Direction::Input;
    }
  }];
}

def OutputPortOp : PortLikeOp<"port.output"> {
  let summary = "Ibis output port";

  let extraPortClassDefinition = [{
    ibis::Direction OutputPortOp::getPortDirection() {
      return ibis::Direction::Output;
    }
  }];
}

class PortRefToInnerTypeConstraint<string lhs, string rhs>
  : TypesMatchWith<"the lhs portref dictates the rhs type",
    lhs, rhs, "$_self.cast<PortRefType>().getPortType()">;

def PortReadOp : IbisOp<"port.read", [
      PortRefToInnerTypeConstraint<"port", "output">,
      Pure
]> {
  let summary = "Ibis port read";
  let description = [{
    Read the value of a local port (i.e. a port of 'this' class).
  }];

  let arguments = (ins PortRefType:$port);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    $port attr-dict `:` qualified(type($port))
  }];
}

def PortWriteOp : IbisOp<"port.write", [
      InnerTypeToPortRefTypeConstraint<"value", "port", "Input">
]> {
  let summary = "Ibis port write";
  let description = [{
    Write a value to a local port (i.e. a port of 'this' class).
  }];

  let arguments = (ins PortRefType:$port, AnyType:$value);
  let assemblyFormat = [{
    $port `,` $value attr-dict `:` type($value)
  }];
}

#endif // CIRCT_DIALECT_IBIS_IBISOPS_TD
