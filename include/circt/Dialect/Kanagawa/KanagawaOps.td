//===- KanagawaOps.td - Definition of Kanagawa dialect operations ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_KANAGAWA_KANAGAWAOPS_TD
#define CIRCT_DIALECT_KANAGAWA_KANAGAWAOPS_TD

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

include "circt/Dialect/Handshake/HandshakeInterfaces.td"
include "circt/Dialect/HW/HWAttributesNaming.td"
include "circt/Dialect/HW/HWOpInterfaces.td"
include "circt/Dialect/Kanagawa/KanagawaInterfaces.td"
include "circt/Dialect/Kanagawa/KanagawaTypes.td"
include "circt/Dialect/Handshake/HandshakeInterfaces.td"
include "circt/Dialect/Seq/SeqTypes.td"
include "circt/Support/InstanceGraphInterface.td"
include "circt/Dialect/HW/HWTypes.td"

class KanagawaOp<string mnemonic, list<Trait> traits = []> :
    Op<KanagawaDialect, mnemonic, traits>;

def HasCustomSSAName :
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>;

def ClassOp : KanagawaOp<"class", [
    IsolatedFromAbove, RegionKindInterface,
    SingleBlock,
    NoTerminator, ScopeOpInterface,
    InstanceGraphModuleOpInterface,
    HasParent<"DesignOp"> ] > {

  let summary = "Kanagawa class";
  let description = [{
    Kanagawa has the notion of a class which can contain methods and member
    variables.

    In the low-level Kanagawa representation, the ClassOp becomes a container for
    `kanagawa.port`s, `kanagawa.container`s, and contain logic for member variables.
  }];

  let arguments = (ins InnerSymAttr:$inner_sym, OptionalAttr<StrAttr>:$name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    ($name^)? `sym` $inner_sym attr-dict-with-keyword $body
  }];

  let extraClassDeclaration = [{
    // Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }

    Block* getBodyBlock() { return &getBody().front(); }

    // For hw:InnerSymbol - the symbol targets the operation and not any result.
    static std::optional<size_t> getTargetResultIndex() {
      return std::nullopt;
    }
  }];
}

class InstanceOpBase<string mnemonic> : KanagawaOp<mnemonic, [
    DeclareOpInterfaceMethods<InnerSymbol,["getTargetResultIndex"]>,
    DeclareOpInterfaceMethods<InnerRefUserOpInterface>,
    InstanceGraphInstanceOpInterface,
    HasCustomSSAName
]> {
  let arguments = (ins InnerSymAttr:$inner_sym, InnerRefAttr:$targetName);
  let results = (outs ScopeRefType:$scopeRef);
  let assemblyFormat = [{
    $inner_sym `,` $targetName attr-dict
    custom<ScopeRefFromName>(type($scopeRef), ref($targetName))
  }];

  let builders = [
    OpBuilder<(ins "hw::InnerSymAttr":$instanceName, "hw::InnerRefAttr":$targetName), [{
      build($_builder, $_state, $_builder.getType<ScopeRefType>(targetName),
        instanceName, targetName);
    }]>
  ];

  code extraInstanceClassDeclaration = ?;
  let extraClassDeclaration = extraInstanceClassDeclaration # [{
    llvm::StringRef getInstanceName() {
      return getInnerSym().getSymName().strref();
    }

    mlir::StringAttr getInstanceNameAttr() {
      return getInnerSymAttr().getSymName();
    }

    StringAttr getReferencedModuleNameAttr() {
      return getTargetNameAttr().getName();
    }

    // Operation* getReferencedModuleSlow();
    Operation* getReferencedModule(const hw::InnerRefNamespace &);
  }];

  code extraInstanceClassDefinition = ?;
  let extraClassDefinition = extraInstanceClassDefinition # [{
    // For hw:InnerSymbol - the symbol targets the operation and not any result.
    std::optional<size_t> $cppClass::getTargetResultIndex() {
      return std::nullopt;
    }
  }];
}

def InstanceOp : InstanceOpBase<"instance"> {
  let summary = "Kanagawa class instance";
  let description = [{
    Instantiates an Kanagawa class.
  }];

  let extraInstanceClassDeclaration = [{
    // Return the class this instance is instantiating.
    ClassOp getClass(const hw::InnerRefNamespace &);
  }];

  let extraInstanceClassDefinition = [{
    ClassOp InstanceOp::getClass(const hw::InnerRefNamespace &ns) {
      return ns.lookupOp<ClassOp>(getTargetNameAttr());
      // return dyn_cast_or_null<ClassOp>(getReferencedModule(&ns));
    }
    Operation* InstanceOp::getReferencedModule(
        const hw::InnerRefNamespace &ns) {
      return getClass(ns);
    }
  }];
}


class MethodOpBase<string mnemonic, list<Trait> traits = []> :
    KanagawaOp<mnemonic, !listconcat(traits, [
      IsolatedFromAbove,
      DeclareOpInterfaceMethods<InnerSymbol,["getTargetResultIndex"]>,
      // FunctionOpInterface @mortbopet: This is very much a function-like
      // operation, but FunctionOpInterface inherits from Symbol which is
      // incompatible with the fact that kanagawa methods define inner symbols.
      MethodLikeOpInterface,
      HasParent<"ClassOp">,
  ])> {

  let arguments = (ins InnerSymAttr:$inner_sym,
                       TypeAttrOf<FunctionType>:$function_type,
                       ArrayAttr:$argNames,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let hasCustomAssemblyFormat = 1;

  code extraMethodClassDeclaration = "";
  let extraClassDeclaration = extraMethodClassDeclaration # [{
    StringAttr getMethodName() {
      return getInnerSymAttr().getSymName();
    }
  }];

  let extraClassDefinition = [{
    // For hw:InnerSymbol - the symbol targets the operation and not any result.
    std::optional<size_t> $cppClass::getTargetResultIndex() {
      return std::nullopt;
    }
  }];
}

def MethodOp : MethodOpBase<"method", [
      AutomaticAllocationScope,
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>
  ]> {

  let summary = "Kanagawa method";
  let description = [{
    Kanagawa methods are a lot like software functions: a list of named arguments
    and unnamed return values with imperative control flow.
  }];

  let regions = (region AnyRegion:$body);
}

def DataflowMethodOp : MethodOpBase<"method.df", [
      SingleBlockImplicitTerminator<"kanagawa::ReturnOp">,
      RegionKindInterface,
      FineGrainedDataflowRegionOpInterface
  ]> {

  let summary = "Kanagawa dataflow method";
  let description = [{
    Kanagawa dataflow methods share the same interface as an `kanagawa.method` but
    without imperative CFG-based control flow. Instead, this method implements a
    graph region, and control flow is expected to be defined by dataflow operations.
  }];
  let regions = (region SizedRegion<1>:$body);

  let extraMethodClassDeclaration = [{
    // Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }
  }];
}

def ReturnOp : KanagawaOp<"return", [
      Pure, ReturnLike, Terminator,
      ParentOneOf<["MethodOp", "DataflowMethodOp"]>]> {
  let summary = "Kanagawa method terminator";

  let arguments = (ins Variadic<AnyType>:$retValues);
  let assemblyFormat = "($retValues^)? attr-dict (`:` type($retValues)^)?";
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins)>,
  ];
}

class BlockLikeOp<string mnemonic, list<Trait> traits = []> : KanagawaOp<mnemonic, !listconcat(traits, [
  SingleBlockImplicitTerminator<"BlockReturnOp">,
  AutomaticAllocationScope,
  BlockOpInterface])> {
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    OptionalAttr<ConfinedAttr<I64Attr, [IntMinValue<1>]>>:$maxThreads
  );
  let results = (outs Variadic<AnyType>:$outputs);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;

  code extraBlockClassDeclaration = ?;
  let extraClassDeclaration = extraBlockClassDeclaration # [{
    // Return the body of this block.
    Block* getBodyBlock() { return &getBody().front(); }
  }];
  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
}

class HighLevelBlockLikeOp<string mnemonic, list<Trait> traits = []> : BlockLikeOp<mnemonic, traits> {
  let builders = [
    OpBuilder<(ins "TypeRange":$outputs, "ValueRange":$inputs,
        CArg<"IntegerAttr", "{}">:$maxThreads), [{
      $_state.addOperands(inputs);
      if (maxThreads)
        $_state.addAttribute(getMaxThreadsAttrName($_state.name), maxThreads);
      auto* region = $_state.addRegion();
      $_state.addTypes(outputs);
      ensureTerminator(*region, $_builder, $_state.location);
      llvm::SmallVector<Location> argLocs;
      for(auto input : inputs)
        argLocs.push_back(input.getLoc());
      region->front().addArguments(inputs.getTypes(), argLocs);
    }]>
  ];

  let extraBlockClassDeclaration = [{
    llvm::SmallVector<Type> getInternalResultTypes() {
      llvm::SmallVector<Type> outTypes;
      llvm::copy(getResultTypes(), std::back_inserter(outTypes));
      return outTypes;
    }
  }];
}

def StaticBlockOp : HighLevelBlockLikeOp<"sblock"> {
  let summary = "Kanagawa block";
  let description = [{
    The `kanagawa.sblock` operation defines a block wherein a group of operations
    are expected to be statically scheduleable.
    The operation is not isolated from above to facilitate ease of construction.
    However, once a program has been constructed and lowered to a sufficient
    level, the user may run `--kanagawa-argify-blocks` to effectively isolate the
    block from above, by converting SSA values referenced through dominanes into
    arguments of the block

    The block may contain additional attributes to specify constraints on
    the block further down the compilation pipeline.
  }];
}

def IsolatedStaticBlockOp : HighLevelBlockLikeOp<"sblock.isolated", [
  IsolatedFromAbove
]> {
  let summary = "Kanagawa isolated block";
  let description = [{
    The `kanagawa.sblock.isolated` operation is like an `kanagawa.sblock` operation
    but with an IsolatedFromAbove condition, meaning that all arguments and
    results are passed through the block as arguments and results.
  }];
}

def DCBlockOp : BlockLikeOp<"sblock.dc", [
  IsolatedFromAbove
]> {
  let summary = "DC-interfaced Kanagawa block";
  let description = [{
    The `kanagawa.sblock.dc` operation is like an `kanagawa.sblock` operation with
    a few differences, being:
    1. The operation is DC-interfaced, meaning that all arguments and results
       are dc-value typed.
    2. The operation is IsolatedFromAbove.
  }];

  let builders = [
    OpBuilder<(ins "TypeRange":$outputs, "ValueRange":$inputs,
        CArg<"IntegerAttr", "{}">:$maxThreads)>
  ];

  let extraBlockClassDeclaration = [{
    // The internal result types of a DC block op is the DC-type stripped
    // outer result values.
    llvm::SmallVector<Type> getInternalResultTypes() {
      llvm::SmallVector<Type> innerRes;
      for(Type outerRes : getResultTypes())
        innerRes.push_back(llvm::cast<dc::ValueType>(outerRes).getInnerType());
      return innerRes;
    }
  }];
}

def InlineStaticBlockBeginOp : KanagawaOp<"sblock.inline.begin", [
  HasParent<"MethodOp">
]> {
  let summary = "Kanagawa inline static block begin marker";
  let description = [{
    The `kanagawa.sblock.inline.begin` operation is a marker that indicates the
    begin of an inline static block.
    The operation is used to maintain `kanagawa.sblocks` while in the Kanagawa inline
    phase (to facilitate e.g. mem2reg).

    The operation:
    1. denotes the begin of the sblock
    2. carries whatever attributes that the source `kanagawa.sblock` carried.
    3. is considered side-effectfull.
  }];

  let assemblyFormat = "attr-dict";
  let extraClassDeclaration = [{
    /// Return the `InlineStaticBlockEndOp` that this operation is referencing.
    InlineStaticBlockEndOp getEndOp();
  }];
}

def InlineStaticBlockEndOp : KanagawaOp<"sblock.inline.end", [
  HasParent<"MethodOp">
]> {
  let summary = "Kanagawa inline static block end marker";
  let description = [{
    The `kanagawa.sblock.inline.end` operation is a marker that indicates the
    end of an inline static block.
    The operation is used to maintain `kanagawa.sblocks` while in the Kanagawa inline
    phase (to facilitate e.g. mem2reg).
  }];

  let assemblyFormat = "attr-dict";

  let extraClassDeclaration = [{
    /// Return the `InlineStaticBlockBeginOp` that this operation is referencing.
    InlineStaticBlockBeginOp getBeginOp();
  }];
}

def BlockReturnOp : KanagawaOp<"sblock.return", [
    Pure, ReturnLike, Terminator,
    ParentOneOf<["StaticBlockOp", "IsolatedStaticBlockOp", "DCBlockOp"]>]> {
  let summary = "Kanagawa static block terminator";

  let arguments = (ins Variadic<AnyType>:$retValues);
  let assemblyFormat = "($retValues^)? attr-dict (`:` type($retValues)^)?";
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins), "/*no-op*/">
  ];
}

def MemRefTypeAttr : TypeAttrBase<"MemRefType", "any memref type">;
def VarOp : KanagawaOp<"var", [
  DeclareOpInterfaceMethods<InnerSymbol,["getTargetResultIndex"]>
]> {
  let summary = "Kanagawa variable definition";
  let description = [{
    Defines an Kanagawa class member variable. The variable is typed with a
    `memref.memref` type, and may define either a singleton or uni-dimensional
    array of values.
    `kanagawa.var` defines a symbol within the encompassing class scope which can
    be dereferenced through a `!kanagawa.scoperef` value of the parent class.
  }];

  let arguments = (ins InnerSymAttr:$inner_sym, MemRefTypeAttr:$type);
  let assemblyFormat = "$inner_sym `:` $type attr-dict";

  let extraClassDefinition = [{
    // For hw:InnerSymbol - the symbol targets the operation and not any result.
    std::optional<size_t> $cppClass::getTargetResultIndex() {
      return std::nullopt;
    }
  }];
}

def GetVarOp : KanagawaOp<"get_var", [
  DeclareOpInterfaceMethods<InnerRefUserOpInterface>,
  HasCustomSSAName
]> {
  let summary = "Dereferences an kanagawa member variable through a scoperef";
  let arguments = (ins ScopeRefType:$instance, FlatSymbolRefAttr:$varName);
  let results = (outs AnyMemRef:$var);
  let assemblyFormat = [{
    $instance `,` $varName attr-dict `:` qualified(type($instance)) `->` qualified(type($var))
  }];

  let extraClassDeclaration = [{
    // Return the `VarOp` that this operation is referencing.
    VarOp getVar(const hw::InnerRefNamespace &symbolTable);
  }];

  let extraClassDefinition = [{
    void $cppClass::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
      setNameFn(getResult(), getVarName());
    }
  }];
}

def CallOp : KanagawaOp<"call", [
    DeclareOpInterfaceMethods<InnerRefUserOpInterface>,
    CallOpInterface
  ]> {
  let summary = "Kanagawa method call";
  let description = [{
    Dispatch a call to an Kanagawa method.
  }];

  let arguments = (ins InnerRefAttr:$callee, Variadic<AnyType>:$operands);
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
      (*this)->setAttr(getCalleeAttrName(), llvm::cast<mlir::SymbolRefAttr>(callee));
    }

    /// Return the callee of this operation.
    MethodOp getTarget(const hw::InnerRefNamespace &symbolTable);

    /// Stub implementations for ArgumentAttributesMethods. If desired,
    /// implement these by defining arg_attrs and res_attrs as arguments to the
    /// operation as OptionalAttr<DictArrayAttr>.
    mlir::ArrayAttr getArgAttrsAttr() { return nullptr; }
    mlir::ArrayAttr getResAttrsAttr() { return nullptr; }
    void setArgAttrsAttr(mlir::ArrayAttr args) {}
    void setResAttrsAttr(mlir::ArrayAttr args) {}
    mlir::Attribute removeArgAttrsAttr() { return nullptr; }
    mlir::Attribute removeResAttrsAttr() { return nullptr; }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def PathDirection : I32EnumAttr<"PathDirection", "path direction", [
  I32EnumAttrCase<"Parent", 0, "parent">,
  I32EnumAttrCase<"Child", 1, "child">]> {
  let cppNamespace = "::circt::kanagawa";
}

def PathStepAttr : AttrDef<KanagawaDialect, "PathStep", [TypedAttrInterface]> {
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

def PathOp : KanagawaOp<"path", [
    DeclareOpInterfaceMethods<InnerRefUserOpInterface>,
    Pure, InferTypeOpInterface, HasCustomSSAName
  ]> {
  let summary = "Kanagawa path";
  let description = [{
    The `kanagawa.path` operation describes an instance hierarchy path relative to
    the current scope. The path is specified by a list of either parent or
    child identifiers (navigating up or down the hierarchy, respectively).

    Scopes along the path are optionally typed, however, An `kanagawa.path` must
    lways terminate in a fully typed specifier, i.e. never an `!kanagawa.scoperef<>`.

    The operation returns a single `!kanagawa.scoperef`-typed value representing
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

  let hasCanonicalizeMethod = 1;
}

// ===---------------------------------------------------------------------===//
// Low-level Kanagawa operations
// ===---------------------------------------------------------------------===//

def DesignOp : KanagawaOp<"design", [
  Symbol,
  SingleBlock,
  InnerSymbolTable,
  NoTerminator,
  NoRegionArguments,
  IsolatedFromAbove
]> {
  let summary = "All Kanagawa containers must be inside this op";

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $sym_name attr-dict-with-keyword $body
  }];
}

def ContainerOp : KanagawaOp<"container", [
  SingleBlock,
  NoTerminator, NoRegionArguments,
  ScopeOpInterface, IsolatedFromAbove,
  InstanceGraphModuleOpInterface,
  RegionKindInterface,
  DeclareOpInterfaceMethods<NamedInnerSymbol>,
  ParentOneOf<["DesignOp", "ClassOp"]>
]> {
  let summary = "Kanagawa container";
  let description = [{
    An kanagawa container describes a collection of logic nested within an Kanagawa class.
  }];

  let arguments = (ins InnerSymAttr:$inner_sym, UnitAttr:$isTopLevel, OptionalAttr<StrAttr>:$name);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    ($name^)? `sym` $inner_sym (`top_level` $isTopLevel^)? attr-dict-with-keyword $body
  }];

  let extraClassDeclaration = [{
    // Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }

    Block* getBodyBlock() { return &getBody().front(); }

    // For hw:InnerSymbol - the symbol targets the operation and not any result.
    static std::optional<size_t> getTargetResultIndex() {
      return std::nullopt;
    }

    // Override InstanceGraphModuleOpInterface::getModuleNameAttr to return the
    // inner_sym name of the container instead of the `name` which is the default
    // behavior of InstanceGraphModuleOpInterface.
    mlir::StringAttr getModuleNameAttr() {
      return getInnerSymAttr().getSymName();
    }
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringRef":$inner_sym, "bool":$isTopLevel, CArg<"mlir::StringAttr", "nullptr">:$name), [{
      build($_builder, $_state,
        hw::InnerSymAttr::get($_builder.getStringAttr(inner_sym)), isTopLevel, name);
    }]>,
    OpBuilder<(ins "hw::InnerSymAttr":$inner_sym, "bool":$isTopLevel, CArg<"mlir::StringAttr", "nullptr">:$name), [{
      auto region = $_state.addRegion();
      region->push_back(new Block());
      $_state.addAttribute(getInnerSymAttrName($_state.name), inner_sym);
      if (isTopLevel)
        $_state.addAttribute(getIsTopLevelAttrName($_state.name), $_builder.getUnitAttr());
      if(name)
        $_state.addAttribute(getNameAttrName($_state.name), name);
    }]>
  ];
}

def ContainerInstanceOp : InstanceOpBase<"container.instance"> {
  let summary = "Kanagawa container instance";
  let description = [{
    Instantiates an Kanagawa container.
  }];

  let extraInstanceClassDeclaration = [{
    // Return the container this instance is instantiating.
    ContainerOp getContainer(const hw::InnerRefNamespace &symbolTable);
  }];

  let extraInstanceClassDefinition = [{
    ContainerOp ContainerInstanceOp::getContainer(
        const hw::InnerRefNamespace &ns) {
      // return dyn_cast_or_null<ContainerOp>(getReferencedModule(&ns));
      return ns.lookupOp<ContainerOp>(getTargetNameAttr());
    }
    Operation* ContainerInstanceOp::getReferencedModule(
        const hw::InnerRefNamespace &ns) {
      return getContainer(ns);
    }
  }];
}

def GetPortOp : KanagawaOp<"get_port", [
      Pure,
      DeclareOpInterfaceMethods<InnerRefUserOpInterface>,
      HasCustomSSAName
]> {
  let summary = "Kanagawa get port";
  let description = [{
    Given an Kanagawa class reference, returns a port of said class. The port
    is specified by the symbol name of the port in the referenced class.

    Importantly, the user must specify how they intend to use the op, by
    specifying the direction of the portref type that this op is generated with.
    If the request port is to be read from, the type must be `!kanagawa.portref<out T>`
    and if the port is to be written to, the type must be `!kanagawa.portref<in T>`.
    This is to ensure that the usage is reflected in the get_port type which in
    turn is used by the tunneling passes to create the proper ports through the
    hierarchy.

    This implies that the portref direction of the get_port op is independent of
    the actual direction of the target port, and only the inner portref type
    must match.
  }];

  let arguments = (ins
    ScopeRefType:$instance,
    FlatSymbolRefAttr:$portSymbol);
  let results = (outs PortRefType:$port);
  let assemblyFormat = [{
    $instance `,` $portSymbol `:` qualified(type($instance)) `->`
      qualified(type($port)) attr-dict
  }];

  let builders = [
    OpBuilder<(ins "Value":$instance, "StringAttr":$portName, "Type":$innerPortType,
        "kanagawa::Direction":$direction), [{
      build($_builder, $_state, $_builder.getType<PortRefType>(
        innerPortType, direction), instance, portName
      );
    }]>
  ];

  let extraClassDeclaration = [{
    // Returns the direction requested by this get_port op.
    kanagawa::Direction getDirection() {
      return llvm::cast<PortRefType>(getPort().getType()).getDirection();
    }

    // Returns the port op that this operation is referencing.
    PortOpInterface getPort(const hw::InnerRefNamespace &symbolTable);
  }];
}

class PortLikeOp<string mnemonic, list<Trait> traits = []> :
    KanagawaOp<mnemonic, !listconcat(traits, [
      PortOpInterface,
      ParentOneOf<["ClassOp", "ContainerOp"]>,
      InferTypeOpInterface,
      HasCustomSSAName,
      NamedInnerSymbol
  ])> {

  let description = [{
    An kanagawa port has an attached 'name' attribute. This is a name-hint used
    to generate the final port name. The port name and port symbol are not
    related, and all references to a port is done through the port symbol.
  }];

  let arguments = (ins InnerSymAttr:$inner_sym, TypeAttrOf<AnyType>:$type, OptionalAttr<StrAttr>:$name);
  let results = (outs PortRefType:$port);
  let assemblyFormat = [{
    ($name^)? `sym` $inner_sym `:` $type attr-dict
  }];

  let extraClassDeclaration = [{
    Type getPortType() {
      return getTypeAttr().getValue();
    }

    static kanagawa::Direction getPortDirection();

    /// Infer the return types of this operation.
    static LogicalResult inferReturnTypes(MLIRContext *context,
                                          std::optional<Location> loc,
                                          ValueRange operands,
                                          DictionaryAttr attrs,
                                          mlir::OpaqueProperties properties,
                                          mlir::RegionRange regions,
                                          SmallVectorImpl<Type> &results) {
      Adaptor adaptor(operands, attrs, properties, regions);
      results.push_back(PortRefType::get(context, adaptor.getType(),
                        getPortDirection()));
      return success();
    }

    // For hw:InnerSymbol - the symbol targets the operation and not any result.
    static std::optional<size_t> getTargetResultIndex() {
      return std::nullopt;
    }
  }];

  let hasVerifier = 1;

  code extraPortClassDefinition = ?;
  let extraClassDefinition =  extraPortClassDefinition # [{
    LogicalResult $cppClass::verify() {
      if(llvm::isa<ScopeRefType>(getType()))
        return emitOpError("port type cannot be a scope reference");

      return success();
    }
  }];
}

def InputPortOp : PortLikeOp<"port.input"> {
  let summary = "Kanagawa input port";

  let extraPortClassDefinition = [{
    kanagawa::Direction InputPortOp::getPortDirection() {
      return kanagawa::Direction::Input;
    }
  }];
}

def OutputPortOp : PortLikeOp<"port.output"> {
  let summary = "Kanagawa output port";
  let extraPortClassDefinition = [{
    kanagawa::Direction OutputPortOp::getPortDirection() {
      return kanagawa::Direction::Output;
    }
  }];
  let hasCanonicalizeMethod = 1;
}

class WireLikeOp<string mnemonic, list<Trait> traits = []> :
    KanagawaOp<mnemonic, !listconcat(traits, [
      PortOpInterface,
      ParentOneOf<["ClassOp", "ContainerOp"]>,
      HasCustomSSAName,
      DeclareOpInterfaceMethods<InnerSymbol,["getTargetResultIndex"]>
  ])> {

  let extraClassDeclaration = [{
    Type getPortType() {
      return getPort().getType();
    }

    static kanagawa::Direction getPortDirection();
  }];

  code extraWireClassDefinition = ?;
  let extraClassDefinition = extraWireClassDefinition # [{
    // For hw:InnerSymbol - the symbol targets the operation and not any result.
    std::optional<size_t> $cppClass::getTargetResultIndex() {
      return std::nullopt;
    }
  }];
}

class InnerTypeToPortrefTypeConstraint<string lhs, string rhs, string dir>
  : TypesMatchWith<"the rhs type dictates the inner type of the rhs portref type",
    lhs, rhs, "PortRefType::get($_ctxt, $_self, kanagawa::Direction::" # dir # ")">;

class PortRefToInnerTypeConstraint<string lhs, string rhs>
  : TypesMatchWith<"the inner type of the lhs portref type dictates the rhs type",
    lhs, rhs, "llvm::cast<PortRefType>($_self).getPortType()">;

def InputWireOp : WireLikeOp<"wire.input", [
  InnerTypeToPortrefTypeConstraint<"output", "port", "Input">
]> {
  let summary = "Kanagawa input wire";
  let description = [{
    An input wire defines an `kanagawa.portref<in T>` port alongside a value
    of type `T` which represents the value to-be-written to the wire.
  }];

  let arguments = (ins InnerSymAttr:$inner_sym, OptionalAttr<StrAttr>:$name);
  let results = (outs PortRefType:$port, AnyType:$output);
  let assemblyFormat = [{
    $inner_sym `:` qualified(type($output)) attr-dict
  }];

  let extraWireClassDefinition =  [{
    kanagawa::Direction $cppClass::getPortDirection() {
      return kanagawa::Direction::Input;
    }
  }];

  let builders = [
    OpBuilder<(ins "StringAttr":$inner_sym, "Type":$innerPortType, CArg<"mlir::StringAttr", "nullptr">:$name), [{
      build($_builder, $_state,
        {$_builder.getType<PortRefType>(
          innerPortType, kanagawa::Direction::Input), innerPortType}, hw::InnerSymAttr::get(inner_sym), name);
    }]>
  ];

  let hasCanonicalizeMethod = 1;
}

def OutputWireOp : WireLikeOp<"wire.output", [
  InnerTypeToPortrefTypeConstraint<"input", "port", "Output">
]> {
  let summary = "Kanagawa output wire";
  let description = [{
    An output wire defines an `kanagawa.portref<out T>` port that can be read.
    The operation takes an input value of type `T` which represents the value
    on the output portref.
  }];

  let arguments = (ins InnerSymAttr:$inner_sym, AnyType:$input, OptionalAttr<StrAttr>:$name);
  let results = (outs PortRefType:$port);
  let assemblyFormat = [{
    $inner_sym `,` $input `:` qualified(type($input)) attr-dict
  }];

  let extraWireClassDefinition = [{
    kanagawa::Direction $cppClass::getPortDirection() {
      return kanagawa::Direction::Output;
    }
  }];

  let hasCanonicalizeMethod = 1;
}

def PortReadOp : KanagawaOp<"port.read", [
      PortRefToInnerTypeConstraint<"port", "output">,
      Pure,
      HasCustomSSAName
]> {
  let summary = "Kanagawa port read";
  let description = [{
    Read the value of a port reference.
  }];

  let arguments = (ins PortRefType:$port);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    $port attr-dict `:` qualified(type($port))
  }];
}

def PortWriteOp : KanagawaOp<"port.write", [
      PortRefToInnerTypeConstraint<"port", "value">
]> {
  let summary = "Kanagawa port write";
  let description = [{
    Write a value to a port reference.
  }];

  let arguments = (ins PortRefType:$port, AnyType:$value);
  let assemblyFormat = [{
    $port `,` $value attr-dict `:` qualified(type($port))
  }];
}

def PipelineHeaderOp : KanagawaOp<"pipeline.header", [
  Pure
]> {
  let summary = "Kanagawa pipeline header operation";
  let description = [{
    This operation defines the hardware-like values used to drive a pipeline,
    such as clock and reset.
    This is an intermediate operation, meaning that it's strictly used to
    facilitate progressive lowering of kanagawa static blocks to scheduled pipelines.
  }];

  let arguments = (ins);
  let results = (outs ClockType:$clock, I1:$reset, I1:$go, I1:$stall);
  let assemblyFormat = "attr-dict";
}

#endif // CIRCT_DIALECT_KANAGAWA_KANAGAWAOPS_TD
