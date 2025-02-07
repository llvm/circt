//===- ArcOps.td - Arc dialect operations ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_ARCOPS_TD
#define CIRCT_DIALECT_ARC_ARCOPS_TD

include "circt/Dialect/Arc/ArcDialect.td"
include "circt/Dialect/Arc/ArcInterfaces.td"
include "circt/Dialect/Arc/ArcTypes.td"
include "circt/Dialect/HW/HWTypes.td"
include "circt/Dialect/Seq/SeqTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class ArcOp<string mnemonic, list<Trait> traits = []> :
  Op<ArcDialect, mnemonic, traits>;

def DefineOp : ArcOp<"define", [
  IsolatedFromAbove,
  FunctionOpInterface,
  Symbol,
  RegionKindInterface,
  SingleBlockImplicitTerminator<"arc::OutputOp">,
  HasParent<"mlir::ModuleOp">
]> {
  let summary = "State transfer arc definition";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;

  let hasRegionVerifier = 1;

  let builders = [
    OpBuilder<(ins "mlir::StringAttr":$sym_name,
                   "mlir::TypeAttr":$function_type), [{
      build($_builder, $_state, sym_name, function_type, mlir::ArrayAttr(),
            mlir::ArrayAttr());
    }]>,
    OpBuilder<(ins "mlir::StringRef":$sym_name,
                   "mlir::FunctionType":$function_type), [{
      build($_builder, $_state, sym_name, function_type, mlir::ArrayAttr(),
            mlir::ArrayAttr());
    }]>,
  ];

  let extraClassDeclaration = [{
    static mlir::RegionKind getRegionKind(unsigned index) {
      return mlir::RegionKind::SSACFG;
    }

    mlir::Block &getBodyBlock() { return getBody().front(); }

    // Get the arc's symbolic name.
    mlir::StringAttr getNameAttr() {
      return (*this)->getAttrOfType<mlir::StringAttr>(
        ::mlir::SymbolTable::getSymbolAttrName());
    }

    // Get the arc's symbolic name.
    mlir::StringRef getName() {
      return getNameAttr().getValue();
    }

    /// Returns the argument types of this function.
    mlir::ArrayRef<mlir::Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }

    /// Returns the result types of this function.
    mlir::ArrayRef<mlir::Type> getResultTypes() {
      return getFunctionType().getResults();
    }

    /// Verify the type attribute of this function. Returns failure and emits
    /// an error if the attribute is invalid.
    mlir::LogicalResult verifyType() {
      auto type = getFunctionTypeAttr().getValue();
      if (!llvm::isa<mlir::FunctionType>(type))
        return emitOpError("requires '") << getFunctionTypeAttrName() <<
                           "' attribute of function type";
      return mlir::success();
    }

    /// Returns true if the arc returns the inputs directly and in the same
    /// order, otherwise false.
    bool isPassthrough();

    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    mlir::Region *getCallableRegion() { return &getBody(); }
  }];
}

def OutputOp : ArcOp<"output", [
  Terminator,
  ParentOneOf<["DefineOp", "LutOp", "ClockDomainOp"]>,
  Pure,
  ReturnLike
]> {
  let summary = "Arc terminator";
  let arguments = (ins Variadic<AnyType>:$outputs);
  let assemblyFormat = [{
    attr-dict ($outputs^ `:` qualified(type($outputs)))?
  }];
  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];
  let hasVerifier = 1;
}

def StateOp : ArcOp<"state", [
  MemRefsNormalizable,
  CallOpInterface,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<ClockedOpInterface>,
  PredOpTrait<"types of initial arguments match result types",
    CPred<[{getInitials().empty() ||
              llvm::equal(getInitials().getType(), getResults().getType())}]>>
]> {
  let summary = "State transfer arc";

  let arguments = (ins
    FlatSymbolRefAttr:$arc,
    Optional<ClockType>:$clock,
    Optional<I1>:$enable,
    Optional<I1>:$reset,
    I32Attr:$latency,
    Variadic<AnyType>:$inputs,
    Variadic<AnyType>:$initials);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $arc `(` $inputs `)` (`clock` $clock^)? (`enable` $enable^)?
    (`reset` $reset^)?
    ( `initial` ` ` `(` $initials^ `:` type($initials) `)`)?
    `latency` $latency attr-dict `:` functional-type($inputs, results)
  }];

  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;

  let builders = [
    OpBuilder<(ins "DefineOp":$arc, "mlir::Value":$clock, "mlir::Value":$enable,
      "unsigned":$latency, CArg<"mlir::ValueRange", "{}">:$inputs,
      CArg<"mlir::ValueRange", "{}">:$initials), [{
      build($_builder, $_state, mlir::SymbolRefAttr::get(arc),
            arc.getFunctionType().getResults(), clock, enable, latency,
            inputs, initials);
    }]>,
    OpBuilder<(ins "mlir::SymbolRefAttr":$arc, "mlir::TypeRange":$results,
      "mlir::Value":$clock, "mlir::Value":$enable, "unsigned":$latency,
      CArg<"mlir::ValueRange", "{}">:$inputs,
      CArg<"mlir::ValueRange", "{}">:$initials
    ), [{
      build($_builder, $_state, arc, results, clock, enable, Value(), latency,
      inputs, initials);
    }]>,
    OpBuilder<(ins "mlir::SymbolRefAttr":$arc, "mlir::TypeRange":$results,
      "mlir::Value":$clock, "mlir::Value":$enable, "mlir::Value":$reset,
      "unsigned":$latency, CArg<"mlir::ValueRange", "{}">:$inputs,
      CArg<"mlir::ValueRange", "{}">:$initials
    ), [{
      if (clock)
        $_state.addOperands(clock);
      if (enable)
        $_state.addOperands(enable);
      if (reset)
        $_state.addOperands(reset);
      $_state.addOperands(inputs);
      $_state.addOperands(initials);
      $_state.addAttribute("arc", arc);
      $_state.addAttribute("latency", $_builder.getI32IntegerAttr(latency));
      $_state.addAttribute(getOperandSegmentSizeAttr(),
        $_builder.getDenseI32ArrayAttr({
          clock ? 1 : 0,
          enable ? 1 : 0,
          reset ? 1 : 0,
          static_cast<int32_t>(inputs.size()),
          static_cast<int32_t>(initials.size())}));
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "mlir::StringAttr":$arc, "mlir::TypeRange":$results,
      "mlir::Value":$clock, "mlir::Value":$enable, "unsigned":$latency,
      CArg<"mlir::ValueRange", "{}">:$inputs,
      CArg<"mlir::ValueRange", "{}">:$initials
    ), [{
      build($_builder, $_state, mlir::SymbolRefAttr::get(arc), results, clock,
            enable, latency, inputs, initials);
    }]>,
    OpBuilder<(ins "mlir::StringRef":$arc, "mlir::TypeRange":$results,
      "mlir::Value":$clock, "mlir::Value":$enable, "unsigned":$latency,
      CArg<"mlir::ValueRange", "{}">:$inputs,
      CArg<"mlir::ValueRange", "{}">:$initials
    ), [{
      build($_builder, $_state,
            mlir::StringAttr::get($_builder.getContext(), arc),
            results, clock, enable, latency, inputs, initials);
    }]>
  ];
  let skipDefaultBuilders = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    operand_range getArgOperands() {
      return getInputs();
    }
    MutableOperandRange getArgOperandsMutable() {
      return getInputsMutable();
    }

    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("arc");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr(getArcAttrName(), llvm::cast<mlir::SymbolRefAttr>(callee));
    }
  }];
}

def CallOp : ArcOp<"call", [
  MemRefsNormalizable, Pure,
  CallOpInterface,
  DeclareOpInterfaceMethods<ClockedOpInterface, ["isClocked", "eraseClock"]>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "calls an arc";

  let arguments = (ins FlatSymbolRefAttr:$arc, Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $arc `(` $inputs `)` attr-dict `:` functional-type(operands, results)
  }];

  let builders = [
    OpBuilder<(ins "DefineOp":$arc, CArg<"mlir::ValueRange", "{}">:$inputs), [{
      build($_builder, $_state, arc.getFunctionType().getResults(),
            mlir::SymbolRefAttr::get(arc), inputs);
    }]>,
  ];

  let extraClassDeclaration = [{
    operand_range getArgOperands() {
      return {operand_begin(), operand_end()};
    }
    MutableOperandRange getArgOperandsMutable() {
      return getInputsMutable();
    }

    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("arc");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr(getArcAttrName(), llvm::cast<mlir::SymbolRefAttr>(callee));
    }
  }];
}

def MemoryOp : ArcOp<"memory", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Memory";
  let results = (outs MemoryType:$memory);
  let assemblyFormat = "type($memory) attr-dict";
}

class MemoryAndDataTypesMatch<string mem, string data> : TypesMatchWith<
  "memory and data types must match", mem, data,
  "llvm::cast<MemoryType>($_self).getWordType()">;
class MemoryAndAddressTypesMatch<string mem, string address> : TypesMatchWith<
  "memory and address types must match", mem, address,
  "llvm::cast<MemoryType>($_self).getAddressType()">;

def MemoryReadPortOp : ArcOp<"memory_read_port", [
  Pure,
  MemoryAndDataTypesMatch<"memory", "data">,
  MemoryAndAddressTypesMatch<"memory", "address">
]> {
  let summary = "Read port from a memory";
  let description = [{
    Represents a combinatorial memory read port. No memory read side-effect
    trait is necessary because at the stage of the Arc lowering where this
    operation is legal to be present, it is guaranteed that all reads from the
    same address produce the same output. This is because all writes are
    reordered to happen at the end of the cycle in LegalizeStateUpdates (or
    alternatively produce the necessary temporaries).
  }];
  let arguments = (ins
    MemoryType:$memory,
    AnyInteger:$address
  );
  let results = (outs AnyInteger:$data);

  let assemblyFormat = [{
    $memory `[` $address `]` attr-dict `:` type($memory)
  }];
}

def MemoryWritePortOp : ArcOp<"memory_write_port", [
  MemoryEffects<[MemWrite]>,
  CallOpInterface,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  AttrSizedOperandSegments,
  ClockedOpInterface
]> {
  let summary = "Write port to a memory";
  let arguments = (ins
    MemoryType:$memory,
    FlatSymbolRefAttr:$arc,
    Variadic<AnyType>:$inputs,
    Optional<ClockType>:$clock,
    UnitAttr:$enable,
    UnitAttr:$mask,
    DefaultValuedAttr<I32Attr, "1">:$latency
  );

  let assemblyFormat = [{
    $memory `,` $arc  `(` $inputs `)` (`clock` $clock^)?  (`enable` $enable^)?
    (`mask` $mask^)? `latency` $latency attr-dict `:`
    type($memory) `,` type($inputs)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    SmallVector<Type> getArcResultTypes();
    static unsigned getAddressIdx() { return 0; }
    static unsigned getDataIdx() { return 1; }
    static unsigned getEnableIdx() { return 2; }
    static unsigned getMaskIdx(bool hasEnable) { return hasEnable ? 3 : 2; }

    operand_range getArgOperands() {
      return getInputs();
    }
    MutableOperandRange getArgOperandsMutable() {
      return getInputsMutable();
    }

    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("arc");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr(getArcAttrName(), llvm::cast<mlir::SymbolRefAttr>(callee));
    }
  }];
}

def MemoryReadOp : ArcOp<"memory_read", [
  MemoryEffects<[MemRead]>,
  MemoryAndDataTypesMatch<"memory", "data">,
  MemoryAndAddressTypesMatch<"memory", "address">
]> {
  let summary = "Read a word from a memory";
  let arguments = (ins
    MemoryType:$memory,
    AnyInteger:$address
  );
  let results = (outs AnyInteger:$data);
  let assemblyFormat = [{
    $memory `[` $address `]` attr-dict `:` type($memory)
  }];
}

def MemoryWriteOp : ArcOp<"memory_write", [
  MemoryEffects<[MemWrite]>,
  MemoryAndDataTypesMatch<"memory", "data">,
  MemoryAndAddressTypesMatch<"memory", "address">
]> {
  let summary = "Write a word to a memory";
  let arguments = (ins
    MemoryType:$memory,
    AnyInteger:$address,
    Optional<I1>:$enable,
    AnyInteger:$data
  );
  let assemblyFormat = [{
    $memory `[` $address `]` `,` $data (`if` $enable^)?
    attr-dict `:` type($memory)
  }];
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// Trigger Grouping
//===----------------------------------------------------------------------===//

def ClockDomainOp : ArcOp<"clock_domain", [
  IsolatedFromAbove,
  RegionKindInterface,
  RecursiveMemoryEffects,
  SingleBlockImplicitTerminator<"arc::OutputOp">
]> {
  let summary = "a clock domain";

  let arguments = (ins Variadic<AnyType>:$inputs, ClockType:$clock);
  let results = (outs Variadic<AnyType>:$outputs);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    ` ` `(` $inputs `)` `clock` $clock attr-dict `:`
    functional-type($inputs, results) $body
  }];

  let extraClassDeclaration = [{
    static mlir::RegionKind getRegionKind(unsigned index) {
      return mlir::RegionKind::Graph;
    }
    mlir::Block &getBodyBlock() { return getBody().front(); }
  }];

  let hasRegionVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// (Pseudo) Clock Trees
//===----------------------------------------------------------------------===//

class ClockTreeLikeOp<string mnemonic, list<Trait> traits = []>:
  ArcOp<mnemonic, !listconcat(traits, [
    RecursiveMemoryEffects, NoTerminator, NoRegionArguments, SingleBlock,
    HasParent<"ModelOp">
])> {
  let regions = (region SizedRegion<1>:$body);
}

def InitialOp : ClockTreeLikeOp<"initial"> {
  let summary = "Region to be executed at the start of simulation";
  let assemblyFormat = [{
    attr-dict-with-keyword $body
  }];
}

def FinalOp : ClockTreeLikeOp<"final"> {
  let summary = "Region to be executed at the end of simulation";
  let assemblyFormat = [{
    attr-dict-with-keyword $body
  }];
}

//===----------------------------------------------------------------------===//
// Storage Allocation
//===----------------------------------------------------------------------===//

def AllocStateOp : ArcOp<"alloc_state", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate internal state";
  let arguments = (ins StorageType:$storage, UnitAttr:$tap);
  let results = (outs StateType:$state);
  let assemblyFormat = [{
    $storage (`tap` $tap^)? attr-dict `:` functional-type($storage, $state)
  }];
}

def AllocMemoryOp : ArcOp<"alloc_memory", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate a memory";
  let arguments = (ins StorageType:$storage);
  let results = (outs MemoryType:$memory);
  let assemblyFormat = [{
    $storage attr-dict `:` functional-type($storage, $memory)
  }];
}

def AllocStorageOp : ArcOp<"alloc_storage", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate contiguous storage space from a larger storage space";
  let arguments = (ins StorageType:$input, OptionalAttr<I32Attr>:$offset);
  let results = (outs StorageType:$output);
  let assemblyFormat = [{
    $input (`[` $offset^ `]`)? attr-dict `:` functional-type($input, $output)
  }];
}

def RootInputOp : ArcOp<"root_input", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
]> {
  let summary = "A root input";
  let arguments = (ins StrAttr:$name, StorageType:$storage);
  let results = (outs StateType:$state);
  let assemblyFormat = [{
    $name `,` $storage attr-dict `:` functional-type($storage, $state)
  }];
}

def RootOutputOp : ArcOp<"root_output", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
]> {
  let summary = "A root output";
  let arguments = (ins StrAttr:$name, StorageType:$storage);
  let results = (outs StateType:$state);
  let assemblyFormat = [{
    $name `,` $storage attr-dict `:` functional-type($storage, $state)
  }];
}

//===----------------------------------------------------------------------===//
// Storage Access
//===----------------------------------------------------------------------===//

def AllocatableType : AnyTypeOf<[StateType, MemoryType, StorageType]>;

def StorageGetOp : ArcOp<"storage.get", [Pure]> {
  let summary = "Access an allocated state, memory, or storage slice";
  let arguments = (ins StorageType:$storage, I32Attr:$offset);
  let results = (outs AllocatableType:$result);
  let assemblyFormat = [{
    $storage `[` $offset `]` attr-dict
    `:` qualified(type($storage)) `->` type($result)
  }];
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// State Read/Write
//===----------------------------------------------------------------------===//

class StateAndValueTypesMatch<string state, string value> : TypesMatchWith<
  "state and value types must match", state, value,
  "llvm::cast<StateType>($_self).getType()">;

def StateReadOp : ArcOp<"state_read", [
  MemoryEffects<[MemRead]>,
  StateAndValueTypesMatch<"state", "value">,
]> {
  let summary = "Read a state's value";
  let arguments = (ins StateType:$state);
  let results = (outs AnyType:$value);
  let assemblyFormat = [{
    $state attr-dict `:` type($state)
  }];
}

def StateWriteOp : ArcOp<"state_write", [
  MemoryEffects<[MemWrite]>,
  StateAndValueTypesMatch<"state", "value">,
]> {
  let summary = "Update a state's value";
  let arguments = (ins
    StateType:$state,
    AnyType:$value,
    Optional<I1>:$condition
  );
  let assemblyFormat = [{
    $state `=` $value (`if` $condition^)? attr-dict `:` type($state)
  }];
}

//===----------------------------------------------------------------------===//
// Simulation Orchestration
//===----------------------------------------------------------------------===//

def SimInstantiateOp : ArcOp<"sim.instantiate",
  [MemoryEffects<[MemAlloc]>, NoTerminator,
   DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Instantiates an Arc model for simulation";
  let description = [{
    Creates an instance of an Arc model in scope, in order to simulate it.
    The model can be used from within the associated region, modelling its
    lifetime.
  }];
  let regions = (region SizedRegion<1>:$body);

  let hasRegionVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def SimSetInputOp : ArcOp<"sim.set_input",
  [MemoryEffects<[MemWrite]>,
   DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Sets the value of an input of the model instance";
  let description = [{
    Sets the value of an input port in a specific instance of a model. The
    provided input port must be of input type on the model and its type must
    match the type of the value operand.
  }];
  let arguments = (ins SimModelInstance:$instance,
                       StrAttr:$input,
                       AnyType:$value);
  let assemblyFormat =
    [{ $instance `,` $input `=` $value attr-dict
       `:` type($value) `,` qualified(type($instance)) }];
}

def SimGetPortOp : ArcOp<"sim.get_port",
  [MemoryEffects<[MemRead]>, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Gets the value of a port of the model instance";
  let description = [{
    Gets the value of the given port in a specific instance of a model. The
    provided port must be of the type of the expected value.
  }];
  let arguments = (ins SimModelInstance:$instance, StrAttr:$port);
  let results = (outs AnyType:$value);
  let assemblyFormat =
    [{ $instance `,` $port attr-dict
       `:` type($value) `,` qualified(type($instance)) }];
}

def SimStepOp : ArcOp<"sim.step",
  [MemoryEffects<[MemRead, MemWrite]>,
   DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary =
    "Evaluates one step of the simulation for the provided model instance";
  let description = [{
    Evaluates one step of the simulation for the provided model instance,
    updating ports accordingly.
  }];
  let arguments = (ins SimModelInstance:$instance);
  let assemblyFormat =
    [{ $instance attr-dict `:` qualified(type($instance)) }];
}

def SimEmitValueOp : ArcOp<"sim.emit"> {
  let summary = "Sends a value to the simulation driver";
  let description = [{
    Sends a named value to the simulation driver. This is notably useful
    for printing values during simulation.
  }];
  let arguments = (ins StrAttr:$valueName, AnyType:$value);
  let assemblyFormat = [{ $valueName `,` $value attr-dict `:` type($value) }];
}

//===----------------------------------------------------------------------===//
// Miscellaneous
//===----------------------------------------------------------------------===//

def TapOp : ArcOp<"tap"> {
  let summary = "A tracker op to observe a value under a given name";
  let arguments = (ins AnyType:$value, StrAttr:$name);
  let assemblyFormat = [{ $value attr-dict `:` type($value) }];
}

def ModelOp : ArcOp<"model", [
  IsolatedFromAbove, NoTerminator, Symbol,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "A model with stratified clocks";
  let description = [{
    A model with stratified clocks. The `io` optional attribute
    specifies the I/O of the module associated to this model.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<ModuleType>:$io,
                       OptionalAttr<FlatSymbolRefAttr>:$initialFn,
                       OptionalAttr<FlatSymbolRefAttr>:$finalFn);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $sym_name `io` $io
    (`initializer` $initialFn^)?
    (`finalizer` $finalFn^)?
    attr-dict-with-keyword $body
  }];

  let extraClassDeclaration = [{
    mlir::Block &getBodyBlock() { return getBody().front(); }
  }];

  let hasVerifier = 1;
}

def LutOp : ArcOp<"lut", [
  IsolatedFromAbove,
  SingleBlockImplicitTerminator<"arc::OutputOp">,
  Pure
]> {
  let summary = "A lookup-table.";
  let description = [{
    Represents a lookup-table as one operation. The operations that map the
    lookup/input values to the corresponding table-entry are collected inside
    the body of this operation.
    Note that the operation is marked to be isolated from above to guarantee
    that all input values have to be passed as an operand. This allows for
    simpler analyses and canonicalizations of the LUT as well as lowering.
    Only combinational operations are allowed inside the LUT, i.e., no
    side-effects, state, time delays, etc.
  }];

  let arguments = (ins Variadic<AnySignlessInteger>:$inputs);
  let results = (outs AnySignlessInteger:$output);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    `(` $inputs `)` `:` functional-type($inputs, $output)
    attr-dict-with-keyword $body
  }];

  let extraClassDeclaration = [{
    mlir::Block *getBodyBlock() { return &getBody().front(); }
  }];

  let hasVerifier = 1;
}

def ZeroCountPredicateLeading  : I32EnumAttrCase<"leading", 0>;
def ZeroCountPredicateTrailing  : I32EnumAttrCase<"trailing", 1>;
def ZeroCountPredicate : I32EnumAttr<
    "ZeroCountPredicate", "arc.zero_count predicate",
    [ZeroCountPredicateLeading, ZeroCountPredicateTrailing]> {
  let cppNamespace = "circt::arc";
}

def ZeroCountOp : ArcOp<"zero_count", [Pure, SameOperandsAndResultType]> {
  let summary = "leading/trailing zero count operation";
  let arguments = (ins AnySignlessInteger:$input,
                       ZeroCountPredicate:$predicate);
  let results = (outs AnySignlessInteger:$output);
  let assemblyFormat = "$predicate $input attr-dict `:` type($input)";
}

def IntOr1DVectorOfInt : AnyTypeOf<[
  AnySignlessInteger,
  VectorOfRankAndType<[1], [AnySignlessInteger]>
]>;

def VectorizeOp : ArcOp<"vectorize", [
    IsolatedFromAbove, RecursiveMemoryEffects
  ]> {
  let summary = "isolated subgraph of operations to be vectorized";
  let description = [{
    This operation represents a vectorized computation DAG. It places a
    convenient boundary between the subgraph to be vectorized and the
    surrounding non-vectorizable parts of the original graph.

    This allows us to split the vectorization transformations into multiple
    parts/passes:
      * Finding an initial set of operations to be vectorized
      * Optimizing this set by pulling in more operations into the nested block,
        splitting it such that the vector width does not exceed a given limit,
        applying a cost model and potentially reverting the decision to
        vectorize this subgraph (e.g., because not enough ops could be pulled
        in)
      * Performing the actual vectorization by lowering this operation. This
        operation allows to perform the lowering of the boundary and the body
        separately and either via 1D `vector` types for SIMD vectorization or
        plain integers for manual vectorization within a scalar register.

    For each block argument of the nested block, there is a list of operands
    that represent the elements of the vector. If the boundary is already
    vectorized each list will only contain a single SSA value of either vector
    type or an integer representing the concatenation of all original operands
    of that vector.

    Example:

    Given the following two AND operations in the IR
    ```mlir
    %0 = arith.and %in0, %in1 : i1
    %1 = arith.and %in2, %in2 : i1
    ```
    they could be vectorized by putting one such AND operation in the body block
    of the `arc.vectorize` operation and forwarding the operands accordingly.
    ```mlir
    %0:2 = arc.vectorize (%in0, %in1), (%in2, %in2) :
      (i1, i1, i1, i1) -> (i1, i1) {
    ^bb0(%arg0: i1, %arg1: i1):
      %1 = arith.and %arg0, %arg1 : i1
      arc.output %1 : i1
    }
    ```
    In a next step, the boundary could be lowered/vectorized. This can happen
    in terms of integers for vectorization within scalar registers:
    ```mlir
    %0 = comb.concat %in0, %in1 : i1, i1
    %1 = comb.replicate %in2 : (i1) -> i2
    %2 = arc.vectorize (%0), (%1) : (i2, i2) -> (i2) {
    ^bb0(%arg0: i1, %arg1: i1):
      %1 = arith.and %arg0, %arg1 : i1
      arc.output %1 : i1
    }
    %3 = comb.extract %2 from 1 : (i2) -> i1
    %4 = comb.extract %2 from 0 : (i2) -> i1
    ```
    Or via `vector` types for SIMD vectorization:
    ```mlir
    %cst = arith.constant dense<0> : vector<2xi1>
    %0 = vector.insert %in0, %cst[0] : i1 into vector<2xi1>
    %1 = vector.insert %in1, %0[1] : i1 into vector<2xi1>
    %2 = vector.broadcast %in2 : i1 to vector<2xi1>
    %3 = arc.vectorize (%1), (%2) :
      (vector<2xi1>, vector<2xi1>) -> (vector<2xi1>) {
    ^bb0(%arg0: i1, %arg1: i1):
      %1 = arith.and %arg0, %arg1 : i1
      arc.output %1 : i1
    }
    %4 = vector.extract %2[0] : vector<2xi1>
    %5 = vector.extract %2[1] : vector<2xi1>
    ```
    Alternatively, the body could be vectorized first. Again, as integers
    ```mlir
    %0:2 = arc.vectorize (%in0, %in1), (%in2, %in2) :
      (i1, i1, i1, i1) -> (i1, i1) {
    ^bb0(%arg0: i2, %arg1: i2):
      %1 = arith.and %arg0, %arg1 : i2
      arc.output %1 : i2
    }
    ```
    or SIMD vectors.
    ```mlir
    %0:2 = arc.vectorize (%in0, %in1), (%in2, %in3) :
      (i1, i1, i1, i1) -> (i1, i1) {
    ^bb0(%arg0: vector<2xi1>, %arg1: vector<2xi1>):
      %1 = arith.and %arg0, %arg1 : vector<2xi1>
      arc.output %1 : vector<2xi1>
    }
    ```
    Once both sides are lowered, the `arc.vectorize` op simply becomes a
    passthrough for the operands and can be removed by inlining the nested
    block. The integer based vectorization would then look like the following:
    ```mlir
    %0 = comb.concat %in0, %in1 : i1, i1
    %1 = comb.replicate %in2 : (i1) -> i2
    %2 = arith.and %0, %1 : i2
    %3 = comb.extract %2 from 1 : (i2) -> i1
    %4 = comb.extract %2 from 0 : (i2) -> i1
    ```
    The SIMD vector based lowering would result in the following IR:
    ```mlir
    %cst = arith.constant dense<0> : vector<2xi1>
    %0 = vector.insert %in0, %cst[0] : i1 into vector<2xi1>
    %1 = vector.insert %in1, %0[1] : i1 into vector<2xi1>
    %2 = vector.broadcast %in2 : i1 to vector<2xi1>
    %3 = arith.and %1, %2 : vector<2xi1>
    %4 = vector.extract %3[0] : vector<2xi1>
    %5 = vector.extract %3[1] : vector<2xi1>
    ```
  }];

  let arguments = (ins
    VariadicOfVariadic<AnyType, "inputOperandSegments">:$inputs,
    DenseI32ArrayAttr:$inputOperandSegments);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $inputs attr-dict `:` functional-type($inputs, $results) $body
  }];

  let extraClassDeclaration = [{
    /// Returns whether the `arc.vectorize` boundary is already vectorized.
    /// The boundary is vectorized if each input vector has only one element
    /// (either of integer or vector type), i.e., the vector itself.
    bool isBoundaryVectorized();

    /// Returns whether the body block of the `arc.vectorize` is already
    /// vectorized. The body is vectorized if the boundary is vectorized and
    /// the body's argument and result types match those of the boundary, or
    /// they are a vectorized version of the boundary types (i.e., a matching
    /// vector type or integer with summed-up bitwidth).
    bool isBodyVectorized();
  }];

  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def VectorizeReturnOp : ArcOp<"vectorize.return", [
  HasParent<"VectorizeOp">, Pure, ReturnLike, Terminator
]> {
  let summary = "arc.vectorized terminator";
  let arguments = (ins AnyType:$value);
  let assemblyFormat = "operands attr-dict `:` qualified(type(operands))";
}

#endif // CIRCT_DIALECT_ARC_ARCOPS_TD
