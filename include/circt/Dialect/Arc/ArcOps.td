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
  ParentOneOf<["DefineOp", "LutOp", "ExecuteOp"]>,
  Pure,
  ReturnLike
]> {
  let summary = "Arc terminator";
  let arguments = (ins Variadic<AnyType>:$outputs);
  let assemblyFormat = [{
    attr-dict ($outputs^ `:` qualified(type($outputs)))?
  }];
  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, mlir::ValueRange());
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
  let summary = "Instantiates a state element with input from a transfer arc";

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
    AnyInteger:$data
  );
  let assemblyFormat = [{
    $memory `[` $address `]` `,` $data attr-dict `:` type($memory)
  }];
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
  let arguments = (ins StorageType:$input,
                       UI32Attr:$size,
                       OptionalAttr<I32Attr>:$offset);
  let results = (outs StorageType:$output);
  let assemblyFormat = [{
    $input (`[` $offset^ `]`)? `,` $size attr-dict
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

def AllocatableType : AnyTypeOf<[StateType, MemoryType, StorageType,
                                 ArrayRefType]>;

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
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Update a state's value";
  let arguments = (ins
    StateType:$state,
    AnyType:$value,
    OptionalAttr<FlatSymbolRefAttr>:$traceTapModel,
    OptionalAttr<UI64Attr>:$traceTapIndex);
  let assemblyFormat = [{
    $state `=` $value
    (`tap` $traceTapModel`[`$traceTapIndex^`]` )? attr-dict `:` type($state)
  }];
  let hasVerifier = true;
  let builders = [
    OpBuilder<(ins  "mlir::Value":$state,
                    "mlir::Value":$value), [{
      build($_builder, $_state, state, value,
              mlir::FlatSymbolRefAttr{}, mlir::IntegerAttr{});
    }]>
  ];
}

def CurrentTimeOp : ArcOp<"current_time", [
  MemoryEffects<[MemRead]>
]> {
  let summary = "Read the current simulation time";
  let description = [{
    Reads the current simulation time from the model's storage. The time is
    represented as an `i64` value in femtoseconds.
  }];
  let arguments = (ins StorageType:$storage);
  let results = (outs I64:$time);
  let assemblyFormat = [{
    $storage attr-dict `:` qualified(type($storage))
  }];
}

def TerminateOp : ArcOp<"terminate", [
  MemoryEffects<[MemWrite]>
]> {
  let summary = "Request simulation termination";
  let description = [{
    Sets a termination flag in the model's storage.
    It unconditionally writes 1 for success or 2 for failure.
    This allows the simulation to exit gracefully after the current
    evaluation cycle.
  }];

  let arguments = (ins
    StorageType:$storage,
    BoolAttr:$success
  );

  let assemblyFormat = [{
    $storage `,` $success attr-dict `:` qualified(type($storage))
  }];
}

def GetNextWakeupOp : ArcOp<"get_next_wakeup", [
  MemoryEffects<[MemRead]>
]> {
  let summary = "Read the model's next wakeup time";
  let description = [{
    Reads the next wakeup time from the model's storage. The time is an
    absolute timestamp in femtoseconds, not a duration relative to the current
    simulation time. A wakeup at the current time requests an immediate
    re-evaluation in the same time step; a value of `UINT64_MAX` indicates that
    no wakeup is pending.
  }];
  let arguments = (ins StorageType:$storage);
  let results = (outs I64:$time);
  let assemblyFormat = [{
    $storage attr-dict `:` qualified(type($storage))
  }];
}

def SetNextWakeupOp : ArcOp<"set_next_wakeup", [
  MemoryEffects<[MemWrite]>
]> {
  let summary = "Set the model's next wakeup time";
  let description = [{
    Writes the next wakeup time into the model's storage. The time is
    represented as an `i64` value in femtoseconds. The model's eval body
    establishes the slot at `UINT64_MAX` ("no wakeup pending") on entry, and
    process suspension code lowers the value to the earliest scheduled
    wakeup over the course of an evaluation.
  }];
  let arguments = (ins StorageType:$storage, I64:$time);
  let assemblyFormat = [{
    $storage `,` $time attr-dict `:` qualified(type($storage))
  }];
}

//===----------------------------------------------------------------------===//
// Coroutines
//===----------------------------------------------------------------------===//

def CoroutineDefineOp : ArcOp<"coroutine.define", [
  FunctionOpInterface,
  IsolatedFromAbove,
  Symbol,
]> {
  let summary = "Coroutine definition";
  let description = [{
    Define a coroutine. Coroutines are suspendable functions that are entered at
    their entry block or re-entered at a resume block. Local state and values
    are persisted by an opaque state held by the caller, alongside an opaque
    program counter value indicating at which point the coroutine should be
    resumed.

    Coroutines can be suspended by the `arc.coroutine.yield` terminator, which
    returns control and local state back to the caller. The caller can then
    re-enter the coroutine by passing that control and local state back into the
    coroutine. Coroutines can be finished by the `arc.coroutine.return` and
    `arc.coroutine.halt` terminators, which return control back to the caller
    with a corresponding program counter indicating return or halt.

    Arguments are passed to the coroutine upon each entry. The entry block's
    arguments hold the values supplied by the caller on the first call. Each
    block targeted by a suspension op must have its first arguments match the
    coroutine's function type. On resumption, those arguments are bound to the
    values supplied by the caller and may differ from the values passed on prior
    entries.

    Results are returned from the coroutine upon each suspension. Each of the
    terminators must provide a set of values to be yielded back to the caller
    upon suspension.

    The local state of a coroutine is represented by `!arc.coroutine_state<@A>`,
    and the program counter for resuming by `!arc.coroutine_pc<@A>`. These types
    are opaque and are expanded to a concrete union/struct of local state and a
    concrete integer PC via a lowering. These types are only used on coroutine
    calls; coroutine definitions define them implicitly by values carried across
    resume points.

    To nest coroutines, a coroutine definition can call another coroutine and
    carry the `!arc.coroutine_state` and `!arc.coroutine_pc` of that call as
    local state across its own suspension points.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let results = (outs);
  let regions = (region MinSizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "mlir::StringAttr":$sym_name,
                   "mlir::TypeAttr":$function_type), [{
      build($_builder, $_state, sym_name, function_type, mlir::ArrayAttr(),
            mlir::ArrayAttr());
    }]>,
    OpBuilder<(ins "mlir::StringRef":$sym_name,
                   "mlir::FunctionType":$function_type), [{
      build($_builder, $_state,
            $_builder.getStringAttr(sym_name),
            mlir::TypeAttr::get(function_type),
            mlir::ArrayAttr(), mlir::ArrayAttr());
    }]>,
  ];

  let extraClassDeclaration = [{
    /// Returns the argument types of this coroutine.
    mlir::ArrayRef<mlir::Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }

    /// Returns the result types of this coroutine.
    mlir::ArrayRef<mlir::Type> getResultTypes() {
      return getFunctionType().getResults();
    }

    mlir::Region *getCallableRegion() { return &getBody(); }
  }];
}

// A coroutine state/PC type whose `coroutine` parameter matches the enclosing
// op's `callee` symbol attribute.
class CoroutineTypeOfCallee<ArcTypeDef baseType> : ConfinedType<baseType, [
  CPred<"::llvm::cast<" # baseType.cppType # ">($_self).getCoroutine() == "
        "($_op).getAttrOfType<::mlir::FlatSymbolRefAttr>(\"callee\")">
], baseType.summary # " bound to the op's callee symbol", baseType.cppType>;

def CoroutineStateOfCallee : CoroutineTypeOfCallee<CoroutineStateType>;
def CoroutinePCOfCallee : CoroutineTypeOfCallee<CoroutinePCType>;

def CoroutineCallOp : ArcOp<"coroutine.call", [
  CallOpInterface,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
]> {
  let summary = "Call a coroutine";
  let description = [{
    Call an `arc.coroutine.define`. The coroutine is resumed at the point
    indicated by the `pc` operand, and local state is restored from the `state`
    operand. The `args` are passed into the coroutine and may differ between
    subsequent re-entries. When the coroutine suspends or finishes, control is
    transferred back to the caller and the call op returns the coroutine's
    resume program counter, resume state, and the values yielded back from the
    coroutine as results.

    Passing a `return` or `halt` program counter to the callee results in
    undefined behavior. These program counter values indicate that the coroutine
    has finished or suspended forever, respectively, and the coroutine must not
    be re-entered in either case.

    The caller is responsible for interpreting the program counter returned from
    the coroutine. A `return` indicates that the coroutine is finished and
    control shall continue in the parent. A `halt` indicates that the coroutine
    suspends forever, and the parent should also return `halt` if it is a
    coroutine itself. Any other value indicates that the callee has suspended
    and expects to be re-entered at a later point, and the caller must suspend
    itself and re-enter the callee if it is a coroutine itself.
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    CoroutineStateOfCallee:$state,
    CoroutinePCOfCallee:$pc,
    Variadic<AnyType>:$args
  );
  let results = (outs
    CoroutineStateOfCallee:$resumeState,
    CoroutinePCOfCallee:$resumePC,
    Variadic<AnyType>:$results
  );
  let assemblyFormat = [{
    $callee `(` $state `,` $pc (`,` $args^)? `)` attr-dict
    `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    operand_range getArgOperands() {
      return getArgs();
    }
    MutableOperandRange getArgOperandsMutable() {
      return getArgsMutable();
    }

    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
    }

    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr(getCalleeAttrName(),
                       llvm::cast<mlir::SymbolRefAttr>(callee));
    }

    /// CallOpInterface requires ArgAndResultAttrsOpInterface. Call sites
    /// don't carry these attributes, so stub them out as no-ops.
    mlir::ArrayAttr getArgAttrsAttr() { return nullptr; }
    mlir::ArrayAttr getResAttrsAttr() { return nullptr; }
    void setArgAttrsAttr(mlir::ArrayAttr args) {}
    void setResAttrsAttr(mlir::ArrayAttr args) {}
    mlir::Attribute removeArgAttrsAttr() { return nullptr; }
    mlir::Attribute removeResAttrsAttr() { return nullptr; }
  }];
}

def CoroutineInstanceOp : ArcOp<"coroutine.instance", [
  CallOpInterface,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  HasParent<"hw::HWModuleOp">,
]> {
  let summary = "Continuously run a coroutine in an hw.module";
  let description = [{
    Execute a coroutine concurrently in an `hw.module`. The program counter and
    state of the coroutine are held implicitly by the instance and passed into
    the coroutine when executed next. The values yielded by the coroutine are
    produced as results of the instance. The callee must produce a wakeup time
    as its last result value. This wakeup time is not exposed as a result of
    the instance op and is instead used to schedule the next execution.
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$args
  );
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $callee `(` $args `)` attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    operand_range getArgOperands() {
      return getArgs();
    }
    MutableOperandRange getArgOperandsMutable() {
      return getArgsMutable();
    }

    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
    }

    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr(getCalleeAttrName(),
                       llvm::cast<mlir::SymbolRefAttr>(callee));
    }

    mlir::ArrayAttr getArgAttrsAttr() { return nullptr; }
    mlir::ArrayAttr getResAttrsAttr() { return nullptr; }
    void setArgAttrsAttr(mlir::ArrayAttr args) {}
    void setResAttrsAttr(mlir::ArrayAttr args) {}
    mlir::Attribute removeArgAttrsAttr() { return nullptr; }
    mlir::Attribute removeResAttrsAttr() { return nullptr; }
  }];
}

def CoroutineYieldOp : ArcOp<"coroutine.yield", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<BranchOpInterface>,
  HasParent<"CoroutineDefineOp">,
  Terminator,
]> {
  let summary = "Suspend a coroutine and request resumption at a block";
  let description = [{
    Suspend a coroutine. Control is transferred back to the caller, alongside
    a program counter value indicating the destination block, and the local
    state needed to restore the values live across the yield op. Additionally,
    the yield operands are returned to the caller and must match the result
    types of the coroutine.

    The destination block's first arguments must match the coroutine's function
    type. On resumption, those arguments are bound to the values supplied by the
    caller. The remaining arguments of the destination block are bound to the
    destination operands of the yield op.
  }];
  let arguments = (ins
    Variadic<AnyType>:$yieldOperands,
    Variadic<AnyType>:$destOperands
  );
  let successors = (successor AnySuccessor:$dest);
  let assemblyFormat = [{
    (` ` `(` $yieldOperands^ `:` type($yieldOperands) `)` `,`)?
    $dest (`(` $destOperands^ `:` type($destOperands) `)`)?
    attr-dict
  }];
  let hasVerifier = 1;
}

def CoroutineReturnOp : ArcOp<"coroutine.return", [
  HasParent<"CoroutineDefineOp">,
  Terminator,
]> {
  let summary = "Return from a coroutine";
  let description = [{
    Returns control from a coroutine to the caller, yielding back a special
    sentinel program counter value indicating that the coroutine has run to
    completion. Additionally, the yield operands are returned to the caller and
    must match the result types of the coroutine.
  }];
  let arguments = (ins Variadic<AnyType>:$yieldOperands);
  let assemblyFormat = [{
    ($yieldOperands^ `:` type($yieldOperands))? attr-dict
  }];
  let hasVerifier = 1;
}

def CoroutineHaltOp : ArcOp<"coroutine.halt", [
  HasParent<"CoroutineDefineOp">,
  Terminator,
]> {
  let summary = "Halt a coroutine permanently";
  let description = [{
    Halts execution of a coroutine forever. Control effectively gets stuck
    indefinitely at the halt operation, also preventing all callers from making
    progress. Yields back a special sentinel program counter value to the caller
    which the caller must translate into either halting itself if it is a
    coroutine, or arranging for the coroutine to never be re-entered again.
  }];
  let arguments = (ins Variadic<AnyType>:$yieldOperands);
  let assemblyFormat = [{
    ($yieldOperands^ `:` type($yieldOperands))? attr-dict
  }];
  let hasVerifier = 1;
}

class CoroutinePCBase<string mnemonic> : ArcOp<mnemonic, [Pure]> {
  let summary = "Check whether a coroutine PC is a sentinel value";
  let arguments = (ins CoroutinePCType:$pc);
  let results = (outs I1:$result);
  let assemblyFormat = [{
    $pc `:` type($pc) attr-dict
  }];
}
def CoroutinePCIsReturnOp : CoroutinePCBase<"coroutine.pc_is_return">;
def CoroutinePCIsHaltOp : CoroutinePCBase<"coroutine.pc_is_halt">;

def CoroutineUndefinedStateOp : ArcOp<"coroutine.undefined_state", [Pure]> {
  let summary = "Create an undefined initial coroutine state";
  let description = [{
    Produces an undefined `!arc.coroutine_state` value. A coroutine establishes
    its local state on first entry and never reads the state passed on that
    first call, such that any initial state value will do. This op provides such
    a value to seed the very first call into a coroutine.
  }];
  let results = (outs CoroutineStateType:$state);
  let assemblyFormat = [{
    `:` type($state) attr-dict
  }];
}

def CoroutineStartPCOp : ArcOp<"coroutine.start_pc", [Pure]> {
  let summary = "Create the start sentinel coroutine program counter";
  let description = [{
    Produces the special `start` sentinel `!arc.coroutine_pc` value. Passing
    this program counter to an `arc.coroutine.call` enters the coroutine for the
    first time, executing its body from the entry block. This op provides such a
    value to seed the very first call into a coroutine.
  }];
  let results = (outs CoroutinePCType:$pc);
  let assemblyFormat = [{
    `:` type($pc) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Procedural Ops
//===----------------------------------------------------------------------===//

def ExecuteOp : ArcOp<"execute", [
  IsolatedFromAbove,
  RecursiveMemoryEffects,
]> {
  let summary = "Execute an SSACFG region";
  let description = [{
    The `arc.execute` op allows an SSACFG region to be embedded in a parent
    graph region, or another SSACFG region. Whenever execution reaches this op,
    its body region is executed and the results yielded from the body are
    produced as the `arc.execute` op's results. The op is isolated from above.
    Any SSA values defined outside the op that are used inside the body have to
    be captured as operands and then referred to as entry block arguments in the
    body.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region MinSizedRegion<1>:$body);
  let assemblyFormat = [{
    (` ` `(` $inputs^ `:` type($inputs) `)`)?
    (`->` `(` type($results)^ `)`)?
    attr-dict-with-keyword $body
  }];
  let hasRegionVerifier = 1;
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

  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$runtimeModel,
                       OptionalAttr<StrAttr>:$runtimeArgs);
  let hasRegionVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  let builders = [OpBuilder<(ins), [{
      build($_builder, $_state, FlatSymbolRefAttr{}, StringAttr{});
    }]>];
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

    If `timePostIncrement` is provided, the instance's simulation time will be
    incremented by the given number in femtoseconds _after_ the step has been
    evaluated.

    Example:

    ```mlir
    %cst5  = arith.constant 5  : i64
    %cst10 = arith.constant 10 : i64
    arc.sim.set_time %instance, %cst10 [...]
    arc.sim.step %instance by %cst5 [...]       // Step at t = 10 fs
    arc.sim.step %instance [...]                // Step at t = 15 fs
    arc.sim.step %instance by %cst5 [...]       // Step at t = 15 fs
    arc.sim.step %instance by %cst5 [...]       // Step at t = 20 fs
    ```
  }];
  let arguments = (ins SimModelInstance:$instance,
                       Optional<I64>:$timePostIncrement);
  let assemblyFormat =
    [{ $instance (`by` $timePostIncrement^)? attr-dict
        `:` qualified(type($instance)) }];
   let builders = [
    OpBuilder<(ins "mlir::Value":$instance), [{
      build($_builder, $_state, instance, /*timePostIncrement=*/ Value{});
    }]>
  ];
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

def SimGetTimeOp : ArcOp<"sim.get_time", [
  MemoryEffects<[MemRead]>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Gets the current simulation time of the model instance";
  let description = [{
    Gets the current simulation time of the given model instance. The time is
    represented as an `i64` value in femtoseconds.
  }];
  let arguments = (ins SimModelInstance:$instance);
  let results = (outs I64:$time);
  let assemblyFormat = [{
    $instance attr-dict `:` qualified(type($instance))
  }];
}

def SimSetTimeOp : ArcOp<"sim.set_time", [
  MemoryEffects<[MemWrite]>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Sets the simulation time of the model instance";
  let description = [{
    Sets the simulation time of the given model instance. The time is
    represented as an `i64` value in femtoseconds.
  }];
  let arguments = (ins SimModelInstance:$instance, I64:$time);
  let assemblyFormat = [{
    $instance `,` $time attr-dict `:` qualified(type($instance))
  }];
}

def SimGetNextWakeupOp : ArcOp<"sim.get_next_wakeup", [
  MemoryEffects<[MemRead]>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Gets the next wakeup time of the model instance";
  let description = [{
    Gets the next wakeup time of the given model instance. The time is an
    absolute timestamp in femtoseconds, not a duration relative to the current
    simulation time. A wakeup at the current time requests an immediate
    re-evaluation in the same time step; a value of `UINT64_MAX` indicates that
    no wakeup is pending. The slot is recomputed on every evaluation, so a
    driver typically reads it after each `arc.sim.step` to decide how far to
    advance simulation time before the next step.
  }];
  let arguments = (ins SimModelInstance:$instance);
  let results = (outs I64:$time);
  let assemblyFormat = [{
    $instance attr-dict `:` qualified(type($instance))
  }];
}

//===----------------------------------------------------------------------===//
// Miscellaneous
//===----------------------------------------------------------------------===//

def TapOp : ArcOp<"tap"> {
  let summary = "A tracker op to observe a value under one or more given names";
  let arguments = (ins AnyType:$value, StrArrayAttr:$names);
  let assemblyFormat = [{ $value attr-dict `:` type($value) }];
  let builders = [OpBuilder<(ins "mlir::Value":$val, "mlir::StringRef":$name),
                            [{
      build($_builder, $_state, val,
       $_builder.getArrayAttr({$_builder.getStringAttr(name)}));
    }]>];
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
                       OptionalAttr<UI32Attr>:$storageBytes,
                       OptionalAttr<FlatSymbolRefAttr>:$initialFn,
                       OptionalAttr<FlatSymbolRefAttr>:$finalFn,
                       OptionalAttr<TraceTapArrayAttr>:$traceTaps);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $sym_name `io` $io
    (`storageBytes` $storageBytes^)?
    (`initializer` $initialFn^)?
    (`finalizer` $finalFn^)?
    (`traceTaps` $traceTaps^)?
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

def RuntimeModelOp : ArcOp<"runtime.model", [
  HasParent<"::mlir::ModuleOp">, Pure, Symbol
]> {
  let summary = "Provides static metadata of an Arc model used by the runtime";
  let description = [{
    Collection of static metadata for a specific Arc model accessed by the
    arcilator runtime library:
    - name: Name of the model
    - numStateBytes: Number of bytes required to store the model's internal
        state
    - traceTaps: Traced signal metadata
  }];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       StrAttr:$name,
                       I64Attr:$numStateBytes,
                       OptionalAttr<TraceTapArrayAttr>:$traceTaps);
  let assemblyFormat = [{
    $sym_name $name `numStateBytes` $numStateBytes
    (`traceTaps` $traceTaps^)? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// ArrayRef ops
//===----------------------------------------------------------------------===//
//
// Ops that deal with bufferized arrays. The !hw.array type has value semantics;
// the !arc.arrayref type is the equivalent but with reference semantics.

def ArrayRefAllocOp : ArcOp<"arrayref.alloc", []> {
  let summary = "array allocation";
  let description = [{
    Allocates an array on the stack. If `init` is set, the array is initialized
    to this value. Otherwise the array's contents are undefined.
  }];
  let arguments = (ins OptionalAttr<ArrayAttr>:$init);
  let results = (outs ArrayRefType:$output);
  let assemblyFormat = "attr-dict (`init` `(` $init^ `)`)? `:` type($output)";
  let hasVerifier = 1;
}

def ArrayRefGetOp : ArcOp<"arrayref.get", [
  MemoryEffects<[MemRead]>,
  TypesMatchWith<"array element type matches the result type", "input", "value",
                 "cast<ArrayRefType>($_self).getElementType()">
]> {
  let summary = "array get";
  let description = [{
    Loads a single value from an arrayref.
  }];
  let arguments = (ins ArrayRefType:$input, Index:$index);
  let results = (outs AnyType:$value);
  let assemblyFormat = [{
    $input `[` $index `]` attr-dict `:` type($input) `->` type($value)
  }];
}

def ArrayRefInjectOp : ArcOp<"arrayref.inject", [
  MemoryEffects<[MemRead, MemWrite]>,
  AllTypesMatch<["input", "output"]>,
  TypesMatchWith<"element type matches input element type", "input", "element",
                 "cast<ArrayRefType>($_self).getElementType()">
]> {
  let summary = "array inject";
  let description = [{
    Stores a single value into an arrayref.
  }];
  let arguments = (ins ArrayRefType:$input, Index:$index, AnyType:$element);
  let results = (outs ArrayRefType:$output);
  let assemblyFormat = [{
    $input `[` $index `]` `,` $element attr-dict `:` type($input) `,` type($element) `->` type($output)
  }];
}

def ArrayRefSliceOp : ArcOp<"arrayref.slice", [
  Pure,
  AllElementTypesMatch<["input", "output"]>
]> {
  let summary = "array slice";
  let description = [{
    Slices an array into a sub-array, returning a view into the original array.
  }];
  let arguments = (ins ArrayRefType:$input, Index:$lowIndex);
  let results = (outs ArrayRefType:$output);
  let assemblyFormat = [{
    $input`[`$lowIndex`]` attr-dict `:` functional-type($input, $output)
  }];
}

def ArrayRefCopyOp : ArcOp<"arrayref.copy", [
  MemoryEffects<[MemRead, MemWrite]>,
  SameOperandsAndResultType,
]> {
  let summary = "array copy";
  let description = [{
    Copies the content of one array into another array. Returns the input array.
  }];
  let arguments = (ins ArrayRefType:$input, ArrayRefType:$source);
  let results = (outs ArrayRefType:$output);
  let assemblyFormat = [{
    $input `=` $source attr-dict `:` type($input)
  }];
}

def ArrayRefCreateOp : ArcOp<"arrayref.create", [
  MemoryEffects<[MemWrite]>,
  AllTypesMatch<["input", "output"]>,
  TypesMatchWith<"operand types match element type",
                 "output", "elements", "SmallVector<Type, 2>("
                 "cast<ArrayRefType>($_self).getNumElements(), "
                 "cast<ArrayRefType>($_self).getElementType())">
]> {
  let summary = "array create";
  let description = [{
    Creates an array from a list of values. The values are ordered from
    most significant to least significant (the same as `hw.array_create`).

    The `input` array is populated with `elements` and is returned.
  }];
  let arguments = (ins ArrayRefType:$input, Variadic<AnyType>:$elements);
  let results = (outs ArrayRefType:$output);
  let assemblyFormat = [{
    $input `=` $elements attr-dict `:` type($output)
  }];
}

def ArrayRefFromArrayOp : ArcOp<"arrayref.from_array", [
  MemoryEffects<[MemWrite]>,
  AllTypesMatch<["input", "output"]>,
  TypesMatchWith<"array type matches output type",
                 "array", "output",
                 "ArrayRefType::get("
                   "cast<hw::ArrayType>($_self).getElementType(), "
                   "cast<hw::ArrayType>($_self).getNumElements())">
]> {
  let summary = "arrayref from array";
  let description = [{
    Populates an arrayref with the elements of an array.
  }];
  let arguments = (ins ArrayRefType:$input, ArrayType:$array);
  let results = (outs ArrayRefType:$output);
  let assemblyFormat = [{
    $input `=` $array attr-dict `:` type($output) `,` type($array)
  }];
}

def ArrayRefToArrayOp : ArcOp<"arrayref.to_array", [
  TypesMatchWith<"array type matches result type",
                 "input", "result",
                 "hw::ArrayType::get("
                   "cast<ArrayRefType>($_self).getElementType(), "
                   "cast<ArrayRefType>($_self).getNumElements())">
]> {
  let summary = "arrayref to array";
  let description = [{
    Creates an array from an arrayref.
  }];
  let arguments = (ins ArrayRefType:$input);
  let results = (outs ArrayType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` functional-type($input, $result)
  }];
}

#endif // CIRCT_DIALECT_ARC_ARCOPS_TD
