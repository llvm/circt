//===- Ops.td - Handshake operation definitions ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//============================================================================//
//
// This file define Handshake ops in tablegen.
//
//===----------------------------------------------------------------------===//

// This is almost exactly like a standard FuncOp, except that it has some
// extra verification conditions.  In particular, each Value must
// only have a single use.  Also, it defines a Dominance-Free Scope
def FuncOp : Op<Handshake_Dialect, "func", [
   IsolatedFromAbove,
   FunctionLike,
   Symbol,
   RegionKindInterface
]> {
  let summary = "Handshake dialect function.";
  let description = [{
    The "handshake.func" operation represents a handshaked function.
    This is almost exactly like a standard FuncOp, except that it has
    some extra verification conditions. In particular, each Value must
    only have a single use.
  }];

  let arguments = (ins);
  let results = (outs);
  let regions = (region AnyRegion : $body);

  let skipDefaultBuilders = 1;

  let builders =
       [OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
                      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>];

  let extraClassDeclaration = [{
    // Add an entry block to an empty function, and set up the block arguments
    // to match the signature of the function.
    Block *addEntryBlock();

    FunctionType getType() {
      return (*this)->getAttrOfType<TypeAttr>(getTypeAttrName())
          .getValue()
          .cast<FunctionType>();
    }

    // This trait needs access to the hooks defined below.
    friend class OpTrait::FunctionLike<handshake::FuncOp>;

    /// Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }

    /// Returns the number of arguments. This is a hook for
    /// OpTrait::FunctionLike.
    unsigned getNumFuncArguments() { return getType().getInputs().size(); }

    /// Returns the number of results. This is a hook for OpTrait::FunctionLike.
    unsigned getNumFuncResults() { return getType().getResults().size(); }

    /// Returns the names of the arguments to this function.
    ArrayAttr getArgNames() {
      return (*this)->getAttrOfType<ArrayAttr>("argNames");
    }

    /// Returns the names of the results from this function.
    ArrayAttr getResNames() {
      return (*this)->getAttrOfType<ArrayAttr>("resNames");
    }

    /// Returns the argument name at the given index.
    StringAttr getArgName(unsigned idx) {
      return getArgNames()[idx].cast<StringAttr>();
    }

    /// Returns the result name at the given index.
    StringAttr getResName(unsigned idx) {
      return getResNames()[idx].cast<StringAttr>();
    }

    /// Resolve argument and result names. This can be used during building of
    /// a handshake.func operation to ensure that names provided by an incoming
    /// operation fulfills the name requirements of the handshake function.
    void resolveArgAndResNames();

    /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
    /// attribute is present and checks if it holds a function type.  Ensures
    /// getType, getNumFuncArguments, and getNumFuncResults can be called
    /// safely.
    LogicalResult verifyType() {
      auto type = getTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError("requires '" + getTypeAttrName() +
                           "' attribute of function type");
      return success();
    }
  }];

  let verifier = [{ return ::verify$cppClass(*this); }];
  let printer = [{ return ::print$cppClass(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

// InstanceOp
def InstanceOp : Handshake_Op<"instance", [CallOpInterface]> {
  let summary = "module instantiate operation";
  let description = [{
    The `instance` operation represents the instantiation of a module.  This
	  is similar to a function call, except that different instances of the
	  same module are guaranteed to have their own distinct state.
    The instantiated module is encoded as a symbol reference attribute named
    "module". An instance operation takes a control input as its last argument
    and returns a control output as its last result.

    Example:
    ```mlir
    %2:2 = handshake.instance @my_add(%0, %1, %ctrl) : (f32, f32, none) -> (f32, none)
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$module, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [OpBuilder<
    (ins "FuncOp":$module, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("module", SymbolRefAttr::get(module));
      $_state.addTypes(module.getType().getResults());
      $_state.addTypes({$_builder.getType<::mlir::NoneType>()});
  }]>, OpBuilder<
    (ins "SymbolRefAttr":$module, "TypeRange":$results,
     CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("module", module);
      $_state.addTypes(results);
      $_state.addTypes({$_builder.getType<::mlir::NoneType>()});
  }]>, OpBuilder<
    (ins "StringRef":$module, "TypeRange":$results,
     CArg<"ValueRange", "{}">:$operands), [{
       build($_builder, $_state,
             SymbolRefAttr::get($_builder.getContext(), module), results,
             operands);
  }]>];

  let extraClassDeclaration = [{
    StringRef getModule() { return module(); }
    FunctionType getModuleType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the module of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("module");
    }

    /// Get the control operand of this instance op
    Value getControl() {
      return getOperands().back();
    }
  }];

  let assemblyFormat = [{
    $module `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];

  let verifier = [{ return ::verify$cppClass(*this); }];
}

// This is almost exactly like a standard FuncOp, except that it has some
// extra verification conditions.  In particular, each Value must
// only have a single use.
def ReturnOp : Handshake_Op<"return", [Terminator]> {
  let summary = "Handshake dialect return.";
  let description = [{
    The "handshake.return" operation represents a handshaked
    function.  This is almost exactly like a standard ReturnOp, except
    that it exists in a handshake.func.  It has the same operands as
    standard ReturnOp which it replaces and an additional control -
    only operand(exit point of control - only network).
  }];

  let arguments = (ins Variadic<AnyType> : $operands, NoneType : $control);

  let skipDefaultBuilders = 1;

  let builders = [OpBuilder<(ins "ArrayRef<Value>":$operands)>];

  let printer = "return ::print$cppClass(p, *this);";
  let verifier = "return ::verify(*this);";
  let parser = "return ::parse$cppClass(parser, result);";
}

def BufferOp : Handshake_Op<"buffer", [NoSideEffect, HasClock]> {
  let summary = "buffer operation";

  let description = [{
    The "handshake.buffer" operation represents a buffer operation. $slots
    must be an unsigned integer larger than 0. $sequantial=True indicates a
    nontransparent buffer, while $sequantial=False indicates a transparent
    buffer.
  }];

  let arguments = (ins AnyType, BoolAttr:$sequential, BoolAttr:$control,
                   Confined<I32Attr, [IntMinValue<1>]>:$slots);
  let results = (outs AnyType);

  let extraClassDeclaration = [{
    bool isSequential() {
      return (*this)->getAttrOfType<BoolAttr>("sequential").getValue();
    }
    bool isControl() {
      return (*this)->getAttrOfType<BoolAttr>("control").getValue();
    }
    APInt getNumSlots() {
      return (*this)->getAttrOfType<IntegerAttr>("slots").getValue();
    }
  }];
}

def ForkOp : Handshake_Op<"fork", [
  NoSideEffect, DeclareOpInterfaceMethods<ExecutableOpInterface>,
  HasClock, DeclareOpInterfaceMethods<GeneralOpInterface>
]> {
  let summary = "fork operation";

  let description = [{
    The "handshake.fork" operation represents a fork operation.  A
    single input is replicated to N outputs and distributed to each
    output as soon as the corresponding successor is available.
  }];

  let arguments = (ins AnyType, BoolAttr : $control);
  let results = (outs Variadic<AnyType>);

  let hasCanonicalizer = 1;
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Value":$operand, "int":$outputs)>];

  let extraClassDeclaration = [{
    bool isControl() { return (*this)->getAttrOfType<BoolAttr>("control").getValue(); }
  }];
}

def LazyForkOp : Handshake_Op<"lazy_fork", [NoSideEffect]> {
  let summary = "lazy fork operation";
  let description = [{
    The "handshake.lfork" operation represents a lazy fork operation.
    A single input is replicated to N outputs and distributed to each
    output when all successors are available.
  }];

  let arguments = (ins AnyType, BoolAttr : $control);
  let results = (outs Variadic<AnyType>);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Value":$operand, "int":$outputs)>];

  let extraClassDeclaration = [{
    bool isControl() {  return (*this)->getAttrOfType<BoolAttr>("control").getValue(); }
  }];
}

def MergeOp : Handshake_Op<"merge", [
  NoSideEffect, MergeLikeOpInterface,
  DeclareOpInterfaceMethods<ExecutableOpInterface>
]> {
  let summary = "merge operation";
  let description = [{
    The "handshake.merge" operation represents a (nondeterministic)
    merge operation. Any input is propagated to the single output. The
    number of inputs corresponds to the number of predecessor
    blocks.
  }];

  let arguments = (ins Variadic<AnyType>:$dataOperands);
  let results = (outs AnyType);

  let hasCanonicalizer = 1;
  let builders = [OpBuilder<(ins "Value":$operand, "int":$inputs)>];
}

def MuxOp : Handshake_Op<"mux", [
  NoSideEffect, MergeLikeOpInterface,
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>
]> {
  let summary = "mux operation";
  let description = [{
    The "handshake.mux" operation represents a(deterministic)
        merge operation.
    Operands: select, data0, data1, data2, ...

    The 'select' operand is received from ControlMerge of the same
    block and it represents the index of the data operand that the mux
    should propagate to its single output.  The number of data inputs
    corresponds to the number of predecessor blocks.
  }];
  let arguments = (ins AnyType : $selectOperand,
							  Variadic<AnyType> : $dataOperands);
  let results = (outs AnyType);

  let builders = [OpBuilder<(ins "Value":$operand, "int":$inputs)>];
  let verifier = "return ::verify(*this);";
}

def ControlMergeOp : Handshake_Op<"control_merge", [
  NoSideEffect, MergeLikeOpInterface, HasClock,
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getResultName"]>
]> {
  let summary = "control merge operation";
  let description = [{
    The "handshake.control_merge" operation represents a
    (nondeterministic) control merge.  Any input is propagated to the
    first output and the index of the propagated input is sent to the
    second output.  The number of inputs corresponds to the number of
    predecessor blocks.  ControlMerge is a control-only
    component(i.e., has no data but only bidirectional handshake).
  }];

  let arguments = (ins Variadic<AnyType>:$dataOperands, BoolAttr : $control);
  let results = (outs Variadic<AnyType>);

  let hasCanonicalizer = 1;
  let builders = [OpBuilder<(ins "Value":$operand, "int":$inputs)>];

  let extraClassDeclaration = [{
    bool isControl() {
      return (*this)->getAttrOfType<BoolAttr>("control").getValue();
    }
  }];
}

def BranchOp : Handshake_Op<"branch", [
  NoSideEffect, DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<GeneralOpInterface>
]> {
  let summary = "branch operation";
  let description = [{
      The "handshake.branch" operation represents an unconditional
      branch.  The single data input is propagated to the single
      successor.  The input must be triggered by some predecessor to
      avoid continous triggering of a successor block.
  }];
  let arguments = (ins AnyType : $dataOperand,
		                 BoolAttr : $control);
  let results = (outs AnyType : $dataResult);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Value":$dataOperand)>];

  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{
    bool isControl() {
      return (*this)->getAttrOfType<BoolAttr>("control").getValue();
    }
  }];
}

def ConditionalBranchOp : Handshake_Op<"conditional_branch", [
  NoSideEffect, DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getResultName"]>
]> {
  let summary = "conditional branch operation";
  let description = [{
     The "handshake.cbranch" operation represents a conditional
     branch.  The data input is propagated to one of the two outputs
     based on the condition input.
  }];

  let arguments = (ins I1 : $conditionOperand,
                       AnyType : $dataOperand,
                       BoolAttr : $control);
  let results = (outs AnyType : $trueResult,
                      AnyType : $falseResult);

  let builders = [OpBuilder<(ins "Value":$condOperand, "Value":$dataOperand)>];

  let extraClassDeclaration = [{
    // These are the indices into the dests list.
    enum { trueIndex = 0, falseIndex = 1 };

    Value getFalseResult() { return getResult(falseIndex); }

    Value getTrueResult() { return getResult(trueIndex); }

    Value getDataOperand() { return getOperand(1); }

    bool isControl() { return (*this)->getAttrOfType<BoolAttr>("control").getValue(); }
  }];
}

def SinkOp
    : Handshake_Op<"sink", [DeclareOpInterfaceMethods<ExecutableOpInterface>]> {
  let summary = "sink operation";
  let description = [{
    The "handshake.sink" operation discards any data that arrives at its
         input.The sink has no successors and it can continuously consume data.
  }];
  let arguments = (ins AnyType);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Value":$operand)>];
}

def SourceOp : Handshake_Op<"source", [NoSideEffect]> {
  let summary = "source operation";
  let description = [{
    The "handshake.source" operation represents continuous data
    source.  The source continously sets a 'valid' signal which the
    successor can consume at any point in time.
  }];

  let results = (outs AnyType);
}

def NeverOp : Handshake_Op<"never", [NoSideEffect]> {
  let summary = "never operation";
  let description = [{
    The "handshake.never" operation represents disconnected data
    source. The source never sets any 'valid' signal which will
    never trigger the successor at any point in time.
  }];

  let results = (outs AnyType);
}

def ConstantOp : Handshake_Op<"constant", [
  NoSideEffect, DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<GeneralOpInterface>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>
]> {
  let summary = "constant operation";
  let description = [{
    The "handshake.const" has a constant value. When triggered by its
    single `ctrl` input, it sends the constant value to its single
    successor.
  }];

  let arguments = (ins AnyAttr:$value, AnyType:$ctrl);
  let results = (outs AnyType);

  // let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Attribute":$value, "Value":$ctrl)>];

  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{
    Attribute getValue() { return (*this)->getAttr("value"); }
  }];
}

def EndOp
    : Handshake_Op<"end", [DeclareOpInterfaceMethods<ExecutableOpInterface>]> {
  let summary = "end operation";
  let description = [{
    The "handshake.end" propagates the result of the appropriate
    return operation from one of its inputs to its single output after
    all memory accesses have completed.  Currently not used(data
    returned through ReturnOp).
  }];
  let arguments = (ins AnyType : $control, Variadic<AnyType>);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Value":$operand)>];
}

def StartOp : Handshake_Op<"start", [
  NoSideEffect, DeclareOpInterfaceMethods<ExecutableOpInterface>
]> {
  let summary = "start operation";
  let description = [{
       Triggers execution of the control - only network.  Placed in entry
    block.  Currently not used( trigger given as function argument)
  }];

  let arguments = (ins BoolAttr : $control);
  let results = (outs NoneType);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins)>];
}

def TerminatorOp : Handshake_Op<"terminator", [Terminator]> {
  let summary = "handshake terminator operation";
  let description = [{
    This op is used as a terminator in every block of the dataflow
    netlist (as a replacement for StandardOp branches). It has no
    functionality and can be removed in some subsequent pass, when the
    block structure is removed.
  }];

  let successors = (successor VariadicSuccessor<AnySuccessor>:$dests);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "ArrayRef<Block *>":$successors)>];
}

def MemRefTypeAttr : TypeAttrBase<"MemRefType", "memref type attribute">;
def MemoryOp : Handshake_Op<"memory", [
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<MemoryOpInterface>, HasClock,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getResultName"]>
]> {
  let summary = "memory";
  let description = [{
    Each MemoryOp represents an independent memory or memory region (BRAM or external memory).
    It receives memory access requests from load and store operations. For every request,
    it returns data (for load) and a data-less token indicating completion.
    The memory op represents a flat, unidimensional memory.
    Operands: all stores (stdata1, staddr1, stdata2, staddr2, ...), then all loads (ldaddr1, ldaddr2,...)
    Outputs: all load outputs, ordered the same as
    load addresses (lddata1, lddata2, ...), followed by all none outputs,
    ordered as operands (stnone1, stnone2,...ldnone1, ldnone2,...)
  }];
  let arguments = (ins Variadic<AnyType>,
                          I32Attr : $ld_count,
                       I32Attr : $st_count,
                       I32Attr : $id,
                       MemRefTypeAttr : $type);
  let results = (outs Variadic<AnyType>);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(
      ins "ArrayRef<Value>":$operands, "int":$outputs, "int":$control_outputs, "bool":$lsq,
      "int":$id, "Value":$memref)>
  ];
  let extraClassDeclaration = [{
    APInt getLdCount() {
      return (*this)->getAttrOfType<IntegerAttr>("ld_count").getValue();
    }
    APInt getStCount() {
      return (*this)->getAttrOfType<IntegerAttr>("st_count").getValue();
    }
    unsigned getID() {
      return (*this)->getAttrOfType<IntegerAttr>("id").getValue().getZExtValue();
    }
    MemRefType getMemRefType() {
      return (*this)->getAttrOfType<TypeAttr>("type").getValue().cast<MemRefType>();
    }
  }];
  let verifier = "return ::verify$cppClass(*this);";
}

def ExternalMemoryOp : Handshake_Op<"extmemory", [
  DeclareOpInterfaceMethods<ExecutableOpInterface>
]> {
  let summary = "external memory";
  let description = [{
    An ExternalMemoryOp represents a wrapper around a memref input to a
    handshake function. The semantics of the load/store operands are identical
    to what is decribed for MemoryOp. The only difference is that the first
    operand to this operand is a `memref` value.
    Upon lowering to FIRRTL, a handshake interface will be created in the
    top-level component for each load- and store which connected to this memory.

    Example:

    ```mlir
    handshake.func @main(%i: index, %v: i32, %mem : memref<10xi32>, %ctrl: none) -> none {
      %stCtrl = "handshake.extmemory"(%mem, %vout, %addr) {id = 0 : i32, ld_count = 0 : i32, lsq = false, st_count = 1 : i32} : (memref<10xi32>, i32, index, index) -> (none)
      %vout, %addr = "handshake.store"(%v, %i, %ctrl) : (i32, index, none) -> (i32, index)
      ...
    }
    ```
  }];
  let arguments = (ins AnyMemRef:$memref,
                       Variadic<AnyType>:$inputs,
                       I32Attr : $ld_count,
                       I32Attr : $st_count,
                       I32Attr : $id);
  let results = (outs Variadic<AnyType>:$outputs);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(
      ins "Value":$memref, "ArrayRef<Value>":$inputs, "int":$ld_count, "int":$st_count,
      "int":$id)>
  ];
  let assemblyFormat = "`[` `ld` `=` $ld_count `,` `st` `=`  $st_count `]` `(` $memref `:` type($memref) `)` `(` $inputs `)` attr-dict `:` functional-type($inputs, $outputs)";

  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return (*this)->getOperand(0).getType().cast<MemRefType>();
    }
  }];
}

def LoadOp
    : Handshake_Op<"load", [DeclareOpInterfaceMethods<ExecutableOpInterface>]> {
  let summary = "load operation";
  let description = [{
     Load memory port, sends load requests to MemoryOp. From dataflow
     predecessor, receives address indices and a control-only value
     which signals completion of all previous memory accesses which
     target the same memory.  When all inputs are received, the load
     sends the address indices to MemoryOp. When the MemoryOp returns
     a piece of data, the load sends it to its dataflow successor.

     Operands: address indices (from predecessor), data (from MemoryOp), control-only input.
     Results: data (to successor), address indices (to MemoryOp).
  }];

  let arguments = (ins Variadic<AnyType>, AnyType, NoneType);
  let results = (outs AnyType, Variadic<AnyType>:$addressResults);

  let builders = [OpBuilder<(ins "Value":$memref, "ArrayRef<Value>":$indices)>];
}

def StoreOp : Handshake_Op<"store", [
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<GeneralOpInterface>
]> {
  let summary = "store operation";
  let description = [{
    Store memory port, sends store requests to MemoryOp. From dataflow
    predecessors, receives address indices, data, and a control-only
    value which signals completion of all previous memory accesses
    which target the same memory.  When all inputs are received, the
    store sends the address and data to MemoryOp.

    Operands: address indices, data, control-only input.
    Results: data and address indices (sent to MemoryOp).
  }];

  let arguments = (ins AnyType, Variadic<AnyType>, NoneType);
  let results = (outs AnyType, Variadic<AnyType>);

  let builders =
      [OpBuilder<(ins "Value":$valueToStore, "ArrayRef<Value>":$indices)>];
}

def JoinOp : Handshake_Op<"join", [
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<GeneralOpInterface>
]> {
  let summary = "join operation";
  let description = [{
    A control-only synchronizer.  Produces a valid output when all
    inputs become available.
  }];
  let arguments = (ins Variadic<NoneType>, BoolAttr : $control);
  let results = (outs NoneType);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "ArrayRef<Value>":$operands)>];
}

def I4 : I<4>;
def I4Attr : SignlessIntegerAttrBase<I4, "4-bit integer attribute">;
