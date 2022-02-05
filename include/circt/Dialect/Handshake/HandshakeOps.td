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

include "mlir/IR/OpAsmInterface.td"

// This is almost exactly like a standard FuncOp, except that it has some
// extra verification conditions.  In particular, each Value must
// only have a single use.  Also, it defines a Dominance-Free Scope
def FuncOp : Op<Handshake_Dialect, "func", [
   IsolatedFromAbove,
   FunctionLike,
   Symbol,
   RegionKindInterface,
   OpAsmOpInterface
]> {
  let summary = "Handshake dialect function.";
  let description = [{
    The func operation represents a handshaked function.
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

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    static ::llvm::StringRef getDefaultDialect() {
      return "handshake";
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
  let skipDefaultBuilders = 1;
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
    The return operation represents a handshaked
    function.  This is almost exactly like a standard ReturnOp, except
    that it exists in a handshake.func.  It has the same operands as
    standard ReturnOp which it replaces and an additional control -
    only operand(exit point of control - only network).
  }];

  let arguments = (ins Variadic<AnyType> : $operands, NoneType : $control);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "ArrayRef<Value>":$operands)>];

  let verifier = "return ::verify(*this);";
  let assemblyFormat = [{ operands attr-dict `:` qualified(type(operands)) }];
}

def BufferOp : Handshake_Op<"buffer", [NoSideEffect, HasClock,
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<GeneralOpInterface>]> {
  let summary = "buffer operation";

  let description = [{
    The buffer operation represents a buffer operation. $slots
    must be an unsigned integer larger than 0. $sequantial=True indicates a
    nontransparent buffer, while $sequantial=False indicates a transparent
    buffer.

    An 'initValues' attribute containing a list of integer values may be provided.
    The list must be of the same length as the number of slots. This will
    initialize the buffer with the given values upon reset.
    For now, only sequential buffers are allowed to have initial values.
    @todo: How to support different init types? these have to be stored (and
    retrieved) as attributes, hence they must be of a known type.
  }];

  let arguments = (ins
    TypeAttr:$dataType,
    Confined<I32Attr, [IntMinValue<1>]>:$size,
    AnyType:$operand,
    BoolAttr:$sequential,
    OptionalAttr<I64ArrayAttr>:$initValues);
  let results = (outs AnyType:$result);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Type":$dataType, "int":$size,
    "Value":$operand, "bool":$sequential)>];

  let extraClassDeclaration = [{
    bool isSequential() {
      return (*this)->getAttrOfType<BoolAttr>("sequential").getValue();
    }
    int getNumSlots() {
      return (*this)->getAttrOfType<IntegerAttr>("size").getValue().getZExtValue();
    }
    SmallVector<int64_t> getInitValues() {
      assert(initValues() && "initValues attribute not set");
      SmallVector<int64_t> values;
      for (auto value : (*this)->getAttrOfType<ArrayAttr>("initValues"))
        values.push_back(value.cast<IntegerAttr>().getValue().getSExtValue());
      return values;
    }

  }];
  let printer = "return ::print$cppClass(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";
  let verifier = "return ::verify$cppClass(*this);";
  let hasCanonicalizer = 1;
}

def ForkOp : Handshake_Op<"fork", [
  NoSideEffect, DeclareOpInterfaceMethods<ExecutableOpInterface>,
  HasClock, DeclareOpInterfaceMethods<GeneralOpInterface>
]> {
  let summary = "fork operation";

  let description = [{
    The fork operation represents a fork operation.  A
    single input is replicated to N outputs and distributed to each
    output as soon as the corresponding successor is available.

    Example:
    ```mlir
    %1:2 = fork [2] %0 : i32
    ```
  }];

  let arguments = (ins
    TypeAttr:$dataType,
    Confined<I32Attr, [IntMinValue<1>]>:$size,
    AnyType : $operand);
  let results = (outs Variadic<AnyType> : $result);

  let hasCanonicalizer = 1;
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Value":$operand, "int":$outputs)>];
  let printer = "return ::print$cppClass(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";
}

def LazyForkOp : Handshake_Op<"lazy_fork", [NoSideEffect]> {
  let summary = "lazy fork operation";
  let description = [{
    The lazy_fork operation represents a lazy fork operation.
    A single input is replicated to N outputs and distributed to each
    output when all successors are available.

    Example:
    ```mlir
    %1:2 = lazy_fork [2] %0 : i32
    ```
  }];

  let arguments = (ins
    TypeAttr:$dataType,
    Confined<I32Attr, [IntMinValue<1>]>:$size,
    AnyType : $operand);
  let results = (outs Variadic<AnyType> : $result);
  let printer = "return ::print$cppClass(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Value":$operand, "int":$outputs)>];
}

def MergeOp : Handshake_Op<"merge", [
  NoSideEffect, MergeLikeOpInterface,
  DeclareOpInterfaceMethods<ExecutableOpInterface>
]> {
  let summary = "merge operation";
  let description = [{
    The merge operation represents a (nondeterministic)
    merge operation. Any input is propagated to the single output. The
    number of inputs corresponds to the number of predecessor
    blocks.

    Example:
    ```
    %0 = merge %a, %b, %c : i32
    ```
  }];

  let arguments = (ins
    TypeAttr:$dataType,
    Confined<I32Attr, [IntMinValue<1>]>:$size,
    Variadic<AnyType>:$dataOperands);
  let results = (outs AnyType:$result);
  let printer = "return ::print$cppClass(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";

  let hasCanonicalizer = 1;
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "ValueRange":$operands)>];
}

def MuxOp : Handshake_Op<"mux", [
  NoSideEffect, MergeLikeOpInterface,
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]> 
]> {
  let summary = "mux operation";
  let description = [{
    The mux operation represents a(deterministic) merge operation.
    Operands: select, data0, data1, data2, ...

    The 'select' operand is received from ControlMerge of the same
    block and it represents the index of the data operand that the mux
    should propagate to its single output.  The number of data inputs
    corresponds to the number of predecessor blocks.

    The mux operation is intended solely for control+dataflow selection.
    For purely dataflow selection, use the 'select' operation instead.

    Example:
    ```mlir
    %0 = mux %select [%data0, %data1, %data2] {attributes}: index, i32
    ```
  }];
  let arguments = (ins
    TypeAttr:$dataType,
    Confined<I32Attr, [IntMinValue<1>]>:$size,
    AnyType : $selectOperand,
		Variadic<AnyType> : $dataOperands);
  let results = (outs AnyType : $result);
  let skipDefaultBuilders = 1;
  let builders = [
    // Note/warning: This builder should only be used by StandardToHandshake
    // since it contains some hardcoded assumptions that the conversion pass
    // will manually adjust its operand later.
    OpBuilder<(ins "Value":$anyInput, "int":$inputs)>,
    // Use this builder for any other use case.
    OpBuilder<(ins "Value":$selectOperand, "ValueRange":$inputs)>];
  let printer = "return ::print$cppClass(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";
  let verifier = "return ::verify(*this);";
  let hasCanonicalizer = 1;
}

def ControlMergeOp : Handshake_Op<"control_merge", [
  NoSideEffect, MergeLikeOpInterface, HasClock,
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getResultName"]>
]> {
  let summary = "control merge operation";
  let description = [{
    The control_merge operation represents a
    (nondeterministic) control merge.  Any input is propagated to the
    first output and the index of the propagated input is sent to the
    second output.  The number of inputs corresponds to the number of
    predecessor blocks.  ControlMerge is a control-only
    component(i.e., has no data but only bidirectional handshake).

    Example:
    ```
    %0, %idx = control_merge %a, %b, %c : i32
    ```
  }];

  let arguments = (ins
    TypeAttr:$dataType,
    Confined<I32Attr, [IntMinValue<1>]>:$size,
    Variadic<AnyType> : $dataOperands);
  let results = (outs AnyType : $result, Index : $index);

  let hasCanonicalizer = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    // @note: Legacy builder used by the std-to-handshake pass. Should be removed once the pass is rewritten.
    OpBuilder<(ins "Value":$operand, "int":$inputs)>,
    OpBuilder<(ins "ValueRange":$inputs)>
  ];
  let printer = "return ::print$cppClass(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";
}

def BranchOp : Handshake_Op<"br", [
  NoSideEffect, DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<GeneralOpInterface>,
  AllTypesMatch<["dataOperand", "dataResult"]>
  ]> {
  let summary = "branch operation";
  let description = [{
      The branch operation represents an unconditional
      branch.  The single data input is propagated to the single
      successor.  The input must be triggered by some predecessor to
      avoid continous triggering of a successor block.

      Example:
      ```mlir
      %1 = br %0 : i32
      ```
  }];
  let arguments = (ins
    TypeAttr:$dataType,
    AnyType : $dataOperand);
  let results = (outs AnyType : $dataResult);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Value":$dataOperand)>];

  let hasCanonicalizer = 1;
  let parser = "return ::parse$cppClass(parser, result);";
  let printer = "return ::print$cppClass(p, *this);";
}

def ConditionalBranchOp : Handshake_Op<"cond_br", [
  NoSideEffect, DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getResultName"]>,
  TypesMatchWith<"data operand type matches true branch result type",
                    "dataOperand", "trueResult", "$_self">,
  TypesMatchWith<"data operand type matches false branch result type",
                    "dataOperand", "falseResult", "$_self">
]> {
  let summary = "conditional branch operation";
  let description = [{
     The cbranch operation represents a conditional
     branch.  The data input is propagated to one of the two outputs
     based on the condition input.

     Example:
     ```mlir
     %true, %false = conditional_branch %cond, %data : i32
     ```
  }];

  let arguments = (ins I1 : $conditionOperand,
                       AnyType : $dataOperand);
  let results = (outs AnyType : $trueResult,
                      AnyType : $falseResult);

  let builders = [OpBuilder<(ins "Value":$condOperand, "Value":$dataOperand)>];
  let skipDefaultBuilders = 1;
  let printer = "return ::print$cppClass(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";
  let extraClassDeclaration = [{
    // These are the indices into the dests list.
    enum { trueIndex = 0, falseIndex = 1 };
  }];
}

def SelectOp : Handshake_Op<"select", [
  NoSideEffect,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>,
  TypesMatchWith<"data operand type matches true branch result type",
                    "trueOperand", "falseOperand", "$_self">,
  TypesMatchWith<"data operand type matches false branch result type",
                    "falseOperand", "result", "$_self">
]> {
  let summary = "Select operation";
  let description = [{
     The select operation will select between two inputs based on an input
     conditional. The select operation differs from a mux in that
     1. All operands must be valid before the operation can transact
     2. All operands will be transacted at simultaneously

     The 'select' operation is intended to handle 'std.select' and other
     ternary-like operators, which considers strictly dataflow. The 'mux' operator
     considers control+dataflow between blocks.

     Example:
     ```mlir
     %res = select %cond, %true, %false : i32
     ```
  }];

  let arguments = (ins I1 : $condOperand,
                       AnyType : $trueOperand, AnyType : $falseOperand);
  let results = (outs AnyType : $result);

  let builders = [OpBuilder<(ins 
    "Value":$condOperand, "Value":$trueOperand, "Value":$falseOperand)>];
  let skipDefaultBuilders = 1;
  let printer = "return ::print$cppClass(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";
}

def SinkOp
    : Handshake_Op<"sink", [DeclareOpInterfaceMethods<ExecutableOpInterface>]> {
  let summary = "sink operation";
  let description = [{
    The sink operation discards any data that arrives at its
         input.The sink has no successors and it can continuously consume data.

    Example:
    ```mlir
    sink %data : i32 
    ```
  }];
  let arguments = (ins AnyType:$operand);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$operand)>,
    // Like ODS-generated builder, needed for instantiation through declarative rewrite patterns. 
    OpBuilder<(ins 
      "TypeRange":$resultTypes, "ValueRange":$operands, "ArrayRef<::mlir::NamedAttribute>":$attributes)>
  ];
  let printer = "return ::print$cppClass(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";
}

def SourceOp : Handshake_Op<"source", [NoSideEffect]> {
  let summary = "source operation";
  let description = [{
    The source operation represents continuous token
    source.  The source continously sets a 'valid' signal which the
    successor can consume at any point in time.
  }];

  let skipDefaultBuilders = 1;
  let results = (outs AnyType:$result);
  let parser = "return ::parse$cppClass(parser, result);";
  let builders = [OpBuilder<(ins)>];
  let printer = "return ::print$cppClass(p, *this);";
}

def NeverOp : Handshake_Op<"never", [NoSideEffect]> {
  let summary = "never operation";
  let description = [{
    The never operation represents disconnected data
    source. The source never sets any 'valid' signal which will
    never trigger the successor at any point in time.
  }];

  let results = (outs AnyType:$result);
  let assemblyFormat = [{ attr-dict `:` qualified(type($result))}];
}

def ConstantOp : Handshake_Op<"constant", [
  NoSideEffect, DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<GeneralOpInterface>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>
]> {
  let summary = "constant operation";
  let description = [{
    The const has a constant value. When triggered by its
    single `ctrl` input, it sends the constant value to its single
    successor.

    Example:
    ```mlir
    %0 = constant %ctrl {value = 42 : i32} : i32
    ```
  }];

  let arguments = (ins AnyAttr:$value, NoneType:$ctrl);
  let results = (outs AnyType : $result);

  let builders = [OpBuilder<(ins "Attribute":$value, "Value":$ctrl)>];
  let skipDefaultBuilders = 1;

  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{
    Attribute getValue() { return (*this)->getAttr("value"); }
  }];
  let assemblyFormat = [{ $ctrl attr-dict `:` qualified(type($result))}];
  let verifier = "return ::verify$cppClass(*this);";
}

def EndOp
    : Handshake_Op<"end", [DeclareOpInterfaceMethods<ExecutableOpInterface>]> {
  let summary = "end operation";
  let description = [{
    The end propagates the result of the appropriate
    return operation from one of its inputs to its single output after
    all memory accesses have completed.  Currently not used(data
    returned through ReturnOp).
  }];
  let arguments = (ins AnyType : $control, Variadic<AnyType> : $operands);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Value":$operand)>];
  let assemblyFormat = [{ operands attr-dict `:` qualified(type($control)) `,` qualified(type($operands))}];
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
  let results = (outs NoneType : $result);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins)>];
  let assemblyFormat = [{ attr-dict `:` qualified(type($result))}];
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
  let arguments = (ins Variadic<AnyType> : $inputs,
                   I32Attr : $ldCount,
                   I32Attr : $stCount,
                   I32Attr : $id,
                   MemRefTypeAttr : $memRefType);
  let results = (outs Variadic<AnyType> : $outputs);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(
      ins "ArrayRef<Value>":$operands, "int":$outputs, "int":$control_outputs, "bool":$lsq,
      "int":$id, "Value":$memref)>
  ];
  let assemblyFormat = "`[` `ld` `=` $ldCount `,` `st` `=`  $stCount `]` `(` $inputs `)` attr-dict `:` $memRefType `,` functional-type($inputs, $outputs)";
  let verifier = "return ::verify$cppClass(*this);";
}

def ExternalMemoryOp : Handshake_Op<"extmemory", [
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getResultName"]>
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
      %stCtrl = extmemory[ld = 0, st = 1](%mem : memref<10xi32>)(%vout, %addr) {id = 0 : i32} : (i32, index) -> (none)
      %vout, %addr = store(%v, %i, %ctrl) : (i32, index, none) -> (i32, index)
      ...
    }
    ```
  }];
  let arguments = (ins AnyMemRef:$memref,
                       Variadic<AnyType>:$inputs,
                       I32Attr : $ldCount,
                       I32Attr : $stCount,
                       I32Attr : $id);
  let results = (outs Variadic<AnyType>:$outputs);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(
      ins "Value":$memref, "ArrayRef<Value>":$inputs, "int":$ldCount, "int":$stCount,
      "int":$id)>
  ];
  let assemblyFormat = "`[` `ld` `=` $ldCount `,` `st` `=`  $stCount `]` `(` $memref `:` qualified(type($memref)) `)` `(` $inputs `)` attr-dict `:` functional-type($inputs, $outputs)";
}

def LoadOp : Handshake_Op<"load", [
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getResultName"]>
]> {
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

    Example:
    ```mlir
    %dataToSucc, %addr1ToMem, %addr2ToMem = load [%addr1, %addr2] %dataFromMem, %ctrl : i8, i16, index
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$addresses, AnyType:$data, NoneType:$ctrl);
  let results = (outs AnyType:$dataResult, Variadic<AnyType>:$addressResults);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Value":$memref, "ArrayRef<Value>":$indices)>];
  let printer = "return ::print$cppClass(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";
  let verifier = "return ::verify$cppClass(*this);";
}

def StoreOp : Handshake_Op<"store", [
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<GeneralOpInterface>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getResultName"]>
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
    Types: data type followed by address type.

    Example:
    ```mlir
    %dataToMem, %addrToMem = store [%addr1, %addr2] %dataFromPred , %ctrl : i8, i16, index
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$addresses, AnyType:$data, NoneType:$ctrl);
  let results = (outs AnyType:$dataResult, Variadic<AnyType>:$addressResult);

  let builders =
      [OpBuilder<(ins "Value":$valueToStore, "ArrayRef<Value>":$indices)>];
  let printer = "return ::print$cppClass(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";
  let verifier = "return ::verify$cppClass(*this);";
}

def JoinOp : Handshake_Op<"join", [
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<GeneralOpInterface>
]> {
  let summary = "join operation";
  let description = [{
    A control-only synchronizer.  Produces a valid output when all
    inputs become available.
    
    Example:
    ```mlir
    %0 = join %a, %b, %c : i32
    ```
  }];
  let arguments = (ins Variadic<NoneType> : $data);
  let results = (outs NoneType : $result);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "ArrayRef<Value>":$operands)>];
  let printer = "return ::print$cppClass(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";
}

def I4 : I<4>;
def I4Attr : SignlessIntegerAttrBase<I4, "4-bit integer attribute">;
