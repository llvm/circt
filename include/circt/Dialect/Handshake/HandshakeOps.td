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

#ifdef HANDSHAKE_OPS
#else
#define HANDSHAKE_OPS

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

include "HandshakeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Handshake_Dialect : Dialect {
  let name = "handshake";
  let cppNamespace = "::circt::handshake";
}

// Base class for Handshake dialect ops.
class Handshake_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<Handshake_Dialect, mnemonic,
         traits # [HasParent<"handshake::FuncOp">]> {
}

// This is almost exactly like a standard FuncOp, except that it has some
// extra verification conditions.  In particular, each Value must
// only have a single use.  Also, it defines a Dominance-Free Scope
def FuncOp : Op<Handshake_Dialect, "func", [
   NativeOpTrait<"IsIsolatedFromAbove">,
   NativeOpTrait<"FunctionLike">,
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

  let builders = [OpBuilderDAG<
    (ins "llvm::StringRef":$name, "mlir::FunctionType":$type,
      CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attrs,
      CArg<"llvm::ArrayRef<mlir::NamedAttrList>", "{}">:$argAttrs),
    [{
      $_state.addAttribute(SymbolTable::getSymbolAttrName(),
      $_builder.getStringAttr(name));
      $_state.addAttribute(getTypeAttrName(), TypeAttr::get(type));
      $_state.attributes.append(attrs.begin(), attrs.end());
      $_state.addRegion();

      // Not needed?? Arguments are already included in attrs
      /* assert(type.getNumInputs() == argAttrs.size());
       SmallString<8> argAttrName;
       for (unsigned i = 0, e = type.getNumInputs(); i != e; ++i)
         if (auto argDict = argAttrs[i].getDictionary())
           $_state.addAttribute(getArgAttrName(i, argAttrName),
       argDict);*/
     }]>];

  let extraClassDeclaration = [{
    // Add an entry block to an empty function, and set up the block arguments
    // to match the signature of the function.
    mlir::Block *addEntryBlock();

    mlir::FunctionType getType() {
      return (*this)->getAttrOfType<mlir::TypeAttr>(getTypeAttrName())
          .getValue()
          .cast<mlir::FunctionType>();
    }
    // bool isVarArg() {
    //   return getType().getUnderlyingType()->isFunctionVarArg();
    // }

    // This trait needs access to the hooks defined below.
    friend class mlir::OpTrait::FunctionLike<handshake::FuncOp>;

    /// Implement RegionKindInterface.
    static mlir::RegionKind getRegionKind(unsigned index) {
      return mlir::RegionKind::Graph;
    }

    /// Returns the number of arguments. This is a hook for
    /// OpTrait::FunctionLike.
    unsigned getNumFuncArguments() { return getType().getInputs().size(); }

    /// Returns the number of results. This is a hook for OpTrait::FunctionLike.
    unsigned getNumFuncResults() { return getType().getResults().size(); }

    /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
    /// attribute is present and checks if it holds a function type.  Ensures
    /// getType, getNumFuncArguments, and getNumFuncResults can be called
    /// safely.
    mlir::LogicalResult verifyType() {
      auto type = getTypeAttr().getValue();
      if (!type.isa<mlir::FunctionType>())
        return emitOpError("requires '" + getTypeAttrName() +
                           "' attribute of function type");
      return mlir::success();
    }
  }];

  let verifier = [{
    // If this function is external there is nothing to do.
    if (isExternal())
      return mlir::success();

    // Verify that the argument list of the function and the arg list of the
    // entry block line up.  The trait already verified that the number of
    // arguments is the same between the signature and the block.
    auto fnInputTypes = getType().getInputs();
    mlir::Block &entryBlock = front();

    for (unsigned i = 0, e = entryBlock.getNumArguments(); i != e; ++i)
      if (fnInputTypes[i] != entryBlock.getArgument(i).getType())
        return emitOpError("type of entry block argument #")
               << i << '(' << entryBlock.getArgument(i).getType()
               << ") must match the type of the corresponding argument in "
               << "function signature(" << fnInputTypes[i] << ')';

    return mlir::success();
  }];
  let printer = [{
    mlir::FunctionType fnType = getType();
    impl::printFunctionLikeOp(p, *this, fnType.getInputs(), /*isVariadic=*/true,
                              fnType.getResults());
  }];
  let parser = [{
    auto buildFuncType =
        [](Builder & builder, llvm::ArrayRef<mlir::Type> argTypes,
            llvm::ArrayRef<mlir::Type> results,
           impl::VariadicFlag, std::string &) {
      return builder.getFunctionType(argTypes, results);
    };

    return impl::parseFunctionLikeOp(parser, result, /*allowVariadic=*/true,
                                     buildFuncType);
  }];
}

// InstanceOp
def InstanceOp : Handshake_Op<"instance", [CallOpInterface]> {
  let summary = "module instantiate operation";
  let description = [{
    The `instance` operation represents the instantiation of a module.  This
	  is similar to a function call, except that different instances of the same
    module are guaranteed to have their own distinct state. The instantiated
    module is encoded as a symbol reference attribute named "module".

    Example:

    ```mlir
    %2 = handshake.instance @my_add(%0, %1) : (f32, f32) -> f32
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$module, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [OpBuilderDAG<
    (ins "mlir::FuncOp":$module, CArg<"mlir::ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("module", $_builder.getSymbolRefAttr(module));
      $_state.addTypes(module.getType().getResults());
  }]>, OpBuilderDAG<
    (ins "mlir::SymbolRefAttr":$module, "llvm::ArrayRef<mlir::Type>":$results,
     CArg<"mlir::ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("module", module);
      $_state.addTypes(results);
  }]>, OpBuilderDAG<
    (ins "llvm::StringRef":$module, "llvm::ArrayRef<mlir::Type>":$results,
     CArg<"mlir::ValueRange", "{}">:$operands), [{
      build($_builder, $_state, $_builder.getSymbolRefAttr(module), results,
            operands);
  }]>];

  let extraClassDeclaration = [{
    llvm::StringRef getModule() { return module(); }
    mlir::FunctionType getModuleType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the module of this operation.
    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("module");
    }
  }];

  let assemblyFormat = [{
    $module `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
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

  let builders = [OpBuilderDAG<(ins "llvm::ArrayRef<mlir::Value>":$operands)>];

  let printer = "return ::print(p, *this);";
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
                   Confined<I32Attr, [IntMinValue<0>]>:$slots);
  let results = (outs AnyType);

  let extraClassDeclaration = [{
    bool isSequential() {
      return (*this)->getAttrOfType<mlir::BoolAttr>("sequential").getValue();
    }
    bool isControl() {
      return (*this)->getAttrOfType<mlir::BoolAttr>("control").getValue();
    }
    llvm::APInt getNumSlots() {
      return (*this)->getAttrOfType<mlir::IntegerAttr>("slots").getValue();
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
  let builders = [OpBuilderDAG<(ins "mlir::Value":$operand, "int":$outputs)>];

  let extraClassDeclaration = [{
    bool isControl() {
      return (*this)->getAttrOfType<mlir::BoolAttr>("control").getValue();
    }
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
  let builders = [OpBuilderDAG<(ins "mlir::Value":$operand, "int":$outputs)>];

  let extraClassDeclaration = [{
    bool isControl() {
      return (*this)->getAttrOfType<mlir::BoolAttr>("control").getValue();
    }
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
  let skipDefaultBuilders = 1;
  let builders = [OpBuilderDAG<(ins "mlir::Value":$operand, "int":$inputs)>];
}

def MuxOp : Handshake_Op<"mux", [
  NoSideEffect, MergeLikeOpInterface,
  DeclareOpInterfaceMethods<ExecutableOpInterface>
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

  let skipDefaultBuilders = 1;
  let builders = [OpBuilderDAG<(ins "mlir::Value":$operand, "int":$inputs)>];
  let verifier = "return ::verify(*this);";
}

def ControlMergeOp : Handshake_Op<"control_merge", [
  NoSideEffect, MergeLikeOpInterface, HasClock,
  DeclareOpInterfaceMethods<ExecutableOpInterface>
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

  let skipDefaultBuilders = 1;
  let builders = [OpBuilderDAG<(ins "mlir::Value":$operand, "int":$inputs)>];

  let extraClassDeclaration = [{
    bool isControl() {
      return (*this)->getAttrOfType<mlir::BoolAttr>("control").getValue();
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
  let builders = [OpBuilderDAG<(ins "mlir::Value":$dataOperand)>];

  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{
    bool isControl() {
      return (*this)->getAttrOfType<mlir::BoolAttr>("control").getValue();
    }
  }];
}

def ConditionalBranchOp : Handshake_Op<"conditional_branch", [
  NoSideEffect, DeclareOpInterfaceMethods<ExecutableOpInterface>
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

  let skipDefaultBuilders = 1;
  let builders = [OpBuilderDAG<(ins "mlir::Value":$condOperand,
                                    "mlir::Value":$dataOperand)>];

  let extraClassDeclaration = [{
    // These are the indices into the dests list.
    enum { trueIndex = 0, falseIndex = 1 };

    mlir::Value getFalseResult() { return getResult(falseIndex); }

    mlir::Value getTrueResult() { return getResult(trueIndex); }

    mlir::Value getDataOperand() { return getOperand(1); }

    bool isControl() {
      return (*this)->getAttrOfType<mlir::BoolAttr>("control").getValue();
    }
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
  let builders = [OpBuilderDAG<(ins "mlir::Value":$operand)>];
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

def ConstantOp : Handshake_Op<"constant", [
  NoSideEffect, DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<GeneralOpInterface>
]> {
  let summary = "constant operation";
  let description = [{
    The "handshake.const" has a constant value.When triggered by its
    single input, it sends the constant value to its single
    successor.
  }];

  let arguments = (ins AnyType, AnyAttr : $value);
  let results = (outs AnyType);

  // let skipDefaultBuilders = 1;
  let builders = [OpBuilderDAG<(ins "mlir::Attribute":$value,
                                    "mlir::Value":$operand)>];

  let extraClassDeclaration = [{
    mlir::Attribute getValue() { return (*this)->getAttr("value"); }
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
  let builders = [OpBuilderDAG<(ins "mlir::Value":$operand)>];
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
  let builders = [OpBuilderDAG<(ins)>];
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
  let builders = [
    OpBuilderDAG<(ins "llvm::ArrayRef<mlir::Block *>":$successors)>
  ];
}

def MemRefTypeAttr : TypeAttrBase<"mlir::MemRefType", "memref type attribute">;
def MemoryOp : Handshake_Op<"memory", [
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<MemoryOpInterface>, HasClock
]> {
  let summary = "memory";
  let description = [{
    Each MemoryOp represents an independent memory or memory region (BRAM or
    external memory). It receives memory access requests from load and store
    operations. For every request, it returns data (for load) and a data-less
    token indicating completion.
    
    Operands: all stores (stdata1, staddr1, stdata2, staddr2, ...), then all
     loads (ldaddr1, ldaddr2,...)
    
    Outputs: all load outputs, ordered the same as load addresses (lddata1,
      lddata2, ...), followed by all none outputs, ordered as operands (stnone1,
      stnone2,...ldnone1, ldnone2,...)
  }];
  let arguments = (ins Variadic<AnyType>,
                          I32Attr : $ld_count,
                       I32Attr : $st_count,
                       I32Attr : $id,
                       MemRefTypeAttr : $type);
  let results = (outs Variadic<AnyType>);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilderDAG<
    (ins "llvm::ArrayRef<mlir::Value>":$operands, "int":$outputs,
      "int":$control_outputs, "bool":$lsq, "int":$id, "mlir::Value":$memref)>
  ];
  let extraClassDeclaration = [{
    llvm::APInt getLdCount() {
      return (*this)->getAttrOfType<mlir::IntegerAttr>("ld_count").getValue();
    }
    llvm::APInt getStCount() {
      return (*this)->getAttrOfType<mlir::IntegerAttr>("st_count").getValue();
    }
    unsigned getID() {
      return (*this)->getAttrOfType<mlir::IntegerAttr>("id").getValue()
        .getZExtValue();
    }
    mlir::MemRefType getMemRefType() {
      return (*this)->getAttrOfType<mlir::TypeAttr>("type").getValue()
        .cast<mlir::MemRefType>();
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

     Operands: address indices (from predecessor), data (from MemoryOp),
      control-only input.
     Results: data (to successor), address indices (to MemoryOp).
  }];

  let arguments = (ins Variadic<Index>, AnyType, NoneType);
  let results = (outs AnyType, Variadic<Index>:$addressResults);

  let builders = [OpBuilderDAG<(ins "mlir::Value":$memref, 
                                    "llvm::ArrayRef<mlir::Value>":$indices)>];
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

  let arguments = (ins AnyType, Variadic<Index>, NoneType);
  let results = (outs AnyType, Variadic<Index>);

  let builders =
      [OpBuilderDAG<(ins "mlir::Value":$valueToStore,
                         "llvm::ArrayRef<mlir::Value>":$indices)>];
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
  let builders = [OpBuilderDAG<(ins "llvm::ArrayRef<mlir::Value>":$operands)>];
}

def I4 : I<4>;
def I4Attr : SignlessIntegerAttrBase<I4, "4-bit integer attribute">;

def HasOneResult : Constraint<CPred<"$_self.size() == 1">, "has one result">;

def EliminateSimpleMergesPattern : Pat<(MergeOp $a), (replaceWithValue $a)>;

def EliminateSimpleBranchesPattern
    : Pat<(BranchOp $a, $attr), (replaceWithValue $a)>;

def EliminateSimpleForksPattern : Pat<(ForkOp
                                       : $op $a, $attr),
                                      (replaceWithValue $a), [(HasOneResult
                                                               : $op)]>;

// def EliminateSimpleControlMergesPattern :
//     Pattern<(ControlMergeOp $a),
//             [(replaceWithValue $a),
//              (ConstantOp $a__0, ConstantAttr<I4Attr,
// "0">)]>;

#endif // HANDSHAKE_OPS
