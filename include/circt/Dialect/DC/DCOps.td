//===- DCOps.td - DC dialect operations --------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DC_OPS_TD
#define CIRCT_DIALECT_DC_OPS_TD

include "circt/Dialect/DC/DCDialect.td"
include "circt/Dialect/DC/DCTypes.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/FunctionInterfaces.td"

class DCOp<string mnemonic, list<Trait> traits = []> :
  Op<DCDialect, mnemonic, !listconcat(traits, [
    Pure
  ])>;

class SameTypeConstraint<string lhs, string rhs>
 : TypesMatchWith<"lhs and rhs types should be equivalent",
                   lhs, rhs, [{ $_self }]>;

def FuncOp : Op<DCDialect, "func", [
      IsolatedFromAbove,
      Symbol,
      FunctionOpInterface,
      RegionKindInterface,
      CallableOpInterface,
  ]> {
  // I really don't want to have to add yet another container operation. However,
  // where else do we place DC ops?
  // We have to place them in a graph region operation, so func.func isn't
  // applicable. We're not committed to being in `hw` yet, so hw.module doesn't
  // apply either...
  let summary = "DC function";
  let description = [{
      The `dc.func` operation represents a function in the DC dialect. It has
      a single block that contains the body of the function. The body is a 
      graph region
  }];

  let arguments = (ins
      SymbolNameAttr:$sym_name,
      TypeAttrOf<FunctionType>:$function_type,
      OptionalAttr<DictArrayAttr>:$arg_attrs,
      OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$resAttrs)
  >];

  let extraClassDeclaration = [{
    // Use FunctionOpInterface traits's getFunctionBody method.
    using mlir::detail::FunctionOpInterfaceTrait<dc::FuncOp>::getFunctionBody;

    /// Return the block corresponding to the region.
    Block *getBodyBlock() { return &getFunctionBody().front(); }

    //===------------------------------------------------------------------===//
    // RegionKindInterface
    //===------------------------------------------------------------------===//
    static RegionKind getRegionKind(unsigned index) {
      return RegionKind::Graph;
    }

    /// Returns the argument types of this function.
    ArrayRef<mlir::Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }

    /// Returns the result types of this function.
    ArrayRef<mlir::Type> getResultTypes() {
      return getFunctionType().getResults();
    }

    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() {
      return isExternal() ? nullptr : &getBody();
    }

    /// Returns the results types that the callable region produces when
    /// executed.
    ArrayRef<Type> getCallableResults() {
      return getFunctionType().getResults();
    }

    /// Returns the argument attributes for all callable region arguments or
    /// null if there are none.
    ArrayAttr getCallableArgAttrs() {
      return getArgAttrs().value_or(nullptr);
    }

    /// Returns the result attributes for all callable region results or
    /// null if there are none.
    ArrayAttr getCallableResAttrs() {
      return getResAttrs().value_or(nullptr);
    }
  }];
}

def CallOp : Op<DCDialect, "call",
    [CallOpInterface, MemRefsNormalizable,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "call operation";
  let description = [{
    The `dc.call` operation represents a direct call to a function that is
    within the same symbol scope as the call. The operands and result types of
    the call must match the specified function type. The callee is encoded as a
    symbol reference attribute named "callee".

    Example:

    ```mlir
    %2 = dc.call @my_add(%0, %1) : (f32, f32) -> f32
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getFunctionType().getResults());
    }]>,
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "StringAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, SymbolRefAttr::get(callee), results, operands);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, StringAttr::get($_builder.getContext(), callee),
            results, operands);
    }]>];

  let extraClassDeclaration = [{
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def ReturnOp : DCOp<"return", [Terminator, ReturnLike, HasParent<"dc::FuncOp">]> {
  let summary = "Return operation";
  let description = [{
    The `dc.return` operation returns from a function. It takes a list of
    values to return. The number of values must match the number of results
    of the function.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}

def BufferOp : DCOp<"buffer",
    [SameTypeConstraint<"input", "output">]> {
  let summary = "Buffer operation";
  let description = [{
    The buffer operation may buffer a control-side token against an arbitrary amount
    of data-side values. In practice, this is realized by buffering a !dc.value<...>.

    Example:
    ```mlir
    %value_out = dc.buffer %value : !dc.value<i32, i1, i4>
    ```
  }];

  let arguments = (ins
    ValueOrTokenType:$input,
    ConfinedAttr<I32Attr, [IntMinValue<1>]>:$size,
    OptionalAttr<AnyAttr>:$initValues
  );
  let results = (outs ValueOrTokenType:$output);

  let assemblyFormat = "$input `[` $size `]` attr-dict `:` type($input)";
  let builders = [OpBuilder<(
    ins "Value":$input, "size_t":$size), [{
        build(odsBuilder, odsState, input.getType(), input, odsBuilder.getI32IntegerAttr(size), {});
    }]>];
}

def JoinOp : DCOp<"join"> {
  let summary = "Synchronizes the incoming tokens with the outgoing token";
  let description = [{
    This operator synchronizes all incoming tokens. Synchronization implies applying
    join semantics in between all in- and output ports.

    Example:
    ```mlir
    %0 = dc.join %a, %b
    ```
  }];

  let arguments = (ins Variadic<TokenType>:$tokens);
  let results = (outs TokenType:$output);

  let assemblyFormat = "$tokens attr-dict";
  let hasCanonicalizer = 1;
  let builders = [OpBuilder<(
    ins "ValueRange":$tokens), [{
        build(odsBuilder, odsState, odsBuilder.getType<dc::TokenType>(), tokens);
    }]>];
}

def ForkOp : DCOp<"fork"> {
    let summary = "Splits the incoming token into multiple outgoing tokens";
    let description = [{
        This operator splits the incoming token into multiple outgoing tokens.
    
        Example:
        ```mlir
        %0, %1 = dc.fork %a : !dc.token, !dc.token
        ```
    }];
    
    let arguments = (ins TokenType:$token);
    let results = (outs Variadic<TokenType>:$outputs);

    let assemblyFormat = "$token attr-dict (`:` qualified(type($outputs))^)?";
    let hasCanonicalizer = 1;

    let builders = [OpBuilder<(
        ins "Value":$token, "size_t":$numOutputs), [{
            llvm::SmallVector<mlir::Type, 4> outputTypes(numOutputs, odsBuilder.getType<dc::TokenType>());
            build(odsBuilder, odsState, outputTypes, token);
        }]>];
}

def BranchOp : DCOp<"branch"> {
    let summary = "Branch operation";
    let description = [{
        The token is propagated to a selected output based on the value of the
        condition signal.
    }];

    let arguments = (ins I1ValueType:$condition, TokenType:$token);
    let results = (outs TokenType:$trueToken, TokenType:$falseToken);

    let assemblyFormat = "$condition `,` $token attr-dict";
    let builders = [OpBuilder<(
        ins "Value":$condition, "Value":$token), [{
            auto tt = odsBuilder.getType<dc::TokenType>();
            build(odsBuilder, odsState, tt, tt, condition, token);
        }]>];
}

def MergeOp : DCOp<"merge"> {
    let summary = "Merge operation";
    let description = [{
        An input token is selected based on the value of the incoming select
        signal, and propagated to the single output. Only the condition value,
        the selected input, and the output will be transacted.
    }];

    let arguments = (ins ValueType:$select, Variadic<TokenType>:$tokens);
    let results = (outs TokenType:$output);
    let assemblyFormat = "$select `[` $tokens `]` attr-dict `:` qualified(type($select))";
    let builders = [OpBuilder<(
        ins "Value":$select, "ValueRange":$tokens), [{
            build(odsBuilder, odsState, odsBuilder.getType<dc::TokenType>(), select, tokens);
        }]>];
    let hasCanonicalizer = 1;
}

def SinkOp : DCOp<"sink"> {
    let summary = "Sink operation";
    let description = [{
        The sink operation will always accept any incoming tokens, and 
        discard them.
    }];

    let arguments = (ins TokenType:$token);
    let results = (outs);
    let assemblyFormat = "$token attr-dict";
}

def SourceOp : DCOp<"source"> {
    let summary = "Source operation";
    let description = [{
        The source operation will always produce a token.
    }];

    let arguments = (ins);
    let results = (outs TokenType:$output);
    let assemblyFormat = "attr-dict";
    let builders = [OpBuilder<(ins), [{
            build(odsBuilder, odsState, odsBuilder.getType<dc::TokenType>());
        }]>];
}

def PackOp : DCOp<"pack"> {
    let summary = "Pack operation";
    let description = [{
        An operation which packs together a !dc.token value with some other
        value - mostly used to facilitate conversion, given the lack of
        1:N value replacement in MLIRs pattern rewrite infrastructure.
    }];

    let arguments = (ins TokenType:$token, Variadic<AnyType>:$inputs);
    let results = (outs ValueType:$output);
    let assemblyFormat = "$token `[` $inputs `]` attr-dict `:` functional-type($inputs, $output)";
    let builders = [OpBuilder<(
        ins "Value":$token, "ValueRange":$inputs)>];
    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

def UnpackOp : DCOp<"unpack"> {
    let summary = "Unpack operation";
    let description = [{
        An operation which unpacks a !dc.value value into a !dc.token value
        and its constituent values - mostly used to facilitate conversion, given the
        lack of 1:N value replacement in MLIRs pattern rewrite infrastructure.
    }];

    let arguments = (ins ValueType:$input);
    let results = (outs TokenType:$token, Variadic<AnyType>:$outputs);
    let assemblyFormat = "$input attr-dict `:` functional-type($input, $outputs)";
    let hasCanonicalizer = 1;
    let hasFolder = 1;

    let builders = [OpBuilder<(
        ins "Value":$input), [{
            llvm::SmallVector<mlir::Type, 4> outputTypes;
            outputTypes.push_back(odsBuilder.getType<dc::TokenType>());
            dc::ValueType vt = input.getType().cast<dc::ValueType>();
            llvm::append_range(outputTypes, vt.getInnerTypes());
            build(odsBuilder, odsState, outputTypes, input);
        }]>];
}

/*
def SymbolOp : DCOp<"symbol", [
        Symbol
    ]> {
    let summary = "Indicates a symbol in the control and dataflow graph";
    let description = [{
        This operator indicates a symbol in the control and dataflow graph.
        A symbol can be referenced from within `dc.node` operations.
    
        Example:
        ```mlir
        dc.node @foo
        ```
    }];

    let arguments = (ins SymbolNameAttr:$sym_name);
    let results = (outs);

    let assemblyFormat = "$sym_name attr-dict";
}

class SameTypeConstraint<string lhs, string rhs>
 : TypesMatchWith<"lhs and rhs types should be equivalent",
                   lhs, rhs, [{ $_self }]>;

// Define `dc.node` which is an operation that contains a list of references to
// `dc.symbol` ops.
class NodeOp<string mnemonic, list<Trait> traits = []> : DCOp<mnemonic,
    !listconcat(traits,[
        SingleBlock,
        SameTypeConstraint<"out", "in">
    ])> {
    let arguments = (ins
        AnyType:$in,
        FlatSymbolRefArrayAttr:$symbols
    );
    let results = (outs AnyType:$out);
    let assemblyFormat = "$in `[` $symbols `]` attr-dict `:` type($out)";
}

def ControlNode : NodeOp<"control"> {}
def DataNode : NodeOp<"data"> {}

*/

#endif // CIRCT_DIALECT_DC_OPS_TD