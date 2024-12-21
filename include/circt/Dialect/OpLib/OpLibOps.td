//===- OpLibOps.td - OpLib Op Definitions ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for OpLib.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_OPLIB_OPS_TD
#define CIRCT_DIALECT_OPLIB_OPS_TD

include "circt/Dialect/OpLib/OpLib.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

class OpLibOp<string mnemonic, list<Trait> traits = []> :
  Op<OpLib_Dialect, mnemonic, traits>;

def LibraryOp : OpLibOp<"library", [IsolatedFromAbove, SingleBlock,
                                    Symbol, SymbolTable, NoTerminator,
                                    NoRegionArguments,
                                    HasParent<"mlir::ModuleOp">]> {
  let summary = "OpLib dialect Library operation";
  let description = [{
    The `oplib.library` operation represents an operator library to be
    used by passes that lower to one of a few hardware design dialects. 
    A library is made up of a single block and contains only
    `oplib.operator` ops.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let results = (outs);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let assemblyFormat = [{
    $sym_name attr-dict-with-keyword $bodyRegion
  }];

  let hasVerifier = 1;
}

def OperatorOp : OpLibOp<"operator", [IsolatedFromAbove, 
                                         SingleBlock,
                                         NoRegionArguments,
                                         Symbol, SymbolTable,
                                         HasParent<"LibraryOp">]> {
  let summary = "OpLib dialect Operator operation";
  let description = [{
    The `oplib.operator` operation represents an operator in the
    operator library.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, 
                       I32Attr:$latency,
                       OptionalAttr<F64Attr>:$incDelay,
                       OptionalAttr<F64Attr>:$outDelay);
  let results = (outs);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let assemblyFormat = [{
    $sym_name `latency` `<` $latency `>` (`,` `incDelay` `<` $incDelay^ `>`)? 
    (`,` `outDelay` `<` $outDelay^ `>`)? attr-dict-with-keyword $bodyRegion
  }];

  let hasVerifier = 1;
}

def TargetOp : OpLibOp<"target", [Symbol, 
                                     FunctionOpInterface,
                                     SingleBlockImplicitTerminator<"OutputOp">,
                                     HasParent<"OperatorOp">]> {
  let summary = "OpLib dialect Target operation";
  let description = [{
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);

  let results = (outs);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() {
      return &getBodyRegion();
    }

    /// Verify the type attribute of this function. Returns failure and emits
    /// an error if the attribute is invalid.
    LogicalResult verifyType() {
      auto type = getFunctionTypeAttr().getValue();
      if (!llvm::isa<FunctionType>(type))
        return emitOpError("requires '" + getFunctionTypeAttrName().getValue() +
                           "' attribute of function type");
      return success();
    }
  }];
  // let assemblyFormat = [{
  //   $sym_name custom<FunctionType>($function_type) attr-dict-with-keyword $bodyRegion
  // }];
  // let hasVerifier = 1;
}

def OperationOp : OpLibOp<"operation", [HasParent<"TargetOp">]> {
  let summary = "OpLib dialect Operation operation";
  let description = [{
  }];
  let arguments = (ins Variadic<AnyType>:$operands,
                       StrAttr:$opName,
                       StrAttr:$dialectName,
                       OptionalAttr<DictionaryAttr>:$opDict);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $opName `in` $dialectName (`with` $opDict^)? `(` ($operands^ `:` qualified(type($operands)))? `)`  
    attr-dict (`:` qualified(type($results))^)?
  }];
}

def CalyxMatchOp : OpLibOp<"calyx_match", 
    [Terminator, SymbolTable,
     HasParent<"OperatorOp">,
     NoRegionArguments,
     SingleBlock,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "OpLib dialect Target operation";
  let description = [{
  }];

  let arguments = (ins FlatSymbolRefAttr:$target,
                       TypeAttrOf<FunctionType>:$targetType);

  let results = (outs);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = [{
    `(` $target `:` $targetType `)` `produce` attr-dict-with-keyword $bodyRegion
  }];
  // let hasVerifier = 1;
}

def OutputOp : OpLibOp<"output", [Terminator,
                                ReturnLike, ParentOneOf<["TargetOp"]>]> {
  let summary = "OpLib dialect Output operation";
  let description = [{
    The `oplib.output` operation.
  }];
  let arguments = (ins Variadic<AnyType>:$outputs);
  let assemblyFormat = [{
    attr-dict ($outputs^ `:` qualified(type($outputs)))?
  }];
}

def YieldOp : OpLibOp<"yield", [Terminator,
                                   ReturnLike, 
                                   AttrSizedOperandSegments,
                                   ParentOneOf<["CalyxMatchOp"]>]> {
  let summary = "OpLib dialect Yield operation";
  let description = [{
    The `oplib.output` operation.
  }];
  let arguments = (ins Optional<AnyType>:$clock,
                       Optional<AnyType>:$clockEnable,
                       Optional<AnyType>:$reset,
                       Variadic<AnyType>:$inputs,
                       Variadic<AnyType>:$outputs);
  let assemblyFormat = [{
    attr-dict (`clk` `(` $clock^ `:` qualified(type($clock)) `)` `,` )? 
              (`ce` `(` $clockEnable^ `:` qualified(type($clockEnable)) `)` `,` )?
              (`reset` `(` $reset^ `:` qualified(type($reset)) `)` `,` )?
              (`ins` `(` $inputs^ `:` qualified(type($inputs)) `)` `,` )?
              (`outs` `(` $outputs^ `:` qualified(type($outputs)) `)`)?
  }];
}

#endif // CIRCT_DIALECT_OPLIB_OPS_TD
