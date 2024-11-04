//===- CalyxLibOps.td - CalyxLib Op Definitions ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for CalyxLib.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CALYXLIB_OPS_TD
#define CIRCT_DIALECT_CALYXLIB_OPS_TD

include "circt/Dialect/CalyxLib/CalyxLib.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

class CalyxLibOp<string mnemonic, list<Trait> traits = []> :
  Op<CalyxLib_Dialect, mnemonic, traits>;

def LibraryOp : CalyxLibOp<"library", [IsolatedFromAbove, SingleBlock,
                                       Symbol, SymbolTable, NoTerminator,
                                       HasParent<"mlir::ModuleOp">]> {
  let summary = "CalyxLib dialect Library operation";
  let description = [{
    The `calyxlib.library` operation represents an operator library to be
    used by passes that lower to Calyx. A library is made up of a single
    block and contains only `calyxlib.operator` ops.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let results = (outs);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let assemblyFormat = [{
    $sym_name attr-dict-with-keyword $bodyRegion
  }];

  // let hasVerifier = 1;
}

def OperatorOp : CalyxLibOp<"operator", [IsolatedFromAbove, SingleBlock,
                                         Symbol, NoTerminator, 
                                         SymbolTable,
                                         HasParent<"LibraryOp">]> {
  let summary = "CalyxLib dialect Operator operation";
  let description = [{
    The `calyxlib.operator` operation represents an operator in the
    operator library.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, 
                       OptionalAttr<ArrayAttr>:$calyxLibProperties);
  let results = (outs);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let assemblyFormat = [{
    $sym_name attr-dict-with-keyword custom<CalyxLibProperties>($calyxLibProperties) $bodyRegion
  }];

  // let hasVerifier = 1;
}

def TargetOp : CalyxLibOp<"target", [Symbol, 
                                     FunctionOpInterface,
                                     HasParent<"OperatorOp">]> {
  let summary = "CalyxLib dialect Target operation";
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

// def MatchOp : CalyxLibOp<"match", [HasParent<"OperatorOp">]> {
//   let summary = "CalyxLib dialect Equivalence operation";
//   let description = [{
//     The `calyxlib.equivalence` operation defines an equivalence between a
//     calyx cell and a `calyxlib.target`. The first region must contain exactly
//     one operation that implements the Calyx `CellInterface`. The second region
//     must contain exactly one `calyxlib.target` op.
//   }];

//   let arguments = (ins TypeAttrOf<FunctionType>:$matchTypes);
//   let results = (outs);
//   let regions = (region SizedRegion<1>:$matchRegion,
//                         SizedRegion<1>:$produceRegion);
  // let assemblyFormat = [{
  //   attr-dict-with-keyword $cellRegion `target` custom<TargetTypes>($targetInputTypes) $targetRegion
  // }];
  // let hasCustomAssemblyFormat = 1;

  // let hasVerifier = 1;
// }

// def LatencyOp : CalyxLibOp<"latency", [HasParent<"OperatorOp">]> {
//   let summary = "CalyxLib dialect Latency operation";
//   let description = [{
//     The `calyxlib.latency` operation defines the latency of the operator
//     it is a child of. Only one `calyxlib.latency` operator may exist
//     for each Operator.
//   }];
//   let arguments = (ins I32Attr:$latency);
//   let assemblyFormat = [{
//     $latency attr-dict
//   }];
// }

// def DelayOp : CalyxLibOp<"delay", [HasParent<"OperatorOp">]> {
//   let summary = "CalyxLib dialect Delay operation";
//   let description = [{
//     The `calyxlib.latency` operation defines the incoming and 
//     outgoing delays of the operator it is a child of. Only one 
//     `calyxlib.delay` operator may exist for each Operator.
//   }];
//   let arguments = (ins F32Attr:$incomingDelay,
//                        F32Attr:$outgoingDelay);
//   let assemblyFormat = [{
//     `incoming` `(`$incomingDelay`)` `,` `outgoing` `(` $outgoingDelay `)` attr-dict
//   }];
// }

def OutputOp : CalyxLibOp<"output", [Terminator,
                                ReturnLike, HasParent<"TargetOp">]> {
  let summary = "CalyxLib dialect Output operation";
  let description = [{
    The `calyxlib.output` operation.
  }];
  let arguments = (ins Variadic<AnyType>:$outputs);
  let assemblyFormat = [{
    attr-dict ($outputs^ `:` qualified(type($outputs)))?
  }];
}

#endif // CIRCT_DIALECT_CALYXLIB_OPS_TD
