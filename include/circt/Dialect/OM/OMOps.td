//===- OMOps.td - Object Model dialect ops --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains the Object Model dialect operation definitions.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_OM_OMOPS_TD
#define CIRCT_DIALECT_OM_OMOPS_TD

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

class OMOp<string mnemonic, list<Trait> traits = []> :
    Op<OMDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Class definitions
//===----------------------------------------------------------------------===//

def ClassOp : OMOp<"class",
    [SingleBlock, NoTerminator, Symbol, SymbolTable,
     HasParent<"mlir::ModuleOp">,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>
    ]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrArrayAttr:$formalParamNames
  );

  let regions = (region
    SizedRegion<1>:$body
  );

  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    mlir::Block *getBodyBlock() { return &getBody().front(); }
  }];
}

def ClassFieldOp : OMOp<"class.field",
    [HasParent<"ClassOp">, Symbol]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    AnyType:$value
  );

  let assemblyFormat = [{
    $sym_name `,` $value `:` type($value) attr-dict
  }];
}

#endif // CIRCT_DIALECT_OM_OMOPS_TD
