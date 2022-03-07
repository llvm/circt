//===- MooreTypesImpl.td - Moore type implementations ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SystemVerilog type implementations for the Moore dialect.
//
//===----------------------------------------------------------------------===//

class MooreType<string name> : TypeDef<MooreDialect, name> { }

def LValueTypeImpl : MooreType<"LValue"> {
  let mnemonic = "lvalue";
  let parameters = (ins "::mlir::Type":$nestedType);
  let assemblyFormat = "`<` $nestedType `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::Type":$nestedType), [{
      return $_get(nestedType.getContext(), nestedType);
    }]>
  ];
}
