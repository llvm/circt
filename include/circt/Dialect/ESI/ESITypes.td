//===- ESITypes.td - ESI type specifications ---------------*- tablegen -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ESI will have a rich, high level type system. Currently, it is much more
// minimal.
//
//===----------------------------------------------------------------------===//

class ESI_Type<string name> : TypeDef<ESI_Dialect, name> { }

def ESIAnyType : ESI_Type<"Any"> {
  let summary = "any type";
  let description = [{
    Used to state that any type is accepted. The specific type will be
    determined later in compilation.
  }];

  let mnemonic = "any";
  let assemblyFormat = "";

  let extraClassDeclaration = [{
    static AnyType get(MLIRContext *context);
  }];
}

def ESIWindowType : ESI_Type<"Window"> {
  let summary = "a window defined by the `esi.data_window` op";
  let mnemonic = "window";

  let parameters = (ins "FlatSymbolRefAttr":$window_sym);
  let assemblyFormat = [{ `<` $window_sym `>` }];
}
