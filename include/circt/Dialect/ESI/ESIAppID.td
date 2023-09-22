//===- ESIChannels.td - All ESI ops related to app channels -- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is where any operations for the ESI dialect live.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_APPID_TD
#define CIRCT_DIALECT_ESI_APPID_TD

include "mlir/IR/AttrTypeBase.td"

def AppIDAttr : ESI_Attr<"AppID"> {
  let summary = "An application relevant instance identifier";
  let description = [{
    Identifies an instance which is visible through multiple hierarchy levels.
    Indended to make locating an instance easier in the instance hierarchy.
  }];

  let parameters = (ins "StringAttr":$name, "uint64_t":$index);
  let mnemonic = "appid";
  let assemblyFormat = [{
    `<` $name `[` $index `]` `>`
  }];

  let extraClassDeclaration = [{
    static constexpr StringRef AppIDAttrName = "esi.appid";
  }];
}

def AppIDPathAttr : ESI_Attr<"AppIDPath"> {
  let summary = "An application-specific hierarchical path through a design";
  let description = [{
    A list of AppID components which specifies a specific dynamic instance
    in the design.
  }];

  let parameters = (ins "FlatSymbolRefAttr":$root,
                        ArrayRefParameter<"AppIDAttr">:$path);
  let mnemonic = "appid_path";
  let assemblyFormat = [{
    `<` $root `[` $path `]` `>`
  }];
}

#endif // CIRCT_DIALECT_ESI_APPID_TD
