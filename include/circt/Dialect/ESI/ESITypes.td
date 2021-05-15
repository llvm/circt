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

def Struct : ESI_Type<"Struct"> {
  let summary = "A HW struct with a name";
  let mnemonic = "struct";
  let parameters = (ins
    StringRefParameter<"struct name">:$name,
    "::circt::hw::StructType":$inner);

  let extraClassDeclaration = [{
    ArrayRef<::circt::hw::StructType::FieldInfo> getElements() const {
      return getInner().getElements();
    }
  }];
}
