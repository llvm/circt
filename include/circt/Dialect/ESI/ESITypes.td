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

//===----------------------------------------------------------------------===//
// Data windows
//===----------------------------------------------------------------------===//

def ESIWindowType : ESI_Type<"Window"> {
  let summary = "a data window";
  let description = [{
    A 'data window' allows designers to break up large messages into multiple
    frames (aka phits) spread across multiple cycles. Windows are specified in
    terms of a mapping of struct fields to frames. The width of a window is the
    maximum frame size + the union tag (log2(#frames)).

    A data window does NOT imply an ESI channel.

    Current restrictions:
    - A field may only appear once.
    - Fields may only be re-ordered (wrt the original message) within a frame.
    - Array fields whose array length is not evenly divisible by 'numItems' will
    have an implicit frame inserted directly after containing the leftover array
    items.
  }];

  let mnemonic = "window";
  let genVerifyDecl = 1;

  let parameters = (ins "StringAttr":$name, "mlir::Type":$into,
                        ArrayRefParameter<"WindowFrameType">:$frames);
  let assemblyFormat = [{ `<` $name `,` $into `,` `[` $frames `]` `>` }];
}

def WindowFrameType : ESI_Type<"WindowFrame"> {
  let summary = "Declare a data window frame";
  let description = [{
    A named list of fields which should appear in a given frame.
  }];
  let mnemonic = "window.frame";

  let parameters = (ins "StringAttr":$name,
                        ArrayRefParameter<"WindowFieldType">:$members);

  let assemblyFormat = [{
    `<` $name `,` `[` $members `]` `>`
  }];
}

def WindowFieldType : ESI_Type<"WindowField"> {
  let summary = "a field-in-frame specifier";
  let description = [{
    Specify that a field should appear within the enclosing frame.
  }];
  let mnemonic = "window.field";

  let parameters = (ins
      "StringAttr":$fieldName,
      OptionalParameter<"uint64_t", "# of items in arrays or lists">:$numItems);

  let assemblyFormat = [{
    `<` $fieldName (`,` $numItems^)? `>`
  }];
}
