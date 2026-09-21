//===- SVTypesImpl.td - SV type definitions ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SV_SVTYPESIMPL
#define CIRCT_DIALECT_SV_SVTYPESIMPL

include "circt/Dialect/SV/SVDialect.td"
include "mlir/IR/AttrTypeBase.td"

class SVType<string name> : TypeDef<SVDialect, name> { }


def NetTypeImpl : SVType<"Net"> {
  let summary = "SystemVerilog net lvalue";
  let description = [{
    Represents a SystemVerilog net handle (e.g. a structural wire). The
    element type must be a valid net element type; it may not itself be a
    net or var handle, since handles do not nest and net/var kinds do not
    mix.
  }];
  let mnemonic = "net";
  let parameters = (ins "::mlir::Type":$elementType);
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      return $_get(elementType.getContext(), elementType);
    }]>
  ];

  let assemblyFormat = "`<` $elementType `>`";
}

def VarTypeImpl : SVType<"Var"> {
  let summary = "SystemVerilog variable lvalue";
  let description = [{
    Represents a SystemVerilog variable handle (e.g. a register or logic
    declaration). The element type must be a valid var element type; it may
    not itself be a net or var handle, since handles do not nest and
    net/var kinds do not mix.
  }];
  let mnemonic = "var";
  let parameters = (ins "::mlir::Type":$elementType);
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      return $_get(elementType.getContext(), elementType);
    }]>
  ];

  let assemblyFormat = "`<` $elementType `>`";
}

def UnpackedOpenArrayTypeImpl : SVType<"UnpackedOpenArray"> {
  let summary = "SystemVerilog unpacked 'open' array";
  let description = [{
    This type represents unpacked 'open' array. Open array is a special type that can be
    used as formal arguments of DPI import statements.

    See SystemVerilog Spec 35.5.6.1.
  }];

  let mnemonic = "open_uarray";
  let parameters = (ins "::mlir::Type":$elementType);

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      auto *ctx = elementType.getContext();
      return $_get(ctx, elementType);
    }]>
  ];

  let assemblyFormat = "`<` $elementType `>`";
}

#endif // CIRCT_DIALECT_SV_SVTYPESIMPL