//===- EmitOps.td - `emit` dialect ops ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for `emit`.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_EMIT_EMITOPS_TD
#define CIRCT_DIALECT_EMIT_EMITOPS_TD

include "circt/Dialect/Emit/EmitDialect.td"
include "mlir/IR/OpAsmInterface.td"

class EmitOp<string mnemonic, list<Trait> traits = []> :
    Op<EmitDialect, mnemonic, traits>;

def FileOp : EmitOp<"file", [
    Symbol,
    SingleBlock,
    NoTerminator,
    NoRegionArguments,
    IsolatedFromAbove
]> {
  let summary = "Represents the contents of an emitted file";

  let description = [{
    This operation groups a set of nested operations to be emitted to a file.

    Other operations (such as file lists)can reference a file to access its
    filename through an optional symbol.
  }];

  let regions = (region SizedRegion<1>:$body);
  let arguments = (ins
    StrAttr:$file_name,
    OptionalAttr<SymbolNameAttr>:$sym_name
  );
  let results = (outs);

  let assemblyFormat = "$file_name (`sym` $sym_name^)? $body attr-dict";

  let skipDefaultBuilders = 1;
  let builders = [
    // Creates a file with a callback. While the callback is executed, the
    // insertion point of the builder is moved inside the body of the file op.
    OpBuilder<(ins "StringRef":$fileName, "StringRef":$symName,
                    CArg<"llvm::function_ref<void()>">:$bodyCtor)>,
    OpBuilder<(ins "StringAttr":$fileName,
                    CArg<"llvm::function_ref<void()>">:$bodyCtor)>,
    OpBuilder<(ins "const Twine &":$fileName,
                    CArg<"llvm::function_ref<void()>", "{}">:$bodyCtor), [{
      return build($_builder, $_state, $_builder.getStringAttr(fileName), bodyCtor);
    }]>
  ];

  let extraClassDeclaration = [{
    // SymbolOpInterface
    static bool isOptionalSymbol() { return true; }

    // Utilities
    Block *getBodyBlock() { return &getBodyRegion().front(); }
  }];
}

def VerbatimOp : EmitOp<"verbatim", [HasParent<"circt::emit::FileOp">]> {
  let summary = "Verbatim opaque text emitted inline.";
  let description = [{
    This operation produces opaque text inline in the file.

    `emit.verbatim` allows symbol reference substitutions with {{0}} syntax.
  }];

  let arguments = (ins
    StrAttr:$text
  );

  let assemblyFormat = [{
    $text attr-dict
  }];
}

def RefOp : EmitOp<"ref", [
  HasParent<"circt::emit::FileOp">,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Print a referenced SV operation inline into the file";
  let description = [{
    The `emit.ref` operation targets an op via a symbol, emitting its
    contents into the file it is part of. The set of targetable
    operations and the emission rules are defined in ExportVerilog.
  }];

  let arguments = (ins FlatSymbolRefAttr:$target);

  let assemblyFormat = [{
    $target attr-dict
  }];
}

def FileListOp : EmitOp<"file_list", [
    Symbol,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Represents a file list";

  let description = [{
    This operation emits a file list referencing a set of files.

    File lists can be references from other ops (including other file lists)
    through an optional symbol.
  }];

  let arguments = (ins
    StrAttr:$file_name,
    FlatSymbolRefArrayAttr:$files,
    OptionalAttr<SymbolNameAttr>:$sym_name
  );
  let results = (outs);

  let assemblyFormat = "$file_name `,` $files (`sym` $sym_name^)? attr-dict";

  let extraClassDeclaration = [{
    // SymbolOpInterface
    static bool isOptionalSymbol() { return true; }
  }];
}

#endif // CIRCT_DIALECT_EMIT_EMITOPS_TD
