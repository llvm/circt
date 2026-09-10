//===- EmitDialect.td - Emit dialect definition ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains the EmitDialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_EMIT_EMITDIALECT
#define CIRCT_DIALECT_EMIT_EMITDIALECT

def EmitDialect : Dialect {
  let name = "emit";
  let cppNamespace = "::circt::emit";

  let summary = "Types and operations for the `emit` dialect";
  let description = [{
    The `emit` dialect is intended to model the structure of the emitted RTL.

    It organizes the files, file lists, directories and collateral.
  }];
}

#endif // CIRCT_DIALECT_EMIT_EMITDIALECT
