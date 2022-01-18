//===- GAADialect.td - GAA dialect definition -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the GAA Dialect.
//
//===----------------------------------------------------------------------===//

def GAADialect : Dialect {
  let name = "gaa";
  let cppNamespace = "::circt::gaa";
  let summary = "Operations for gaa dialect";
  let description = [{
    This dialect defines the `gaa` dialect, which is used to lower from
    GAA IR to HW.
    For more information, refer to [RationalGAA Documentation](https://github.com/llvm/circt/blob/main/docs/RationaleGAA.md).
  }];
}

// Base class for the operation in this dialect.
class GAAOp<
  string mnemonic,
  list<Trait> traits = []
  > : Op<GAADialect, mnemonic, traits>;
