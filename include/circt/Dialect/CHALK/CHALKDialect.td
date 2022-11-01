//===- CHALKDialect.td - CHALK dialect definition --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the CHALK dialect.
//
//===----------------------------------------------------------------------===//

def CHALKDialect : Dialect {
  let name = "chalk";
  let summary = "Types and operations for chalk dialect";
  let description = [{
    This dialect defines the `chalk` dialect, which is intended to represent
    2D VLSI layouts.
  }];

  // let hasConstantMaterializer = 1;
  // let dependentDialects = [
  //   "circt::hw::HWDialect"
  // ];
  // let extraClassDeclaration = [{
  //   /// Register all CHALK types.
  //   void registerTypes();
  //   /// Register all attributes.
  //   void registerAttributes();
  // }];
  let cppNamespace = "::circt::chalk";
}

// Base class for the operations in this dialect.
class CHALKOp<string mnemonic, list<Trait> traits = []> :
    Op<CHALKDialect, mnemonic, traits>;
