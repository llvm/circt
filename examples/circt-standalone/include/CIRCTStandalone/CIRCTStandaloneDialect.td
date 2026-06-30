//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_STANDALONE_DIALECT
#define CIRCT_STANDALONE_DIALECT

include "mlir/IR/OpBase.td"

def CIRCTStandaloneDialect : Dialect {
  let name = "circt_standalone";
  let summary = "An out-of-tree CIRCT example dialect";
  let description = [{
    This dialect demonstrates the boilerplate required to build an out-of-tree
    CIRCT project with a dialect, passes, an optimizer driver, and a plugin.
  }];
  let cppNamespace = "::circt::standalone";
}

class CIRCTStandaloneOp<string mnemonic, list<Trait> traits = []> :
    Op<CIRCTStandaloneDialect, mnemonic, traits>;

#endif // CIRCT_STANDALONE_DIALECT
