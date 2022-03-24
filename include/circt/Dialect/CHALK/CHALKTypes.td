//===- CHALKTypes.td - Definition of CHALK dialect types ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CHALK_CHALKTYPES_TD
#define CIRCT_DIALECT_CHALK_CHALKTYPES_TD

// Base class for the types in this dialect.
class CHALKType<string name> : TypeDef<CHALKDialect, name> {}

def InstanceType : CHALKType<"Instance"> {
  let summary = "An CHALK instance type";
  let description = [{
    Represents an CHALK instance.
  }];
  let mnemonic = "instance";
  let parameters = (ins);
}

#endif // CIRCT_DIALECT_CHALK_CHALKTYPES_TD
