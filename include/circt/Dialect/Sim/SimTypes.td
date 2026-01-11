//===- SimTypes.td - Sim dialect types ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_SIMTYPES_TD
#define CIRCT_DIALECT_SIM_SIMTYPES_TD

include "circt/Dialect/Sim/SimDialect.td"
include "mlir/IR/AttrTypeBase.td"

class SimTypeDef<string name> : TypeDef<SimDialect, name> { }

def FormatStringType : SimTypeDef<"FormatString"> {
  let mnemonic = "fstring";

  let summary = "Format string type";
  let description = [{
    A format string type represents either a single formatting fragment or the
    concatenation of an arbitrary but finite number of fragments.
    A formatting fragment is either a static string literal or the association
    of a dynamic hardware value with a format specifier.
  }];
}

def DynamicStringType : SimTypeDef<"DynamicString"> {
  let mnemonic = "dstring";

  let summary = "String type with variable length"; 
  let description = [{
    The dynamic string type represents a string with a variable length that is 
    only known at runtime. The contents of the string are held as an array of
    `i8` bytes. The interpretation of these bytes as a concrete character 
    encoding is undefined, but treating the bytes as a UTF-8 string is 
    recommended. A string's length must fit into an `i64`.
    Even though practical lowerings of this type likely have to hold the string
    contents in dynamically-allocated heap memory, *this type has value
    semantics*. If the same string SSA value is passed to multiple users, any
    mutation performed by one user is not visible to other users.
  }];
}

#endif // CIRCT_DIALECT_SIM_SIMTYPES_TD
