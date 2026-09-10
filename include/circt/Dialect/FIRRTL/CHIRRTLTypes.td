//===- CHIRRTLTypes.td - CHIRRTL Types Definition ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the CHIRRTL types.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_CHIRRTLTYPES_TD
#define CIRCT_DIALECT_FIRRTL_CHIRRTLTYPES_TD

include "CHIRRTL.td"

//===----------------------------------------------------------------------===//
// CHIRRTL Types Definitions
//===----------------------------------------------------------------------===//

def CMemoryType : TypeDef<CHIRRTLDialect, "CMemory"> {
  let mnemonic = "cmemory";
  let summary = "a behavioral memory";
  let description = [{
    Syntax:
    ```
    cmemory-type ::= `cmemory` `<` element-type, element-count `>`
    ```

    The value of a `cmemory` type represents a behavioral memory with unknown
    ports.  This is produced by `combmem` and `seqmem` declarations and used by
    `memoryport` declarations to define memories and their ports. A CMemory is
    similar to a vector of passive element types.

    Examples:

    ```mlir
    !chirrtl.cmemory<uint<32>, 16>
    !chirrtl.cmemory<bundle<a : uint<1>>, 16>
    ```
  }];

  let parameters = (ins "firrtl::FIRRTLBaseType":$elementType,
                        "uint64_t":$numElements);

  let builders = [
    TypeBuilderWithInferredContext<(ins "firrtl::FIRRTLBaseType":$elementType,
                                        "uint64_t":$numElements), [{
      return $_get(elementType.getContext(), elementType, numElements);
    }]>
  ];

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

def CMemoryPortType : TypeDef<CHIRRTLDialect, "CMemoryPort"> {
  let mnemonic = "cmemoryport";
  let summary = "a behavioral memory port";
  let description = [{
    Syntax:
    ```
    cmemoryport-type ::= `cmemoryport`
    ```

    The value of a `cmemoryport` type represents a port which has been
    declared on a `cmemory`.  This value is used to set the memory port access
    conditions.
  }];
}



#endif // CIRCT_DIALECT_FIRRTL_CHIRRTLTYPES_TD
