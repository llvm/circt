//===- CHIRRTL.td - CHIRRTL dialect definition -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the CHIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CHIRRTL_TD
#define CHIRRTL_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "circt/Dialect/FIRRTL/FIRRTLDialect.td"
include "circt/Dialect/FIRRTL/FIRRTLTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def CHIRRTLDialect : Dialect {
  let name = "chirrtl";

  let summary = "Types and operations for the chirrtl dialect";
  let description = [{
    This dialect defines the `chirrtl` dialect, which contains high-level
    memory defintions which can be lowered to FIRRTL.
  }];
  let dependentDialects = ["circt::firrtl::FIRRTLDialect"];
  let cppNamespace = "::circt::chirrtl";
}

//===----------------------------------------------------------------------===//
// CHIRRTL Types
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

  let parameters = (ins "firrtl::FIRRTLType":$elementType,
                        "unsigned":$numElements);

  let builders = [
    TypeBuilderWithInferredContext<(ins "firrtl::FIRRTLType":$elementType,
                                        "unsigned":$numElements), [{
      return $_get(elementType.getContext(), elementType, numElements);
    }]>
  ];

  let genVerifyDecl = 1;
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

//===----------------------------------------------------------------------===//
// CHIRRTL Operations
//===----------------------------------------------------------------------===//

// Base class for the operation in this dialect.
class CHIRRTLOp<string mnemonic, list<OpTrait> traits = []> :
    Op<CHIRRTLDialect, mnemonic, traits>;

def CombMemOp : CHIRRTLOp<"combmem"> {
  let summary = "Define a new combinational memory";
  let description = [{
    Define a new behavioral combinational memory. Combinational memories have a
    write latency of 1 and a read latency of 0.
  }];
  let arguments = (ins StrAttr:$name, AnnotationArrayAttr:$annotations);
  let results = (outs CMemoryType:$result);
  let assemblyFormat = "custom<CombMemOp>(attr-dict) `:` type($result)";
}

def SeqMemOp : CHIRRTLOp<"seqmem"> {
  let summary = "Define a new sequential memory";
  let description = [{
    Define a new behavioral sequential memory.  Sequential memories have a
    write latency and a read latency of 1.
  }];
  let arguments = (ins RUWAttr:$ruw, StrAttr:$name,
                       AnnotationArrayAttr:$annotations);
  let results = (outs CMemoryType:$result);
  let assemblyFormat = "$ruw custom<SeqMemOp>(attr-dict) `:` type($result)";
}

def MemoryPortOp : CHIRRTLOp<"memoryport", [InferTypeOpInterface,
         DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Declares a memory port on a memory";

  let summary = "Defines a memory port on CHIRRTL memory";
  let description = [{
    This operation defines a new memory port on a `seqmem` or `combmem`CHISEL.
    `data` is the data returned from the memory port.  

    The memory port requires an access point, which sets the enable condition
    of the port, the clock, and the address.  This is done by passing the the
    `port` argument to a `chirrtl.memoryport.access operation`.

  }];

  let arguments = (ins CMemoryType:$memory, MemDirAttr:$direction,
                    StrAttr:$name, AnnotationArrayAttr:$annotations);

  let results = (outs FIRRTLType:$data, CMemoryPortType:$port);

  let assemblyFormat = [{
    $direction $memory custom<MemoryPortOp>(attr-dict) `:`
       functional-type(operands, results)
  }];

  let builders = [
    OpBuilder<(ins "::mlir::Type":$dataType, "::mlir::Value":$memory,
                   "MemDirAttr":$direction, CArg<"StringRef", "{}">:$name,
                   CArg<"ArrayRef<Attribute>","{}">:$annotations)>
  ];

  let verifier = "return ::verifyMemoryPortOp(*this);";

  let extraClassDeclaration = [{
    /// Get the assocated access op.
    MemoryPortAccessOp getAccess();

    /// Infer the return types of this operation.
    static LogicalResult inferReturnTypes(MLIRContext *context,
                                          Optional<Location> loc,
                                          ValueRange operands,
                                          DictionaryAttr attrs,
                                          mlir::RegionRange regions,
                                          SmallVectorImpl<Type> &results);
  }];
}

def MemoryPortAccessOp : CHIRRTLOp<"memoryport.access"> {
  let summary = "Enables a memory port";

  let description = [{
    This operation is used to conditionally enable a memory port, and associate
    it with a `clock` and `index`.  The memory port will be actuve on the
    positive edge of the clock. The index is the address of the memory
    accessed. See the FIRRTL rational for more information about why this
    operation exists.
  }];

  let arguments = (ins CMemoryPortType:$port, IntType:$index,
                   ClockType:$clock);

  let assemblyFormat = [{
    $port `[` $index `]` `,` $clock attr-dict `:` type(operands)
  }];
}

#endif // CHIRRTL_TD