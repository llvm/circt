//===- CHIRRTLOps.td - CHIRRTL expression ops --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for CHIRRTL expressions and statements.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_CHIRRTLOPS_TD
#define CIRCT_DIALECT_FIRRTL_CHIRRTLOPS_TD

include "CHIRRTLDialect.td"
include "CHIRRTLTypes.td"
include "circt/Dialect/FIRRTL/FIRRTLAttributes.td"
include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.td"
include "circt/Dialect/FIRRTL/FIRRTLTypes.td"
include "circt/Dialect/FIRRTL/FIRRTLEnums.td"
include "circt/Dialect/HW/HWOpInterfaces.td"
include "circt/Dialect/HW/HWTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// CHIRRTL Operations
//===----------------------------------------------------------------------===//

// Base class for the operation in this dialect.
class CHIRRTLOp<string mnemonic, list<Trait> traits = []> :
    Op<CHIRRTLDialect, mnemonic, traits>;

def CombMemOp : CHIRRTLOp<"combmem", [HasCustomSSAName, DeclareOpInterfaceMethods<InnerSymbol,["getTargetResultIndex"]>, FNamableOp]> {
  let summary = "Define a new combinational memory";
  let description = [{
    Define a new behavioral combinational memory. Combinational memories have a
    write latency of 1 and a read latency of 0.
  }];
  let arguments = (ins StrAttr:$name, NameKindAttr:$nameKind,
                       AnnotationArrayAttr:$annotations,
                       OptionalAttr<InnerSymAttr>:$inner_sym,
                       OptionalAttr<MemoryInitAttr>:$init,
                       OptionalAttr<StrAttr>:$prefix);
  let results = (outs CMemoryType:$result);
  let assemblyFormat = [{(`sym` $inner_sym^)? `` custom<NameKind>($nameKind)
                         `` custom<CombMemOp>(attr-dict) `:` qualified(type($result))}];
  let builders = [
    OpBuilder<(ins "firrtl::FIRRTLBaseType":$elementType, "uint64_t":$numElements,
                   "mlir::StringRef":$name, "firrtl::NameKindEnum":$nameKind,
                   "ArrayAttr":$annotations,
                   CArg<"StringAttr", "StringAttr()">:$innerSym,
                   CArg<"firrtl::MemoryInitAttr", "firrtl::MemoryInitAttr{}">:$init)>
  ];
}

def SeqMemOp : CHIRRTLOp<"seqmem", [HasCustomSSAName, DeclareOpInterfaceMethods<InnerSymbol,["getTargetResultIndex"]>, FNamableOp]> {
  let summary = "Define a new sequential memory";
  let description = [{
    Define a new behavioral sequential memory.  Sequential memories have a
    write latency and a read latency of 1.
  }];
  let arguments = (ins RUWBehaviorAttr:$ruw, StrAttr:$name,
      NameKindAttr:$nameKind, AnnotationArrayAttr:$annotations,
      OptionalAttr<InnerSymAttr>:$inner_sym, OptionalAttr<MemoryInitAttr>:$init,
      OptionalAttr<StrAttr>:$prefix);
  let results = (outs CMemoryType:$result);
  let assemblyFormat = [{(`sym` $inner_sym^)? `` custom<NameKind>($nameKind) $ruw
                         custom<SeqMemOp>(attr-dict) `:` qualified(type($result))}];
  let builders = [OpBuilder<(ins "firrtl::FIRRTLBaseType":$elementType,
      "uint64_t":$numElements, "firrtl::RUWBehavior":$ruw,
      "mlir::StringRef":$name, "firrtl::NameKindEnum":$nameKind,
      "ArrayAttr":$annotations, CArg<"StringAttr", "StringAttr()">:$innerSym,
      CArg<"firrtl::MemoryInitAttr", "firrtl::MemoryInitAttr{}">:$init)>];
}

def MemoryPortOp : CHIRRTLOp<"memoryport", [InferTypeOpInterface,
      HasCustomSSAName]> {
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

  let results = (outs FIRRTLBaseType:$data, CMemoryPortType:$port);

  let assemblyFormat = [{
    $direction $memory `` custom<MemoryPortOp>(attr-dict) `:`
       functional-type(operands, results)
  }];

  let builders = [
    OpBuilder<(ins "::mlir::Type":$dataType, "::mlir::Value":$memory,
                   "firrtl::MemDirAttr":$direction, CArg<"StringRef", "{}">:$name,
                   CArg<"ArrayRef<Attribute>","{}">:$annotations)>
  ];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Get the assocated access op.
    MemoryPortAccessOp getAccess();

    /// Infer the return types of this operation.
    static LogicalResult inferReturnTypes(MLIRContext *context,
                                          std::optional<Location> loc,
                                          ValueRange operands,
                                          DictionaryAttr attrs,
                                          mlir::OpaqueProperties properties,
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
    $port `[` $index `]` `,` $clock attr-dict `:` qualified(type(operands))
  }];
}

def MemoryDebugPortOp : CHIRRTLOp<"debugport", [InferTypeOpInterface,
      HasCustomSSAName]> {
  let summary = "Declares a memory debug port on a memory";

  let summary = "Defines a debug memory port on CHIRRTL memory";
  let description = [{
    This operation defines a new debug memory port on a `combmem`CHISEL.
    `data` is the data returned from the memory port.
  }];

  let arguments = (ins CMemoryType:$memory,
                    StrAttr:$name, AnnotationArrayAttr:$annotations);

  let results = (outs RefType:$data);

  let assemblyFormat = [{
    $memory custom<MemoryDebugPortOp>(attr-dict) `:`
       functional-type(operands, results)
  }];

  let builders = [
    OpBuilder<(ins "::mlir::Type":$dataType, "::mlir::Value":$memory,
                   CArg<"StringRef", "{}">:$name,
                   CArg<"ArrayRef<Attribute>","{}">:$annotations)>
  ];

  let extraClassDeclaration = [{
    /// Infer the return types of this operation.
    static LogicalResult inferReturnTypes(MLIRContext *context,
                                          std::optional<Location> loc,
                                          ValueRange operands,
                                          DictionaryAttr attrs,
                                          mlir::OpaqueProperties properties,
                                          mlir::RegionRange regions,
                                          SmallVectorImpl<Type> &results);
  }];
}

#endif // CIRCT_DIALECT_FIRRTL_CHIRRTLOPS_TD
