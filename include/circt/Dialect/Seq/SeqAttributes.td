//===- SeqAttributes.td - Attributes for Seq dialect -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SEQ_SEQATTRIBUTES_TD
#define CIRCT_DIALECT_SEQ_SEQATTRIBUTES_TD

include "circt/Dialect/Seq/SeqDialect.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

let cppNamespace = "circt::seq" in {

// Read-under-write behavior
def RUW_Undefined : I32EnumAttrCase<"Undefined", 0, "undefined">;
def RUW_Old : I32EnumAttrCase<"Old", 1, "old">;
def RUW_New : I32EnumAttrCase<"New", 2, "new">;
def RUWAttr : I32EnumAttr<"RUW", "Read-Under-Write Behavior",
                          [RUW_Undefined, RUW_Old, RUW_New]>;

// Write-under-write behavior
def WUW_Undefined : I32EnumAttrCase<"Undefined", 0, "undefined">;
def WUW_PortOrder : I32EnumAttrCase<"PortOrder", 1, "port_order">;
def WUWAttr : I32EnumAttr<"WUW", "Write-Under-Write Behavior",
                          [WUW_Undefined, WUW_PortOrder]>;

}

/// An attribute holding information about memory initialization.
def FirMemInitAttr : AttrDef<SeqDialect, "FirMemInit"> {
  let mnemonic = "firmem.init";
  let summary = "Memory initialization information";
  let description = [{
    This attribute captures what the initial contents of a memory should be.
    At the moment this is modeled primarily with simulation in mind, where the
    memory contents are pre-loaded from a file at simulation startup.

    The `filename` specifies a file on disk that contains the initial contents
    for this memory. If `isBinary` is set, the file is interpreted as a binary
    file, otherwise it is treated as hexadecimal. This is modeled after the
    `$readmemh` and `$readmemb` SystemVerilog functions. If `isInline` is set,
    the initialization is emitted directly in the memory model; otherwise it is
    split out into a separate module that can be bound in.
  }];
  let parameters = (ins
    "mlir::StringAttr":$filename,
    "bool":$isBinary,
    "bool":$isInline
  );
  let assemblyFormat = "`<` $filename `,` $isBinary `,` $isInline `>`";
}

/// An attribute holding information about rom initialization.
def FirRomInitAttr : AttrDef<SeqDialect, "FirRomInit"> {
  let mnemonic = "firrom.init";
  let summary = "ROM initialization information";
  let description = [{
    This attribute captures what the initial content of the ROM should be.
    Currently, this property is used in simulation, FPGA XPM ROM initialization,
    and memory compiler to generate physical ROM. If this is the case in a
    simulation where the memory contents are preloaded from a file when the
    simulation starts. If it is used in the case of XPM ROM initialization, the
    path string will be passed in the parameter of the black box. If it is used
    in the memory compiler to generate a physical ROM, the parameters will be
    passed to the configuration file.

    The `filename` specifies a file on disk that contains the initial contents
    for this ROM. If `isBinary` is set, the file is interpreted as a binary
    file, otherwise it is treated as hexadecimal. This is modeled after the
    `$readmemh` and `$readmemb` SystemVerilog functions. For some fpga may
    not support isBinary is set, additional conversion is required. For memory
    compilers, depending on the vendor, some only support `isBinary` to be
    true, and some only support `isBinary` to be false, and additional
    conversion may be necessary. If `isInline` is set, the initialization is
    emitted directly in the ROM model; otherwise it is split out into a separate
    module that can be bound in.
  }];
  let parameters = (ins
    "mlir::StringAttr":$filename,
    "bool":$isBinary,
    "bool":$isInline
  );
  let assemblyFormat = "`<` $filename `,` $isBinary `,` $isInline `>`";
}

#endif // CIRCT_DIALECT_SEQ_SEQATTRIBUTES_TD
