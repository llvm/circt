//===- BLIFOps.td - BLIF ops ============-------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This defines the BLIF ops.  This defines module-like operations, connections,
// and logic.
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_BLIF_BLIFOPS_TD
#define CIRCT_DIALECT_BLIF_BLIFOPS_TD

include "circt/Dialect/BLIF/BLIFDialect.td"
include "circt/Dialect/HW/HWTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/RegionKindInterface.td"

def ModelOp : BLIFOp<"model", [IsolatedFromAbove, RegionKindInterface, 
                               SingleBlockImplicitTerminator<"OutputOp">]> {
  let summary = "A model, which is a module";
  let description = [{
    The basic container.  Is like a module.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<ModuleType>:$module_type,
                       APIntAttr:$clocks);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let builders = [
    OpBuilder<(ins "StringRef":$name, 
                   "ArrayRef<StringRef>":$inputs,
                   "ArrayRef<StringRef>":$outputs,
                   "ArrayRef<StringRef>":$clocks)>
  ];


  let extraClassDeclaration = [{
    static mlir::RegionKind getRegionKind(unsigned index) {
      return mlir::RegionKind::Graph;
    }
  }];
}

def OutputOp : BLIFOp<"output", [Terminator]> {
  let arguments = (ins Variadic<I1>:$inputs);
}

def I8Property : IntProperty<"int8_t">;

def LogicGateOp: BLIFOp<"logic_gate", []> {
  let summary = "Combinatorial logic";
  let description = [{
    A logic gate represents a logic function in sum-of-products
    form.  Each entry in $func is a vector applied to the inputs
    where 0 means invert the input, 1 means use the input, and 2 
    means don't use the input.
  }];
  
  let arguments = (ins ArrayProperty<IntArrayProperty<I8Property>>:$func,
                       Variadic<I1>:$inputs);
  let results = (outs I1:$result);
}

def LatchMode : I32EnumAttr<
  "LatchModeEnum", "Latch Mode", [
    I32EnumAttrCase<"Unspecified", 0, "un">,
    I32EnumAttrCase<"FallingEdge", 1, "fe">,
    I32EnumAttrCase<"RisingEdge", 2, "re">,
    I32EnumAttrCase<"ActiveHigh", 3, "ah">,
    I32EnumAttrCase<"ActiveLow", 4, "al">,
    I32EnumAttrCase<"Asynchronous", 5, "as">
  ]> {
    let cppNamespace = "circt::blif";
  }

def LatchGateOp: BLIFOp<"latch_gate", []> {

  let arguments = (ins I1:$input, LatchMode:$mode, Optional<I1>:$clock, I32Property:$initVal);
  let results = (outs I1:$output);
}

#endif // CIRCT_DIALECT_BLIF_BLIFOPS_TD
