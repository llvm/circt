//===- MSFTConstructs.td - MSFT high-level constructs ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "circt/Dialect/HW/HWTypes.td"
include "circt/Dialect/Seq/SeqTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"

// A handle to refer to hw::ArrayType of hw::ArrayType in ODS.
def MatrixType : DialectType<HWDialect,
    CPred<[{
      hw::type_isa<hw::ArrayType>($_self) &&
      hw::type_isa<hw::ArrayType>(
        hw::type_cast<hw::ArrayType>($_self).getElementType())
    }]>, "an array of arrays",
    "hw::TypeAliasOr<hw::ArrayType>">;


// SystolicArray starting with an MVP then we'll be making it more generic.
def SystolicArrayOp : MSFTOp<"systolic.array",
          [SingleBlockImplicitTerminator<"PEOutputOp"> ]> {
  let summary = "Model of a row/column broadcast systolic array";
  // TODO: flesh out description once we've proved this op out.
  let description = [{
    Note: the PE region is NOT a graph region. This was intentional since
    systolic arrays are entirely feed-forward.
  }];

  let arguments = (ins ArrayType:$rowInputs,
                       ArrayType:$colInputs);
  let results = (outs MatrixType:$peOutputs);
  let regions = (region SizedRegion<1>:$pe);

  let hasCustomAssemblyFormat = 1;
}

def PEOutputOp: MSFTOp<"pe.output", [Terminator]> {
  let summary = "Set the outputs from a PE block";
  let arguments = (ins AnyType:$output);
  let assemblyFormat = "$output attr-dict `:` type($output)";
}

// Linear, pipelineable datapath.
def LinearOp : MSFTOp<"hlc.linear", [
    SingleBlockImplicitTerminator<"OutputOp">
  ]> {
  let summary = "Model of a linear datapath which can be arbitrarily pipelined";
  let description = [{
    Defines a feed-forward datapath which can be scheduled into a pipeline.
    Due to the feed-forwardness, the inner region is NOT a graph region.
    Internally, only combinational operations (`comb`, `msft`, `hw`) are allowed.

    Example:
    ```mlir
    msft.module @foo(%in0 : i32, %in1 : i32, %in2 : i32, %clk : i1) -> (out: i32) -> {
      %0 = msft.hlc.linear(%a = %in0, %b = %in1, %c = %in2) clock %clk (i32, i32, i32) -> (i32) {
        %0 = comb.mul %a, %b : i32
        %1 = comb.add %0, %c : i32
        msft.output %1 : i32
      }
    }
    ```
  }];

  let arguments = (ins ClockType:$clock);
  let results = (outs Variadic<AnyType>:$outs);
  let regions = (region SizedRegion<1>:$datapath);

  let extraClassDeclaration = [{
    Block *getBodyBlock() { return &getDatapath().front(); }
  }];

  let hasVerifier = 1;
  let assemblyFormat = [{ `clock` $clock attr-dict `:` type($outs) $datapath }];
}

def OutputOp : MSFTOp<"output", [Terminator,
                                Pure, ReturnLike]> {
  let summary = "termination operation";

  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [
    OpBuilder<(ins)>
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` qualified(type($operands)))?";
}
