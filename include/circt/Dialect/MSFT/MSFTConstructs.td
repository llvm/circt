//===- MSFTConstructs.td - MSFT high-level constructs ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "circt/Dialect/HW/HWTypes.td"

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
