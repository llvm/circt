//===- SVGenerate.td - SV generate block ops ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the ops for SystemVerilog generate blocks.
//
//===----------------------------------------------------------------------===//

def GenerateCaseOp : SVOp<"generate.case",
                          [SingleBlock, NoTerminator, NoRegionArguments]> {

  let summary = "A 'case' statement inside of a generate block";
  let description = "See SystemVerilog 2017 27.5.";

  let regions = (region VariadicRegion<SizedRegion<1>>:$caseRegions);
  let arguments = (ins StrAttr:$block_name, AnyAttr:$cond,
                       ArrayAttr:$casePatterns, ArrayAttr:$caseNames);

  let assemblyFormat = [{
    $block_name `:` `(` $cond `)` attr-dict `[`
      custom<CaseRegions>($casePatterns, $caseNames, $caseRegions)
    `]`
  }];

  let hasVerifier = true;
}
