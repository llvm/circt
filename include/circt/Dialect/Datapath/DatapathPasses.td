//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DATAPATH_PASSES_TD
#define CIRCT_DIALECT_DATAPATH_PASSES_TD

include "mlir/Pass/PassBase.td"

def DatapathReduceDelay : Pass<"datapath-reduce-delay"> {
  let summary = "Reduce datapath delay at a cost of potentially greater area.";
  let description = [{
    This pass performs datapath transformations to reduce the delay of any 
    matching paths. The transformations include logic replication/duplication to 
    operate on carry-save values, potentially increasing area usage.
  }];
  let constructor = "circt::datapath::createDatapathReduceDelayPass()";
  let dependentDialects = ["datapath::DatapathDialect"];
}

#endif // CIRCT_DIALECT_DATAPATH_PASSES_TD
