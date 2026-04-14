//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_PROCDURALREGIONTRAIT_TD
#define CIRCT_SUPPORT_PROCDURALREGIONTRAIT_TD

include "mlir/IR/OpBase.td"

// Note:  The procedural and non-procedural traits are intended to be mutually
//        exclusive. If an operation or region is not strictly procedural or
//        non-procedural, it should carry neither trait.

def ProceduralRegion : NativeOpTrait<"ProceduralRegion"> {
  let cppNamespace = "::circt";
}

def NonProceduralRegion : NativeOpTrait<"NonProceduralRegion"> {
  let cppNamespace = "::circt";
}

def ProceduralOp : NativeOpTrait<"ProceduralOp"> {
  let cppNamespace = "::circt";
}

def NonProceduralOp : NativeOpTrait<"NonProceduralOp"> {
  let cppNamespace = "::circt";
}

#endif // CIRCT_SUPPORT_PROCDURALREGIONTRAIT_TD
