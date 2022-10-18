//===- InteropOpInterfaces.td - Interop Interfaces ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes interop operation interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_INTEROP_INTEROPOPINTERFACES
#define CIRCT_DIALECT_INTEROP_INTEROPOPINTERFACES

include "mlir/IR/OpBase.td"

def ProceduralContainerInteropOpInterface :
    OpInterface<"ProceduralContainerInteropOpInterface"> {
  let cppNamespace = "circt::interop";
  let description = "Provides the lowerings for container-side interop.";

  let methods = [
    InterfaceMethod<"Lower an unrealized allocation interop operation.",
      "::mlir::LogicalResult", "lowerAllocation",
      (ins "::mlir::PatternRewriter&":$rewriter,
           "::circt::interop::ProceduralAllocOp":$op,
           "::circt::interop::ProceduralAllocOpAdaptor":$adaptor)>,

    InterfaceMethod<"Lower an unrealized initialization interop operation.",
      "::mlir::LogicalResult", "lowerInitialization",
      (ins "::mlir::PatternRewriter&":$rewriter,
           "::circt::interop::ProceduralInitOp":$op,
           "::circt::interop::ProceduralInitOpAdaptor":$adaptor)>,

    InterfaceMethod<"Lower an unrealized update interop operation.",
      "::mlir::LogicalResult", "lowerUpdate",
      (ins "::mlir::PatternRewriter&":$rewriter,
           "::circt::interop::ProceduralUpdateOp":$op,
           "::circt::interop::ProceduralUpdateOpAdaptor":$adaptor)>,

    InterfaceMethod<"Lower an unrealized deallocation interop operation.",
      "::mlir::LogicalResult", "lowerDeallocation",
      (ins "::mlir::PatternRewriter&":$rewriter,
           "::circt::interop::ProceduralDeallocOp":$op,
           "::circt::interop::ProceduralDeallocOpAdaptor":$adaptor)>,
  ];
}

#endif // CIRCT_DIALECT_INTEROP_INTEROPOPINTERFACES
