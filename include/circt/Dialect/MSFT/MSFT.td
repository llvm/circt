//===- MSFT.td - MSFT dialect definition -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the MSFT dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MSFT_TD
#define MSFT_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Pass/PassBase.td"

def MSFTDialect : Dialect {
  let name = "msft";

  let summary = "Microsoft internal support dialect";
  let description = [{
    Umbrella dialect for everything needed to support Microsoft development
    but not thoroughly discussed. Most (if not everything) in this dialect is
    a candidate for generalization and re-homing.
  }];

  let hasConstantMaterializer = 1;
  let cppNamespace = "::circt::msft";
  let extraClassDeclaration = [{
    /// Register all MSFT attributes.
    void registerAttributes();

    void registerGenerator(StringRef opName, StringRef generatorName,
                           GeneratorCallback cb, Attribute parameters);

    /// Generator details don't need to be exposed.
    llvm::ManagedStatic<detail::Generators> generators;
  }];
}

include "circt/Dialect/MSFT/MSFTAttributes.td"
include "circt/Dialect/MSFT/MSFTOps.td"
include "circt/Dialect/MSFT/MSFTPasses.td"

#endif // MSFT_TD
