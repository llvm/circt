//===- MSFTDialect.td - MSFT dialect base definitions ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MSFT dialect and base op class.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MSFT_MSFTDIALECT_TD
#define CIRCT_DIALECT_MSFT_MSFTDIALECT_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def MSFTDialect : Dialect {
  let name = "msft";
  let cppNamespace = "::circt::msft";

  let summary = "Microsoft internal support dialect";
  let description = [{
    Umbrella dialect for everything needed to support Microsoft development
    but not thoroughly discussed. Most (if not everything) in this dialect is
    a candidate for generalization and re-homing.
  }];

  let hasConstantMaterializer = 1;
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    /// Register all MSFT attributes.
    void registerAttributes();
  }];

  let dependentDialects = ["::circt::hw::HWDialect"];
}

// Base class for the operation in this dialect.
class MSFTOp<string mnemonic, list<Trait> traits = []> :
    Op<MSFTDialect, mnemonic, traits>;

#endif // CIRCT_DIALECT_MSFT_MSFTDIALECT_TD
