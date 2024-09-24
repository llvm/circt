//===- RTGOps.td - RTG operations --------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the RTG MLIR operations.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/EnumAttr.td"

// Base class for the operation in this dialect.
class RTGTestOp<string mnemonic, list<Trait> traits = []> :
    Op<RTGTestDialect, mnemonic, traits>;

