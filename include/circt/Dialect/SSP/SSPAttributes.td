//===- SSPAttributes.td - SSP attribute definitions --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the SSP (static scheduling problem) dialect attributes.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Dialect attributes
//===----------------------------------------------------------------------===//

def DummyAttr : AttrDef<SSPDialect, "Dummy"> {
  let summary = "Trigger generation of dialect attribute parser & printer.";
  let mnemonic = "dummy";
}
