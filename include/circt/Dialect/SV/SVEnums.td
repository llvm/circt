//===- SVEnums.td - Enums for SV dialect -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines SV dialect specific enums.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SV_SVENUM_TD
#define CIRCT_DIALECT_SV_SVENUM_TD

include "mlir/IR/EnumAttr.td"

let cppNamespace = "circt::sv" in {

def NoReset: I32EnumAttrCase<"NoReset", 0, "noreset">;
def SyncReset: I32EnumAttrCase<"SyncReset", 1, "syncreset">;
def AsyncReset: I32EnumAttrCase<"AsyncReset", 2, "asyncreset">;

def ResetTypeAttr : I32EnumAttr<"ResetType", "reset type",
                                   [NoReset, SyncReset, AsyncReset]>;

}

def CaseStmt: I32EnumAttrCase<"CaseStmt", 0, "case">;
def CaseXStmt: I32EnumAttrCase<"CaseXStmt", 1, "casex">;
def CaseZStmt: I32EnumAttrCase<"CaseZStmt", 2, "casez">;

def CaseStmtTypeAttr : I32EnumAttr<"CaseStmtType", "case type",
                                   [CaseStmt, CaseXStmt, CaseZStmt]>;

def ValidationQualifierPlain: I32EnumAttrCase<"ValidationQualifierPlain", 0, "plain">;
def ValidationQualifierUnique: I32EnumAttrCase<"ValidationQualifierUnique", 1, "unique">;
def ValidationQualifierUnique0: I32EnumAttrCase<"ValidationQualifierUnique0", 2, "unique0">;
def ValidationQualifierPriority: I32EnumAttrCase<"ValidationQualifierPriority", 3, "priority">;

def ValidationQualifierTypeEnum: I32EnumAttr<"ValidationQualifierTypeEnum", "validation qualifier type",
              [ValidationQualifierPlain, ValidationQualifierUnique,
               ValidationQualifierUnique0, ValidationQualifierPriority]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::circt::sv";
}

def MemBaseBin: I32EnumAttrCase<"MemBaseBin", 0>;
def MemBaseHex: I32EnumAttrCase<"MemBaseHex", 1>;
def MemBaseTypeAttr :
    I32EnumAttr<"MemBaseTypeAttr", "the numeric base of a memory file",
                [MemBaseBin, MemBaseHex]>;

def ModportDirectionInput : I32EnumAttrCase<"input", 0>;
def ModportDirectionOutput : I32EnumAttrCase<"output", 1>;
def ModportDirectionInOut : I32EnumAttrCase<"inout", 2>;

def ModportDirection : I32EnumAttr<"ModportDirection",
  "Defines direction in a modport",
  [ModportDirectionInput, ModportDirectionOutput, ModportDirectionInOut]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::circt::sv";
}

/// Immediate assertions, like `assert`.
def ImmediateAssert: I32EnumAttrCase<"Immediate", 0, "immediate">;
/// Observed deferred assertions, like `assert #0`.
def ObservedAssert: I32EnumAttrCase<"Observed", 1, "observed">;
/// Final deferred assertions, like `assert final`.
def FinalAssert: I32EnumAttrCase<"Final", 2, "final">;

/// A mode specifying how immediate/deferred assertions operate.
def DeferAssertAttr : I32EnumAttr<
  "DeferAssert", "assertion deferring mode",
  [ImmediateAssert, ObservedAssert, FinalAssert]>
{
  let cppNamespace = "circt::sv";
}

#endif // CIRCT_DIALECT_SV_SVENUM_TD
