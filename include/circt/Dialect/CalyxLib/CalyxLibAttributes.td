//===- CalyxLibAttributes.td - CalyxLib attributes ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CalyxLib dialect attributes.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Dialect attributes
//===----------------------------------------------------------------------===//

include "mlir/IR/AttrTypeBase.td"

class OperatorProperty<Dialect dialect,
                   string name, string type,
                   list<Trait> traits = []>
  : AttrDef<dialect, name, traits> {
  
  string propertyName = name;
  string propertyType = type;

  // Unwrap the data in this attribute's `$value` parameter in order to pass it 
  // to the corresponding setter on the problem class.
  // code unwrapValue = [{ getValue() }];
  // Wrap the `value` returned by the getter on the problem class in order to
  // store it in this attribute's `$value` parameter.
  // code wrapValue = [{ value }];

  let summary = "Models the `" # propertyName # "` property for a CalyxLib operator.";

  let parameters = (ins propertyType:$value);
  let mnemonic = propertyName;
  let assemblyFormat = [{ `<` $value `>` }];
}

def IncomingDelayProp : OperatorProperty<CalyxLib_Dialect,
  "IncomingDelay", "::mlir::FloatAttr"> {
  let mnemonic = "incDelay";
}

def OutgoingDelayProp : OperatorProperty<CalyxLib_Dialect,
  "OutgoingDelay", "::mlir::FloatAttr"> {
  let mnemonic = "outDelay";
}

def LimitProp : OperatorProperty<CalyxLib_Dialect,
  "Limit", "unsigned"> {
  let mnemonic = "limit";
}

def LatencyProp : OperatorProperty<CalyxLib_Dialect,
  "Latency", "unsigned"> {
  let mnemonic = "latency";
}
