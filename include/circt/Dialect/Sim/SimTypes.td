//===- SimTypes.td - Sim dialect types ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_SIMTYPES_TD
#define CIRCT_DIALECT_SIM_SIMTYPES_TD

include "circt/Dialect/Sim/SimDialect.td"
include "mlir/IR/AttrTypeBase.td"

class SimTypeDef<string name> : TypeDef<SimDialect, name> { }

def FormatStringType : SimTypeDef<"FormatString"> {
  let mnemonic = "fstring";

  let summary = "Format string type";
  let description = [{
    A format string type represents either a single formatting fragment or the
    concatenation of an arbitrary but finite number of fragments.
    A formatting fragment is either a static string literal or the association
    of a dynamic hardware value with a format specifier.
  }];
}


def EdgeTriggerType : SimTypeDef<"EdgeTrigger"> {
  let summary = "Trigger derived from an edge event.";
  let parameters = (ins "::circt::hw::EventControl":$edgeEvent);
  let mnemonic = "trigger.edge";
  let assemblyFormat = "`<` $edgeEvent `>`";
}

def InitTriggerType : SimTypeDef<"InitTrigger"> {
  let summary = "Trigger derived from the simulation start event.";
  let mnemonic = "trigger.init";
}

def AnyTriggerType : AnyTypeOf<[EdgeTriggerType, InitTriggerType]>;

#endif // CIRCT_DIALECT_SIM_SIMTYPES_TD
