//===- StaticLogicAttributes.td - StaticLogic attributes ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STATICLOGICATTRIBUTES_TD
#define STATICLOGICATTRIBUTES_TD

include "mlir/IR/OpBase.td"

include "circt/Dialect/StaticLogic/StaticLogicDialect.td"

def InitiationIntervalAttr :
    AttrDef<StaticLogic_Dialect, "InitiationInterval"> {
  let summary = "Initiation interval attribute";
  let description = [{
    This attribute represents the initiation interval (II) of a pipeline. For
    now, it is a simple integer representing the II computed by the Scheduling
    infrastructure.
  }];
  let mnemonic = "II";
  let parameters = (ins "mlir::IntegerAttr":$value);
}

#endif // STATICLOGICATTRIBUTES_TD
