//===- RTGInterfaces.td - Interfaces used in RTG -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTG_IR_RTGINTERFACES_TD
#define CIRCT_DIALECT_RTG_IR_RTGINTERFACES_TD

include "mlir/IR/Interfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

/// Context resources can only be defined inside the `rtg.target` operation.
def ContextResourceDefining : TraitList<[
  HasParent<"::circt::rtg::TargetOp">,
]>;

def ContextResourceTypeInterface : TypeInterface<
    "ContextResourceTypeInterface"> {
  let description = [{
    This interface should be implemented by types that represent context
    resources. The `on_context` operation takes an operand of a type
    implementing this interface or a set of such a type.
    Any operation that creates a value of a type implementing this interface
    must implement the `ContextResourceOpInterface` (does not apply to
    operations that just forward a value of such type).
  }];
  let cppNamespace = "::circt::rtg";
}

def ContextResourceAttrInterface : AttrInterface<
    "ContextResourceAttrInterface", [TypedAttrInterface]> {
  let description = [{
    This interface should be implemented by attributes that represent context
    resources.
    Any attribute implementing this interface must be of a type implementing
    the `ContextResourceTypeInterface`.

    TODO: properly verify this; unfortunately, we don't have a 'verify' field
    here like the 'OpInterface' has.
  }];
  let cppNamespace = "::circt::rtg";
}

#endif // CIRCT_DIALECT_RTG_IR_RTGINTERFACES_TD
