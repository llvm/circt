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

def ValidationTypeInterface : TypeInterface<"ValidationTypeInterface"> {
  let description = [{
    This interface should be implemented by types that are allowed to be used
    as the type of the 'ref' operands of the 'rtg.validate' operation.
  }];
  let cppNamespace = "::circt::rtg";

  let methods = [
    InterfaceMethod<[{
      Returns whether the specified type is a valid content type, e.g., if this
      type is an integer register it is a handle to a bit-vector stored in that
      register, thus an immediate of the width of the register is a valid
      content type.
    }], "bool", "isValidContentType", (ins "mlir::Type":$contentType)>,

    InterfaceMethod<[{
      Parses the given string as a value of the given content type and returns
      an Attribute representing the value for that type. A null attribute may be
      returned if the string cannot be parsed as a value of the given type.
    }], "mlir::TypedAttr", "parseContentValue",
      (ins "mlir::StringRef":$valueString, "mlir::Type":$contentType)>,

    // NOTE: this method should probably be factored out into a separate
    // interface that inherits from ValidationTypeInterface as it is specific to
    // ISA targets.
    InterfaceMethod<[{
      Returns the name of the intrinsic label that the desired simulator
      understands to print the content of this ref value annotated with the
      given ID. The 'value' attribute represents the concrete value, e.g., the
      exact register to print the content of instead of. Note: this might be
      generalized to an SSA value argument in the future if needed.
    }], "std::string", "getIntrinsicLabel",
      (ins "mlir::Attribute":$value, "mlir::StringRef":$id)>,
  ];
}

def ImplicitConstraintOpInterface : OpInterface<"ImplicitConstraintOpInterface"> {
  let description = [{
    This interface should be implemented by operations that implicitly enforce
    a constraint. This is used to automatically materialize constraints that
    are not explicitly specified by the user.
  }];
  let cppNamespace = "::circt::rtg";

  let methods = [
    InterfaceMethod<[{
      Returns whether the constraint enforced by this operation has already
      been materialized.
    }], "bool", "isConstraintMaterialized">,
    InterfaceMethod<[{
      Materialize the constraint enforced by this operation. The operation
      itself may be modified in place (return itself) or a new operation may
      be returned to be replaced with this operation. Returning `nullptr` means
      the operation should be erased without being replaced (only valid for
      operations without results).
    }], "mlir::Operation *", "materializeConstraint", (ins
      "mlir::OpBuilder &":$builder)>,
  ];
}

#endif // CIRCT_DIALECT_RTG_IR_RTGINTERFACES_TD
