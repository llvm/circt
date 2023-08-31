//===- HWTypeInterfaces.td - HW Type Interfaces ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the type interfaces of the HW dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWTYPEINTERFACES_TD
#define CIRCT_DIALECT_HW_HWTYPEINTERFACES_TD

include "mlir/IR/OpBase.td"

def FieldIDTypeInterface : TypeInterface<"FieldIDTypeInterface"> {
  let cppNamespace = "circt::hw";
  let description = [{
    Common methods for types which can be indexed by a FieldID.
    FieldID is a depth-first numbering of the elements of a type.  For example:
    ```
    struct a  /* 0 */ {
      int b; /* 1 */
      struct c /* 2 */ {
        int d; /* 3 */
      }
    }

    int e; /* 0 */
    ```
  }];

  let methods = [
    InterfaceMethod<"Get the maximum field ID for this type",
      "uint64_t", "getMaxFieldID">,

    InterfaceMethod<[{
      Get the sub-type of a type for a field ID, and the subfield's ID. Strip
      off a single layer of this type and return the sub-type and a field ID
      targeting the same field, but rebased on the sub-type.

      The resultant type *may* not be a FieldIDTypeInterface if the resulting 
      fieldID is zero.  This means that leaf types may be ground without 
      implementing an interface.  An empty aggregate will also appear as a 
      zero.
    }], "std::pair<::mlir::Type, uint64_t>",
    "getSubTypeByFieldID", (ins "uint64_t":$fieldID)>,

    InterfaceMethod<[{
      Returns the effective field id when treating the index field as the
      root of the type.  Essentially maps a fieldID to a fieldID after a
      subfield op. Returns the new id and whether the id is in the given
      child.
    }], "std::pair<uint64_t, bool>", "projectToChildFieldID",
    (ins "uint64_t":$fieldID, "uint64_t":$index)>,

    InterfaceMethod<[{
      Returns the index (e.g. struct or vector element) for a given FieldID.
      This returns the containing index in the case that the fieldID points to a
      child field of a field.
    }], "uint64_t", "getIndexForFieldID", (ins "uint64_t":$fieldID)>,

    InterfaceMethod<[{
      Return the fieldID of a given index (e.g. struct or vector element).
      Field IDs start at 1, and are assigned
      to each field in a recursive depth-first walk of all
      elements. A field ID of 0 is used to reference the type itself.
    }], "uint64_t", "getFieldID", (ins "uint64_t":$fieldID)>,

    InterfaceMethod<[{
      Find the index of the element that contains the given fieldID.
      As well, rebase the fieldID to the element.
    }], "std::pair<uint64_t, uint64_t>", "getIndexAndSubfieldID",
    (ins "uint64_t":$fieldID)>,

  ];
}

#endif // CIRCT_DIALECT_HW_HWTYPEINTERFACES_TD
