//===- FieldRef.h - FIRRTL Field References ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines FieldRefs and helpers for them.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIELDREF_H
#define CIRCT_DIALECT_FIRRTL_FIELDREF_H

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SmallVector.h"

namespace circt {
namespace firrtl {

/// This class represents a reference to a specific field or element of an
/// aggregate value.
///
/// This is capable of referring to any field in a bundle, or a element in a
/// vector. There is no restriction that the reference must be to a leaf field,
/// it may be pointing to a field that is itself a bundle. The FieldRef is
/// constructable from the OpResult of any (static) indexing operation. All
/// indexing operations referring to the same field in an aggregate result in
/// the same FieldRef.
///
/// This can be used as the key in a hashtable to store field specific
/// information. It is not possible to go from a FieldRef back to the
/// Subfield op it was created from.
class FieldRef {
public:
  /// Get a null FieldRef.
  FieldRef(){};

  /// Get a FieldRef location for the specified value.  If the value is the
  /// result of indexing in to an aggregate type, this will resolve the location
  /// to be a member of that aggregate.
  FieldRef(Value value);

  /// Get the Value which created this location.
  Value getValue() const { return value; }

  /// Get the operation which defines this field.
  Operation *getDefiningOp() const;

  /// Get a string representation of this field.
  std::string getFieldName() const;

  /// Get the field ID of this FieldRef, which is a unique identifier mapped to
  /// a specific field in a bundle.
  unsigned getFieldID() const { return id; }

  /// Set the field ID number of this FieldRef.
  void setFieldID(unsigned fieldID) { id = fieldID; }

  bool operator==(const FieldRef &other) const {
    return value == other.value && id == other.id;
  }

  /// Print the FieldRef.
  void print(raw_ostream &os) const;
  void dump() const;

private:
  /// A pointer to the value which created this.
  Value value;

  /// A unique field ID.
  unsigned id = 0;
};

inline raw_ostream &operator<<(raw_ostream &os, FieldRef fieldRef) {
  fieldRef.print(os);
  return os;
}

inline ::llvm::hash_code hash_value(const FieldRef &fieldRef) {
  return llvm::hash_combine(fieldRef.getValue(), fieldRef.getFieldID());
}

} // namespace firrtl
} // namespace circt

namespace llvm {
template <>
struct DenseMapInfo<circt::firrtl::FieldRef> {
  static inline circt::firrtl::FieldRef getEmptyKey() {
    return circt::firrtl::FieldRef();
  }
  static inline circt::firrtl::FieldRef getTombstoneKey() {
    return circt::firrtl::FieldRef();
  }
  static unsigned getHashValue(const circt::firrtl::FieldRef &val) {
    return circt::firrtl::hash_value(val);
  }
  static bool isEqual(const circt::firrtl::FieldRef &lhs,
                      const circt::firrtl::FieldRef &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

#endif // CIRCT_DIALECT_FIRRTL_FIELDREF_H
