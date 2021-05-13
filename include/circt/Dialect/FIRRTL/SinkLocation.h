//===- SinkLocation.h - FIRRTL Sink Locations -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines SinkLocation and helpers for them.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_SINKLOCATION_H
#define CIRCT_DIALECT_FIRRTL_SINKLOCATION_H

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SmallVector.h"

namespace circt {
namespace firrtl {

/// This class represents a reference to a specific field or element of a sink
/// value. This is used to distinguish between the individual elements of Values
/// which are aggregate sink types.  This is capable of referring to a field in
/// a bundle, or a element in a vector. All indexing operations referring to the
/// same field in an aggregate have the same SinkLocation.  This can be used as
/// the key in a hashtable to store field specific information. It is not
/// possible to go from a SinkLocation back to the Subfield op it was created
/// from.
class SinkLocation {
public:
  /// Get a null SinkLocation.
  SinkLocation(){};

  /// Get a sink location for the specified value.  If the value is the result
  /// of indexing in to an aggregate type, this will resolve the location to be
  /// a member of that aggregate.
  SinkLocation(Value value);

  /// Get the Value which created this sink location.
  Value getSink() const { return sink; }

  /// Get the operation which defines this sink.
  Operation *getDefiningOp() const;

  /// Get a string representation of this sink location.
  std::string getFieldName() const;

  /// Get the path of indices to this specific field.
  std::vector<unsigned> &getPath() { return path; }
  const std::vector<unsigned> &getPath() const { return path; }

  bool operator==(const SinkLocation &other) const {
    if (sink != other.sink)
      return false;
    return std::equal(path.begin(), path.end(), other.path.begin(),
                      other.path.end());
  }

  /// Print the location.
  void print(raw_ostream &os) const;
  void dump() const;

private:
  /// A pointer to the value which created this sink location.
  Value sink;

  /// The path through bundle and array accesses that this sink represents.
  std::vector<unsigned> path;
};

inline raw_ostream &operator<<(raw_ostream &os, SinkLocation sink) {
  sink.print(os);
  return os;
}

inline ::llvm::hash_code hash_value(const SinkLocation &sink) {
  auto &path = sink.getPath();
  return llvm::hash_combine(sink.getSink(),
                            llvm::hash_combine_range(path.begin(), path.end()));
}

} // namespace firrtl
} // namespace circt

namespace llvm {
template <>
struct DenseMapInfo<circt::firrtl::SinkLocation> {
  static inline circt::firrtl::SinkLocation getEmptyKey() {
    return circt::firrtl::SinkLocation();
  }
  static inline circt::firrtl::SinkLocation getTombstoneKey() {
    return circt::firrtl::SinkLocation();
  }
  static unsigned getHashValue(const circt::firrtl::SinkLocation &val) {
    return circt::firrtl::hash_value(val);
  }
  static bool isEqual(const circt::firrtl::SinkLocation &lhs,
                      const circt::firrtl::SinkLocation &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

#endif // CIRCT_DIALECT_FIRRTL_SINKLOCATION_H
