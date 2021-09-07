//===- FIRRTLAttributes.h - FIRRTL dialect attributes -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the FIRRTL dialect custom attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLATTRIBUTES_H
#define CIRCT_DIALECT_FIRRTL_FIRRTLATTRIBUTES_H

#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "llvm/ADT/BitVector.h"

namespace circt {
namespace firrtl {

//===----------------------------------------------------------------------===//
// PortDirections
//===----------------------------------------------------------------------===//

/// This represents the direction of a single port.
enum class Direction { In, Out };

/// Prints the Direction to the stream as either "in" or "out".
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Direction &dir);

/// This represents the directions of ports in a module.  This is a simple type
/// safe wrapper around a BitVector.
class PortDirections {
public:
  /// Iterator for PortDirections returning Directions.
  struct Iterator
      : public llvm::iterator_facade_base<Iterator, std::forward_iterator_tag,
                                          Direction> {
    explicit Iterator(const PortDirections &directions, size_t index = 0)
        : directions(&directions), index(index) {}
    Iterator &operator=(const Iterator &) = default;
    using llvm::iterator_facade_base<Iterator, std::forward_iterator_tag,
                                     Direction>::operator++;
    Iterator &operator++() { return ++index, *this; }
    Direction operator*() const { return directions->at(index); }
    bool operator==(const Iterator &rhs) const { return index == rhs.index; }
    bool operator!=(const Iterator &rhs) const { return index != rhs.index; }

  private:
    const PortDirections *directions;
    size_t index;
  };

  using value_type = Direction;
  using iterator = Iterator;

  /// Construct an empty PortDirections.
  PortDirections() = default;

  /// Construct a PortDirections from a array of Directions.
  PortDirections(ArrayRef<Direction> directions) {
    reserve(directions.size());
    llvm::copy(directions, std::back_inserter(*this));
  }

  bool operator==(const PortDirections &rhs) const {
    return directions == rhs.directions;
  }

  /// Get the direction of a specifc port.
  Direction at(unsigned index) const { return Direction(directions[index]); }

  /// Get the direction of a specifc port.
  Direction operator[](unsigned index) const { return at(index); }

  /// Get the number of ports.
  size_t size() const { return directions.size(); }

  /// Reserve storage without initializing the elements.
  void reserve(unsigned amount) { directions.reserve(amount); }

  /// Append a port direction.
  void push_back(Direction direction) {
    directions.push_back(static_cast<bool>(direction));
  }

  /// Unpack the the port directions into a vector of Directions.
  SmallVector<Direction> unpack() {
    return SmallVector<Direction>(begin(), end());
  }

  Iterator begin() const { return Iterator(*this); }
  Iterator end() const { return Iterator(*this, size()); }

  llvm::hash_code hash_value() const {
    auto hash = llvm::hash_value(directions.size());
    if (directions.size())
      hash = hash_combine(hash, directions.getData());
    return hash;
  }

private:
  llvm::BitVector directions;
};

inline llvm::hash_code hash_value(const PortDirections &portDirections) {
  return portDirections.hash_value();
}

} // namespace firrtl
} // namespace circt

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h.inc"

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLATTRIBUTES_H
