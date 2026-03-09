//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares utilities for analyzing RTG instruction operations,
// including operand type classification, register effect decorators, and
// helper functions for querying operation properties from TableGen definitions.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_CIRCT_TBLGEN_RTGTYPEANALYZER_H
#define CIRCT_TOOLS_CIRCT_TBLGEN_RTGTYPEANALYZER_H

#include "circt/Support/LLVM.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/TableGen/Record.h"

namespace circt {
namespace tblgen {
namespace rtg {

//===----------------------------------------------------------------------===//
// Decorators
//===----------------------------------------------------------------------===//

/// Helper class to represent a register effect decorator.
class RegisterEffect : public mlir::tblgen::Operator::VariableDecorator {
public:
  /// Get the name of the register effect.
  StringRef getEffectName() const { return def->getValueAsString("effect"); }

  static bool classof(const mlir::tblgen::Operator::VariableDecorator *var) {
    return var->getDef().isSubClassOf("RegisterEffect");
  }
};

/// Check if an operand is a source register.
bool isSourceRegister(const mlir::tblgen::Operator &op, unsigned argIndex);

/// Check if an operand is a destination register.
bool isDestinationRegister(const mlir::tblgen::Operator &op, unsigned argIndex);

//===----------------------------------------------------------------------===//
// Operand Type Classification
//===----------------------------------------------------------------------===//

/// Represents a register operand type.
struct Register {
  Register(StringRef name, StringRef typeName, size_t binaryEncodingWidth)
      : name(name), typeName(typeName),
        binaryEncodingWidth(binaryEncodingWidth) {}

  /// The name of a Python class representing this register type.
  StringRef name;
  /// The name of a Python class representing the type of this register type.
  StringRef typeName;
  /// The number of bits needed to encode this register type in binary format.
  size_t binaryEncodingWidth;

  bool operator==(const Register &other) const {
    return name == other.name && typeName == other.typeName &&
           binaryEncodingWidth == other.binaryEncodingWidth;
  }

  bool operator<(const Register &other) const {
    if (name != other.name)
      return name < other.name;
    if (typeName != other.typeName)
      return typeName < other.typeName;
    return binaryEncodingWidth < other.binaryEncodingWidth;
  }
};

/// Represents an immediate operand type with a specific bit width.
struct Immediate {
  explicit Immediate(size_t bitWidth) : bitWidth(bitWidth) {}

  size_t bitWidth;

  bool operator==(const Immediate &other) const {
    return bitWidth == other.bitWidth;
  }

  bool operator<(const Immediate &other) const {
    return bitWidth < other.bitWidth;
  }
};

/// Represents a memory operand type with a specific address width.
struct Memory {
  explicit Memory(size_t addressWidth) : addressWidth(addressWidth) {}

  size_t addressWidth;

  bool operator==(const Memory &other) const {
    return addressWidth == other.addressWidth;
  }

  bool operator<(const Memory &other) const {
    return addressWidth < other.addressWidth;
  }
};

/// Represents an immediate operand type with any bit width.
struct AnyImmediate {
  bool operator==(const AnyImmediate &) const { return true; }
  bool operator<(const AnyImmediate &) const { return false; }
};

/// Represents a memory operand type with any address width.
struct AnyMemory {
  bool operator==(const AnyMemory &) const { return true; }
  bool operator<(const AnyMemory &) const { return false; }
};

/// Represents a label operand type.
struct Label {
  bool operator==(const Label &) const { return true; }
  bool operator<(const Label &) const { return false; }
};

/// Classification of operand types in RTG instructions.
/// Note: don't use an empty struct as first option as it is used by the
/// variant dense map info for the empty and tombstone key and those empty
/// structs don't have meaningful keys.
using OperandType =
    std::variant<Register, Immediate, Memory, Label, AnyImmediate, AnyMemory>;

/// A set of operand types, supporting queries for specific type alternatives.
/// After insertion the entries have to be sorted and uniqued manually.
struct OperandTypeSet : public SmallVector<OperandType> {
  /// Check if the set contains any of the specified operand type alternatives.
  template <typename Tp, typename... Ts>
  bool contains() const {
    return llvm::any_of(*this, [](const OperandType &el) {
      return (std::holds_alternative<Tp>(el) || ... ||
              std::holds_alternative<Ts>(el));
    });
  }

  /// Append an operand type to the set and uniquify.
  void appendAndUnique(const OperandType &kind) {
    push_back(kind);
    llvm::sort(*this);
    erase(llvm::unique(*this), end());
  }
};

/// Classify a type Record into one or more type kinds. Multiple kinds meaning
/// that the type is a union of the kinds.
void classifyOperandType(const llvm::Record &typeRec, OperandTypeSet &kinds);

/// Find an operand by name in an operation's arguments.
const mlir::tblgen::NamedTypeConstraint *
findOperandByName(const mlir::tblgen::Operator &op, llvm::StringRef name);

/// Check if an operation has a specific interface trait.
bool hasOperatorInterface(const mlir::tblgen::Operator &op,
                          llvm::StringRef interfaceName);

/// Check if an operation has the InstructionOpInterface.
inline bool hasInstructionOpInterface(const mlir::tblgen::Operator &op) {
  return hasOperatorInterface(op, "circt::rtg::InstructionOpInterface");
}

/// Check if an operation has the RegisterAllocationOpInterface.
inline bool hasRegisterAllocationOpInterface(const mlir::tblgen::Operator &op) {
  return hasOperatorInterface(op, "circt::rtg::RegisterAllocationOpInterface");
}

} // namespace rtg
} // namespace tblgen
} // namespace circt

#endif // CIRCT_TOOLS_CIRCT_TBLGEN_RTGTYPEANALYZER_H
