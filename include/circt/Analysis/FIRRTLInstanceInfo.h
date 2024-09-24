//===- FIRRTLInstanceInfo.h - Instance info analysis ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the InstanceInfo analysis.  This is an analysis that
// depends on the InstanceGraph analysis, but provides additional information
// about FIRRTL operations.  This is useful if you find yourself needing to
// selectively iterate over parts of the design.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLINSTANCEINFO_H
#define CIRCT_DIALECT_FIRRTL_FIRRTLINSTANCEINFO_H

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/Pass/AnalysisManager.h"

namespace circt {
namespace firrtl {

class InstanceInfo {

public:
  explicit InstanceInfo(Operation *op, mlir::AnalysisManager &am);

  /// A lattice value to record the value of a property.
  class LatticeValue {

  public:
    enum Kind {
      // Indicates that we have no information about this property.  Properties
      // start in this state and then are changed to Constant or Mixed.
      Unknown = 0,

      // Indicates that the property has a true or false value.
      Constant,

      // Indicates that the property was found to be both true and false.
      Mixed
    };

  private:
    /// Whether or not the property holds.
    Kind kind = Kind::Unknown;

    /// The value of the property if `kind` is `Constant`.
    bool value = false;

  public:
    /// Return true if the kind is Unknown.
    bool isUnknown() const;

    /// Return true if the kind is Constant.
    bool isConstant() const;

    /// Return true if the kind is Mixed.
    bool isMixed() const;

    /// Return the value.  This should only be used if the kind is Constant.
    bool getConstant() const;

    /// Set this LatticeValue to a constant.
    void markConstant(bool constant);

    /// Merge attributes from another LatticeValue into this one.
    void mergeIn(LatticeValue that);

    /// Merge a constant value into this one.
    void mergeIn(bool value);
  };

  /// Information about a circuit
  struct CircuitAttributes {
    /// The design-under-test if one is defined.
    igraph::InstanceGraphNode *dutNode;

    /// The design-under-test if one is defined or the top module.
    igraph::InstanceGraphNode *effectiveDutNode;
  };

  /// Information about a module
  struct ModuleAttributes {
    /// Indicates if this module is instantiated under the design-under-test.
    InstanceInfo::LatticeValue underDut;

    /// Indicates if this module is instantiated under a layer.
    InstanceInfo::LatticeValue underLayer;
  };

  //===--------------------------------------------------------------------===//
  // Circuit Attribute Queries
  //===--------------------------------------------------------------------===//

  /// Return true if this circuit has a design-under-test.
  bool hasDut();

  /// Return the design-under-test if one is defined for the circuit, otherwise
  /// return null.
  igraph::InstanceGraphNode *getDut();

  /// Return the "effective" design-under-test.  This will be the
  /// design-under-test if one is defined.  Otherwise, this will be the root
  /// node of the instance graph.
  igraph::InstanceGraphNode *getEffectiveDut();

  //===--------------------------------------------------------------------===//
  // Module Attribute Queries
  //===--------------------------------------------------------------------===//

  /// Return true if this module is the design-under-test.
  bool isDut(igraph::ModuleOpInterface op);

  /// Return true if this module is the design-under-test and the circuit has a
  /// design-under-test.  If the circuit has no design-under-test, then return
  /// true if this is the top module.
  bool isEffectiveDut(igraph::ModuleOpInterface op);

  /// Return true if at least one instance of this module is under (or
  /// transitively under) the design-under-test.  This is true if the module is
  /// the design-under-test.
  bool atLeastOneInstanceUnderDut(igraph::ModuleOpInterface op);

  /// Return true if all instances of this module are under (or transitively
  /// under) the design-under-test.  This is true if the module is the
  /// design-under-test.
  bool allInstancesUnderDut(igraph::ModuleOpInterface op);

  /// Return true if at least one instance of this module is under (or
  /// transitively under) a layer.
  bool atLeastOneInstanceUnderLayer(igraph::ModuleOpInterface op);

  /// Return true if all instances of this module are under (or transitively
  /// under) layer blocks.
  bool allInstancesUnderLayer(igraph::ModuleOpInterface op);

private:
  /// Stores circuit-level attributes.
  CircuitAttributes circuitAttributes;

  /// Internal mapping of operations to module attributes.
  DenseMap<Operation *, ModuleAttributes> moduleAttributes;

  /// Return the module attributes associated with a module.
  const ModuleAttributes &getModuleAttributes(igraph::ModuleOpInterface op);
};

#ifndef NDEBUG
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const InstanceInfo::LatticeValue &value) {
  if (value.isUnknown())
    return os << "unknown";
  if (value.isConstant())
    return os << "constant<" << (value.getConstant() ? "true" : "false") << ">";
  return os << "mixed";
}
#endif

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLINSTANCEINFO_H
