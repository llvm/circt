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

using namespace detail;

class InstanceInfo {

public:
  explicit InstanceInfo(Operation *op, mlir::AnalysisManager &am);

  /// A lattice value recording whether or not a property is true for a set of
  /// operations.
  struct LatticeValue {
    enum Kind {
      // Indicates that we have no information about this instance.  If the
      // circuit has been completely analyzed, then this is equivalent to the
      // property holds for zero instances.
      Unknown = 0,
      // Indicates that the property has a true or false value for all instances
      // observed.
      Constant,
      // Indicates that the property holds for some, but not all instances.
      Mixed
    };

    /// Whether or not the property holds.
    Kind kind = Unknown;

    /// The value of the property if `kind` is `Constant`.
    bool constant = false;

    // Combine this lattice value with another lattice value.
    void merge(LatticeValue that);

    // Merge evidence that a property is true.
    void merge(bool property);
  };

  struct ModuleAttributes {
    /// Indicates that this module is the design-under-test.  This is indicated
    /// with a `sifive.enterprise.firrtl.MarkDUTAnnotation`.
    bool isDut;

    /// Indicates if this module is instantiated under the design-under-test.
    InstanceInfo::LatticeValue underDut = {LatticeValue::Kind::Unknown};

    /// Indicates if this module is instantiated under a layer.
    InstanceInfo::LatticeValue underLayer = {LatticeValue::Kind::Unknown};
  };

  /// Return true if this module is the design-under-test.
  bool isDut(FModuleOp op);

  /// Return true if this module is instantiated under the design-under-test.
  bool isUnderDut(FModuleOp op);

  /// Return true if all instantiations of this module are under the
  /// design-under-test.
  bool isFullyUnderDut(FModuleOp op);

  /// Return true if this module is instantiated under a layer block.
  bool isUnderLayer(FModuleOp op);

  /// Return true if all instantiations of this module are under layer blocks.
  bool isFullyUnderLayer(FModuleOp op);

private:
  /// Internal mapping of operations to module attributes.
  DenseMap<Operation *, ModuleAttributes> moduleAttributes;

  /// Return the module attributes associated with a module.
  const ModuleAttributes &getModuleAttributes(FModuleOp op);
};

#ifndef NDEBUG
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const InstanceInfo::LatticeValue &value) {
  switch (value.kind) {
  case InstanceInfo::LatticeValue::Kind::Unknown:
    return os << "unknown";
  case InstanceInfo::LatticeValue::Kind::Constant:
    return os << "constant<" << (value.constant ? "true" : "false") << ">";
  case InstanceInfo::LatticeValue::Kind::Mixed:
    return os << "mixed";
  }
  return os;
}
#endif

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLINSTANCEINFO_H
