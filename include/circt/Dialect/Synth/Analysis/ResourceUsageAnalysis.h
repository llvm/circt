//===- ResourceUsageAnalysis.h - Resource Usage Analysis --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the resource usage analysis for the Synth dialect.
// The analysis computes resource utilization including and-inverter gates,
// DFF bits, and LUTs across module hierarchies.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_ANALYSIS_RESOURCEUSAGEANALYSIS_H
#define CIRCT_DIALECT_SYNTH_ANALYSIS_RESOURCEUSAGEANALYSIS_H

#include "circt/Support/InstanceGraph.h"
#include "circt/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include <memory>

namespace mlir {
class AnalysisManager;
} // namespace mlir

namespace circt {
namespace synth {

/// Analysis that computes resource usage for Synth dialect operations.
/// This analysis walks module hierarchies and counts resources such as:
/// - And-inverter graph (AIG)
/// - Majority-inverter graph (MIG)
/// - Truth tables (LUTs)
/// - Sequential elements (DFFs)
class ResourceUsageAnalysis {
public:
  ResourceUsageAnalysis(mlir::Operation *moduleOp, mlir::AnalysisManager &am);

  /// Resource usage counts for a set of operations.
  struct ResourceUsage {
    ResourceUsage(llvm::StringMap<uint64_t> counts)
        : counts(std::move(counts)) {}
    ResourceUsage() = default;

    /// Accumulate resource counts from another ResourceUsage.
    ResourceUsage &operator+=(const ResourceUsage &other) {
      for (const auto &count : other.counts)
        counts[count.getKey()] += count.second;
      return *this;
    }

    const auto &getCounts() const { return counts; }

  private:
    llvm::StringMap<uint64_t> counts;
  };

  /// Resource usage for a single module, including local and total counts.
  struct ModuleResourceUsage {
    ModuleResourceUsage(StringAttr moduleName, ResourceUsage local,
                        ResourceUsage total)
        : moduleName(moduleName), local(std::move(local)),
          total(std::move(total)) {}

    StringAttr moduleName;
    ResourceUsage local; ///< Resources used directly in this module.
    ResourceUsage total; ///< Resources including all child instances.

    /// Information about a child module instance.
    struct InstanceResource {
      StringAttr moduleName, instanceName;
      ModuleResourceUsage *usage;
      InstanceResource(StringAttr moduleName, StringAttr instanceName,
                       ModuleResourceUsage *usage)
          : moduleName(moduleName), instanceName(instanceName), usage(usage) {}
    };

    SmallVector<InstanceResource> instances;
    const ResourceUsage &getTotal() const { return total; }
    const ResourceUsage &getLocal() const { return local; }
    void emitJSON(raw_ostream &os) const;
  };

  /// Get resource usage for a module.
  ModuleResourceUsage *getResourceUsage(igraph::ModuleOpInterface module);
  ModuleResourceUsage *getResourceUsage(StringAttr moduleName);

  /// Get the instance graph used by this analysis.
  igraph::InstanceGraph *getInstanceGraph() const { return instanceGraph; }

private:
  /// Cache of computed resource usage per module.
  DenseMap<StringAttr, std::unique_ptr<ModuleResourceUsage>> designUsageCache;

  /// Instance graph for module hierarchy traversal.
  igraph::InstanceGraph *instanceGraph;
};

} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_ANALYSIS_RESOURCEUSAGEANALYSIS_H
