//===- ModuleSummary.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"

#include <mutex>
#include <numeric>

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_MODULESUMMARY
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace firrtl;

static size_t knownWidths(Type type) {
  std::function<size_t(Type)> getWidth = [&](Type type) -> size_t {
    return TypeSwitch<Type, size_t>(type)
        .Case<BundleType>([&](BundleType bundle) -> size_t {
          size_t width = 0;
          for (auto &elt : bundle) {
            auto w = getWidth(elt.type);
            width += w;
          }
          return width;
        })
        .Case<FEnumType>([&](FEnumType fenum) -> size_t {
          size_t width = 0;
          for (auto &elt : fenum) {
            auto w = getWidth(elt.type);
            width = std::max(width, w);
          }
          return width + fenum.getTagWidth();
        })
        .Case<FVectorType>([&](auto vector) -> size_t {
          auto w = getWidth(vector.getElementType());
          return w * vector.getNumElements();
        })
        .Case<IntType>([&](IntType iType) {
          auto v = iType.getWidth();
          if (v)
            return *v;
          return 0;
        })
        .Case<ClockType, ResetType, AsyncResetType>([](Type) { return 1; })
        .Default([&](auto t) { return 0; });
  };
  return getWidth(type);
}

namespace {
struct ModuleSummaryPass
    : public circt::firrtl::impl::ModuleSummaryBase<ModuleSummaryPass> {
  struct KeyTy {
    SmallVector<size_t> portSizes;
    size_t opcount;
    bool operator==(const KeyTy &rhs) const {
      return portSizes == rhs.portSizes && opcount == rhs.opcount;
    }
  };

  size_t countOps(FModuleOp mod) {
    size_t retval = 0;
    mod.walk([&](Operation *op) { retval += 1; });
    return retval;
  }

  SmallVector<size_t> portSig(FModuleOp mod) {
    SmallVector<size_t> ports;
    for (auto p : mod.getPortTypes())
      ports.push_back(knownWidths(cast<TypeAttr>(p).getValue()));
    return ports;
  }

  void runOnOperation() override;
};
} // namespace

namespace mlir {
// NOLINTNEXTLINE(readability-identifier-naming)
inline llvm::hash_code hash_value(const ModuleSummaryPass::KeyTy &element) {
  return llvm::hash_combine(element.portSizes.size(),
                            llvm::hash_combine_range(element.portSizes.begin(),
                                                     element.portSizes.end()),
                            element.opcount);
}
} // namespace mlir

namespace llvm {
// Type hash just like pointers.
template <>
struct DenseMapInfo<ModuleSummaryPass::KeyTy> {
  using KeyTy = ModuleSummaryPass::KeyTy;
  static KeyTy getEmptyKey() { return {{}, ~0ULL}; }
  static KeyTy getTombstoneKey() { return {{}, ~0ULL - 1}; }
  static unsigned getHashValue(const KeyTy &val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(const KeyTy &lhs, const KeyTy &rhs) { return lhs == rhs; }
};
} // namespace llvm

void ModuleSummaryPass::runOnOperation() {
  auto circuit = getOperation();

  using MapTy = DenseMap<KeyTy, SmallVector<FModuleOp>>;
  MapTy data;

  std::mutex dataMutex; // protects data

  mlir::parallelForEach(circuit.getContext(),
                        circuit.getBodyBlock()->getOps<FModuleOp>(),
                        [&](auto mod) {
                          auto p = portSig(mod);
                          auto n = countOps(mod);
                          const std::lock_guard<std::mutex> lock(dataMutex);
                          data[{p, n}].push_back(mod);
                        });

  SmallVector<MapTy::value_type> sortedData(data.begin(), data.end());
  std::sort(sortedData.begin(), sortedData.end(),
            [](const MapTy::value_type &lhs, const MapTy::value_type &rhs) {
              return std::get<0>(lhs).opcount * std::get<1>(lhs).size() *
                         std::get<1>(lhs).size() >
                     std::get<0>(rhs).opcount * std::get<1>(rhs).size() *
                         std::get<1>(rhs).size();
            });
  llvm::errs() << "cost, opcount, portcount, modcount, portBits, examplename\n";
  for (auto &p : sortedData)
    if (p.second.size() > 1) {
      llvm::errs() << (p.first.opcount * p.second.size() * p.second.size())
                   << "," << p.first.opcount << "," << p.first.portSizes.size()
                   << "," << p.second.size() << ","
                   << std::accumulate(p.first.portSizes.begin(),
                                      p.first.portSizes.end(), 0)
                   << "," << p.second[0].getName() << "\n";
    }
  markAllAnalysesPreserved();
}
