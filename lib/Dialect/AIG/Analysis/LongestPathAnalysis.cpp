//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This analysis computes the longest combinational paths through a circuit
// represented in the AIG (And-Inverter Graph) dialect. The key aspects are:
//
// - Each AIG and-inverter operation is considered to have a unit delay of 1
// - The analysis traverses the circuit graph from inputs/registers to outputs
// - It handles hierarchical designs by analyzing modules bottom-up
// - Results include full path information with delays and debug points
// - Caching is used extensively to improve performance on large designs
//
// The core algorithm works as follows:
// 1. Build an instance graph of the full design hierarchy
// 2. Analyze modules in post-order (children before parents)
// 3. For each module:
//    - Trace paths from inputs and registers
//    - Propagate delays through logic and across module boundaries
//    - Record maximum delay and path info for each node
// 4. Combine results across hierarchy to get full chip critical paths
//
// The analysis handles both closed paths (register-to-register) and open
// paths (input-to-register, register-to-output) across the full hierarchy.
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AIG/AIGAnalysis.h"
#include "circt/Dialect/AIG/AIGPasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/InstanceGraph.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ImmutableList.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

#define DEBUG_TYPE "aig-longest-path-analysis"
using namespace circt;
using namespace aig;

static StringAttr getNameImpl(Value value) {
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    auto op = dyn_cast<hw::HWModuleOp>(arg.getParentBlock()->getParentOp());
    if (!op) {
      // TODO: Handle other operations.
      return StringAttr::get(value.getContext(), "<unknown-argument>");
    }

    return op.getArgName(arg.getArgNumber());
  }
  return TypeSwitch<Operation *, StringAttr>(value.getDefiningOp())
      .Case<seq::CompRegOp, seq::FirRegOp>(
          [](auto op) { return op.getNameAttr(); })
      .Case<hw::InstanceOp>([&](hw::InstanceOp op) {
        SmallString<16> str;
        str += op.getInstanceName();
        str += ".";
        str += cast<StringAttr>(
            op.getResultNamesAttr()[cast<OpResult>(value).getResultNumber()]);
        return StringAttr::get(op.getContext(), str);
      })
      .Case<seq::FirMemReadOp>([&](seq::FirMemReadOp op) {
        llvm::SmallString<16> str;
        str += op.getMemory().getDefiningOp<seq::FirMemOp>().getNameAttr();
        str += ".read_port";
        return StringAttr::get(value.getContext(), str);
      })
      .Case<seq::FirMemReadWriteOp>([&](seq::FirMemReadWriteOp op) {
        llvm::SmallString<16> str;
        str += op.getMemory().getDefiningOp<seq::FirMemOp>().getNameAttr();
        str += ".rw_port";
        return StringAttr::get(value.getContext(), str);
      })
      .Case<seq::FirMemOp>([&](seq::FirMemOp op) {
        llvm::SmallString<16> str;
        str += op.getMemory().getDefiningOp<seq::FirMemOp>().getNameAttr();
        str += ".write_port";
        return StringAttr::get(value.getContext(), str);
      })
      .Default([&](auto op) {
        llvm::errs() << "Unknown op: " << *op << "\n";
        return StringAttr::get(value.getContext(), "");
      });
}

static void printObjectImpl(llvm::raw_ostream &os, const Object &object,
                            int64_t delay = -1,
                            llvm::ImmutableList<DebugPoint> history = {},
                            StringRef comment = "") {
  std::string pathString;
  llvm::raw_string_ostream osPath(pathString);
  object.instancePath.print(osPath);
  os << "Object(" << pathString << "." << getNameImpl(object.value).getValue()
     << "[" << object.bitPos << "]";
  if (delay != -1)
    os << ", delay=" << delay;
  if (!history.isEmpty()) {
    os << ", history=[";
    llvm::interleaveComma(history, os, [&](DebugPoint p) { p.print(os); });
    os << "]";
  }
  if (!comment.empty())
    os << ", comment=\"" << comment << "\"";
  os << ")";
}

namespace circt {
namespace aig {
#define GEN_PASS_DEF_PRINTLONGESTPATHANALYSIS
#include "circt/Dialect/AIG/AIGPasses.h.inc"
} // namespace aig
} // namespace circt

using namespace circt;
using namespace aig;

//===----------------------------------------------------------------------===//
// Printing
//===----------------------------------------------------------------------===//

void OpenPath::print(llvm::raw_ostream &os) const {
  printObjectImpl(os, fanIn, delay, history);
}

void DebugPoint::print(llvm::raw_ostream &os) const {
  printObjectImpl(os, object, delay, {}, comment);
}

void Object::print(llvm::raw_ostream &os) const { printObjectImpl(os, *this); }

void DataflowPath::print(llvm::raw_ostream &os) {
  os << "root=" << root.getModuleName() << ", ";
  os << "fanOut=";
  fanOut.print(os);
  os << ", ";
  os << "fanIn=";
  path.print(os);
}

template <>
struct llvm::DenseMapInfo<Object> {
  using Info = llvm::DenseMapInfo<
      std::tuple<circt::igraph::InstancePath, Value, size_t>>;
  static Object getEmptyKey() {
    auto [path, value, bitPos] = Info::getEmptyKey();
    return Object(path, value, bitPos);
  }

  static Object getTombstoneKey() {
    auto [path, value, bitPos] = Info::getTombstoneKey();
    return Object(path, value, bitPos);
  }
  static llvm::hash_code getHashValue(Object object) {
    return Info::getHashValue(
        {object.instancePath, object.value, object.bitPos});
  }
  static bool isEqual(const Object &a, const Object &b) {
    return Info::isEqual({a.instancePath, a.value, a.bitPos},
                         {b.instancePath, b.value, b.bitPos});
  }
};

//===----------------------------------------------------------------------===//
// LongestPathAnalysis
//===----------------------------------------------------------------------===//

LongestPathAnalysis::~LongestPathAnalysis() {}

LongestPathAnalysis::LongestPathAnalysis(Operation *moduleOp,
                                         mlir::AnalysisManager &am) {}

bool LongestPathAnalysis::isAnalysisAvailable(StringAttr moduleName) const {
  return false;
}

int64_t LongestPathAnalysis::getMaxDelay(Value value) const {
  assert(false && "Not implemented");
  return 0;
}

int64_t LongestPathAnalysis::getAverageMaxDelay(Value value) const {
  assert(false && "Not implemented");
  return 0;
}

LogicalResult LongestPathAnalysis::getClosedPaths(
    StringAttr moduleName, SmallVectorImpl<DataflowPath> &results) const {
  assert(false && "Not implemented");
  return failure();
}

LogicalResult LongestPathAnalysis::getOpenPaths(
    StringAttr moduleName,
    SmallVectorImpl<std::pair<Object, OpenPath>> &openPathsToFF,
    SmallVectorImpl<std::tuple<size_t, size_t, OpenPath>>
        &openPathsFromOutputPorts) const {
  assert(false && "Not implemented");
  return failure();
}

ArrayRef<hw::HWModuleOp> LongestPathAnalysis::getTopModules() const {
  assert(false && "Not implemented");
  return {};
}

namespace {
struct PrintLongestPathAnalysisPass
    : public impl::PrintLongestPathAnalysisBase<PrintLongestPathAnalysisPass> {
  using PrintLongestPathAnalysisBase::outputFile;
  using PrintLongestPathAnalysisBase::PrintLongestPathAnalysisBase;
  using PrintLongestPathAnalysisBase::showTopKPercent;
  void runOnOperation() override;
};

} // namespace

void PrintLongestPathAnalysisPass::runOnOperation() {
  auto &analysis = getAnalysis<circt::aig::LongestPathAnalysis>();
  (void)analysis;
  return markAllAnalysesPreserved();
}
