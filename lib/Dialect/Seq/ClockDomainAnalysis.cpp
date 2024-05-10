//===- ClockDomainAnalysis.cpp ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/Seq/SeqOps.h"

using namespace circt;
using namespace seq;
using namespace hw;

class ClockDomainAnalysis final {
  using InstancePath = SmallVector<HWInstanceLike>;
  using ClockSet = llvm::SmallSetVector<Value, 4>;

public:
  ClockDomainAnalysis(InstanceGraph &graph) : graph(graph) {}

private:
  DenseMap<std::pair<InstancePath, Value>, Value> clockSourceCache;
  DenseMap<Value, ClockSet> clocksAtVal;

  InstanceGraph &graph;
};
