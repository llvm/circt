//===- CPSATSchedulers.h - Schedulers using external CPSAT solvers --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Scheduling/Problems.h"
#include "circt/Scheduling/Schedulers.h"

namespace circt::scheduling {

class CPSATScheduler : public Scheduler<SharedOperatorsProblem> {
public:
  LogicalResult schedule(SharedOperatorsProblem &problem,
                         Operation *lastOp) override;
};

}; // namespace circt::scheduling