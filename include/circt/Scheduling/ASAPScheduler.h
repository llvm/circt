//===- ASAPchedulers.h - Linear programming-based schedulers --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Scheduling/Schedulers.h"

namespace circt::scheduling {

class ASAPScheduler : public Scheduler<Problem> {
public:
  LogicalResult schedule(Problem &problem, Operation *lastOp) override;
};

}; // namespace circt::scheduling