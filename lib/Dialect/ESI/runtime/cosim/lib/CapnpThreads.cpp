//===- CapnpThreads.cpp - Cosim RPC common code -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cosim/CapnpThreads.h"
#include "CosimDpi.capnp.h"
#include <capnp/ez-rpc.h>
#include <thread>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

using namespace capnp;
using namespace esi::cosim;

CapnpCosimThread::CapnpCosimThread(FILE *logFile)
    : logFile(logFile), myThread(nullptr), stopSig(false) {}
CapnpCosimThread::~CapnpCosimThread() { stop(); }

void CapnpCosimThread::loop(kj::WaitScope &waitScope,
                            std::function<void()> poll) {
  // OK, this is uber hacky, but it unblocks me and isn't _too_ inefficient. The
  // problem is that I can't figure out how read the stop signal from libkj
  // asyncrony land.
  //
  // IIRC the main libkj wait loop uses `select()` (or something similar on
  // Windows) on its FDs. As a result, any code which checks the stop variable
  // doesn't run until there is some I/O. Probably the right way is to set up a
  // pipe to deliver a shutdown signal.
  //
  // TODO: Figure out how to do this properly, if possible.
  while (!stopSig) {
    waitScope.poll();
    poll();
    waitScope.poll();
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
}

/// Signal the RPC server thread to stop. Wait for it to exit.
void CapnpCosimThread::stop() {
  Lock g(m);
  if (myThread == nullptr) {
    fprintf(stderr, "CapnpCosimThread not Run()\n");
  } else if (!stopSig) {
    stopSig = true;
    myThread->join();
  }
}
