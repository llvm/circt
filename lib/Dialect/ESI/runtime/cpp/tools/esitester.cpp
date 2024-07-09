//===- esitester.cpp - ESI accelerator test/example tool ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DO NOT EDIT!
// This file is distributed as part of an ESI runtime package. The source for
// this file should always be modified within CIRCT
// (lib/dialect/ESI/runtime/cpp/tools/esitester.cpp).
//
//===----------------------------------------------------------------------===//
//
// This application isn't a utility so much as a test driver for an ESI system.
// It is also useful as an example of how to use the ESI C++ API. esiquery.cpp
// is also useful as an example.
//
//===----------------------------------------------------------------------===//

#include "esi/Accelerator.h"
#include "esi/Manifest.h"
#include "esi/Services.h"

#include <iostream>
#include <map>
#include <stdexcept>

using namespace esi;

static void registerCallbacks(Accelerator *);

int main(int argc, const char *argv[]) {
  // TODO: find a command line parser library rather than doing this by hand.
  if (argc < 3) {
    std::cerr << "Expected usage: " << argv[0]
              << " <backend> <connection specifier> [command]" << std::endl;
    return -1;
  }

  const char *backend = argv[1];
  const char *conn = argv[2];
  std::string cmd;
  if (argc > 3)
    cmd = argv[3];

  try {
    Context ctxt;
    std::unique_ptr<AcceleratorConnection> acc = ctxt.connect(backend, conn);
    const auto &info = *acc->getService<services::SysInfo>();
    Manifest manifest(ctxt, info.getJsonManifest());
    std::unique_ptr<Accelerator> accel = manifest.buildAccelerator(*acc);

    registerCallbacks(accel.get());

    if (cmd == "loop") {
      while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    } else if (cmd == "wait") {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    acc->disconnect();
    std::cerr << "Exiting successfully\n";
    return 0;

  } catch (std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }
}

void registerCallbacks(Accelerator *accel) {
  auto ports = accel->getPorts();
  auto f = ports.find(AppID("PrintfExample"));
  if (f != ports.end()) {
    auto callPort = f->second.getAs<services::CallService::Callback>();
    if (callPort)
      callPort->connect(
          [](const MessageData &data) -> MessageData {
            std::cout << "PrintfExample: " << *data.as<uint32_t>() << std::endl;
            return MessageData();
          },
          true);
  }
}
