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

static void registerCallbacks(AcceleratorConnection *, Accelerator *);
static void dmaTest(AcceleratorConnection *, Accelerator *);

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
    Context ctxt = Context::withLogger<StreamLogger>(Logger::Level::Debug);
    std::unique_ptr<AcceleratorConnection> acc = ctxt.connect(backend, conn);
    const auto &info = *acc->getService<services::SysInfo>();
    Manifest manifest(ctxt, info.getJsonManifest());
    Accelerator *accel = manifest.buildAccelerator(*acc);
    acc->getServiceThread()->addPoll(*accel);

    registerCallbacks(acc.get(), accel);

    if (cmd == "loop") {
      while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    } else if (cmd == "wait") {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    } else if (cmd == "dmatest") {
      dmaTest(acc.get(), accel);
    }

    acc->disconnect();
    std::cerr << "Exiting successfully\n";
    return 0;

  } catch (std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }
}

void registerCallbacks(AcceleratorConnection *conn, Accelerator *accel) {
  auto ports = accel->getPorts();
  auto f = ports.find(AppID("PrintfExample"));
  if (f != ports.end()) {
    auto callPort = f->second.getAs<services::CallService::Callback>();
    if (callPort)
      callPort->connect(
          [conn](const MessageData &data) -> MessageData {
            conn->getLogger().debug(
                [&](std::string &subsystem, std::string &msg,
                    std::unique_ptr<std::map<std::string, std::any>> &details) {
                  subsystem = "ESITESTER";
                  msg = "Received PrintfExample message";
                  details = std::make_unique<std::map<std::string, std::any>>();
                  details->emplace("data", data);
                });
            std::cout << "PrintfExample: " << *data.as<uint32_t>() << std::endl;
            return MessageData();
          },
          true);
  }
}

void dmaTest(AcceleratorConnection *conn, Accelerator *acc) {
  // Enable the host memory service.
  auto hostmem = conn->getService<services::HostMem>();
  hostmem->start();

  // Initiate a test read.
  auto *readMem =
      acc->getPorts().at(AppID("ReadMem")).getAs<services::MMIO::MMIORegion>();
  uint64_t *dataPtr = new uint64_t;
  *dataPtr = 0x12345678;
  readMem->write(8, (uint64_t)dataPtr);

  // Wait for the accelerator to read the correct value. Timeout and fail after
  // 10ms.
  uint64_t val = 0;
  for (int i = 0; i < 100; ++i) {
    val = readMem->read(0);
    if (val == *dataPtr)
      break;
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  if (val != *dataPtr)
    throw std::runtime_error("DMA test failed");
}
