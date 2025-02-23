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

#include <chrono>
#include <iostream>
#include <map>
#include <stdexcept>

using namespace esi;

static void registerCallbacks(AcceleratorConnection *, Accelerator *);
static void hostmemTest(AcceleratorConnection *, Accelerator *, bool read,
                        bool write);
static void dmaWriteTest(AcceleratorConnection *, Accelerator *);
static void bandwidthTest(AcceleratorConnection *, Accelerator *);

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
    // TODO: Use proper command line parsing to set debug level.
    Context ctxt = Context::withLogger<StreamLogger>(Logger::Level::Debug);
    // Context ctxt = Context::withLogger<StreamLogger>(Logger::Level::Info);
    std::unique_ptr<AcceleratorConnection> acc = ctxt.connect(backend, conn);
    const auto &info = *acc->getService<services::SysInfo>();
    Manifest manifest(ctxt, info.getJsonManifest());
    Accelerator *accel = manifest.buildAccelerator(*acc);
    acc->getServiceThread()->addPoll(*accel);

    if (cmd == "loop") {
      registerCallbacks(acc.get(), accel);
      while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    } else if (cmd == "wait") {
      registerCallbacks(acc.get(), accel);
      std::this_thread::sleep_for(std::chrono::seconds(1));
    } else if (cmd == "hostmemtest") {
      hostmemTest(acc.get(), accel, true, true);
    } else if (cmd == "hostmemreadtest") {
      hostmemTest(acc.get(), accel, true, false);
    } else if (cmd == "hostmemwritetest") {
      hostmemTest(acc.get(), accel, false, true);
    } else if (cmd == "dmawritetest") {
      dmaWriteTest(acc.get(), accel);
    } else if (cmd == "bandwidth") {
      bandwidthTest(acc.get(), accel);
    } else {
      std::cerr << "Unknown command: " << cmd << std::endl;
      return -1;
    }

    acc->disconnect();
    std::cout << "Exiting successfully\n";
    return 0;

  } catch (std::exception &e) {
    std::cout << "Error: " << e.what() << std::endl;
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
    else
      std::cerr << "PrintfExample port is not a CallService::Callback"
                << std::endl;
  } else {
    std::cerr << "No PrintfExample port found" << std::endl;
  }
}

/// Initiate a test read.
void hostmemTest(Accelerator *acc,
                 esi::services::HostMem::HostMemRegion &region, uint32_t width,
                 void *devicePtr, bool read, bool write) {
  std::cout << "Running hostmem test with width " << width << std::endl;
  uint64_t *dataPtr = static_cast<uint64_t *>(region.getPtr());

  if (read) {
    auto readMemChildIter = acc->getChildren().find(AppID("readmem", width));
    if (readMemChildIter == acc->getChildren().end())
      throw std::runtime_error("hostmem test failed. No readmem child found");
    auto &readMemPorts = readMemChildIter->second->getPorts();
    auto readMemPortIter = readMemPorts.find(AppID("ReadMem"));
    if (readMemPortIter == readMemPorts.end())
      throw std::runtime_error("hostmem test failed. No ReadMem port found");
    auto *readMem = readMemPortIter->second.getAs<services::MMIO::MMIORegion>();
    if (!readMem)
      throw std::runtime_error("hostmem test failed. ReadMem port is not MMIO");

    for (size_t i = 0; i < 8; ++i) {
      dataPtr[0] = 0x12345678 << i;
      dataPtr[1] = 0xDEADBEEF << i;
      region.flush();
      readMem->write(8, reinterpret_cast<uint64_t>(devicePtr));

      // Wait for the accelerator to read the correct value. Timeout and fail
      // after 10ms.
      uint64_t val = 0;
      uint64_t expected = dataPtr[0];
      if (width < 64)
        expected &= ((1ull << width) - 1);
      for (int i = 0; i < 100; ++i) {
        val = readMem->read(0);
        if (val == expected)
          break;
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }

      if (val != expected)
        throw std::runtime_error("hostmem read test failed. Expected " +
                                 esi::toHex(expected) + ", got " +
                                 esi::toHex(val));
    }
  }

  // Initiate a test write.
  if (write) {
    assert(width % 8 == 0);
    auto check = [&](bool print) {
      bool ret = true;
      for (size_t i = 0; i < 8; ++i) {
        if (print)
          std::cout << "dataPtr[" << i << "] = 0x" << esi::toHex(dataPtr[i])
                    << std::endl;
        if (i < (width + 63) / 64 && dataPtr[i] == 0xFFFFFFFFFFFFFFFFull)
          ret = false;
      }
      return ret;
    };

    auto writeMemChildIter = acc->getChildren().find(AppID("writemem", width));
    if (writeMemChildIter == acc->getChildren().end())
      throw std::runtime_error("hostmem test failed. No writemem child found");
    auto &writeMemPorts = writeMemChildIter->second->getPorts();
    auto writeMemPortIter = writeMemPorts.find(AppID("WriteMem"));
    if (writeMemPortIter == writeMemPorts.end())
      throw std::runtime_error("hostmem test failed. No WriteMem port found");
    auto *writeMem =
        writeMemPortIter->second.getAs<services::MMIO::MMIORegion>();
    if (!writeMem)
      throw std::runtime_error(
          "hostmem test failed. WriteMem port is not MMIO");

    for (size_t i = 0, e = 8; i < e; ++i)
      dataPtr[i] = 0xFFFFFFFFFFFFFFFFull;
    region.flush();

    // Command the accelerator to write to 'devicePtr', the pointer which the
    // device should use for 'dataPtr'.
    writeMem->write(0, reinterpret_cast<uint64_t>(devicePtr));
    // Wait for the accelerator to write. Timeout and fail after 10ms.
    for (int i = 0; i < 100; ++i) {
      if (check(false))
        break;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    if (!check(true))
      throw std::runtime_error("hostmem write test failed");

    // Check that the accelerator didn't write too far.
    size_t widthInBytes = width / 8;
    uint8_t *dataPtr8 = reinterpret_cast<uint8_t *>(region.getPtr());
    for (size_t i = widthInBytes; i < 64; ++i) {
      std::cout << "endcheck dataPtr8[" << i << "] = 0x"
                << esi::toHex(dataPtr8[i]) << std::endl;
      if (dataPtr8[i] != 0xFF)
        throw std::runtime_error(
            "hostmem write test failed -- write went too far");
    }
  }
}

void hostmemTest(AcceleratorConnection *conn, Accelerator *acc, bool read,
                 bool write) {
  // Enable the host memory service.
  auto hostmem = conn->getService<services::HostMem>();
  hostmem->start();
  auto scratchRegion = hostmem->allocate(/*size(bytes)=*/512, /*memOpts=*/{});
  uint64_t *dataPtr = static_cast<uint64_t *>(scratchRegion->getPtr());
  for (size_t i = 0; i < scratchRegion->getSize() / 8; ++i)
    dataPtr[i] = 0;
  scratchRegion->flush();

  hostmemTest(acc, *scratchRegion, 32, scratchRegion->getDevicePtr(), read,
              write);
  hostmemTest(acc, *scratchRegion, 64, scratchRegion->getDevicePtr(), read,
              write);
  hostmemTest(acc, *scratchRegion, 96, scratchRegion->getDevicePtr(), read,
              write);
  hostmemTest(acc, *scratchRegion, 504, scratchRegion->getDevicePtr(), read,
              write);
}

static void dmaWriteTest(AcceleratorConnection *conn, Accelerator *acc,
                         size_t width) {
  Logger &logger = conn->getLogger();
  logger.info("esitester",
              "== Running DMA write test with width " + std::to_string(width));
  auto dmaTestChildIter =
      acc->getChildren().find(AppID("tohostdmatest", width));
  if (dmaTestChildIter == acc->getChildren().end())
    throw std::runtime_error("dma test failed. No tohostdma child found");
  auto &toHostDMA = dmaTestChildIter->second->getPorts();
  auto toHostMMIOIter = toHostDMA.find(AppID("ToHostDMATest"));
  if (toHostMMIOIter == toHostDMA.end())
    throw std::runtime_error("dma write test failed. No MMIO port found");
  auto *toHostMMIO = toHostMMIOIter->second.getAs<services::MMIO::MMIORegion>();
  if (!toHostMMIO)
    throw std::runtime_error("dma write test failed. MMIO port is not MMIO");
  auto outPortIter = toHostDMA.find(AppID("out"));
  if (outPortIter == toHostDMA.end())
    throw std::runtime_error("dma test failed. No out port found");
  ReadChannelPort &outPort = outPortIter->second.getRawRead("data");
  outPort.connect();

  size_t xferCount = 16;
  uint64_t last = 0;
  MessageData data;
  toHostMMIO->write(0, xferCount);
  if (width == 64) {
    for (size_t i = 0; i < xferCount; ++i) {
      outPort.read(data);
      uint64_t val = *data.as<uint64_t>();
      if (val < last)
        throw std::runtime_error("dma write test failed. Out of order data");
      last = val;
      logger.info("esitester",
                  "Cycle count [" + std::to_string(i) + "] = 0x" + toHex(val));
    }
  }
  outPort.disconnect();
  logger.info("esitester", "==   DMA write test complete");
}

static void dmaWriteTest(AcceleratorConnection *conn, Accelerator *acc) {
  for (size_t width : {32, 64, 128, 256, 384, 504, 512})
    dmaWriteTest(conn, acc, width);
}

static void bandWidthTest(AcceleratorConnection *conn, Accelerator *acc,
                          size_t width, size_t xferCount) {

  AppIDPath lastPath;
  BundlePort *toHostMMIOPort = acc->resolvePort(
      {AppID("tohostdmatest", width), AppID("ToHostDMATest")}, lastPath);
  if (!toHostMMIOPort)
    throw std::runtime_error("bandwidth test failed. No tohostdmatest[" +
                             std::to_string(width) + "] found");
  auto *toHostMMIO = toHostMMIOPort->getAs<services::MMIO::MMIORegion>();
  if (!toHostMMIO)
    throw std::runtime_error("dma write test failed. MMIO port is not MMIO");
  lastPath.clear();
  BundlePort *outPortBundle =
      acc->resolvePort({AppID("tohostdmatest", width), AppID("out")}, lastPath);
  ReadChannelPort &outPort = outPortBundle->getRawRead("data");
  outPort.connect();

  Logger &logger = conn->getLogger();
  logger.info("esitester", "Starting bandwidth test with " +
                               std::to_string(xferCount) + " x " +
                               std::to_string(width) + " bit transfers");
  MessageData data;
  auto start = std::chrono::high_resolution_clock::now();
  toHostMMIO->write(0, xferCount);
  for (size_t i = 0; i < xferCount; ++i) {
    outPort.read(data);
    logger.debug(
        [i, &data](std::string &subsystem, std::string &msg,
                   std::unique_ptr<std::map<std::string, std::any>> &details) {
          subsystem = "esitester";
          msg = "Cycle count [" + std::to_string(i) + "] = 0x" + data.toHex();
        });
  }
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock::now() - start);
  logger.info("esitester",
              "  Bandwidth test: " + std::to_string(xferCount) + " x " +
                  std::to_string(width) + " bit transfers in " +
                  std::to_string(duration.count()) + " microseconds");
  logger.info("esitester",
              "    bandwidth: " +
                  std::to_string((xferCount * (width / 8) * 1000000) /
                                 duration.count()) +
                  " bytes/sec");
}

static void bandwidthTest(AcceleratorConnection *conn, Accelerator *acc) {
  for (size_t width : {32, 64, 128, 256, 384, 504, 512})
    // for (size_t width : {504})
    bandWidthTest(conn, acc, width, 10);
}
