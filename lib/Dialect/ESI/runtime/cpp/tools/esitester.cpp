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
#include "esi/CLI.h"
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
static void dmaTest(AcceleratorConnection *, Accelerator *, bool read,
                    bool write);
static void bandwidthTest(AcceleratorConnection *, Accelerator *,
                          uint32_t xferCount,
                          const std::vector<uint32_t> &widths, bool read,
                          bool write);

int main(int argc, const char *argv[]) {
  CliParser cli("esitester");
  cli.description("Test an ESI system running the ESI tester image.");
  cli.require_subcommand(1);

  CLI::App *loopSub = cli.add_subcommand("loop", "Loop indefinitely");

  CLI::App *waitSub =
      cli.add_subcommand("wait", "Wait for a certain number of seconds");
  uint32_t waitTime = 1;
  waitSub->add_option("-t,--time", waitTime, "Number of seconds to wait");

  CLI::App *hostmemtestSub =
      cli.add_subcommand("hostmem", "Run the host memory test");
  bool hmRead = false;
  bool hmWrite = false;
  hostmemtestSub->add_flag("-w,--write", hmWrite,
                           "Enable host memory write test");
  hostmemtestSub->add_flag("-r,--read", hmRead, "Enable host memory read test");

  CLI::App *dmatestSub = cli.add_subcommand("dma", "Run the DMA test");
  bool dmaRead = false;
  bool dmaWrite = false;
  dmatestSub->add_flag("-w,--write", dmaWrite, "Enable dma write test");
  dmatestSub->add_flag("-r,--read", dmaRead, "Enable dma read test");

  CLI::App *bandwidthSub =
      cli.add_subcommand("bandwidth", "Run the bandwidth test");
  uint32_t xferCount = 1000;
  bandwidthSub->add_option("-c,--count", xferCount,
                           "Number of transfers to perform");
  bool bandwidthRead = false;
  bool bandwidthWrite = false;
  std::vector<uint32_t> widths = {32, 64, 128, 256, 384, 504, 512};
  bandwidthSub->add_option("--widths", widths,
                           "Width of the transfers to perform (default: 32, "
                           "64, 128, 256, 384, 504, 512)");
  bandwidthSub->add_flag("-w,--write", bandwidthWrite,
                         "Enable bandwidth write");
  bandwidthSub->add_flag("-r,--read", bandwidthRead, "Enable bandwidth read");

  if (int rc = cli.esiParse(argc, argv))
    return rc;
  if (!cli.get_help_ptr()->empty())
    return 0;

  Context &ctxt = cli.getContext();
  try {
    std::unique_ptr<AcceleratorConnection> acc = cli.connect();
    const auto &info = *acc->getService<services::SysInfo>();
    Manifest manifest(ctxt, info.getJsonManifest());
    Accelerator *accel = manifest.buildAccelerator(*acc);
    acc->getServiceThread()->addPoll(*accel);

    if (*loopSub) {
      registerCallbacks(acc.get(), accel);
      while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    } else if (*waitSub) {
      registerCallbacks(acc.get(), accel);
      std::this_thread::sleep_for(std::chrono::seconds(waitTime));
    } else if (*hostmemtestSub) {
      hostmemTest(acc.get(), accel, hmRead, hmWrite);
    } else if (*dmatestSub) {
      dmaTest(acc.get(), accel, dmaRead, dmaWrite);
    } else if (*bandwidthSub) {
      bandwidthTest(acc.get(), accel, xferCount, widths, bandwidthRead,
                    bandwidthWrite);
    }

    acc->disconnect();
    std::cout << "Exiting successfully\n";
    return 0;
  } catch (std::exception &e) {
    ctxt.getLogger().error("esitester", e.what());
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

  for (size_t width : {32, 64, 128, 256, 384, 504, 512, 513})
    hostmemTest(acc, *scratchRegion, width, scratchRegion->getDevicePtr(), read,
                write);
}

static void dmaReadTest(AcceleratorConnection *conn, Accelerator *acc,
                        size_t width) {
  Logger &logger = conn->getLogger();
  logger.info("esitester",
              "== Running DMA read test with width " + std::to_string(width));
  AppIDPath lastPath;
  BundlePort *toHostMMIOPort = acc->resolvePort(
      {AppID("tohostdmatest", width), AppID("ToHostDMATest")}, lastPath);
  if (!toHostMMIOPort)
    throw std::runtime_error("dma read test failed. No tohostdmatest[" +
                             std::to_string(width) + "] found");
  auto *toHostMMIO = toHostMMIOPort->getAs<services::MMIO::MMIORegion>();
  if (!toHostMMIO)
    throw std::runtime_error("dma read test failed. MMIO port is not MMIO");
  lastPath.clear();
  BundlePort *outPortBundle =
      acc->resolvePort({AppID("tohostdmatest", width), AppID("out")}, lastPath);
  ReadChannelPort &outPort = outPortBundle->getRawRead("data");
  outPort.connect();

  size_t xferCount = 24;
  uint64_t last = 0;
  MessageData data;
  toHostMMIO->write(0, xferCount);
  for (size_t i = 0; i < xferCount; ++i) {
    outPort.read(data);
    if (width == 64) {
      uint64_t val = *data.as<uint64_t>();
      if (val < last)
        throw std::runtime_error("dma read test failed. Out of order data");
      last = val;
    }
    logger.debug("esitester",
                 "Cycle count [" + std::to_string(i) + "] = 0x" + data.toHex());
  }
  outPort.disconnect();
  logger.info("esitester", "==   DMA read test complete");
}

static void dmaWriteTest(AcceleratorConnection *conn, Accelerator *acc,
                         size_t width) {
  Logger &logger = conn->getLogger();
  logger.info("esitester",
              "== Running DMA write test with width " + std::to_string(width));
  AppIDPath lastPath;
  BundlePort *fromHostMMIOPort = acc->resolvePort(
      {AppID("fromhostdmatest", width), AppID("FromHostDMATest")}, lastPath);
  if (!fromHostMMIOPort)
    throw std::runtime_error("dma read test for " + toString(width) +
                             " bits failed. No fromhostdmatest[" +
                             std::to_string(width) + "] found");
  auto *fromHostMMIO = fromHostMMIOPort->getAs<services::MMIO::MMIORegion>();
  if (!fromHostMMIO)
    throw std::runtime_error("dma write test for " + toString(width) +
                             " bits failed. MMIO port is not MMIO");
  lastPath.clear();
  BundlePort *outPortBundle = acc->resolvePort(
      {AppID("fromhostdmatest", width), AppID("in")}, lastPath);
  if (!outPortBundle)
    throw std::runtime_error("dma write test for " + toString(width) +
                             " bits failed. No out port found");
  WriteChannelPort &writePort = outPortBundle->getRawWrite("data");
  writePort.connect();

  size_t xferCount = 24;
  uint8_t *data = new uint8_t[width];
  for (size_t i = 0; i < width / 8; ++i)
    data[i] = 0;
  fromHostMMIO->read(8);
  fromHostMMIO->write(0, xferCount);
  for (size_t i = 1; i < xferCount + 1; ++i) {
    data[0] = i;
    bool successWrite;
    size_t attempts = 0;
    do {
      successWrite = writePort.tryWrite(MessageData(data, width / 8));
      if (!successWrite) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    } while (!successWrite && ++attempts < 100);
    if (!successWrite)
      throw std::runtime_error("dma write test for " + toString(width) +
                               " bits failed. Write failed");
    uint64_t lastReadMMIO;
    for (size_t a = 0; a < 20; ++a) {
      lastReadMMIO = fromHostMMIO->read(8);
      if (lastReadMMIO == i)
        break;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      if (a >= 19)
        throw std::runtime_error("dma write for " + toString(width) +
                                 " bits test failed. Read from MMIO failed");
    }
  }
  writePort.disconnect();
  delete[] data;
  logger.info("esitester",
              "==   DMA write test for " + toString(width) + " bits complete");
}

static void dmaTest(AcceleratorConnection *conn, Accelerator *acc, bool read,
                    bool write) {
  bool success = true;
  if (write)
    for (size_t width : {32, 64, 128, 256, 384, 504, 512})
      try {
        dmaWriteTest(conn, acc, width);
      } catch (std::exception &e) {
        success = false;
        std::cerr << "DMA write test for " << width
                  << "bits failed: " << e.what() << std::endl;
      }
  if (read)
    for (size_t width : {32, 64, 128, 256, 384, 504, 512})
      dmaReadTest(conn, acc, width);
  if (!success)
    throw std::runtime_error("DMA test failed");
}

static void bandwidthReadTest(AcceleratorConnection *conn, Accelerator *acc,
                              size_t width, size_t xferCount) {

  AppIDPath lastPath;
  BundlePort *toHostMMIOPort = acc->resolvePort(
      {AppID("tohostdmatest", width), AppID("ToHostDMATest")}, lastPath);
  if (!toHostMMIOPort)
    throw std::runtime_error("bandwidth test failed. No tohostdmatest[" +
                             std::to_string(width) + "] found");
  auto *toHostMMIO = toHostMMIOPort->getAs<services::MMIO::MMIORegion>();
  if (!toHostMMIO)
    throw std::runtime_error("bandwidth test failed. MMIO port is not MMIO");
  lastPath.clear();
  BundlePort *outPortBundle =
      acc->resolvePort({AppID("tohostdmatest", width), AppID("out")}, lastPath);
  ReadChannelPort &outPort = outPortBundle->getRawRead("data");
  outPort.connect();

  Logger &logger = conn->getLogger();
  logger.info("esitester", "Starting bandwidth test read with " +
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
                                 duration.count() / 1024) +
                  " kbytes/sec");
}

static void bandwidthWriteTest(AcceleratorConnection *conn, Accelerator *acc,
                               size_t width, size_t xferCount) {

  AppIDPath lastPath;
  BundlePort *fromHostMMIOPort = acc->resolvePort(
      {AppID("fromhostdmatest", width), AppID("FromHostDMATest")}, lastPath);
  if (!fromHostMMIOPort)
    throw std::runtime_error("bandwidth test failed. No tohostdmatest[" +
                             std::to_string(width) + "] found");
  auto *fromHostMMIO = fromHostMMIOPort->getAs<services::MMIO::MMIORegion>();
  if (!fromHostMMIO)
    throw std::runtime_error("bandwidth test failed. MMIO port is not MMIO");
  lastPath.clear();
  BundlePort *inPortBundle = acc->resolvePort(
      {AppID("fromhostdmatest", width), AppID("in")}, lastPath);
  WriteChannelPort &outPort = inPortBundle->getRawWrite("data");
  outPort.connect();

  Logger &logger = conn->getLogger();
  logger.info("esitester", "Starting bandwidth write read with " +
                               std::to_string(xferCount) + " x " +
                               std::to_string(width) + " bit transfers");
  std::vector<uint8_t> dataVec(width / 8);
  for (size_t i = 0; i < width / 8; ++i)
    dataVec[i] = i;
  MessageData data(dataVec);
  auto start = std::chrono::high_resolution_clock::now();
  fromHostMMIO->write(0, xferCount);
  for (size_t i = 0; i < xferCount; ++i) {
    outPort.write(data);
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
                                 duration.count() / 1024) +
                  " kbytes/sec");
}

static void bandwidthTest(AcceleratorConnection *conn, Accelerator *acc,
                          uint32_t xferCount,
                          const std::vector<uint32_t> &widths, bool read,
                          bool write) {
  if (read)
    for (size_t width : widths)
      bandwidthReadTest(conn, acc, width, xferCount);
  if (write)
    for (size_t width : widths)
      bandwidthWriteTest(conn, acc, width, xferCount);
}
