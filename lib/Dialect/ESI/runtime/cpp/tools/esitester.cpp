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

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace esi;

// Forward declarations of test functions.
static void callbackTest(AcceleratorConnection *, Accelerator *,
                         uint32_t iterations);
static void hostmemTest(AcceleratorConnection *, Accelerator *,
                        const std::vector<uint32_t> &widths, bool write,
                        bool read);
static void hostmemBandwidthTest(AcceleratorConnection *conn, Accelerator *acc,
                                 uint32_t xferCount,
                                 const std::vector<uint32_t> &widths, bool read,
                                 bool write);
static void dmaTest(AcceleratorConnection *, Accelerator *,
                    const std::vector<uint32_t> &widths, bool read, bool write);
static void bandwidthTest(AcceleratorConnection *, Accelerator *,
                          const std::vector<uint32_t> &widths,
                          uint32_t xferCount, bool read, bool write);
static void loopbackAddTest(AcceleratorConnection *, Accelerator *,
                            uint32_t iterations, bool pipeline);
static void aggregateHostmemBandwidthTest(AcceleratorConnection *,
                                          Accelerator *, uint32_t width,
                                          uint32_t xferCount, bool read,
                                          bool write);
static void streamingAddTest(AcceleratorConnection *, Accelerator *,
                             uint32_t addAmt, uint32_t numItems);
static void streamingAddTranslatedTest(AcceleratorConnection *, Accelerator *,
                                       uint32_t addAmt, uint32_t numItems);
static void coordTranslateTest(AcceleratorConnection *, Accelerator *,
                               uint32_t xTrans, uint32_t yTrans,
                               uint32_t numCoords);

// Default widths and default widths string for CLI help text.
constexpr std::array<uint32_t, 5> defaultWidths = {32, 64, 128, 256, 512};
static std::string defaultWidthsStr() {
  std::string s;
  for (size_t i = 0; i < defaultWidths.size(); ++i) {
    s += std::to_string(defaultWidths[i]);
    if (i + 1 < defaultWidths.size())
      s += ",";
  }
  return s;
}

// Helper to format bandwidth with appropriate units.
static std::string formatBandwidth(double bytesPerSec) {
  const char *unit = "B/s";
  double value = bytesPerSec;
  if (bytesPerSec >= 1e9) {
    unit = "GB/s";
    value = bytesPerSec / 1e9;
  } else if (bytesPerSec >= 1e6) {
    unit = "MB/s";
    value = bytesPerSec / 1e6;
  } else if (bytesPerSec >= 1e3) {
    unit = "KB/s";
    value = bytesPerSec / 1e3;
  }
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(2);
  oss << value << " " << unit;
  return oss.str();
}

// Human-readable size from bytes.
static std::string humanBytes(uint64_t bytes) {
  const char *units[] = {"B", "KB", "MB", "GB", "TB"};
  double v = (double)bytes;
  int u = 0;
  while (v >= 1024.0 && u < 4) {
    v /= 1024.0;
    ++u;
  }
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(u == 0 ? 0 : 2);
  oss << v << " " << units[u];
  return oss.str();
}

// Human-readable time from microseconds.
static std::string humanTimeUS(uint64_t us) {
  if (us < 1000)
    return std::to_string(us) + " us";
  double ms = us / 1000.0;
  if (ms < 1000.0) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(ms < 10.0 ? 2 : (ms < 100.0 ? 1 : 0));
    oss << ms << " ms";
    return oss.str();
  }
  double sec = ms / 1000.0;
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(sec < 10.0 ? 3 : 2);
  oss << sec << " s";
  return oss.str();
}

// MSVC does not implement std::aligned_malloc, even though it's part of the
// C++17 standard. Provide a compatibility layer.
static void *alignedAllocCompat(std::size_t alignment, std::size_t size) {
#if defined(_MSC_VER)
  void *ptr = _aligned_malloc(size, alignment);
  if (!ptr)
    throw std::bad_alloc();
  return ptr;
#else
  void *ptr = std::aligned_alloc(alignment, size);
  if (!ptr)
    throw std::bad_alloc();
  return ptr;
#endif
}

static void alignedFreeCompat(void *ptr) {
#if defined(_MSC_VER)
  _aligned_free(ptr);
#else
  std::free(ptr);
#endif
}

int main(int argc, const char *argv[]) {
  CliParser cli("esitester");
  cli.description("Test an ESI system running the ESI tester image.");
  cli.require_subcommand(1);

  CLI::App *callback_test =
      cli.add_subcommand("callback", "initiate callback test");
  uint32_t cb_iters = 1;
  callback_test->add_option("-i,--iters", cb_iters,
                            "Number of iterations to run");

  CLI::App *hostmemtestSub =
      cli.add_subcommand("hostmem", "Run the host memory test");
  bool hmRead = false;
  bool hmWrite = false;
  std::vector<uint32_t> hostmemWidths(defaultWidths.begin(),
                                      defaultWidths.end());
  hostmemtestSub->add_flag("-w,--write", hmWrite,
                           "Enable host memory write test");
  hostmemtestSub->add_flag("-r,--read", hmRead, "Enable host memory read test");
  hostmemtestSub->add_option(
      "--widths", hostmemWidths,
      "Hostmem test widths (default: " + defaultWidthsStr() + ")");

  CLI::App *dmatestSub = cli.add_subcommand("dma", "Run the DMA test");
  bool dmaRead = false;
  bool dmaWrite = false;
  std::vector<uint32_t> dmaWidths(defaultWidths.begin(), defaultWidths.end());
  dmatestSub->add_flag("-w,--write", dmaWrite, "Enable dma write test");
  dmatestSub->add_flag("-r,--read", dmaRead, "Enable dma read test");
  dmatestSub->add_option("--widths", dmaWidths,
                         "DMA test widths (default: " + defaultWidthsStr() +
                             ")");

  CLI::App *bandwidthSub =
      cli.add_subcommand("bandwidth", "Run the bandwidth test");
  uint32_t xferCount = 1000;
  bandwidthSub->add_option("-c,--count", xferCount,
                           "Number of transfers to perform");
  bool bandwidthRead = false;
  bool bandwidthWrite = false;
  std::vector<uint32_t> bandwidthWidths(defaultWidths.begin(),
                                        defaultWidths.end());
  bandwidthSub->add_option("--widths", bandwidthWidths,
                           "Width of the transfers to perform (default: " +
                               defaultWidthsStr() + ")");
  bandwidthSub->add_flag("-w,--write", bandwidthWrite,
                         "Enable bandwidth write");
  bandwidthSub->add_flag("-r,--read", bandwidthRead, "Enable bandwidth read");

  CLI::App *hostmembwSub =
      cli.add_subcommand("hostmembw", "Run the host memory bandwidth test");
  uint32_t hmBwCount = 1000;
  bool hmBwRead = false;
  bool hmBwWrite = false;
  std::vector<uint32_t> hmBwWidths(defaultWidths.begin(), defaultWidths.end());
  hostmembwSub->add_option("-c,--count", hmBwCount,
                           "Number of hostmem transfers");
  hostmembwSub->add_option(
      "--widths", hmBwWidths,
      "Hostmem bandwidth widths (default: " + defaultWidthsStr() + ")");
  hostmembwSub->add_flag("-w,--write", hmBwWrite,
                         "Measure hostmem write bandwidth");
  hostmembwSub->add_flag("-r,--read", hmBwRead,
                         "Measure hostmem read bandwidth");

  CLI::App *loopbackSub =
      cli.add_subcommand("loopback", "Test LoopbackInOutAdd function service");
  uint32_t loopbackIters = 10;
  bool loopbackPipeline = false;
  loopbackSub->add_option("-i,--iters", loopbackIters,
                          "Number of function invocations (default 10)");
  loopbackSub->add_flag("-p,--pipeline", loopbackPipeline,
                        "Pipeline all calls then collect results");

  CLI::App *aggBwSub = cli.add_subcommand(
      "aggbandwidth",
      "Aggregate hostmem bandwidth across four units (readmem*, writemem*)");
  uint32_t aggWidth = 512;
  uint32_t aggCount = 1000;
  bool aggRead = false;
  bool aggWrite = false;
  aggBwSub->add_option(
      "--width", aggWidth,
      "Bit width (default 512; other widths ignored if absent)");
  aggBwSub->add_option("-c,--count", aggCount, "Flits per unit (default 1000)");
  aggBwSub->add_flag("-r,--read", aggRead, "Include read units");
  aggBwSub->add_flag("-w,--write", aggWrite, "Include write units");

  CLI::App *streamingAddSub = cli.add_subcommand(
      "streaming_add", "Test StreamingAdder function service with list input");
  uint32_t streamingAddAmt = 5;
  uint32_t streamingNumItems = 5;
  bool streamingTranslate = false;
  streamingAddSub->add_option("-a,--add", streamingAddAmt,
                              "Amount to add to each element (default 5)");
  streamingAddSub->add_option("-n,--num-items", streamingNumItems,
                              "Number of random items in the list (default 5)");
  streamingAddSub->add_flag("-t,--translate", streamingTranslate,
                            "Use message translation (list translation)");

  CLI::App *coordTranslateSub = cli.add_subcommand(
      "translate_coords",
      "Test CoordTranslator function service with list of coordinates");
  uint32_t coordXTrans = 10;
  uint32_t coordYTrans = 20;
  uint32_t coordNumItems = 5;
  coordTranslateSub->add_option("-x,--x-translation", coordXTrans,
                                "X translation amount (default 10)");
  coordTranslateSub->add_option("-y,--y-translation", coordYTrans,
                                "Y translation amount (default 20)");
  coordTranslateSub->add_option("-n,--num-coords", coordNumItems,
                                "Number of random coordinates (default 5)");

  if (int rc = cli.esiParse(argc, argv))
    return rc;
  if (!cli.get_help_ptr()->empty())
    return 0;

  Context &ctxt = cli.getContext();
  AcceleratorConnection *acc = cli.connect();
  try {
    const auto &info = *acc->getService<services::SysInfo>();
    ctxt.getLogger().info("esitester", "Connected to accelerator.");
    Manifest manifest(ctxt, info.getJsonManifest());
    Accelerator *accel = manifest.buildAccelerator(*acc);
    ctxt.getLogger().info("esitester", "Built accelerator.");
    acc->getServiceThread()->addPoll(*accel);

    if (*callback_test) {
      callbackTest(acc, accel, cb_iters);
    } else if (*hostmemtestSub) {
      hostmemTest(acc, accel, hostmemWidths, hmWrite, hmRead);
    } else if (*loopbackSub) {
      loopbackAddTest(acc, accel, loopbackIters, loopbackPipeline);
    } else if (*dmatestSub) {
      dmaTest(acc, accel, dmaWidths, dmaRead, dmaWrite);
    } else if (*bandwidthSub) {
      bandwidthTest(acc, accel, bandwidthWidths, xferCount, bandwidthRead,
                    bandwidthWrite);
    } else if (*hostmembwSub) {
      hostmemBandwidthTest(acc, accel, hmBwCount, hmBwWidths, hmBwRead,
                           hmBwWrite);
    } else if (*aggBwSub) {
      aggregateHostmemBandwidthTest(acc, accel, aggWidth, aggCount, aggRead,
                                    aggWrite);
    } else if (*streamingAddSub) {
      if (streamingTranslate)
        streamingAddTranslatedTest(acc, accel, streamingAddAmt,
                                   streamingNumItems);
      else
        streamingAddTest(acc, accel, streamingAddAmt, streamingNumItems);
    } else if (*coordTranslateSub) {
      coordTranslateTest(acc, accel, coordXTrans, coordYTrans, coordNumItems);
    }

    acc->disconnect();
  } catch (std::exception &e) {
    ctxt.getLogger().error("esitester", e.what());
    acc->disconnect();
    return -1;
  }
  std::cout << "Exiting successfully\n";
  return 0;
}

static void callbackTest(AcceleratorConnection *conn, Accelerator *accel,
                         uint32_t iterations) {
  auto cb_test = accel->getChildren().find(AppID("cb_test"));
  if (cb_test == accel->getChildren().end())
    throw std::runtime_error("No cb_test child found in accelerator");
  auto &ports = cb_test->second->getPorts();
  auto cmd_port = ports.find(AppID("cmd"));
  if (cmd_port == ports.end())
    throw std::runtime_error("No cmd port found in cb_test child");
  auto *cmdMMIO = cmd_port->second.getAs<services::MMIO::MMIORegion>();
  if (!cmdMMIO)
    throw std::runtime_error("cb_test cmd port is not MMIO");

  auto f = ports.find(AppID("cb"));
  if (f == ports.end())
    throw std::runtime_error("No cb port found in accelerator");

  auto *callPort = f->second.getAs<services::CallService::Callback>();
  if (!callPort)
    throw std::runtime_error("cb port is not a CallService::Callback");

  std::atomic<uint32_t> callbackCount = 0;
  callPort->connect(
      [conn, &callbackCount](const MessageData &data) mutable -> MessageData {
        callbackCount.fetch_add(1);
        conn->getLogger().debug(
            [&](std::string &subsystem, std::string &msg,
                std::unique_ptr<std::map<std::string, std::any>> &details) {
              subsystem = "ESITESTER";
              msg = "Received callback";
              details = std::make_unique<std::map<std::string, std::any>>();
              details->emplace("data", data);
            });
        std::cout << "callback: " << *data.as<uint64_t>() << std::endl;
        return MessageData();
      },
      true);

  for (uint32_t i = 0; i < iterations; ++i) {
    conn->getLogger().info("esitester", "Issuing callback command iteration " +
                                            std::to_string(i) + "/" +
                                            std::to_string(iterations));
    cmdMMIO->write(0x10, i); // Command the callback
    // Wait up to 1 second for the callback to be invoked.
    for (uint32_t wait = 0; wait < 1000; ++wait) {
      if (callbackCount.load() > i)
        break;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (callbackCount.load() <= i)
      throw std::runtime_error("Callback test failed. No callback received");
  }
}

/// Test the hostmem write functionality.
static void hostmemWriteTest(Accelerator *acc,
                             esi::services::HostMem::HostMemRegion &region,
                             uint32_t width) {
  std::cout << "Running hostmem WRITE test with width " << width << std::endl;
  uint64_t *dataPtr = static_cast<uint64_t *>(region.getPtr());
  auto check = [&](bool print) {
    bool ret = true;
    for (size_t i = 0; i < 9; ++i) {
      if (print)
        printf("[write] dataPtr[%zu] = 0x%016lx\n", i, dataPtr[i]);
      if (i < (width + 63) / 64 && dataPtr[i] == 0xFFFFFFFFFFFFFFFFull)
        ret = false;
    }
    return ret;
  };

  auto writeMemChildIter = acc->getChildren().find(AppID("writemem", width));
  if (writeMemChildIter == acc->getChildren().end())
    throw std::runtime_error(
        "hostmem write test failed. No writemem child found");
  auto &writeMemPorts = writeMemChildIter->second->getPorts();

  auto cmdPortIter = writeMemPorts.find(AppID("cmd", width));
  if (cmdPortIter == writeMemPorts.end())
    throw std::runtime_error(
        "hostmem write test failed. No (cmd,width) MMIO port");
  auto *cmdMMIO = cmdPortIter->second.getAs<services::MMIO::MMIORegion>();
  if (!cmdMMIO)
    throw std::runtime_error(
        "hostmem write test failed. (cmd,width) port not MMIO");

  auto issuedPortIter = writeMemPorts.find(AppID("addrCmdIssued"));
  if (issuedPortIter == writeMemPorts.end())
    throw std::runtime_error(
        "hostmem write test failed. addrCmdIssued missing");
  auto *addrCmdIssuedPort =
      issuedPortIter->second.getAs<services::TelemetryService::Metric>();
  if (!addrCmdIssuedPort)
    throw std::runtime_error(
        "hostmem write test failed. addrCmdIssued not telemetry");
  addrCmdIssuedPort->connect();

  auto responsesPortIter = writeMemPorts.find(AppID("addrCmdResponses"));
  if (responsesPortIter == writeMemPorts.end())
    throw std::runtime_error(
        "hostmem write test failed. addrCmdResponses missing");
  auto *addrCmdResponsesPort =
      responsesPortIter->second.getAs<services::TelemetryService::Metric>();
  if (!addrCmdResponsesPort)
    throw std::runtime_error(
        "hostmem write test failed. addrCmdResponses not telemetry");
  addrCmdResponsesPort->connect();

  for (size_t i = 0, e = 9; i < e; ++i)
    dataPtr[i] = 0xFFFFFFFFFFFFFFFFull;
  region.flush();
  cmdMMIO->write(0x10, reinterpret_cast<uint64_t>(region.getDevicePtr()));
  cmdMMIO->write(0x18, 1);
  cmdMMIO->write(0x20, 1);
  bool done = false;
  for (int i = 0; i < 100; ++i) {
    auto issued = addrCmdIssuedPort->readInt();
    auto responses = addrCmdResponsesPort->readInt();
    if (issued == 1 && responses == 1) {
      done = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  if (!done) {
    check(true);
    throw std::runtime_error("hostmem write test (" + std::to_string(width) +
                             " bits) timeout waiting for completion");
  }
  if (!check(true))
    throw std::runtime_error("hostmem write test failed (" +
                             std::to_string(width) + " bits)");
}

static void hostmemReadTest(Accelerator *acc,
                            esi::services::HostMem::HostMemRegion &region,
                            uint32_t width) {
  std::cout << "Running hostmem READ test with width " << width << std::endl;
  auto readMemChildIter = acc->getChildren().find(AppID("readmem", width));
  if (readMemChildIter == acc->getChildren().end())
    throw std::runtime_error(
        "hostmem read test failed. No readmem child found");

  auto &readMemPorts = readMemChildIter->second->getPorts();
  auto addrCmdPortIter = readMemPorts.find(AppID("cmd", width));
  if (addrCmdPortIter == readMemPorts.end())
    throw std::runtime_error(
        "hostmem read test failed. No AddressCommand MMIO port");
  auto *addrCmdMMIO =
      addrCmdPortIter->second.getAs<services::MMIO::MMIORegion>();
  if (!addrCmdMMIO)
    throw std::runtime_error(
        "hostmem read test failed. AddressCommand port not MMIO");

  auto lastReadPortIter = readMemPorts.find(AppID("lastReadLSB"));
  if (lastReadPortIter == readMemPorts.end())
    throw std::runtime_error("hostmem read test failed. lastReadLSB missing");
  auto *lastReadPort =
      lastReadPortIter->second.getAs<services::TelemetryService::Metric>();
  if (!lastReadPort)
    throw std::runtime_error(
        "hostmem read test failed. lastReadLSB not telemetry");
  lastReadPort->connect();

  auto issuedPortIter = readMemPorts.find(AppID("addrCmdIssued"));
  if (issuedPortIter == readMemPorts.end())
    throw std::runtime_error("hostmem read test failed. addrCmdIssued missing");
  auto *addrCmdIssuedPort =
      issuedPortIter->second.getAs<services::TelemetryService::Metric>();
  if (!addrCmdIssuedPort)
    throw std::runtime_error(
        "hostmem read test failed. addrCmdIssued not telemetry");
  addrCmdIssuedPort->connect();

  auto responsesPortIter = readMemPorts.find(AppID("addrCmdResponses"));
  if (responsesPortIter == readMemPorts.end())
    throw std::runtime_error(
        "hostmem read test failed. addrCmdResponses missing");
  auto *addrCmdResponsesPort =
      responsesPortIter->second.getAs<services::TelemetryService::Metric>();
  if (!addrCmdResponsesPort)
    throw std::runtime_error(
        "hostmem read test failed. addrCmdResponses not telemetry");
  addrCmdResponsesPort->connect();

  for (size_t i = 0; i < 8; ++i) {
    auto *dataPtr = static_cast<uint64_t *>(region.getPtr());
    dataPtr[0] = 0x12345678ull << i;
    dataPtr[1] = 0xDEADBEEFull << i;
    region.flush();
    addrCmdMMIO->write(0x10, reinterpret_cast<uint64_t>(region.getDevicePtr()));
    addrCmdMMIO->write(0x18, 1);
    addrCmdMMIO->write(0x20, 1);
    bool done = false;
    for (int waitLoop = 0; waitLoop < 100; ++waitLoop) {
      auto issued = addrCmdIssuedPort->readInt();
      auto responses = addrCmdResponsesPort->readInt();
      if (issued == 1 && responses == 1) {
        done = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!done)
      throw std::runtime_error("hostmem read (" + std::to_string(width) +
                               " bits) timeout waiting for completion");
    uint64_t captured = lastReadPort->readInt();
    uint64_t expected = dataPtr[0];
    if (width < 64)
      expected &= ((1ull << width) - 1);
    if (captured != expected)
      throw std::runtime_error("hostmem read test (" + std::to_string(width) +
                               " bits) failed. Expected " +
                               esi::toHex(expected) + ", got " +
                               esi::toHex(captured));
  }
}

static void hostmemTest(AcceleratorConnection *conn, Accelerator *acc,
                        const std::vector<uint32_t> &widths, bool write,
                        bool read) {
  // Enable the host memory service.
  auto hostmem = conn->getService<services::HostMem>();
  hostmem->start();
  auto scratchRegion = hostmem->allocate(/*size(bytes)=*/1024 * 1024,
                                         /*memOpts=*/{.writeable = true});
  uint64_t *dataPtr = static_cast<uint64_t *>(scratchRegion->getPtr());
  conn->getLogger().info("esitester",
                         "Running host memory test with region size " +
                             std::to_string(scratchRegion->getSize()) +
                             " bytes at 0x" + toHex(dataPtr));
  for (size_t i = 0; i < scratchRegion->getSize() / 8; ++i)
    dataPtr[i] = 0;
  scratchRegion->flush();

  bool passed = true;
  for (size_t width : widths) {
    try {
      if (write)
        hostmemWriteTest(acc, *scratchRegion, width);
      if (read)
        hostmemReadTest(acc, *scratchRegion, width);
    } catch (std::exception &e) {
      conn->getLogger().error("esitester", "Hostmem test failed for width " +
                                               std::to_string(width) + ": " +
                                               e.what());
      passed = false;
    }
  }
  if (!passed)
    throw std::runtime_error("Hostmem test failed");
  std::cout << "Hostmem test passed" << std::endl;
}

static void dmaReadTest(AcceleratorConnection *conn, Accelerator *acc,
                        size_t width) {
  Logger &logger = conn->getLogger();
  logger.info("esitester",
              "== Running DMA read test with width " + std::to_string(width));
  AppIDPath lastPath;
  BundlePort *toHostMMIOPort =
      acc->resolvePort({AppID("tohostdma", width), AppID("cmd")}, lastPath);
  if (!toHostMMIOPort)
    throw std::runtime_error("dma read test failed. No tohostdma[" +
                             std::to_string(width) + "] found");
  auto *toHostMMIO = toHostMMIOPort->getAs<services::MMIO::MMIORegion>();
  if (!toHostMMIO)
    throw std::runtime_error("dma read test failed. MMIO port is not MMIO");
  lastPath.clear();
  BundlePort *outPortBundle =
      acc->resolvePort({AppID("tohostdma", width), AppID("out")}, lastPath);
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
  std::cout << "  DMA read test for " << width << " bits passed" << std::endl;
}

static void dmaWriteTest(AcceleratorConnection *conn, Accelerator *acc,
                         size_t width) {
  Logger &logger = conn->getLogger();
  logger.info("esitester",
              "Running DMA write test with width " + std::to_string(width));
  AppIDPath lastPath;
  BundlePort *fromHostMMIOPort =
      acc->resolvePort({AppID("fromhostdma", width), AppID("cmd")}, lastPath);
  if (!fromHostMMIOPort)
    throw std::runtime_error("dma read test for " + toString(width) +
                             " bits failed. No fromhostdma[" +
                             std::to_string(width) + "] found");
  auto *fromHostMMIO = fromHostMMIOPort->getAs<services::MMIO::MMIORegion>();
  if (!fromHostMMIO)
    throw std::runtime_error("dma write test for " + toString(width) +
                             " bits failed. MMIO port is not MMIO");
  lastPath.clear();
  BundlePort *outPortBundle =
      acc->resolvePort({AppID("fromhostdma", width), AppID("in")}, lastPath);
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
  std::cout << "  DMA write test for " << width << " bits passed" << std::endl;
}

static void dmaTest(AcceleratorConnection *conn, Accelerator *acc,
                    const std::vector<uint32_t> &widths, bool read,
                    bool write) {
  bool success = true;
  if (write)
    for (size_t width : widths)
      try {
        dmaWriteTest(conn, acc, width);
      } catch (std::exception &e) {
        success = false;
        std::cerr << "DMA write test for " << width
                  << " bits failed: " << e.what() << std::endl;
      }
  if (read)
    for (size_t width : widths)
      dmaReadTest(conn, acc, width);
  if (!success)
    throw std::runtime_error("DMA test failed");
  std::cout << "DMA test passed" << std::endl;
}

//
// DMA bandwidth test
//

static void bandwidthReadTest(AcceleratorConnection *conn, Accelerator *acc,
                              size_t width, size_t xferCount) {

  AppIDPath lastPath;
  BundlePort *toHostMMIOPort =
      acc->resolvePort({AppID("tohostdma", width), AppID("cmd")}, lastPath);
  if (!toHostMMIOPort)
    throw std::runtime_error("bandwidth test failed. No tohostdma[" +
                             std::to_string(width) + "] found");
  auto *toHostMMIO = toHostMMIOPort->getAs<services::MMIO::MMIORegion>();
  if (!toHostMMIO)
    throw std::runtime_error("bandwidth test failed. MMIO port is not MMIO");
  lastPath.clear();
  BundlePort *outPortBundle =
      acc->resolvePort({AppID("tohostdma", width), AppID("out")}, lastPath);
  ReadChannelPort &outPort = outPortBundle->getRawRead("data");
  outPort.connect();

  Logger &logger = conn->getLogger();
  logger.info("esitester", "Starting read bandwidth test with " +
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
  double bytesPerSec =
      (double)xferCount * (width / 8.0) * 1e6 / (double)duration.count();
  logger.info("esitester",
              "  Bandwidth test: " + std::to_string(xferCount) + " x " +
                  std::to_string(width) + " bit transfers in " +
                  std::to_string(duration.count()) + " microseconds");
  logger.info("esitester", "    bandwidth: " + formatBandwidth(bytesPerSec));
}

static void bandwidthWriteTest(AcceleratorConnection *conn, Accelerator *acc,
                               size_t width, size_t xferCount) {

  AppIDPath lastPath;
  BundlePort *fromHostMMIOPort =
      acc->resolvePort({AppID("fromhostdma", width), AppID("cmd")}, lastPath);
  if (!fromHostMMIOPort)
    throw std::runtime_error("bandwidth test failed. No fromhostdma[" +
                             std::to_string(width) + "] found");
  auto *fromHostMMIO = fromHostMMIOPort->getAs<services::MMIO::MMIORegion>();
  if (!fromHostMMIO)
    throw std::runtime_error("bandwidth test failed. MMIO port is not MMIO");
  lastPath.clear();
  BundlePort *inPortBundle =
      acc->resolvePort({AppID("fromhostdma", width), AppID("in")}, lastPath);
  WriteChannelPort &outPort = inPortBundle->getRawWrite("data");
  outPort.connect();

  Logger &logger = conn->getLogger();
  logger.info("esitester", "Starting write bandwidth test with " +
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
  double bytesPerSec =
      (double)xferCount * (width / 8.0) * 1e6 / (double)duration.count();
  logger.info("esitester",
              "  Bandwidth test: " + std::to_string(xferCount) + " x " +
                  std::to_string(width) + " bit transfers in " +
                  std::to_string(duration.count()) + " microseconds");
  logger.info("esitester", "    bandwidth: " + formatBandwidth(bytesPerSec));
}

static void bandwidthTest(AcceleratorConnection *conn, Accelerator *acc,
                          const std::vector<uint32_t> &widths,
                          uint32_t xferCount, bool read, bool write) {
  if (read)
    for (uint32_t w : widths)
      bandwidthReadTest(conn, acc, w, xferCount);
  if (write)
    for (uint32_t w : widths)
      bandwidthWriteTest(conn, acc, w, xferCount);
}

//
// Hostmem bandwidth test
//

static void
hostmemWriteBandwidthTest(AcceleratorConnection *conn, Accelerator *acc,
                          esi::services::HostMem::HostMemRegion &region,
                          uint32_t width, uint32_t xferCount) {
  Logger &logger = conn->getLogger();
  logger.info("esitester", "Starting hostmem WRITE bandwidth test: " +
                               std::to_string(xferCount) + " x " +
                               std::to_string(width) + " bits");

  auto writeMemChildIter = acc->getChildren().find(AppID("writemem", width));
  if (writeMemChildIter == acc->getChildren().end())
    throw std::runtime_error("hostmem write bandwidth: writemem child missing");
  auto &writeMemPorts = writeMemChildIter->second->getPorts();

  auto cmdPortIter = writeMemPorts.find(AppID("cmd", width));
  if (cmdPortIter == writeMemPorts.end())
    throw std::runtime_error("hostmem write bandwidth: cmd MMIO missing");
  auto *cmdMMIO = cmdPortIter->second.getAs<services::MMIO::MMIORegion>();
  if (!cmdMMIO)
    throw std::runtime_error("hostmem write bandwidth: cmd not MMIO");

  auto issuedIter = writeMemPorts.find(AppID("addrCmdIssued"));
  auto respIter = writeMemPorts.find(AppID("addrCmdResponses"));
  auto cycleCount = writeMemPorts.find(AppID("addrCmdCycles"));
  if (issuedIter == writeMemPorts.end() || respIter == writeMemPorts.end() ||
      cycleCount == writeMemPorts.end())
    throw std::runtime_error("hostmem write bandwidth: telemetry missing");
  auto *issuedPort =
      issuedIter->second.getAs<services::TelemetryService::Metric>();
  auto *respPort = respIter->second.getAs<services::TelemetryService::Metric>();
  auto *cyclePort =
      cycleCount->second.getAs<services::TelemetryService::Metric>();
  if (!issuedPort || !respPort || !cyclePort)
    throw std::runtime_error(
        "hostmem write bandwidth: telemetry type mismatch");

  issuedPort->connect();
  respPort->connect();
  cyclePort->connect();

  // Initialize pattern (optional).
  uint64_t *dataPtr = static_cast<uint64_t *>(region.getPtr());
  size_t words = region.getSize() / 8;
  for (size_t i = 0; i < words; ++i)
    dataPtr[i] = i + 0xA5A50000;
  region.flush();

  auto start = std::chrono::high_resolution_clock::now();
  // Fire off xferCount write commands (one flit each).
  uint64_t devPtr = reinterpret_cast<uint64_t>(region.getDevicePtr());
  cmdMMIO->write(0x10, devPtr);    // address
  cmdMMIO->write(0x18, xferCount); // flits
  cmdMMIO->write(0x20, 1);         // start

  // Wait for responses counter to reach target.
  bool completed = false;
  for (int wait = 0; wait < 100000; ++wait) {
    uint64_t respNow = respPort->readInt();
    if (respNow == xferCount) {
      completed = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  if (!completed)
    throw std::runtime_error("hostmem write bandwidth timeout");
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock::now() - start);
  double bytesPerSec =
      (double)xferCount * (width / 8.0) * 1e6 / (double)duration.count();
  uint64_t cycles = cyclePort->readInt();
  double bytesPerCycle = (double)xferCount * (width / 8.0) / (double)cycles;
  std::cout << "[WRITE] Hostmem bandwidth (" << std::to_string(width)
            << "): " << formatBandwidth(bytesPerSec) << " "
            << std::to_string(xferCount) << " flits in "
            << std::to_string(duration.count()) << " us, "
            << std::to_string(cycles) << " cycles, " << bytesPerCycle
            << " bytes/cycle" << std::endl;
}

static void
hostmemReadBandwidthTest(AcceleratorConnection *conn, Accelerator *acc,
                         esi::services::HostMem::HostMemRegion &region,
                         uint32_t width, uint32_t xferCount) {
  Logger &logger = conn->getLogger();
  logger.info("esitester", "Starting hostmem READ bandwidth test: " +
                               std::to_string(xferCount) + " x " +
                               std::to_string(width) + " bits");

  auto readMemChildIter = acc->getChildren().find(AppID("readmem", width));
  if (readMemChildIter == acc->getChildren().end())
    throw std::runtime_error("hostmem read bandwidth: readmem child missing");
  auto &readMemPorts = readMemChildIter->second->getPorts();

  auto cmdPortIter = readMemPorts.find(AppID("cmd", width));
  if (cmdPortIter == readMemPorts.end())
    throw std::runtime_error("hostmem read bandwidth: cmd MMIO missing");
  auto *cmdMMIO = cmdPortIter->second.getAs<services::MMIO::MMIORegion>();
  if (!cmdMMIO)
    throw std::runtime_error("hostmem read bandwidth: cmd not MMIO");

  auto issuedIter = readMemPorts.find(AppID("addrCmdIssued"));
  auto respIter = readMemPorts.find(AppID("addrCmdResponses"));
  auto cyclePort = readMemPorts.find(AppID("addrCmdCycles"));
  if (issuedIter == readMemPorts.end() || respIter == readMemPorts.end() ||
      cyclePort == readMemPorts.end())
    throw std::runtime_error("hostmem read bandwidth: telemetry missing");
  auto *issuedPort =
      issuedIter->second.getAs<services::TelemetryService::Metric>();
  auto *respPort = respIter->second.getAs<services::TelemetryService::Metric>();
  auto *cycleCntPort =
      cyclePort->second.getAs<services::TelemetryService::Metric>();
  if (!issuedPort || !respPort || !cycleCntPort)
    throw std::runtime_error("hostmem read bandwidth: telemetry type mismatch");
  issuedPort->connect();
  respPort->connect();
  cycleCntPort->connect();

  // Prepare memory pattern (optional).
  uint64_t *dataPtr = static_cast<uint64_t *>(region.getPtr());
  size_t words64 = region.getSize() / 8;
  for (size_t i = 0; i < words64; ++i)
    dataPtr[i] = 0xCAFEBABE0000ull + i;
  region.flush();
  uint64_t devPtr = reinterpret_cast<uint64_t>(region.getDevicePtr());
  auto start = std::chrono::high_resolution_clock::now();

  cmdMMIO->write(0x10, devPtr);
  cmdMMIO->write(0x18, xferCount);
  cmdMMIO->write(0x20, 1);

  bool timeout = true;
  for (int wait = 0; wait < 100000; ++wait) {
    uint64_t respNow = respPort->readInt();
    if (respNow == xferCount) {
      timeout = false;
      break;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  if (timeout)
    throw std::runtime_error("hostmem read bandwidth timeout");
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock::now() - start);
  double bytesPerSec =
      (double)xferCount * (width / 8.0) * 1e6 / (double)duration.count();
  uint64_t cycles = cycleCntPort->readInt();
  double bytesPerCycle = (double)xferCount * (width / 8.0) / (double)cycles;
  std::cout << "[ READ] Hostmem bandwidth (" << width
            << "): " << formatBandwidth(bytesPerSec) << ", " << xferCount
            << " flits in " << duration.count() << " us, " << cycles
            << " cycles, " << bytesPerCycle << " bytes/cycle" << std::endl;
}

static void hostmemBandwidthTest(AcceleratorConnection *conn, Accelerator *acc,
                                 uint32_t xferCount,
                                 const std::vector<uint32_t> &widths, bool read,
                                 bool write) {
  auto hostmemSvc = conn->getService<services::HostMem>();
  hostmemSvc->start();
  auto region = hostmemSvc->allocate(/*size(bytes)=*/1024 * 1024 * 1024,
                                     /*memOpts=*/{.writeable = true});
  for (uint32_t w : widths) {
    if (write)
      hostmemWriteBandwidthTest(conn, acc, *region, w, xferCount);
    if (read)
      hostmemReadBandwidthTest(conn, acc, *region, w, xferCount);
  }
}

static void loopbackAddTest(AcceleratorConnection *conn, Accelerator *accel,
                            uint32_t iterations, bool pipeline) {
  Logger &logger = conn->getLogger();
  auto loopbackChild = accel->getChildren().find(AppID("loopback"));
  if (loopbackChild == accel->getChildren().end())
    throw std::runtime_error("Loopback test: no 'loopback' child");
  auto &ports = loopbackChild->second->getPorts();
  auto addIter = ports.find(AppID("add"));
  if (addIter == ports.end())
    throw std::runtime_error("Loopback test: no 'add' port");

  // Use FuncService::Func instead of raw channels.
  auto *funcPort = addIter->second.getAs<services::FuncService::Function>();
  if (!funcPort)
    throw std::runtime_error(
        "Loopback test: 'add' port not a FuncService::Function");
  funcPort->connect();
  if (iterations == 0) {
    logger.info("esitester", "Loopback add test: 0 iterations (skipped)");
    return;
  }
  std::mt19937_64 rng(0xC0FFEE);
  std::uniform_int_distribution<uint32_t> dist(0, (1u << 24) - 1);

  if (!pipeline) {
    auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < iterations; ++i) {
      uint32_t argVal = dist(rng);
      uint32_t expected = (argVal + 11) & 0xFFFF;
      uint8_t argBytes[3] = {
          static_cast<uint8_t>(argVal & 0xFF),
          static_cast<uint8_t>((argVal >> 8) & 0xFF),
          static_cast<uint8_t>((argVal >> 16) & 0xFF),
      };
      MessageData argMsg(argBytes, 3);
      MessageData resMsg = funcPort->call(argMsg).get();
      uint16_t got = *resMsg.as<uint16_t>();
      std::cout << "[loopback] i=" << i << " arg=0x" << esi::toHex(argVal)
                << " got=0x" << esi::toHex(got) << " exp=0x"
                << esi::toHex(expected) << std::endl;
      if (got != expected)
        throw std::runtime_error("Loopback mismatch (non-pipelined)");
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();
    double callsPerSec = (double)iterations * 1e6 / (double)us;
    logger.info("esitester", "Loopback add test passed (non-pipelined, " +
                                 std::to_string(iterations) + " calls, " +
                                 std::to_string(us) + " us, " +
                                 std::to_string(callsPerSec) + " calls/s)");
  } else {
    // Pipelined mode: launch all calls first, then collect.
    std::vector<std::future<MessageData>> futures;
    futures.reserve(iterations);
    std::vector<uint32_t> expectedVals;
    expectedVals.reserve(iterations);

    auto issueStart = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < iterations; ++i) {
      uint32_t argVal = dist(rng);
      uint32_t expected = (argVal + 11) & 0xFFFF;
      uint8_t argBytes[3] = {
          static_cast<uint8_t>(argVal & 0xFF),
          static_cast<uint8_t>((argVal >> 8) & 0xFF),
          static_cast<uint8_t>((argVal >> 16) & 0xFF),
      };
      futures.emplace_back(funcPort->call(MessageData(argBytes, 3)));
      expectedVals.emplace_back(expected);
    }
    auto issueEnd = std::chrono::high_resolution_clock::now();

    for (uint32_t i = 0; i < iterations; ++i) {
      MessageData resMsg = futures[i].get();
      uint16_t got = *resMsg.as<uint16_t>();
      uint16_t exp = (uint16_t)expectedVals[i];
      std::cout << "[loopback-pipelined] i=" << i << " got=0x"
                << esi::toHex(got) << " exp=0x" << esi::toHex(exp) << std::endl;
      if (got != exp)
        throw std::runtime_error("Loopback mismatch (pipelined) idx=" +
                                 std::to_string(i));
    }
    auto collectEnd = std::chrono::high_resolution_clock::now();

    auto issueUs = std::chrono::duration_cast<std::chrono::microseconds>(
                       issueEnd - issueStart)
                       .count();
    auto totalUs = std::chrono::duration_cast<std::chrono::microseconds>(
                       collectEnd - issueStart)
                       .count();

    double issueRate = (double)iterations * 1e6 / (double)issueUs;
    double completionRate = (double)iterations * 1e6 / (double)totalUs;

    logger.info("esitester", "Loopback add test passed (pipelined). Issued " +
                                 std::to_string(iterations) + " in " +
                                 std::to_string(issueUs) + " us (" +
                                 std::to_string(issueRate) +
                                 " calls/s), total " + std::to_string(totalUs) +
                                 " us (" + std::to_string(completionRate) +
                                 " calls/s effective)");
  }
}

static void aggregateHostmemBandwidthTest(AcceleratorConnection *conn,
                                          Accelerator *acc, uint32_t width,
                                          uint32_t xferCount, bool read,
                                          bool write) {
  Logger &logger = conn->getLogger();
  if (!read && !write) {
    std::cout << "aggbandwidth: nothing to do (enable --read and/or --write)\n";
    return;
  }
  logger.info(
      "esitester",
      "Aggregate hostmem bandwidth start width=" + std::to_string(width) +
          " count=" + std::to_string(xferCount) +
          " read=" + (read ? "Y" : "N") + " write=" + (write ? "Y" : "N"));

  auto hostmemSvc = conn->getService<services::HostMem>();
  hostmemSvc->start();

  struct Unit {
    std::string prefix;
    bool isRead = false;
    bool isWrite = false;
    std::unique_ptr<esi::services::HostMem::HostMemRegion> region;
    services::TelemetryService::Metric *resp = nullptr;
    services::TelemetryService::Metric *cycles = nullptr;
    services::MMIO::MMIORegion *cmd = nullptr;
    bool launched = false;
    bool done = false;
    uint64_t bytes = 0;
    uint64_t duration_us = 0;
    uint64_t cycleCount = 0;
    std::chrono::high_resolution_clock::time_point start;
  };
  std::vector<Unit> units;
  const std::vector<std::string> readPrefixes = {"readmem", "readmem_0",
                                                 "readmem_1", "readmem_2"};
  const std::vector<std::string> writePrefixes = {"writemem", "writemem_0",
                                                  "writemem_1", "writemem_2"};

  auto addUnits = [&](const std::vector<std::string> &pref, bool doRead,
                      bool doWrite) {
    for (auto &p : pref) {
      AppID id(p, width);
      auto childIt = acc->getChildren().find(id);
      if (childIt == acc->getChildren().end())
        continue; // silently skip missing variants
      auto &ports = childIt->second->getPorts();
      auto cmdIt = ports.find(AppID("cmd", width));
      auto respIt = ports.find(AppID("addrCmdResponses"));
      auto cycIt = ports.find(AppID("addrCmdCycles"));
      if (cmdIt == ports.end() || respIt == ports.end() || cycIt == ports.end())
        continue;
      auto *cmd = cmdIt->second.getAs<services::MMIO::MMIORegion>();
      auto *resp = respIt->second.getAs<services::TelemetryService::Metric>();
      auto *cyc = cycIt->second.getAs<services::TelemetryService::Metric>();
      if (!cmd || !resp || !cyc)
        continue;
      resp->connect();
      cyc->connect();
      Unit u;
      u.prefix = p;
      u.isRead = doRead;
      u.isWrite = doWrite;
      u.region = hostmemSvc->allocate(1024 * 1024 * 1024, {.writeable = true});
      // Init pattern.
      uint64_t *ptr = static_cast<uint64_t *>(u.region->getPtr());
      size_t words = u.region->getSize() / 8;
      for (size_t i = 0; i < words; ++i)
        ptr[i] =
            (p[0] == 'w' ? (0xA5A500000000ull + i) : (0xCAFEBABE0000ull + i));
      u.region->flush();
      u.cmd = cmd;
      u.resp = resp;
      u.cycles = cyc;
      u.bytes = uint64_t(xferCount) * (width / 8);
      units.emplace_back(std::move(u));
    }
  };
  if (read)
    addUnits(readPrefixes, true, false);
  if (write)
    addUnits(writePrefixes, false, true);
  if (units.empty()) {
    std::cout << "aggbandwidth: no matching units present for width " << width
              << "\n";
    return;
  }

  auto wallStart = std::chrono::high_resolution_clock::now();
  // Launch sequentially.
  for (auto &u : units) {
    uint64_t devPtr = reinterpret_cast<uint64_t>(u.region->getDevicePtr());
    u.cmd->write(0x10, devPtr);
    u.cmd->write(0x18, xferCount);
    u.cmd->write(0x20, 1);
    u.start = std::chrono::high_resolution_clock::now();
    u.launched = true;
  }

  // Poll all until complete.
  const uint64_t timeoutLoops = 200000; // ~10s at 50us sleep
  uint64_t loops = 0;
  while (true) {
    bool allDone = true;
    for (auto &u : units) {
      if (u.done)
        continue;
      if (u.resp->readInt() == xferCount) {
        auto end = std::chrono::high_resolution_clock::now();
        u.duration_us =
            std::chrono::duration_cast<std::chrono::microseconds>(end - u.start)
                .count();
        u.cycleCount = u.cycles->readInt();
        u.done = true;
      } else {
        allDone = false;
      }
    }
    if (allDone)
      break;
    if (++loops >= timeoutLoops)
      throw std::runtime_error("aggbandwidth: timeout");
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  auto wallUs = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - wallStart)
                    .count();

  uint64_t totalBytes = 0;
  uint64_t totalReadBytes = 0;
  uint64_t totalWriteBytes = 0;
  for (auto &u : units) {
    totalBytes += u.bytes;
    if (u.isRead)
      totalReadBytes += u.bytes;
    if (u.isWrite)
      totalWriteBytes += u.bytes;
    double unitBps = (double)u.bytes * 1e6 / (double)u.duration_us;
    std::cout << "[agg-unit] " << u.prefix << "[" << width << "] "
              << (u.isRead ? "READ" : (u.isWrite ? "WRITE" : "UNK"))
              << " bytes=" << humanBytes(u.bytes) << " (" << u.bytes << " B)"
              << " time=" << humanTimeUS(u.duration_us) << " (" << u.duration_us
              << " us) cycles=" << u.cycleCount
              << " throughput=" << formatBandwidth(unitBps) << std::endl;
  }
  // Compute aggregate bandwidths as total size / total wall time (not sum of
  // unit throughputs).
  double aggReadBps =
      totalReadBytes ? (double)totalReadBytes * 1e6 / (double)wallUs : 0.0;
  double aggWriteBps =
      totalWriteBytes ? (double)totalWriteBytes * 1e6 / (double)wallUs : 0.0;
  double aggCombinedBps =
      totalBytes ? (double)totalBytes * 1e6 / (double)wallUs : 0.0;

  std::cout << "[agg-total] units=" << units.size()
            << " read_bytes=" << humanBytes(totalReadBytes) << " ("
            << totalReadBytes << " B)"
            << " read_bw=" << formatBandwidth(aggReadBps)
            << " write_bytes=" << humanBytes(totalWriteBytes) << " ("
            << totalWriteBytes << " B)"
            << " write_bw=" << formatBandwidth(aggWriteBps)
            << " combined_bytes=" << humanBytes(totalBytes) << " ("
            << totalBytes << " B)"
            << " combined_bw=" << formatBandwidth(aggCombinedBps)
            << " wall_time=" << humanTimeUS(wallUs) << " (" << wallUs << " us)"
            << std::endl;
  logger.info("esitester", "Aggregate hostmem bandwidth test complete");
}

/// Packed struct representing a parallel window argument for StreamingAdder.
/// Layout in SystemVerilog (so it must be reversed in C):
///   { add_amt: UInt(32), input: UInt(32), last: UInt(8) }
#pragma pack(push, 1)
struct StreamingAddArg {
  uint8_t last;
  uint32_t input;
  uint32_t addAmt;
};
#pragma pack(pop)
static_assert(sizeof(StreamingAddArg) == 9,
              "StreamingAddArg must be 9 bytes packed");

/// Packed struct representing a parallel window result for StreamingAdder.
/// Layout in SystemVerilog (so it must be reversed in C):
///   { data: UInt(32), last: UInt(8) }
#pragma pack(push, 1)
struct StreamingAddResult {
  uint8_t last;
  uint32_t data;
};
#pragma pack(pop)
static_assert(sizeof(StreamingAddResult) == 5,
              "StreamingAddResult must be 5 bytes packed");

/// Test the StreamingAdder module. This module takes a struct containing
/// an add_amt and a list of uint32s, adds add_amt to each element, and
/// returns the resulting list. The data is streamed using windowed types.
static void streamingAddTest(AcceleratorConnection *conn, Accelerator *accel,
                             uint32_t addAmt, uint32_t numItems) {
  Logger &logger = conn->getLogger();
  logger.info("esitester", "Starting streaming add test with add_amt=" +
                               std::to_string(addAmt) +
                               ", num_items=" + std::to_string(numItems));

  // Generate random input data.
  std::mt19937 rng(0xDEADBEEF);
  std::uniform_int_distribution<uint32_t> dist(0, 1000000);
  std::vector<uint32_t> inputData;
  inputData.reserve(numItems);
  for (uint32_t i = 0; i < numItems; ++i)
    inputData.push_back(dist(rng));

  // Find the streaming_adder child.
  auto streamingAdderChild =
      accel->getChildren().find(AppID("streaming_adder"));
  if (streamingAdderChild == accel->getChildren().end())
    throw std::runtime_error(
        "Streaming add test: no 'streaming_adder' child found");

  auto &ports = streamingAdderChild->second->getPorts();
  auto addIter = ports.find(AppID("streaming_add"));
  if (addIter == ports.end())
    throw std::runtime_error(
        "Streaming add test: no 'streaming_add' port found");

  // Get the raw read/write channel ports for the windowed function.
  // The argument channel expects parallel windowed data where each message
  // contains: struct { add_amt: UInt(32), input: UInt(32), last: bool }
  WriteChannelPort &argPort = addIter->second.getRawWrite("arg");
  ReadChannelPort &resultPort = addIter->second.getRawRead("result");

  argPort.connect(ChannelPort::ConnectOptions(std::nullopt, false));
  resultPort.connect(ChannelPort::ConnectOptions(std::nullopt, false));

  // Send each list element with add_amt repeated in every message.
  for (size_t i = 0; i < inputData.size(); ++i) {
    StreamingAddArg arg;
    arg.addAmt = addAmt;
    arg.input = inputData[i];
    arg.last = (i == inputData.size() - 1) ? 1 : 0;
    argPort.write(
        MessageData(reinterpret_cast<const uint8_t *>(&arg), sizeof(arg)));
    logger.debug("esitester", "Sent {add_amt=" + std::to_string(arg.addAmt) +
                                  ", input=" + std::to_string(arg.input) +
                                  ", last=" + (arg.last ? "true" : "false") +
                                  "}");
  }

  // Read the result list (also windowed).
  std::vector<uint32_t> results;
  bool lastSeen = false;
  while (!lastSeen) {
    MessageData resMsg;
    resultPort.read(resMsg);
    if (resMsg.getSize() < sizeof(StreamingAddResult))
      throw std::runtime_error(
          "Streaming add test: unexpected result message size");

    const auto *res =
        reinterpret_cast<const StreamingAddResult *>(resMsg.getBytes());
    lastSeen = res->last != 0;
    results.push_back(res->data);
    logger.debug("esitester", "Received result=" + std::to_string(res->data) +
                                  " (last=" + (lastSeen ? "true" : "false") +
                                  ")");
  }

  // Verify results.
  if (results.size() != inputData.size())
    throw std::runtime_error(
        "Streaming add test: result size mismatch. Expected " +
        std::to_string(inputData.size()) + ", got " +
        std::to_string(results.size()));

  bool passed = true;
  std::cout << "Streaming add test results:" << std::endl;
  for (size_t i = 0; i < inputData.size(); ++i) {
    uint32_t expected = inputData[i] + addAmt;
    std::cout << "  input[" << i << "]=" << inputData[i] << " + " << addAmt
              << " = " << results[i] << " (expected " << expected << ")";
    if (results[i] != expected) {
      std::cout << " MISMATCH!";
      passed = false;
    }
    std::cout << std::endl;
  }

  argPort.disconnect();
  resultPort.disconnect();

  if (!passed)
    throw std::runtime_error("Streaming add test failed: result mismatch");

  logger.info("esitester", "Streaming add test passed");
  std::cout << "Streaming add test passed" << std::endl;
}

/// Test the StreamingAdder module using message translation.
/// This version uses the list translation support where the message format is:
///   Argument: { add_amt (4 bytes), input_length (8 bytes), input_data[] }
///   Result: { data_length (8 bytes), data[] }
/// The translation layer automatically converts between this format and the
/// parallel windowed frames used by the hardware.

/// Translated argument struct for StreamingAdder.
/// Memory layout (standard C struct ordering, fields in declaration order):
///   ESI type: struct { add_amt: UInt(32), input: List<UInt(32)> }
/// becomes host struct:
///   { input_length (size_t, 8 bytes on 64-bit), add_amt (uint32_t),
///   input_data[] }
/// Note: The translation layer handles the conversion between this C struct
/// layout and the hardware's SystemVerilog frame format.
/// Note: size_t is used for list lengths, so this format is platform-dependent.
#pragma pack(push, 1)
struct StreamingAddTranslatedArg {
  size_t inputLength;
  uint32_t addAmt;
  // Trailing array data follows immediately after the struct in memory.
  // Use inputData() accessor to access it.

  /// Get pointer to trailing input data array.
  uint32_t *inputData() { return reinterpret_cast<uint32_t *>(this + 1); }
  const uint32_t *inputData() const {
    return reinterpret_cast<const uint32_t *>(this + 1);
  }
  /// Get span view of input data (requires inputLength to be set first).
  std::span<uint32_t> inputDataSpan() { return {inputData(), inputLength}; }
  std::span<const uint32_t> inputDataSpan() const {
    return {inputData(), inputLength};
  }

  static size_t allocSize(size_t numItems) {
    return sizeof(StreamingAddTranslatedArg) + numItems * sizeof(uint32_t);
  }
};
#pragma pack(pop)

/// Translated result struct for StreamingAdder.
/// Memory layout:
///   struct { data: List<UInt(32)> }
/// becomes:
///   { data_length (size_t, 8 bytes on 64-bit), data[] }
#pragma pack(push, 1)
struct StreamingAddTranslatedResult {
  size_t dataLength;
  // Trailing array data follows immediately after the struct in memory.

  /// Get pointer to trailing result data array.
  uint32_t *data() { return reinterpret_cast<uint32_t *>(this + 1); }
  const uint32_t *data() const {
    return reinterpret_cast<const uint32_t *>(this + 1);
  }
  /// Get span view of result data (requires dataLength to be set first).
  std::span<uint32_t> dataSpan() { return {data(), dataLength}; }
  std::span<const uint32_t> dataSpan() const { return {data(), dataLength}; }

  static size_t allocSize(size_t numItems) {
    return sizeof(StreamingAddTranslatedResult) + numItems * sizeof(uint32_t);
  }
};
#pragma pack(pop)

static void streamingAddTranslatedTest(AcceleratorConnection *conn,
                                       Accelerator *accel, uint32_t addAmt,
                                       uint32_t numItems) {
  Logger &logger = conn->getLogger();
  logger.info("esitester",
              "Starting streaming add test (translated) with add_amt=" +
                  std::to_string(addAmt) +
                  ", num_items=" + std::to_string(numItems));

  // Generate random input data.
  std::mt19937 rng(0xDEADBEEF);
  std::uniform_int_distribution<uint32_t> dist(0, 1000000);
  std::vector<uint32_t> inputData;
  inputData.reserve(numItems);
  for (uint32_t i = 0; i < numItems; ++i)
    inputData.push_back(dist(rng));

  // Find the streaming_adder child.
  auto streamingAdderChild =
      accel->getChildren().find(AppID("streaming_adder"));
  if (streamingAdderChild == accel->getChildren().end())
    throw std::runtime_error(
        "Streaming add test: no 'streaming_adder' child found");

  auto &ports = streamingAdderChild->second->getPorts();
  auto addIter = ports.find(AppID("streaming_add"));
  if (addIter == ports.end())
    throw std::runtime_error(
        "Streaming add test: no 'streaming_add' port found");

  // Get the raw read/write channel ports with translation enabled (default).
  WriteChannelPort &argPort = addIter->second.getRawWrite("arg");
  ReadChannelPort &resultPort = addIter->second.getRawRead("result");

  // Connect with translation enabled (the default).
  argPort.connect();
  resultPort.connect();

  // Allocate the argument struct with proper alignment for the struct members.
  // We use aligned_alloc to ensure the buffer meets alignment requirements.
  size_t argSize = StreamingAddTranslatedArg::allocSize(numItems);
  constexpr size_t alignment = alignof(StreamingAddTranslatedArg);
  // aligned_alloc requires size to be a multiple of alignment
  size_t allocSize = ((argSize + alignment - 1) / alignment) * alignment;
  void *argRaw = alignedAllocCompat(alignment, allocSize);
  if (!argRaw)
    throw std::bad_alloc();
  auto argDeleter = [](void *p) { alignedFreeCompat(p); };
  std::unique_ptr<void, decltype(argDeleter)> argBuffer(argRaw, argDeleter);
  auto *arg = static_cast<StreamingAddTranslatedArg *>(argRaw);
  arg->inputLength = numItems;
  arg->addAmt = addAmt;
  for (uint32_t i = 0; i < numItems; ++i)
    arg->inputData()[i] = inputData[i];

  logger.debug("esitester",
               "Sending translated argument: " + std::to_string(argSize) +
                   " bytes, list_length=" + std::to_string(arg->inputLength) +
                   ", add_amt=" + std::to_string(arg->addAmt));

  // Send the complete message - translation will split it into frames.
  argPort.write(MessageData(reinterpret_cast<const uint8_t *>(arg), argSize));
  // argBuffer automatically freed when it goes out of scope

  // Read the translated result.
  MessageData resMsg;
  resultPort.read(resMsg);

  logger.debug("esitester", "Received translated result: " +
                                std::to_string(resMsg.getSize()) + " bytes");

  if (resMsg.getSize() < sizeof(StreamingAddTranslatedResult))
    throw std::runtime_error(
        "Streaming add test (translated): result too small");

  const auto *result =
      reinterpret_cast<const StreamingAddTranslatedResult *>(resMsg.getBytes());

  if (resMsg.getSize() <
      StreamingAddTranslatedResult::allocSize(result->dataLength))
    throw std::runtime_error(
        "Streaming add test (translated): result data truncated");

  // Verify results.
  if (result->dataLength != inputData.size())
    throw std::runtime_error(
        "Streaming add test (translated): result size mismatch. Expected " +
        std::to_string(inputData.size()) + ", got " +
        std::to_string(result->dataLength));

  bool passed = true;
  std::cout << "Streaming add test results:" << std::endl;
  for (size_t i = 0; i < inputData.size(); ++i) {
    uint32_t expected = inputData[i] + addAmt;
    std::cout << "  input[" << i << "]=" << inputData[i] << " + " << addAmt
              << " = " << result->data()[i] << " (expected " << expected << ")";
    if (result->data()[i] != expected) {
      std::cout << " MISMATCH!";
      passed = false;
    }
    std::cout << std::endl;
  }

  argPort.disconnect();
  resultPort.disconnect();

  if (!passed)
    throw std::runtime_error(
        "Streaming add test (translated) failed: result mismatch");

  logger.info("esitester", "Streaming add test passed (translated)");
  std::cout << "Streaming add test passed" << std::endl;
}

/// Test the CoordTranslator module using message translation.
/// This version uses the list translation support where the message format is:
///   Argument: { x_translation, y_translation, coords_length, coords[] }
///   Result: { coords_length, coords[] }
/// Each coord is a struct { x, y }.

/// Coordinate struct for CoordTranslator.
/// SV ordering means y comes before x in memory.
#pragma pack(push, 1)
struct Coord {
  uint32_t y; // SV ordering: last declared field first in memory
  uint32_t x;
};
#pragma pack(pop)
static_assert(sizeof(Coord) == 8, "Coord must be 8 bytes packed");

/// Translated argument struct for CoordTranslator.
/// Memory layout (standard C struct ordering):
///   ESI type: struct { x_translation: UInt(32), y_translation: UInt(32),
///                      coords: List<struct{x, y}> }
/// becomes host struct:
///   { coords_length (size_t, 8 bytes on 64-bit), y_translation (uint32_t),
///     x_translation (uint32_t), coords[] }
/// Note: Fields are in reverse order due to SV struct ordering.
/// Note: size_t is used for list lengths, so this format is platform-dependent.
#pragma pack(push, 1)
struct CoordTranslateArg {
  size_t coordsLength;
  uint32_t yTranslation; // SV ordering: last declared field first in memory
  uint32_t xTranslation;
  // Trailing array data follows immediately after the struct in memory.

  /// Get pointer to trailing coords array.
  Coord *coords() { return reinterpret_cast<Coord *>(this + 1); }
  const Coord *coords() const {
    return reinterpret_cast<const Coord *>(this + 1);
  }
  /// Get span view of coords (requires coordsLength to be set first).
  std::span<Coord> coordsSpan() { return {coords(), coordsLength}; }
  std::span<const Coord> coordsSpan() const { return {coords(), coordsLength}; }

  static size_t allocSize(size_t numCoords) {
    return sizeof(CoordTranslateArg) + numCoords * sizeof(Coord);
  }
};
#pragma pack(pop)

/// Translated result struct for CoordTranslator.
/// Memory layout:
///   ESI type: List<struct{x, y}>
/// becomes host struct:
///   { coords_length (size_t, 8 bytes on 64-bit), coords[] }
#pragma pack(push, 1)
struct CoordTranslateResult {
  size_t coordsLength;
  // Trailing array data follows immediately after the struct in memory.

  /// Get pointer to trailing coords array.
  Coord *coords() { return reinterpret_cast<Coord *>(this + 1); }
  const Coord *coords() const {
    return reinterpret_cast<const Coord *>(this + 1);
  }
  /// Get span view of coords (requires coordsLength to be set first).
  std::span<Coord> coordsSpan() { return {coords(), coordsLength}; }
  std::span<const Coord> coordsSpan() const { return {coords(), coordsLength}; }

  static size_t allocSize(size_t numCoords) {
    return sizeof(CoordTranslateResult) + numCoords * sizeof(Coord);
  }
};
#pragma pack(pop)

static void coordTranslateTest(AcceleratorConnection *conn, Accelerator *accel,
                               uint32_t xTrans, uint32_t yTrans,
                               uint32_t numCoords) {
  Logger &logger = conn->getLogger();
  logger.info("esitester", "Starting coord translate test with x_trans=" +
                               std::to_string(xTrans) +
                               ", y_trans=" + std::to_string(yTrans) +
                               ", num_coords=" + std::to_string(numCoords));

  // Generate random input coordinates.
  // Note: Coord struct has y before x due to SV ordering, but we generate
  // and display as (x, y) for human readability.
  std::mt19937 rng(0xDEADBEEF);
  std::uniform_int_distribution<uint32_t> dist(0, 1000000);
  std::vector<Coord> inputCoords;
  inputCoords.reserve(numCoords);
  for (uint32_t i = 0; i < numCoords; ++i) {
    Coord c;
    c.x = dist(rng);
    c.y = dist(rng);
    inputCoords.push_back(c);
  }

  // Find the coord_translator child.
  auto coordTranslatorChild =
      accel->getChildren().find(AppID("coord_translator"));
  if (coordTranslatorChild == accel->getChildren().end())
    throw std::runtime_error(
        "Coord translate test: no 'coord_translator' child found");

  auto &ports = coordTranslatorChild->second->getPorts();
  auto translateIter = ports.find(AppID("translate_coords"));
  if (translateIter == ports.end())
    throw std::runtime_error(
        "Coord translate test: no 'translate_coords' port found");

  // Use FuncService::Function which handles connection and translation.
  auto *funcPort =
      translateIter->second.getAs<services::FuncService::Function>();
  if (!funcPort)
    throw std::runtime_error(
        "Coord translate test: 'translate_coords' port not a "
        "FuncService::Function");
  funcPort->connect();

  // Allocate the argument struct with proper alignment for the struct members.
  size_t argSize = CoordTranslateArg::allocSize(numCoords);
  constexpr size_t alignment = alignof(CoordTranslateArg);
  // aligned_alloc requires size to be a multiple of alignment
  size_t allocSize = ((argSize + alignment - 1) / alignment) * alignment;
  void *argRaw = alignedAllocCompat(alignment, allocSize);
  if (!argRaw)
    throw std::bad_alloc();
  auto argDeleter = [](void *p) { alignedFreeCompat(p); };
  std::unique_ptr<void, decltype(argDeleter)> argBuffer(argRaw, argDeleter);
  auto *arg = static_cast<CoordTranslateArg *>(argRaw);
  arg->coordsLength = numCoords;
  arg->xTranslation = xTrans;
  arg->yTranslation = yTrans;
  for (uint32_t i = 0; i < numCoords; ++i)
    arg->coords()[i] = inputCoords[i];

  logger.debug(
      "esitester",
      "Sending coord translate argument: " + std::to_string(argSize) +
          " bytes, coords_length=" + std::to_string(arg->coordsLength) +
          ", x_trans=" + std::to_string(arg->xTranslation) +
          ", y_trans=" + std::to_string(arg->yTranslation));

  // Call the function - translation happens automatically.
  MessageData resMsg =
      funcPort
          ->call(MessageData(reinterpret_cast<const uint8_t *>(arg), argSize))
          .get();
  // argBuffer automatically freed when it goes out of scope

  logger.debug("esitester", "Received coord translate result: " +
                                std::to_string(resMsg.getSize()) + " bytes");

  if (resMsg.getSize() < sizeof(CoordTranslateResult))
    throw std::runtime_error("Coord translate test: result too small");

  const auto *result =
      reinterpret_cast<const CoordTranslateResult *>(resMsg.getBytes());

  if (resMsg.getSize() < CoordTranslateResult::allocSize(result->coordsLength))
    throw std::runtime_error("Coord translate test: result data truncated");

  // Verify results.
  if (result->coordsLength != inputCoords.size())
    throw std::runtime_error(
        "Coord translate test: result size mismatch. Expected " +
        std::to_string(inputCoords.size()) + ", got " +
        std::to_string(result->coordsLength));

  bool passed = true;
  std::cout << "Coord translate test results:" << std::endl;
  for (size_t i = 0; i < inputCoords.size(); ++i) {
    uint32_t expectedX = inputCoords[i].x + xTrans;
    uint32_t expectedY = inputCoords[i].y + yTrans;
    std::cout << "  coord[" << i << "]=(" << inputCoords[i].x << ","
              << inputCoords[i].y << ") + (" << xTrans << "," << yTrans
              << ") = (" << result->coords()[i].x << ","
              << result->coords()[i].y << ")";
    if (result->coords()[i].x != expectedX ||
        result->coords()[i].y != expectedY) {
      std::cout << " MISMATCH! (expected (" << expectedX << "," << expectedY
                << "))";
      passed = false;
    }
    std::cout << std::endl;
  }

  if (!passed)
    throw std::runtime_error("Coord translate test failed: result mismatch");

  logger.info("esitester", "Coord translate test passed");
  std::cout << "Coord translate test passed" << std::endl;
}
