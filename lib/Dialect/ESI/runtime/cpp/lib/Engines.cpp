//===- Engines.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT (lib/dialect/ESI/runtime/cpp/).
//
//===----------------------------------------------------------------------===//

#include "esi/Engines.h"
#include "esi/Accelerator.h"

#include <algorithm>
#include <cstring>

using namespace esi;

//===----------------------------------------------------------------------===//
// Unknown Engine
//===----------------------------------------------------------------------===//

namespace {
/// Created by default when the DMA engine cannot be resolved. Throws the error
/// upon trying to connect and creates ports which throw errors on their
/// connection attempts.
class UnknownEngine : public Engine {
public:
  UnknownEngine(AcceleratorConnection &conn, std::string engineName)
      : Engine(conn), engineName(engineName) {}

  void connect() override {
    throw std::runtime_error("Unknown engine '" + engineName + "'");
  }

  std::unique_ptr<ChannelPort> createPort(AppIDPath idPath,
                                          const std::string &channelName,
                                          BundleType::Direction dir,
                                          const Type *type) override {
    if (BundlePort::isWrite(dir))
      return std::make_unique<UnknownWriteChannelPort>(
          type,
          "Unknown engine '" + engineName + "': cannot create write port");
    else
      return std::make_unique<UnknownReadChannelPort>(
          type, "Unknown engine '" + engineName + "': cannot create read port");
  }

protected:
  std::string engineName;
};
} // namespace

//===----------------------------------------------------------------------===//
// OneItemBuffersToHost engine
//
// Protocol:
//  1) Host sends address of buffer address via MMIO write.
//  2) Device writes data on channel with a byte '1' to said buffer address.
//  3) Host polls the last byte in buffer for '1'.
//  4) Data is copied out of buffer, last byte is set to '0', goto 1.
//===----------------------------------------------------------------------===//

namespace {
class OneItemBuffersToHost;
class OneItemBuffersToHostReadPort : public ReadChannelPort {
public:
  /// Offset into the MMIO space to which the buffer pointer is written.
  static constexpr size_t BufferPtrOffset = 8;

  OneItemBuffersToHostReadPort(const Type *type, OneItemBuffersToHost *engine,
                               AppIDPath idPath, const std::string &channelName)
      : ReadChannelPort(type), engine(engine), idPath(idPath),
        channelName(channelName) {
    bufferSize = (type->getBitWidth() / 8) + 1;
  }

  // Write the location of the buffer to the MMIO space.
  void writeBufferPtr();
  // Connect allocate a buffer and prime the pump.
  void connectImpl(const ChannelPort::ConnectOptions &options) override;
  // Check for buffer fill.
  bool pollImpl() override;

  std::string identifier() const { return idPath.toStr() + "." + channelName; }

protected:
  // Size of buffer based on type.
  size_t bufferSize;
  // Owning engine.
  OneItemBuffersToHost *engine;
  // Single buffer.
  std::unique_ptr<services::HostMem::HostMemRegion> buffer;
  // Number of times the poll function has been called.
  uint64_t pollCount = 0;

  // Identifing information.
  AppIDPath idPath;
  std::string channelName;
};

class OneItemBuffersToHost : public Engine {
  friend class OneItemBuffersToHostReadPort;

public:
  OneItemBuffersToHost(AcceleratorConnection &conn, AppIDPath idPath,
                       const ServiceImplDetails &details)
      : Engine(conn), thisPath(idPath) {
    // Get the MMIO path but don't try to resolve it yet.
    auto mmioIDIter = details.find("mmio");
    if (mmioIDIter != details.end())
      mmioID = std::any_cast<AppID>(mmioIDIter->second);
  }

  static std::unique_ptr<Engine> create(AcceleratorConnection &conn,
                                        AppIDPath idPath,
                                        const ServiceImplDetails &details,
                                        const HWClientDetails &clients) {
    return std::make_unique<OneItemBuffersToHost>(conn, idPath, details);
  }

  // Only throw errors on connect.
  void connect() override;

  // Create a real read port if requested. Create an error-throwing write port
  // if requested.
  std::unique_ptr<ChannelPort> createPort(AppIDPath idPath,
                                          const std::string &channelName,
                                          BundleType::Direction dir,
                                          const Type *type) override {
    if (BundlePort::isWrite(dir))
      return std::make_unique<UnknownWriteChannelPort>(
          type, idPath.toStr() + "." + channelName +
                    " OneItemBuffersToHost: cannot create write port");
    return std::make_unique<OneItemBuffersToHostReadPort>(type, this, idPath,
                                                          channelName);
  }

protected:
  AppIDPath thisPath;
  std::optional<AppID> mmioID;
  services::MMIO::MMIORegion *mmio;
  services::HostMem *hostMem;
};
} // namespace

void OneItemBuffersToHost::connect() {
  // This is where we throw errors.
  if (connected)
    return;
  if (!mmioID)
    throw std::runtime_error("OneItemBuffersToHost: no mmio path specified");
  hostMem = conn.getService<services::HostMem>();
  if (!hostMem)
    throw std::runtime_error("OneItemBuffersToHost: no host memory service");
  hostMem->start();

  // Resolve the MMIO port.
  Accelerator &acc = conn.getAccelerator();
  AppIDPath mmioPath = thisPath;
  mmioPath.pop_back();
  mmioPath.push_back(*mmioID);
  AppIDPath lastPath;
  BundlePort *port = acc.resolvePort(mmioPath, lastPath);
  if (port == nullptr)
    throw std::runtime_error(
        thisPath.toStr() +
        " OneItemBuffersToHost: could not find MMIO port at " +
        mmioPath.toStr());
  mmio = dynamic_cast<services::MMIO::MMIORegion *>(port);
  if (!mmio)
    throw std::runtime_error(
        thisPath.toStr() +
        " OneItemBuffersToHost: MMIO port is not an MMIO port");

  // If we have a valid MMIO port, we can connect.
  connected = true;
}

void OneItemBuffersToHostReadPort::writeBufferPtr() {
  uint8_t *bufferData = reinterpret_cast<uint8_t *>(buffer->getPtr());
  bufferData[bufferSize - 1] = 0;
  // Write the *device-visible* address of the buffer to the MMIO space.
  engine->mmio->write(BufferPtrOffset,
                      reinterpret_cast<uint64_t>(buffer->getDevicePtr()));
}

void OneItemBuffersToHostReadPort::connectImpl(
    const ChannelPort::ConnectOptions &options) {
  engine->connect();
  buffer = engine->hostMem->allocate(bufferSize, {true, false});
  writeBufferPtr();
}

bool OneItemBuffersToHostReadPort::pollImpl() {
  // Check to see if the buffer has been filled.
  uint8_t *bufferData = reinterpret_cast<uint8_t *>(buffer->getPtr());
  if (bufferData[bufferSize - 1] == 0)
    return false;

  Logger &logger = engine->conn.getLogger();

  // If it has, copy the data out. If the consumer (callback) reports that it
  // has accepted the data, re-use the buffer.
  MessageData data(bufferData, bufferSize - 1);
  logger.trace(
      [this, data](std::string &subsystem, std::string &msg,
                   std::unique_ptr<std::map<std::string, std::any>> &details) {
        subsystem = "OneItemBuffersToHost";
        msg = "recieved message";
        details = std::make_unique<std::map<std::string, std::any>>();
        (*details)["channel"] = identifier();
        (*details)["data_size"] = data.getSize();
        (*details)["message_data"] = data.toHex();
      });
  if (callback(std::move(data))) {
    writeBufferPtr();
    logger.trace(
        [this](std::string &subsystem, std::string &msg,
               std::unique_ptr<std::map<std::string, std::any>> &details) {
          subsystem = "OneItemBuffersToHost";
          msg = "callback accepted data";
          details = std::make_unique<std::map<std::string, std::any>>();
          (*details)["channel"] = identifier();
        });
    return true;
  }
  logger.trace(
      [this](std::string &subsystem, std::string &msg,
             std::unique_ptr<std::map<std::string, std::any>> &details) {
        subsystem = "OneItemBuffersToHost";
        msg = "callback rejected data";
        details = std::make_unique<std::map<std::string, std::any>>();
        (*details)["channel"] = identifier();
      });
  return false;
}

REGISTER_ENGINE("OneItemBuffersToHost", OneItemBuffersToHost);

//===----------------------------------------------------------------------===//
// OneItemBuffersFromHost engine
//
// Protocol:
//  1) Host allocates two buffers: one for data and one for the completion byte.
//  2) On write, if the completion byte is '0', the write fails since that
//     implies there's data currently in the buffer which the device has not yet
//     picked up. If the completion byte is '1', continue on.
//  3) The host writes the data to the buffer and sets the completion byte to
//     '0'.
//  4) The device reads the data into the data buffer and sets the completion
//     byte to '1'.
//  5) Host sends address of buffer via MMIO write to register 0x8.
//  6) Host sends address of completion buffer via MMIO write to register 0x10.
//  7) Device reads data from said buffer address.
//  8) Device writes '1' to the first byte of the completion buffer.
//
//===----------------------------------------------------------------------===//

namespace {

class OneItemBuffersFromHost;
class OneItemBuffersFromHostWritePort : public WriteChannelPort {
public:
  /// Offset into the MMIO space to which the buffer pointer is written.
  static constexpr size_t BufferPtrOffset = 8;
  static constexpr size_t CompletionPtrOffset = 16;

  OneItemBuffersFromHostWritePort(const Type *type,
                                  OneItemBuffersFromHost *engine,
                                  AppIDPath idPath,
                                  const std::string &channelName)
      : WriteChannelPort(type), engine(engine), idPath(idPath),
        channelName(channelName) {
    bufferSize = type->getBitWidth() / 8;
  }

  // Connect allocate a buffer and prime the pump.
  void connectImpl(const ChannelPort::ConnectOptions &options) override;

  void write(const MessageData &) override;
  bool tryWrite(const MessageData &data) override;

  std::string identifier() const { return idPath.toStr() + "." + channelName; }

protected:
  // Size of buffer based on type.
  size_t bufferSize;
  // Owning engine.
  OneItemBuffersFromHost *engine;
  // Single buffer.
  std::unique_ptr<services::HostMem::HostMemRegion> data_buffer;
  std::unique_ptr<services::HostMem::HostMemRegion> completion_buffer;

  // Thread protection.
  std::mutex bufferMutex;

  // Identifing information.
  AppIDPath idPath;
  std::string channelName;
};

class OneItemBuffersFromHost : public Engine {
  friend class OneItemBuffersFromHostWritePort;

public:
  OneItemBuffersFromHost(AcceleratorConnection &conn, AppIDPath idPath,
                         const ServiceImplDetails &details)
      : Engine(conn), thisPath(idPath) {
    // Get the MMIO path but don't try to resolve it yet.
    auto mmioIDIter = details.find("mmio");
    if (mmioIDIter != details.end())
      mmioID = std::any_cast<AppID>(mmioIDIter->second);
  }

  static std::unique_ptr<Engine> create(AcceleratorConnection &conn,
                                        AppIDPath idPath,
                                        const ServiceImplDetails &details,
                                        const HWClientDetails &clients) {
    return std::make_unique<OneItemBuffersFromHost>(conn, idPath, details);
  }

  // Only throw errors on connect.
  void connect() override;

  // Create a real write port if requested. Create an error-throwing read port
  // if requested.
  std::unique_ptr<ChannelPort> createPort(AppIDPath idPath,
                                          const std::string &channelName,
                                          BundleType::Direction dir,
                                          const Type *type) override {
    if (!BundlePort::isWrite(dir))
      return std::make_unique<UnknownReadChannelPort>(
          type, idPath.toStr() + "." + channelName +
                    " OneItemBuffersFromHost: cannot create read port");
    return std::make_unique<OneItemBuffersFromHostWritePort>(type, this, idPath,
                                                             channelName);
  }

protected:
  AppIDPath thisPath;
  std::optional<AppID> mmioID;
  services::MMIO::MMIORegion *mmio;
  services::HostMem *hostMem;
};

} // namespace

void OneItemBuffersFromHost::connect() {
  // This is where we throw errors.
  if (connected)
    return;
  if (!mmioID)
    throw std::runtime_error("OneItemBuffersFromHost: no mmio path specified");
  hostMem = conn.getService<services::HostMem>();
  if (!hostMem)
    throw std::runtime_error("OneItemBuffersFromHost: no host memory service");
  hostMem->start();

  // Resolve the MMIO port.
  Accelerator &acc = conn.getAccelerator();
  AppIDPath mmioPath = thisPath;
  mmioPath.pop_back();
  mmioPath.push_back(*mmioID);
  AppIDPath lastPath;
  BundlePort *port = acc.resolvePort(mmioPath, lastPath);
  if (port == nullptr)
    throw std::runtime_error(
        thisPath.toStr() +
        " OneItemBuffersFromHost: could not find MMIO port at " +
        mmioPath.toStr());
  mmio = dynamic_cast<services::MMIO::MMIORegion *>(port);
  if (!mmio)
    throw std::runtime_error(
        thisPath.toStr() +
        " OneItemBuffersFromHost: MMIO port is not an MMIO port");

  // If we have a valid MMIO port, we can connect.
  connected = true;
}

void OneItemBuffersFromHostWritePort::connectImpl(
    const ChannelPort::ConnectOptions &options) {
  engine->connect();
  data_buffer = engine->hostMem->allocate(std::max(bufferSize, (size_t)512),
                                          {false, false});
  completion_buffer = engine->hostMem->allocate(512, {true, false});
  // Set the last byte to '1' to indicate that the buffer is ready to be filled
  // and sent to the device.
  *static_cast<uint8_t *>(completion_buffer->getPtr()) = 1;
}

void OneItemBuffersFromHostWritePort::write(const MessageData &data) {
  while (!tryWrite(data))
    // Wait for the device to read the last data we sent.
    std::this_thread::yield();
}

bool OneItemBuffersFromHostWritePort::tryWrite(const MessageData &data) {
  Logger &logger = engine->conn.getLogger();

  // Check to see if there's an outstanding write.
  completion_buffer->flush();
  uint8_t *completion =
      reinterpret_cast<uint8_t *>(completion_buffer->getPtr());
  if (*completion == 0) {
    logger.trace(
        [this](std::string &subsystem, std::string &msg,
               std::unique_ptr<std::map<std::string, std::any>> &details) {
          subsystem = "OneItemBuffersFromHost";
          msg = identifier() + " write failed: buffer not ready";
          details = std::make_unique<std::map<std::string, std::any>>();
          (*details)["channel"] = identifier();
        });
    return false;
  }

  // If the buffer is empty, use it.
  std::lock_guard<std::mutex> lock(bufferMutex);
  void *bufferData = data_buffer->getPtr();
  std::memcpy(bufferData, data.getBytes(), data.getSize());
  data_buffer->flush();
  // Indicate that the buffer is now in use.
  *completion = 0;
  completion_buffer->flush();
  // Write the *device-visible* address of the data buffer and completion buffer
  // to the MMIO space.
  engine->mmio->write(BufferPtrOffset,
                      reinterpret_cast<uint64_t>(data_buffer->getDevicePtr()));
  engine->mmio->write(
      CompletionPtrOffset,
      reinterpret_cast<uint64_t>(completion_buffer->getDevicePtr()));

  logger.trace(
      [this, data](std::string &subsystem, std::string &msg,
                   std::unique_ptr<std::map<std::string, std::any>> &details) {
        subsystem = "OneItemBuffersFromHost";
        msg = "initiated transfer of message";
        details = std::make_unique<std::map<std::string, std::any>>();
        (*details)["channel"] = identifier();
        (*details)["data_size"] = data.getSize();
        (*details)["message_data"] = data.toHex();
      });
  return true;
}

REGISTER_ENGINE("OneItemBuffersFromHost", OneItemBuffersFromHost);

//===----------------------------------------------------------------------===//
// Engine / Bundle Engine Map
//===----------------------------------------------------------------------===//

ChannelPort &Engine::requestPort(AppIDPath idPath,
                                 const std::string &channelName,
                                 BundleType::Direction dir, const Type *type) {
  auto portIter = ownedPorts.find(std::make_pair(idPath, channelName));
  if (portIter != ownedPorts.end())
    return *portIter->second;
  std::unique_ptr<ChannelPort> port =
      createPort(idPath, channelName, dir, type);
  ChannelPort &ret = *port;
  ownedPorts.emplace(std::make_pair(idPath, channelName), std::move(port));
  return ret;
}

PortMap BundleEngineMap::requestPorts(const AppIDPath &idPath,
                                      const BundleType *bundleType) const {
  PortMap ports;
  for (auto [channelName, dir, type] : bundleType->getChannels()) {
    auto engineIter = bundleEngineMap.find(channelName);
    if (engineIter == bundleEngineMap.end())
      continue;

    ports.emplace(channelName, engineIter->second->requestPort(
                                   idPath, channelName, dir, type));
  }
  return ports;
}

void BundleEngineMap::setEngine(const std::string &channelName,
                                Engine *engine) {
  auto [it, inserted] = bundleEngineMap.try_emplace(channelName, engine);
  if (!inserted)
    throw std::runtime_error("Channel already exists in engine map");
}

//===----------------------------------------------------------------------===//
// Registry
//===----------------------------------------------------------------------===//

namespace {
class EngineRegistry {
public:
  static std::map<std::string, registry::internal::EngineCreate> &get() {
    static EngineRegistry instance;
    return instance.engineRegistry;
  }

private:
  std::map<std::string, registry::internal::EngineCreate> engineRegistry;
};
} // namespace

std::unique_ptr<Engine>
registry::createEngine(AcceleratorConnection &conn,
                       const std::string &dmaEngineName, AppIDPath idPath,
                       const ServiceImplDetails &details,
                       const HWClientDetails &clients) {
  auto &reg = EngineRegistry::get();
  auto it = reg.find(dmaEngineName);
  if (it == reg.end())
    return std::make_unique<UnknownEngine>(conn, dmaEngineName);
  return it->second(conn, idPath, details, clients);
}

void registry::internal::registerEngine(const std::string &name,
                                        EngineCreate create) {
  auto tried = EngineRegistry::get().try_emplace(name, create);
  if (!tried.second)
    throw std::runtime_error("Engine already exists in registry");
}
