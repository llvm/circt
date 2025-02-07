//===- Ports.h - ESI communication channels ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef ESI_PORTS_H
#define ESI_PORTS_H

#include "esi/Common.h"
#include "esi/Types.h"
#include "esi/Utils.h"

#include <cassert>
#include <future>

namespace esi {

class ChannelPort;
using PortMap = std::map<std::string, ChannelPort &>;

/// Unidirectional channels are the basic communication primitive between the
/// host and accelerator. A 'ChannelPort' is the host side of a channel. It can
/// be either read or write but not both. At this level, channels are untyped --
/// just streams of bytes. They are not intended to be used directly by users
/// but used by higher level APIs which add types.
class ChannelPort {
public:
  ChannelPort(const Type *type) : type(type) {}
  virtual ~ChannelPort() {}

  /// Set up a connection to the accelerator. The buffer size is optional and
  /// should be considered merely a hint. Individual implementations use it
  /// however they like. The unit is number of messages of the port type.
  virtual void connect(std::optional<unsigned> bufferSize = std::nullopt) = 0;
  virtual void disconnect() = 0;
  virtual bool isConnected() const = 0;

  /// Poll for incoming data. Returns true if data was read or written into a
  /// buffer as a result of the poll. Calling the call back could (will) also
  /// happen in that case. Some backends need this to be called periodically. In
  /// the usual case, this will be called by a background thread, but the ESI
  /// runtime does not want to assume that the host processes use standard
  /// threads. If the user wants to provide their own threads, they need to call
  /// this on each port occasionally. This is also called from the 'master' poll
  /// method in the Accelerator class.
  bool poll() {
    if (isConnected())
      return pollImpl();
    return false;
  }

  const Type *getType() const { return type; }

protected:
  const Type *type;

  /// Method called by poll() to actually poll the channel if the channel is
  /// connected.
  virtual bool pollImpl() { return false; }

  /// Called by all connect methods to let backends initiate the underlying
  /// connections.
  virtual void connectImpl(std::optional<unsigned> bufferSize) {}
};

/// A ChannelPort which sends data to the accelerator.
class WriteChannelPort : public ChannelPort {
public:
  using ChannelPort::ChannelPort;

  virtual void
  connect(std::optional<unsigned> bufferSize = std::nullopt) override {
    connectImpl(bufferSize);
    connected = true;
  }
  virtual void disconnect() override { connected = false; }
  virtual bool isConnected() const override { return connected; }

  /// A very basic blocking write API. Will likely change for performance
  /// reasons.
  virtual void write(const MessageData &) = 0;

  /// A basic non-blocking write API. Returns true if the data was written.
  /// It is invalid for backends to always return false (i.e. backends must
  /// eventually ensure that writes may succeed).
  virtual bool tryWrite(const MessageData &data) = 0;

private:
  volatile bool connected = false;
};

/// Instantiated when a backend does not know how to create a write channel.
class UnknownWriteChannelPort : public WriteChannelPort {
public:
  UnknownWriteChannelPort(const Type *type, std::string errmsg)
      : WriteChannelPort(type), errmsg(errmsg) {}

  void connect(std::optional<unsigned> bufferSize = std::nullopt) override {
    throw std::runtime_error(errmsg);
  }
  void write(const MessageData &) override { throw std::runtime_error(errmsg); }
  bool tryWrite(const MessageData &) override {
    throw std::runtime_error(errmsg);
  }

protected:
  std::string errmsg;
};

/// A ChannelPort which reads data from the accelerator. It has two modes:
/// Callback and Polling which cannot be used at the same time. The mode is set
/// at connect() time. To change the mode, disconnect() and then connect()
/// again.
class ReadChannelPort : public ChannelPort {

public:
  ReadChannelPort(const Type *type)
      : ChannelPort(type), mode(Mode::Disconnected) {}
  virtual void disconnect() override { mode = Mode::Disconnected; }
  virtual bool isConnected() const override {
    return mode != Mode::Disconnected;
  }

  //===--------------------------------------------------------------------===//
  // Callback mode: To use a callback, connect with a callback function which
  // will get called with incoming data. This function can be called from any
  // thread. It shall return true to indicate that the data was consumed. False
  // if it could not accept the data and should be tried again at some point in
  // the future. Callback is not allowed to block and needs to execute quickly.
  //
  // TODO: Have the callback return something upon which the caller can check,
  // wait, and be notified.
  //===--------------------------------------------------------------------===//

  virtual void connect(std::function<bool(MessageData)> callback,
                       std::optional<unsigned> bufferSize = std::nullopt);

  //===--------------------------------------------------------------------===//
  // Polling mode methods: To use futures or blocking reads, connect without any
  // arguments. You will then be able to use readAsync() or read().
  //===--------------------------------------------------------------------===//

  /// Default max data queue size set at connect time.
  static constexpr uint64_t DefaultMaxDataQueueMsgs = 32;

  /// Connect to the channel in polling mode.
  virtual void
  connect(std::optional<unsigned> bufferSize = std::nullopt) override;

  /// Asynchronous read.
  virtual std::future<MessageData> readAsync();

  /// Specify a buffer to read into. Blocking. Basic API, will likely change
  /// for performance and functionality reasons.
  virtual void read(MessageData &outData) {
    std::future<MessageData> f = readAsync();
    f.wait();
    outData = std::move(f.get());
  }

  /// Set maximum number of messages to store in the dataQueue. 0 means no
  /// limit. This is only used in polling mode and is set to default of 32 upon
  /// connect. While it may seem redundant to have this and bufferSize, there
  /// may be (and are) backends which have a very small amount of memory which
  /// are accelerator accessible and want to move messages out as quickly as
  /// possible.
  void setMaxDataQueueMsgs(uint64_t maxMsgs) { maxDataQueueMsgs = maxMsgs; }

protected:
  /// Indicates the current mode of the channel.
  enum Mode { Disconnected, Callback, Polling };
  volatile Mode mode;

  /// Backends call this callback when new data is available.
  std::function<bool(MessageData)> callback;

  //===--------------------------------------------------------------------===//
  // Polling mode members.
  //===--------------------------------------------------------------------===//

  /// Mutex to protect the two queues used for polling.
  std::mutex pollingM;
  /// Store incoming data here if there are no outstanding promises to be
  /// fulfilled.
  std::queue<MessageData> dataQueue;
  /// Maximum number of messages to store in dataQueue. 0 means no limit.
  uint64_t maxDataQueueMsgs;
  /// Promises to be fulfilled when data is available.
  std::queue<std::promise<MessageData>> promiseQueue;
};

/// Instantiated when a backend does not know how to create a read channel.
class UnknownReadChannelPort : public ReadChannelPort {
public:
  UnknownReadChannelPort(const Type *type, std::string errmsg)
      : ReadChannelPort(type), errmsg(errmsg) {}

  void connect(std::function<bool(MessageData)> callback,
               std::optional<unsigned> bufferSize = std::nullopt) override {
    throw std::runtime_error(errmsg);
  }
  void connect(std::optional<unsigned> bufferSize = std::nullopt) override {
    throw std::runtime_error(errmsg);
  }
  std::future<MessageData> readAsync() override {
    throw std::runtime_error(errmsg);
  }

protected:
  std::string errmsg;
};

/// Services provide connections to 'bundles' -- collections of named,
/// unidirectional communication channels. This class provides access to those
/// ChannelPorts.
class BundlePort {
public:
  /// Compute the direction of a channel given the bundle direction and the
  /// bundle port's direction.
  static bool isWrite(BundleType::Direction bundleDir) {
    return bundleDir == BundleType::Direction::To;
  }

  /// Construct a port.
  BundlePort(AppID id, const BundleType *type, PortMap channels);
  virtual ~BundlePort() = default;

  /// Get the ID of the port.
  AppID getID() const { return id; }

  /// Get access to the raw byte streams of a channel. Intended for internal
  /// usage and binding to other languages (e.g. Python) which have their own
  /// message serialization code. Exposed publicly as an escape hatch, but
  /// ordinary users should not use. You have been warned.
  WriteChannelPort &getRawWrite(const std::string &name) const;
  ReadChannelPort &getRawRead(const std::string &name) const;
  const PortMap &getChannels() const { return channels; }

  /// Cast this Bundle port to a subclass which is actually useful. Returns
  /// nullptr if the cast fails.
  // TODO: this probably shouldn't be 'const', but bundle ports' user access are
  // const. Change that.
  template <typename T>
  T *getAs() const {
    return const_cast<T *>(dynamic_cast<const T *>(this));
  }

  /// Calls `poll` on all channels in the bundle and returns true if any of them
  /// returned true.
  bool poll() {
    bool result = false;
    for (auto &channel : channels)
      result |= channel.second.poll();
    return result;
  }

protected:
  AppID id;
  const BundleType *type;
  PortMap channels;
};

} // namespace esi

#endif // ESI_PORTS_H
