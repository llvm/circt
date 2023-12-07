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

namespace esi {

/// Unidirectional channels are the basic communication primitive between the
/// host and accelerator. A 'ChannelPort' is the host side of a channel. It can
/// be either read or write but not both. At this level, channels are untyped --
/// just streams of bytes. They are not intended to be used directly by users
/// but used by higher level APIs which add types.
class ChannelPort {
public:
  ChannelPort(const Type &type) : type(type) {}
  virtual ~ChannelPort() = default;

  virtual void connect() {}
  virtual void disconnect() {}

  const Type &getType() const { return type; }

private:
  const Type &type;
};

/// A ChannelPort which sends data to the accelerator.
class WriteChannelPort : public ChannelPort {
public:
  using ChannelPort::ChannelPort;

  /// A very basic write API. Will likely change for performance reasons.
  virtual void write(const void *data, size_t size) = 0;
};

/// A ChannelPort which reads data from the accelerator.
class ReadChannelPort : public ChannelPort {
public:
  using ChannelPort::ChannelPort;

  /// Specify a buffer to read into and a maximum size to read. Returns the
  /// number of bytes read, or -1 on error. Basic API, will likely change for
  /// performance reasons.
  virtual std::ptrdiff_t read(void *data, size_t maxSize) = 0;
};

/// Services provide connections to 'bundles' -- collections of named,
/// unidirectional communication channels. This class provides access to those
/// ChannelPorts.
class BundlePort {
public:
  /// Direction of a bundle. This -- combined with the channel direction in the
  /// bundle -- can be used to determine if a channel should be writing to or
  /// reading from the accelerator.
  enum Direction { ToServer, ToClient };

  /// Compute the direction of a channel given the bundle direction and the
  /// bundle port's direction.
  static bool isWrite(BundleType::Direction bundleDir, Direction svcDir) {
    if (svcDir == Direction::ToClient)
      return bundleDir == BundleType::Direction::To;
    return bundleDir == BundleType::Direction::From;
  }

  /// Construct a port.
  BundlePort(AppID id, std::map<std::string, ChannelPort &> channels);

  /// Get the ID of the port.
  AppID getID() const { return _id; }

  /// Get access to the raw byte streams of a channel. Intended for internal
  /// usage and binding to other languages (e.g. Python) which have their own
  /// message serialization code.
  WriteChannelPort &getRawWrite(const std::string &name) const;
  ReadChannelPort &getRawRead(const std::string &name) const;
  const std::map<std::string, ChannelPort &> &getChannels() const {
    return _channels;
  }

private:
  AppID _id;
  std::map<std::string, ChannelPort &> _channels;
};

} // namespace esi

#endif // ESI_PORTS_H
