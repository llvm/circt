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
  ChannelPort(const Type *type) : type(type) {}
  virtual ~ChannelPort() = default;

  virtual void connect() {}
  virtual void disconnect() {}

  const Type *getType() const { return type; }

private:
  const Type *type;
};

/// A ChannelPort which sends data to the accelerator.
class WriteChannelPort : public ChannelPort {
public:
  using ChannelPort::ChannelPort;

  /// A very basic write API. Will likely change for performance reasons.
  virtual void write(const MessageData &) = 0;
};

/// A ChannelPort which reads data from the accelerator.
class ReadChannelPort : public ChannelPort {
public:
  using ChannelPort::ChannelPort;

  /// Specify a buffer to read into. Non-blocking. Returns true if message
  /// successfully recieved. Basic API, will likely change for performance
  /// and functionality reasons.
  virtual bool read(MessageData &) = 0;
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
  BundlePort(AppID id, std::map<std::string, ChannelPort &> channels);
  virtual ~BundlePort() = default;

  /// Get the ID of the port.
  AppID getID() const { return id; }

  /// Get access to the raw byte streams of a channel. Intended for internal
  /// usage and binding to other languages (e.g. Python) which have their own
  /// message serialization code. Exposed publicly as an escape hatch, but
  /// ordinary users should not use. You have been warned.
  WriteChannelPort &getRawWrite(const std::string &name) const;
  ReadChannelPort &getRawRead(const std::string &name) const;
  const std::map<std::string, ChannelPort &> &getChannels() const {
    return channels;
  }

private:
  AppID id;
  std::map<std::string, ChannelPort &> channels;
};

} // namespace esi

#endif // ESI_PORTS_H
