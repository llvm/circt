//===- Ports.cpp - ESI communication channels  -------------------------===//
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

#include "esi/Ports.h"

#include <chrono>
#include <stdexcept>

using namespace esi;

BundlePort::BundlePort(AppID id, const BundleType *type, PortMap channels)
    : id(id), type(type), channels(channels) {}

WriteChannelPort &BundlePort::getRawWrite(const std::string &name) const {
  auto f = channels.find(name);
  if (f == channels.end())
    throw std::runtime_error("Channel '" + name + "' not found");
  auto *write = dynamic_cast<WriteChannelPort *>(&f->second);
  if (!write)
    throw std::runtime_error("Channel '" + name + "' is not a write channel");
  return *write;
}

ReadChannelPort &BundlePort::getRawRead(const std::string &name) const {
  auto f = channels.find(name);
  if (f == channels.end())
    throw std::runtime_error("Channel '" + name + "' not found");
  auto *read = dynamic_cast<ReadChannelPort *>(&f->second);
  if (!read)
    throw std::runtime_error("Channel '" + name + "' is not a read channel");
  return *read;
}
void ReadChannelPort::connect(std::function<bool(MessageData)> callback,
                              std::optional<unsigned> bufferSize) {
  if (mode != Mode::Disconnected)
    throw std::runtime_error("Channel already connected");
  this->callback = callback;
  connectImpl(bufferSize);
  mode = Mode::Callback;
}

void ReadChannelPort::connect(std::optional<unsigned> bufferSize) {
  maxDataQueueMsgs = DefaultMaxDataQueueMsgs;
  this->callback = [this](MessageData data) {
    std::scoped_lock<std::mutex> lock(pollingM);
    assert(!(!promiseQueue.empty() && !dataQueue.empty()) &&
           "Both queues are in use.");

    if (!promiseQueue.empty()) {
      // If there are promises waiting, fulfill the first one.
      std::promise<MessageData> p = std::move(promiseQueue.front());
      promiseQueue.pop();
      p.set_value(std::move(data));
    } else {
      // If not, add it to the data queue, unless the queue is full.
      if (dataQueue.size() >= maxDataQueueMsgs && maxDataQueueMsgs != 0)
        return false;
      dataQueue.push(std::move(data));
    }
    return true;
  };
  connectImpl(bufferSize);
  mode = Mode::Polling;
}

std::future<MessageData> ReadChannelPort::readAsync() {
  if (mode == Mode::Callback)
    throw std::runtime_error(
        "Cannot read from a callback channel. `connect()` without a callback "
        "specified to use polling mode.");

  std::scoped_lock<std::mutex> lock(pollingM);
  assert(!(!promiseQueue.empty() && !dataQueue.empty()) &&
         "Both queues are in use.");

  if (!dataQueue.empty()) {
    // If there's data available, fulfill the promise immediately.
    std::promise<MessageData> p;
    std::future<MessageData> f = p.get_future();
    p.set_value(std::move(dataQueue.front()));
    dataQueue.pop();
    return f;
  } else {
    // Otherwise, add a promise to the queue and return the future.
    promiseQueue.emplace();
    return promiseQueue.back().get_future();
  }
}
