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
#include "esi/Types.h"

#include <chrono>
#include <cstring>
#include <map>
#include <stdexcept>

using namespace esi;

ChannelPort::ChannelPort(const Type *type) {
  if (auto chanType = dynamic_cast<const ChannelType *>(type))
    type = chanType->getInner();
  translationType = dynamic_cast<const WindowType *>(type);
  if (translationType)
    this->type = translationType->getIntoType();
  else
    this->type = type;
}

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
                              const ConnectOptions &options) {
  if (mode != Mode::Disconnected)
    throw std::runtime_error("Channel already connected");

  if (options.translateMessage && translationType) {
    this->callback = [this, cb = std::move(callback)](MessageData data) {
      if (translateIncoming(data))
        return cb(MessageData(std::move(translationBuffer)));
      return true;
    };
  } else {
    this->callback = callback;
  }
  connectImpl(options);
  mode = Mode::Callback;
}

void ReadChannelPort::connect(const ConnectOptions &options) {
  maxDataQueueMsgs = DefaultMaxDataQueueMsgs;
  bool translate = options.translateMessage && translationType;
  this->callback = [this, translate](MessageData data) {
    if (translate) {
      if (!translateIncoming(data))
        return true;
      data = MessageData(std::move(translationBuffer));
    }

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
  connectImpl(options);
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

//===----------------------------------------------------------------------===//
// Translation of incoming data for window types.
//===----------------------------------------------------------------------===//

bool ReadChannelPort::translateIncoming(MessageData &data) {
  assert(translationType &&
         "Translation type must be set for window translation.");

  const auto &frames = translationType->getFrames();
  const Type *intoType = translationType->getIntoType();

  // Frames arrive sequentially in order. We track which frame we're expecting
  // with nextFrameIndex.
  size_t numFrames = frames.size();

  // Get the frame data directly.
  const uint8_t *frameData = data.getBytes();
  size_t frameDataSize = data.size();

  // Get the frame metadata for the current expected frame.
  const WindowType::Frame &frame = frames[nextFrameIndex];

  // Check if this is the first frame of a new message.
  // If so, we need to initialize the translation buffer.
  bool isFirstFrame = (nextFrameIndex == 0);

  if (isFirstFrame) {
    // Initialize the translation buffer to hold the complete intoType.
    // The buffer will be filled in as frames arrive.
    std::ptrdiff_t intoTypeBits = intoType->getBitWidth();
    if (intoTypeBits < 0)
      throw std::runtime_error(
          "Cannot translate window with dynamically-sized intoType");
    size_t intoTypeBytes = (static_cast<size_t>(intoTypeBits) + 7) / 8;
    translationBuffer.resize(intoTypeBytes, 0);
  }

  // Get the intoType as a struct to find field offsets.
  const StructType *intoStruct = dynamic_cast<const StructType *>(intoType);
  if (!intoStruct)
    throw std::runtime_error(
        "Window intoType must be a struct for translation");

  // Build a map of field name to (offset, type) in the intoType.
  // Offset is in bytes from the start of the struct.
  // SystemVerilog ordering: last field is at LSB (offset 0).
  const auto &intoFields = intoStruct->getFields();
  std::map<std::string, std::pair<size_t, const Type *>> fieldInfo;
  size_t currentOffset = 0;

  // If the struct is reversed (SV ordering), iterate in reverse to compute
  // offsets.
  if (intoStruct->isReverse()) {
    for (auto it = intoFields.rbegin(); it != intoFields.rend(); ++it) {
      const auto &[name, fieldType] = *it;
      std::ptrdiff_t fieldBits = fieldType->getBitWidth();
      if (fieldBits < 0)
        throw std::runtime_error("Cannot translate field with dynamic size: " +
                                 name);
      size_t fieldBytes = (static_cast<size_t>(fieldBits) + 7) / 8;
      fieldInfo[name] = {currentOffset, fieldType};
      currentOffset += fieldBytes;
    }
  } else {
    for (const auto &[name, fieldType] : intoFields) {
      std::ptrdiff_t fieldBits = fieldType->getBitWidth();
      if (fieldBits < 0)
        throw std::runtime_error("Cannot translate field with dynamic size: " +
                                 name);
      size_t fieldBytes = (static_cast<size_t>(fieldBits) + 7) / 8;
      fieldInfo[name] = {currentOffset, fieldType};
      currentOffset += fieldBytes;
    }
  }

  // Copy each field from the frame into the appropriate position in the
  // translation buffer. The frame struct also follows SV ordering.
  size_t frameOffset = 0;
  // Frame fields are in the order specified in the window definition.
  // We iterate them in reverse for SV ordering (last field at LSB).
  for (auto fieldIt = frame.fields.rbegin(); fieldIt != frame.fields.rend();
       ++fieldIt) {
    const WindowType::Field &field = *fieldIt;
    auto infoIt = fieldInfo.find(field.name);
    if (infoIt == fieldInfo.end())
      throw std::runtime_error("Frame field '" + field.name +
                               "' not found in intoType");

    const auto &[destOffset, fieldType] = infoIt->second;
    std::ptrdiff_t fieldBits = fieldType->getBitWidth();
    size_t fieldBytes = (static_cast<size_t>(fieldBits) + 7) / 8;

    // Check bounds.
    if (frameOffset + fieldBytes > frameDataSize)
      throw std::runtime_error("Frame data too small for field: " + field.name);
    if (destOffset + fieldBytes > translationBuffer.size())
      throw std::runtime_error("Translation buffer too small for field: " +
                               field.name);

    // Copy the field data.
    std::memcpy(translationBuffer.data() + destOffset, frameData + frameOffset,
                fieldBytes);
    frameOffset += fieldBytes;
  }

  // Move to the next frame.
  nextFrameIndex++;

  // Check if all frames have been received.
  if (nextFrameIndex >= numFrames) {
    // Reset for the next message.
    nextFrameIndex = 0;
    return true;
  }

  return false;
}
