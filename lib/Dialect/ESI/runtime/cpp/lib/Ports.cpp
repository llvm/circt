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

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <stdexcept>

using namespace esi;

ChannelPort::ChannelPort(const Type *type) {
  if (auto chanType = dynamic_cast<const ChannelType *>(type))
    type = chanType->getInner();
  auto translationType = dynamic_cast<const WindowType *>(type);
  if (translationType) {
    this->type = translationType->getIntoType();
    this->translationInfo = std::make_unique<TranslationInfo>(translationType);
  } else {
    this->type = type;
    this->translationInfo = nullptr;
  }
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

  if (options.translateMessage && translationInfo) {
    translationInfo->precomputeFrameInfo();
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
  bool translate = options.translateMessage && translationInfo;
  if (translate)
    translationInfo->precomputeFrameInfo();
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
// Window translation support
//===----------------------------------------------------------------------===//

void ChannelPort::TranslationInfo::precomputeFrameInfo() {
  const Type *intoType = windowType->getIntoType();

  std::ptrdiff_t intoTypeBits = intoType->getBitWidth();
  if (intoTypeBits < 0)
    throw std::runtime_error(
        "Cannot translate window with dynamically-sized intoType");
  intoTypeBytes = (static_cast<size_t>(intoTypeBits) + 7) / 8;

  const StructType *intoStruct = dynamic_cast<const StructType *>(intoType);
  if (!intoStruct)
    throw std::runtime_error(
        "Window intoType must be a struct for translation");

  const auto &intoFields = intoStruct->getFields();
  std::map<std::string, std::pair<size_t, const Type *>> fieldMap;
  size_t currentOffset = 0;

  if (intoStruct->isReverse()) {
    for (auto it = intoFields.rbegin(); it != intoFields.rend(); ++it) {
      const auto &[name, fieldType] = *it;
      std::ptrdiff_t fieldBits = fieldType->getBitWidth();
      if (fieldBits < 0)
        throw std::runtime_error("Cannot translate field with dynamic size: " +
                                 name);
      size_t fieldBytes = (static_cast<size_t>(fieldBits) + 7) / 8;
      fieldMap[name] = {currentOffset, fieldType};
      currentOffset += fieldBytes;
    }
  } else {
    for (const auto &[name, fieldType] : intoFields) {
      std::ptrdiff_t fieldBits = fieldType->getBitWidth();
      if (fieldBits < 0)
        throw std::runtime_error("Cannot translate field with dynamic size: " +
                                 name);
      size_t fieldBytes = (static_cast<size_t>(fieldBits) + 7) / 8;
      fieldMap[name] = {currentOffset, fieldType};
      currentOffset += fieldBytes;
    }
  }

  const auto &windowFrames = windowType->getFrames();
  frames.clear();
  frames.reserve(windowFrames.size());

  for (const auto &frame : windowFrames) {
    FrameInfo frameInfo;
    size_t frameOffset = 0;

    // Iterate fields in reverse (SV ordering)
    for (auto fieldIt = frame.fields.rbegin(); fieldIt != frame.fields.rend();
         ++fieldIt) {
      const WindowType::Field &field = *fieldIt;
      auto it = fieldMap.find(field.name);
      if (it == fieldMap.end())
        throw std::runtime_error("Frame field '" + field.name +
                                 "' not found in intoType");

      size_t bufferOffset = it->second.first;
      const Type *fieldType = it->second.second;
      std::ptrdiff_t fieldBits = fieldType->getBitWidth();
      size_t fieldBytes = (static_cast<size_t>(fieldBits) + 7) / 8;

      frameInfo.copyOps.push_back({frameOffset, bufferOffset, fieldBytes});
      frameOffset += fieldBytes;
    }
    frameInfo.expectedSize = frameOffset;

    // Sort by frameOffset to ensure processing order matches frame layout.
    std::sort(frameInfo.copyOps.begin(), frameInfo.copyOps.end(),
              [](const CopyOp &a, const CopyOp &b) {
                return a.frameOffset < b.frameOffset;
              });

    // Merge adjacent ops
    if (!frameInfo.copyOps.empty()) {
      std::vector<CopyOp> mergedOps;
      mergedOps.reserve(frameInfo.copyOps.size());
      mergedOps.push_back(frameInfo.copyOps[0]);

      for (size_t i = 1; i < frameInfo.copyOps.size(); ++i) {
        CopyOp &last = mergedOps.back();
        const CopyOp &current = frameInfo.copyOps[i];

        if (last.frameOffset + last.size == current.frameOffset &&
            last.bufferOffset + last.size == current.bufferOffset) {
          last.size += current.size;
        } else {
          mergedOps.push_back(current);
        }
      }

      frameInfo.copyOps = std::move(mergedOps);
    }

    frames.push_back(std::move(frameInfo));
  }
}

bool ReadChannelPort::translateIncoming(MessageData &data) {
  assert(translationInfo &&
         "Translation type must be set for window translation.");

  // Frames arrive sequentially in order. We track which frame we're expecting
  // with nextFrameIndex.
  size_t numFrames = translationInfo->frames.size();

  // Get the frame data directly.
  const uint8_t *frameData = data.getBytes();
  size_t frameDataSize = data.size();

  // Get the frame metadata for the current expected frame.
  const auto &frameInfo = translationInfo->frames[nextFrameIndex];

  // Check size
  if (frameDataSize < frameInfo.expectedSize)
    throw std::runtime_error("Frame data too small: expected at least " +
                             std::to_string(frameInfo.expectedSize) +
                             " bytes, got " + std::to_string(frameDataSize) +
                             " bytes");

  // Check if this is the first frame of a new message.
  // If so, we need to initialize the translation buffer.
  bool isFirstFrame = (nextFrameIndex == 0);

  if (isFirstFrame) {
    // Initialize the translation buffer to hold the complete intoType.
    // The buffer will be filled in as frames arrive.
    translationBuffer.resize(translationInfo->intoTypeBytes, 0);
  }

  // Execute copy ops
  for (const auto &op : frameInfo.copyOps)
    std::memcpy(translationBuffer.data() + op.bufferOffset,
                frameData + op.frameOffset, op.size);

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

void WriteChannelPort::translateOutgoing(const MessageData &data) {
  assert(translationInfo &&
         "Translation type must be set for window translation.");

  const uint8_t *srcData = data.getBytes();
  size_t srcDataSize = data.size();

  if (srcDataSize < translationInfo->intoTypeBytes)
    throw std::runtime_error("Source data too small: expected at least " +
                             std::to_string(translationInfo->intoTypeBytes) +
                             " bytes, got " + std::to_string(srcDataSize) +
                             " bytes");

  for (const auto &frameInfo : translationInfo->frames) {
    std::vector<uint8_t> frameBuffer(frameInfo.expectedSize, 0);
    for (const auto &op : frameInfo.copyOps)
      std::memcpy(frameBuffer.data() + op.frameOffset,
                  srcData + op.bufferOffset, op.size);
    translationBuffer.emplace_back(std::move(frameBuffer));
  }
}
