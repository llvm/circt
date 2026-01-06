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

void ReadChannelPort::resetTranslationState() {
  nextFrameIndex = 0;
  accumulatingListData = false;
  translationBuffer.clear();
  listDataBuffer.clear();
}

void ReadChannelPort::connect(std::function<bool(MessageData)> callback,
                              const ConnectOptions &options) {
  if (mode != Mode::Disconnected)
    throw std::runtime_error("Channel already connected");

  resetTranslationState();

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

  resetTranslationState();

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

  const StructType *intoStruct = dynamic_cast<const StructType *>(intoType);
  if (!intoStruct)
    throw std::runtime_error(
        "Window intoType must be a struct for translation");

  const auto &intoFields = intoStruct->getFields();

  // Build a map from field name to (offset, type, isListField).
  // For list fields, the offset is where the list_length field will be stored,
  // and we also track the element type.
  struct FieldInfo {
    size_t offset;
    const Type *type;
    bool isList;
    const Type *listElementType; // Only valid if isList is true
    size_t listElementSize;      // Only valid if isList is true
  };
  std::map<std::string, FieldInfo> fieldMap;
  size_t currentOffset = 0;
  hasListField = false;

  auto processField = [&](const std::string &name, const Type *fieldType) {
    auto *listType = dynamic_cast<const ListType *>(fieldType);
    if (listType) {
      hasListField = true;
      // For list fields in the intoType:
      // - We store a list_length (size_t / 8 bytes) followed by the list data
      // - The offset here is where the list_length goes
      const Type *elemType = listType->getElementType();
      std::ptrdiff_t elemBits = elemType->getBitWidth();
      if (elemBits < 0)
        throw std::runtime_error(
            "Cannot translate list with dynamically-sized element type: " +
            name);
      if (elemBits % 8 != 0)
        throw std::runtime_error(
            "Cannot translate list element with non-byte-aligned size: " +
            name);
      size_t elemBytes = (static_cast<size_t>(elemBits) + 7) / 8;
      fieldMap[name] = {currentOffset, fieldType, true, elemType, elemBytes};
      // Reserve space for list_length. We use size_t (8 bytes on 64-bit
      // platforms) for consistency with standard C/C++ container sizes.
      // Note: This means the translated message format is platform-dependent.
      currentOffset += sizeof(size_t);
      // List data will be appended dynamically after the fixed header
    } else {
      std::ptrdiff_t fieldBits = fieldType->getBitWidth();
      if (fieldBits < 0)
        throw std::runtime_error("Cannot translate field with dynamic size: " +
                                 name);
      if (fieldBits % 8 != 0)
        throw std::runtime_error(
            "Cannot translate field with non-byte-aligned size: " + name);
      size_t fieldBytes = (static_cast<size_t>(fieldBits) + 7) / 8;
      fieldMap[name] = {currentOffset, fieldType, false, nullptr, 0};
      currentOffset += fieldBytes;
    }
  };

  if (intoStruct->isReverse()) {
    for (auto it = intoFields.rbegin(); it != intoFields.rend(); ++it)
      processField(it->first, it->second);
  } else {
    for (const auto &[name, fieldType] : intoFields)
      processField(name, fieldType);
  }

  // intoTypeBytes is now the size of the fixed header portion
  // (for types with lists, this excludes the variable-length list data)
  intoTypeBytes = currentOffset;

  const auto &windowFrames = windowType->getFrames();
  frames.clear();
  frames.reserve(windowFrames.size());

  for (const auto &frame : windowFrames) {
    FrameInfo frameInfo;
    size_t frameOffset = 0;

    // Calculate frame layout in SV memory order.
    // SV structs are laid out with the last declared field at the lowest
    // address. So when we iterate fields in reverse order (rbegin to rend),
    // we get the memory layout order.
    //
    // For list fields, the lowered struct in CIRCT adds fields in this order:
    //   1. list_data[numItems] - array of list elements
    //   2. list_size - (if numItems > 1) count of valid items
    //   3. last - indicates end of list
    //
    // In SV memory layout (reversed), this becomes:
    //   offset 0: last (1 byte)
    //   offset 1: list_size (if present)
    //   offset after size: list_data[numItems]
    //   offset after list: header fields...
    struct FrameFieldLayout {
      std::string name;
      size_t frameOffset;
      size_t size;
      bool isList;
      size_t numItems;             // From window spec
      size_t listElementSize;      // Size of each list element
      size_t bufferOffset;         // Offset in translation buffer
      const Type *listElementType; // Element type for list fields
    };
    std::vector<FrameFieldLayout> fieldLayouts;

    for (auto fieldIt = frame.fields.rbegin(); fieldIt != frame.fields.rend();
         ++fieldIt) {
      const WindowType::Field &field = *fieldIt;
      auto it = fieldMap.find(field.name);
      if (it == fieldMap.end())
        throw std::runtime_error("Frame field '" + field.name +
                                 "' not found in intoType");

      const FieldInfo &fieldInfo = it->second;
      FrameFieldLayout layout;
      layout.name = field.name;
      layout.bufferOffset = fieldInfo.offset;
      layout.isList = fieldInfo.isList;
      layout.numItems = field.numItems;
      layout.listElementType = fieldInfo.listElementType;
      layout.listElementSize = fieldInfo.listElementSize;

      if (fieldInfo.isList) {
        // For list fields, the frame layout is (in memory order):
        //   1. 'last' field (1 byte)
        //   2. list_data (one element per frame)
        // Note: numItems > 1 is not yet supported.
        size_t numItems = field.numItems > 0 ? field.numItems : 1;
        if (numItems != 1)
          throw std::runtime_error(
              "List translation with numItems > 1 is not yet supported. "
              "Field '" +
              field.name + "' has numItems=" + std::to_string(numItems));

        // 'last' field comes first in memory
        size_t lastOffset = frameOffset;
        frameOffset += 1;

        // List data comes after 'last'
        layout.frameOffset = frameOffset;
        layout.size = fieldInfo.listElementSize;
        fieldLayouts.push_back(layout);
        frameOffset += layout.size;

        // Create ListFieldInfo for this list field
        ListFieldInfo listInfo;
        listInfo.fieldName = layout.name;
        listInfo.dataOffset = layout.frameOffset;
        listInfo.elementSize = fieldInfo.listElementSize;
        listInfo.listLengthBufferOffset = layout.bufferOffset;
        listInfo.listDataBufferOffset = intoTypeBytes;
        listInfo.lastFieldOffset = lastOffset;
        frameInfo.listField = listInfo;
      } else {
        // Regular (non-list) field
        std::ptrdiff_t fieldBits = fieldInfo.type->getBitWidth();
        layout.frameOffset = frameOffset;
        layout.size = (static_cast<size_t>(fieldBits) + 7) / 8;
        fieldLayouts.push_back(layout);
        frameOffset += layout.size;
      }
    }

    frameInfo.expectedSize = frameOffset;

    // Second pass: build copy ops for non-list fields
    for (const auto &layout : fieldLayouts) {
      if (!layout.isList) {
        // Regular field - add copy op
        frameInfo.copyOps.push_back(
            {layout.frameOffset, layout.bufferOffset, layout.size});
      }
      // List fields were already handled in the first pass
    }

    // Sort copy ops by frameOffset to ensure processing order matches frame
    // layout.
    std::sort(frameInfo.copyOps.begin(), frameInfo.copyOps.end(),
              [](const CopyOp &a, const CopyOp &b) {
                return a.frameOffset < b.frameOffset;
              });

    // Merge adjacent copy ops
    if (!frameInfo.copyOps.empty()) {
      std::vector<CopyOp> mergedOps;
      mergedOps.push_back(frameInfo.copyOps[0]);

      for (size_t i = 1; i < frameInfo.copyOps.size(); ++i) {
        CopyOp &last = mergedOps.back();
        CopyOp current = frameInfo.copyOps[i];

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
  // For list frames, we use accumulatingListData to track whether we're
  // continuing to accumulate list items.
  bool isFirstFrame = (nextFrameIndex == 0) && !accumulatingListData;

  if (isFirstFrame) {
    // Initialize the translation buffer to hold the fixed header portion.
    translationBuffer.resize(translationInfo->intoTypeBytes, 0);
    // Clear list data buffer for types with lists
    if (translationInfo->hasListField)
      listDataBuffer.clear();
  }

  // Execute copy ops for non-list fields
  for (const auto &op : frameInfo.copyOps)
    std::memcpy(translationBuffer.data() + op.bufferOffset,
                frameData + op.frameOffset, op.size);

  // Handle list field if present in this frame
  if (frameInfo.listField.has_value()) {
    const auto &listInfo = frameInfo.listField.value();

    // With numItems == 1, each frame contains exactly one list element
    size_t bytesToCopy = listInfo.elementSize;
    // Additional check: dataOffset must not be beyond frameDataSize
    if (listInfo.dataOffset > frameDataSize)
      throw std::runtime_error("List data offset is beyond frame bounds");
    // Bounds check to prevent buffer overflow from corrupted _size field
    if (listInfo.dataOffset + bytesToCopy > frameDataSize)
      throw std::runtime_error("List data extends beyond frame bounds");
    size_t oldSize = listDataBuffer.size();
    listDataBuffer.resize(oldSize + bytesToCopy);
    std::memcpy(listDataBuffer.data() + oldSize,
                frameData + listInfo.dataOffset, bytesToCopy);

    // Check if this is the last frame of the list
    uint8_t lastFlag = frameData[listInfo.lastFieldOffset];
    if (lastFlag) {
      // List is complete. Build the final message:
      // [fixed header with list_length (size_t)][list data...]
      // Write the actual length to the listLengthBufferOffset position.
      size_t listLength = listDataBuffer.size() / listInfo.elementSize;
      size_t *listLengthPtr = reinterpret_cast<size_t *>(
          translationBuffer.data() + listInfo.listLengthBufferOffset);
      *listLengthPtr = listLength;

      // Append list data to translation buffer
      size_t headerSize = translationBuffer.size();
      translationBuffer.resize(headerSize + listDataBuffer.size());
      std::memcpy(translationBuffer.data() + headerSize, listDataBuffer.data(),
                  listDataBuffer.size());

      // Reset for next message
      nextFrameIndex = 0;
      listDataBuffer.clear();
      accumulatingListData = false;
      return true;
    }

    // Not the last frame - stay on this frame index for list accumulation
    // (list frames repeat until last=true)
    accumulatingListData = true;
    return false;
  }

  // No list field in this frame - advance to next frame
  nextFrameIndex++;
  size_t numFrames = translationInfo->frames.size();

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

  // Check if we have list fields
  if (!translationInfo->hasListField) {
    // No list fields - simple fixed-size translation
    for (const auto &frameInfo : translationInfo->frames) {
      std::vector<uint8_t> frameBuffer(frameInfo.expectedSize, 0);
      for (const auto &op : frameInfo.copyOps)
        std::memcpy(frameBuffer.data() + op.frameOffset,
                    srcData + op.bufferOffset, op.size);
      translationBuffer.emplace_back(std::move(frameBuffer));
    }
    return;
  }

  // Handle list fields - need to split list data into multiple frames
  for (const auto &frameInfo : translationInfo->frames) {
    if (!frameInfo.listField.has_value()) {
      // Non-list frame - copy as normal
      std::vector<uint8_t> frameBuffer(frameInfo.expectedSize, 0);
      for (const auto &op : frameInfo.copyOps)
        std::memcpy(frameBuffer.data() + op.frameOffset,
                    srcData + op.bufferOffset, op.size);
      translationBuffer.emplace_back(std::move(frameBuffer));
    } else {
      // List frame - need to generate multiple frames
      const auto &listInfo = frameInfo.listField.value();

      // Read the list length from the source data
      size_t listLength = 0;
      std::memcpy(&listLength, srcData + listInfo.listLengthBufferOffset,
                  sizeof(size_t));

      // Check that the buffer is large enough for the list data
      if (translationInfo->intoTypeBytes + (listLength * listInfo.elementSize) >
          srcDataSize) {
        throw std::runtime_error(
            "Source buffer too small for list data: possible corrupted or "
            "inconsistent list length field");
      }
      // Get pointer to list data (after the fixed header)
      const uint8_t *listData = srcData + translationInfo->intoTypeBytes;

      // Generate frames for the list
      size_t itemsRemaining = listLength;
      size_t listDataOffset = 0;

      // Handle empty list case - still need to send one frame with last=true
      if (listLength == 0)
        throw std::runtime_error(
            "Cannot send empty lists - parallel ESI list encoding requires at "
            "least one frame to be sent, and each frame must contain at least "
            "one element.");

      while (itemsRemaining > 0) {
        std::vector<uint8_t> frameBuffer(frameInfo.expectedSize, 0);

        // Copy non-list fields (header data) to each frame
        for (const auto &op : frameInfo.copyOps)
          std::memcpy(frameBuffer.data() + op.frameOffset,
                      srcData + op.bufferOffset, op.size);

        // With numItems == 1, each frame contains exactly one list element
        size_t bytesInThisFrame = listInfo.elementSize;

        // Copy list data
        std::memcpy(frameBuffer.data() + listInfo.dataOffset,
                    listData + listDataOffset, bytesInThisFrame);

        // Update remaining count
        itemsRemaining -= 1;
        listDataOffset += bytesInThisFrame;

        // Set last field
        frameBuffer[listInfo.lastFieldOffset] = (itemsRemaining == 0) ? 1 : 0;

        translationBuffer.emplace_back(std::move(frameBuffer));
      }
    }
  }
}
