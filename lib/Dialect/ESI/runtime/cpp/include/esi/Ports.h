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

#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>

namespace esi {

class ChannelPort;
using PortMap = std::map<std::string, ChannelPort &>;

namespace detail {

/// Shared queue/promise helper for polling-style read APIs.
///
/// Producers either fulfill the oldest waiting reader immediately or enqueue
/// the value for a later `readAsync()`. Consumers either receive buffered data
/// immediately or install a promise which is fulfilled by the next enqueue.
template <typename BufferedT>
class PollingBuffer {
public:
  explicit PollingBuffer(uint64_t maxQueued) : maxQueued(maxQueued) {}

  /// Return the current bounded queue size. `0` means unbounded.
  uint64_t getMaxQueued() {
    std::scoped_lock<std::mutex> lock(mutex);
    return maxQueued;
  }

  /// Update the bounded queue size. `0` means unbounded.
  void setMaxQueued(uint64_t maxMsgs) {
    std::scoped_lock<std::mutex> lock(mutex);
    maxQueued = maxMsgs;
  }

  /// Try to deliver or enqueue a produced value.
  ///
  /// Returns `false` only when the bounded queue is full.
  bool enqueue(BufferedT &value) {
    std::scoped_lock<std::mutex> lock(mutex);
    assert(!(!promiseQueue.empty() && !dataQueue.empty()) &&
           "Both queues are in use.");

    if (!promiseQueue.empty()) {
      std::promise<BufferedT> promise = std::move(promiseQueue.front());
      promiseQueue.pop();
      promise.set_value(std::move(value));
    } else {
      if (dataQueue.size() >= maxQueued && maxQueued != 0)
        return false;
      dataQueue.push(std::move(value));
    }
    return true;
  }

  /// Read the next value asynchronously.
  ///
  /// If a value is already buffered, the returned future is ready
  /// immediately. Otherwise the future is backed by an internal promise which
  /// is fulfilled by a later `enqueue()`.
  std::future<BufferedT> readAsync() {
    std::scoped_lock<std::mutex> lock(mutex);
    assert(!(!promiseQueue.empty() && !dataQueue.empty()) &&
           "Both queues are in use.");

    if (!dataQueue.empty()) {
      std::promise<BufferedT> promise;
      std::future<BufferedT> future = promise.get_future();
      promise.set_value(std::move(dataQueue.front()));
      dataQueue.pop();
      return future;
    }

    promiseQueue.emplace();
    return promiseQueue.back().get_future();
  }

private:
  std::mutex mutex;
  std::queue<BufferedT> dataQueue;
  uint64_t maxQueued;
  std::queue<std::promise<BufferedT>> promiseQueue;
};

} // namespace detail

/// Unidirectional channels are the basic communication primitive between the
/// host and accelerator. A 'ChannelPort' is the host side of a channel. It can
/// be either read or write but not both. At this level, channels are untyped --
/// just streams of bytes. They are not intended to be used directly by users
/// but used by higher level APIs which add types.
class ChannelPort {
public:
  ChannelPort(const Type *type);
  virtual ~ChannelPort() {}

  struct ConnectOptions {
    /// The buffer size is optional and should be considered merely a hint.
    /// Individual implementations use it however they like. The unit is number
    /// of messages of the port type.
    std::optional<unsigned> bufferSize = std::nullopt;

    /// If the type of this port is a window, translate the incoming/outgoing
    /// data into its underlying ('into') type. For 'into' types without lists,
    /// just re-arranges the data fields from the lowered type to the 'into'
    /// type.
    ///
    /// If this option is false, no translation is done and the data is
    /// passed through as-is. Same is true for non-windowed types.
    ///
    /// For messages with lists, only two types are supported:
    ///   1) Parallel encoding includes any 'header' data with each frame. Said
    ///      header data is the same across all frames, so this encoding is
    ///      inefficient but is commonly used for on-chip streaming interfaces.
    ///      Each frame contains a 'last' field to indicate the end of the list.
    ///      In cases where 'numItems' is greater than 1, a field named
    ///      '<listField>_size' indicates the number of valid items in that
    ///      frame.
    ///   2) Serial (bulk transfer) encoding, where a 'header' frame precedes
    ///      the list data frame. Said header frame contains a 'count' field
    ///      indicating the number of items in the list.  Importantly, the
    ///      header frame is always re-transmitted after the specified number of
    ///      list items have been sent. If the 'count' field is zero, the end of
    ///      the list has been reached. If it is non-zero, the message has not
    ///      been completely transmitted and reading should continue until a
    ///      'count' of zero is received.
    ///
    /// In both cases, the host-side MessageData contains the complete header
    /// followed by the list data. In other words, header data is not duplicated
    /// in the returned message. So for a windowed type with header fields and
    /// a list of (x,y) coordinates, the host memory layout would be:
    /// ```
    /// struct ExampleList {
    ///   uint32_t headerField2; // SystemVerilog ordering
    ///   uint32_t headerField1;
    ///   size_t   list_length; // Number list items
    ///   struct { uint16_t y, x; } list_data[];
    /// }
    /// ```
    ///
    /// In a parallel encoding, each frame's wire format (from hardware) is:
    /// ```
    /// struct ExampleListFrame {
    ///   uint8_t list_last; // Non-zero indicates last item in list
    ///   struct { uint16_t y, x; } list_data[numItems]; // SV field ordering
    ///   uint32_t headerField2; // SV struct ordering (reversed)
    ///   uint32_t headerField1;
    /// }
    /// ```
    ///
    /// Important note: for consistency, preserves SystemVerilog struct field
    /// ordering! So it's the opposite of C struct ordering.
    ///
    /// Implementation status:
    ///   - Only parallel list encoding is supported.
    ///   - Fields must be byte-aligned.
    ///
    /// See the CIRCT documentation (or td files) for more details on windowed
    /// messages.
    bool translateMessage = true;

    ConnectOptions(std::optional<unsigned> bufferSize = std::nullopt,
                   bool translateMessage = true)
        : bufferSize(bufferSize), translateMessage(translateMessage) {}
  };

  /// Set up a connection to the accelerator.
  virtual void connect(const ConnectOptions &options = ConnectOptions()) = 0;
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

  /// Get the size of each frame in bytes. For windowed types, this is the
  /// lowered type's width; otherwise, the port type's width. Transports
  /// (DMA engines, cosim, etc.) require every message to be at least one
  /// byte, so void / zero-width port types are reported as 1 byte here; the
  /// transport pads or strips that placeholder byte transparently.
  size_t getFrameSizeBytes() const {
    size_t bytes = translationInfo ? translationInfo->frameBytes
                                   : utils::bitsToBytes(type->getBitWidth());
    return std::max<size_t>(1, bytes);
  }
  const Type *getType() const { return type; }

  /// If this port carries a windowed type, return the original WindowType
  /// (whose `intoType` is what `getType()` returns). Returns nullptr for
  /// non-windowed ports.
  const WindowType *getWindowType() const {
    return translationInfo ? translationInfo->windowType : nullptr;
  }

protected:
  const Type *type;

  /// Instructions for translating windowed types. Precomputes and optimizes a
  /// list of copy operations.
  struct TranslationInfo {
    TranslationInfo(const WindowType *windowType) : windowType(windowType) {}

    /// Precompute and optimize the copy operations for translating frames.
    void precomputeFrameInfo();

    /// Throw if this window cannot be translated by the runtime translator
    /// (e.g. uses serial/bulk list encoding which is not yet supported).
    void requireTranslationSupported() const;

    /// The window type being translated.
    const WindowType *windowType;

    /// A copy operation for translating between frame data and the translation.
    /// Run this during the translation.
    struct CopyOp {
      /// Offset in the incoming/outgoing frame data.
      size_t frameOffset;
      /// Offset in the translation buffer.
      size_t bufferOffset;
      /// Number of bytes to copy.
      size_t size;
    };

    /// Information about a list field within a frame (for parallel encoding).
    /// Note: Currently only numItems == 1 is supported (one list element per
    /// frame).
    struct ListFieldInfo {
      /// Name of the list field.
      std::string fieldName;
      /// Offset of the list data array in the frame.
      size_t dataOffset;
      /// Size of each list element in bytes.
      size_t elementSize;
      /// Offset of the 'last' field in the frame.
      size_t lastFieldOffset;
      /// Offset in the translation buffer where list length is stored.
      size_t listLengthBufferOffset;
      /// Offset in the translation buffer where list data starts.
      size_t listDataBufferOffset;
    };

    /// Information about each frame in the windowed type.
    struct FrameInfo {
      /// The total size of a frame in bytes.
      size_t expectedSize;
      /// Precomputed copy operations for translating this frame (non-list
      /// fields).
      std::vector<CopyOp> copyOps;
      /// Information about list fields in this frame (parallel encoding).
      /// Currently only one list field per frame is supported.
      std::optional<ListFieldInfo> listField;
    };
    /// Precomputed information about each frame.
    std::vector<FrameInfo> frames;
    /// Size of the 'into' type in bytes (for fixed-size types).
    /// For types with lists, this is the size of the fixed header portion.
    size_t intoTypeBytes = 0;
    /// Number of bytes per wire frame (from the lowered type's bit width).
    size_t frameBytes = 0;
    /// True if the window contains a list field (variable-size message).
    bool hasListField = false;
  };
  std::unique_ptr<TranslationInfo> translationInfo;

  /// Method called by poll() to actually poll the channel if the channel is
  /// connected.
  virtual bool pollImpl() { return false; }

  /// Called by all connect methods to let backends initiate the underlying
  /// connections.
  virtual void connectImpl(const ConnectOptions &options) {}
};

/// A ChannelPort which sends data to the accelerator. It has two modes: Sync
/// and Callback, which cannot be used at the same time. The mode is set at
/// connect() time. To change the mode, disconnect() and then connect() again.
class WriteChannelPort : public ChannelPort {
public:
  using ChannelPort::ChannelPort;
  /// Reports a fetched message which hasn't been completely written to the
  /// error callback.
  ~WriteChannelPort() override;

  /// Callback mode data source. Called whenever the port can accept a message:
  /// returns the next message to write (ownership is transferred), or nullptr
  /// if none is available yet. After returning nullptr (or throwing), it isn't
  /// called again until notifyAsyncDataAvailable() is. Must return
  /// immediately.
  using DataFetchCallback =
      std::function<std::unique_ptr<SegmentedMessageData>()>;

  /// Callback mode error handler. Called with the exception which prevented a
  /// message returned by the DataFetchCallback from being written completely
  /// (including the port being disconnected or destroyed first, or the
  /// DataFetchCallback throwing). The message is dropped. Every fetched message
  /// is either handed to the backend in full or reported here.
  using WriteErrorCallback = std::function<void(std::exception_ptr)>;

  /// Disconnect the port. In callback mode, reports a fetched message which
  /// hasn't been completely written to the error callback. Unless called from
  /// a callback, blocks until a concurrent poll() or notifyAsyncDataAvailable()
  /// (including the callbacks it invokes) returns. Subclasses which override
  /// this must call it.
  virtual void disconnect() override;
  virtual bool isConnected() const override {
    Mode m = mode;
    return m == Mode::Sync || m == Mode::Callback;
  }

  //===--------------------------------------------------------------------===//
  // Sync mode: connect without a callback to use write(), tryWrite(), and
  // flush(). These throw in callback mode.
  //===--------------------------------------------------------------------===//

  /// Connect in sync mode. Throws if connected in callback mode.
  virtual void connect(const ConnectOptions &options = {}) override;

  /// A very basic blocking write API. Will likely change for performance
  /// reasons.
  void write(const MessageData &data) {
    requireNotCallbackMode("write");
    if (translateMessages) {
      assert(translationBuffer.empty() &&
             "Cannot call write() with pending translated messages");
      translateOutgoing(data);
      for (auto &msg : translationBuffer)
        writeImpl(maybePadEmptyMessage(msg));
      translationBuffer.clear();
    } else {
      writeImpl(maybePadEmptyMessage(data));
    }
  }

  /// Write a multi-segment message. Takes ownership so the backend can hold
  /// the message across partial writes / async completions. Default flattens
  /// and calls the regular write(). Backends override for scatter-gather /
  /// chunked-DMA support.
  virtual void write(std::unique_ptr<SegmentedMessageData> msg) {
    if (!msg)
      throw std::runtime_error(
          "WriteChannelPort::write: null SegmentedMessageData");
    write(msg->toMessageData());
  }

  /// A basic non-blocking write API. Returns true if any of the data was
  /// sent; false means none of it was, and the write should be retried. If
  /// the data type is a window a 'true' return does not indicate that the
  /// message has been completely written. The 'flush' method can be used to
  /// check that the entire buffer has been written. It is invalid for
  /// backends to always return false (i.e. backends must eventually ensure
  /// that writes may succeed).
  bool tryWrite(const MessageData &data) {
    requireNotCallbackMode("tryWrite");
    return tryWriteInternal(data);
  }

  /// Non-blocking write of a multi-segment message. On success, takes
  /// ownership of `msg` (leaving it null) and returns true. On failure, returns
  /// false and leaves `msg` untouched so the caller can retry with the same
  /// message. Same completion semantics as `tryWrite(const MessageData &)`.
  /// Default flattens and calls the regular tryWrite(). Backends override for
  /// scatter-gather / chunked-DMA support.
  virtual bool tryWrite(std::unique_ptr<SegmentedMessageData> &msg) {
    if (!msg)
      throw std::runtime_error(
          "WriteChannelPort::tryWrite: null SegmentedMessageData");
    MessageData scratch;
    if (!tryWrite(msg->asMessageData(scratch)))
      return false;
    msg.reset();
    return true;
  }

  /// Flush any buffered data. Returns true if all data was flushed.
  ///
  /// If `translateMessages` is false, calling `flush()` will immediately return
  /// true and perform no action, as there is no buffered data to flush.
  bool flush() {
    requireNotCallbackMode("flush");
    return flushInternal();
  }

  //===--------------------------------------------------------------------===//
  // Callback mode: connect with a data fetch callback, which the port calls to
  // get messages to write whenever it can accept them. The port holds at most
  // one fetched message which the backend hasn't accepted.
  //===--------------------------------------------------------------------===//

  /// Connect in callback mode. Throws if already connected or if either
  /// callback is null.
  ///
  /// Messages are written with the segmented tryWrite(), so backends which
  /// override it for scatter-gather support receive them as-is. The port calls
  /// `fetch` and writes what it returns until `fetch` returns nullptr or the
  /// backend backpressures. It does so from poll() (typically called by the
  /// accelerator's service thread via `AcceleratorServiceThread::addPoll()`),
  /// which also retries a message the backend didn't accept, and from
  /// notifyAsyncDataAvailable(). `fetch` is first called by the first of
  /// either after connecting.
  ///
  /// Callbacks run on the thread which is writing, one at a time and never
  /// recursively. They must not throw and must return quickly. They may call
  /// notifyAsyncDataAvailable() and disconnect() (and then connect()), but
  /// must not wait on another thread which uses the port.
  virtual void connect(DataFetchCallback fetch, WriteErrorCallback onError,
                       const ConnectOptions &options = {});

  /// Notify the port that the DataFetchCallback has data again. Unless the
  /// port is already writing on another thread (which then picks the data up),
  /// writes it immediately on the calling thread, calling the callbacks. No-op
  /// unless connected in callback mode.
  void notifyAsyncDataAvailable();

protected:
  /// In callback mode, write fetched data (see connect()). Then, if the port is
  /// still connected (a callback may disconnect it), call pollWriteImpl().
  bool pollImpl() final;

  /// Backend-specific polling. Write port backends override this rather than
  /// pollImpl(). Returns true if any progress was made.
  virtual bool pollWriteImpl() { return false; }

  /// Indicates the current mode of the channel. `Destroyed` is set while the
  /// destructor reports a pending async write, preventing the error callback
  /// from using the partially destroyed port.
  enum Mode { Disconnected, Sync, Callback, Destroyed };
  std::atomic<Mode> mode = Mode::Disconnected;

  /// Pad an empty payload with a single zero byte if the port type has a
  /// zero-width logical width (e.g. `VoidType`). Transports (DMA engines,
  /// cosim, etc.) require every message to carry at least one byte.
  const MessageData &maybePadEmptyMessage(const MessageData &data) {
    if (!data.getSize() && type && type->getBitWidth() == 0)
      return transportPad0Bytes;
    return data;
  }

  /// Backing storage for `maybePadEmptyMessage` return so that the returned
  /// reference outlives the call.
  inline static const MessageData transportPad0Bytes = MessageData({0});

  /// Implementation for write(). Subclasses must implement this.
  virtual void writeImpl(const MessageData &) = 0;

  /// Implementation for tryWrite(). Subclasses must implement this.
  virtual bool tryWriteImpl(const MessageData &data) = 0;

  /// Break a message into its frames.
  std::vector<MessageData> getMessageFrames(const MessageData &data);

  /// Whether to translate outgoing data if the port type is a window type. Set
  /// by the connect() method.
  bool translateMessages = false;
  /// If tryWrite cannot write all the messages of a windowed type at once, it
  /// stores them here and writes them out one by one on subsequent calls.
  std::vector<MessageData> translationBuffer;
  /// Index of the next message to write in translationBuffer.
  size_t translationBufferIdx = 0;
  /// Translate outgoing data if the port type is a window type. Append the new
  /// message 'chunks' to translationBuffer.
  void translateOutgoing(const MessageData &data);

private:
  /// Throw if the port is destroyed, or connected in callback mode (unless
  /// called by the callback mode writer's own tryWrite()).
  void requireNotCallbackMode(const char *method) const;
  /// Set up translation and call connectImpl(). Shared by both modes.
  void connectCommon(const ConnectOptions &options);
  /// tryWrite() and flush() without the mode check.
  bool tryWriteInternal(const MessageData &data);
  bool flushInternal();

  //===--------------------------------------------------------------------===//
  // Callback mode state.
  //===--------------------------------------------------------------------===//

  /// Write fetched data until the DataFetchCallback has none, the backend
  /// backpressures, or the port leaves callback mode, then release `draining`.
  /// The caller must have claimed `draining`. Returns true if progress was
  /// made.
  bool drainAsyncWrites();
  /// Wait (with `asyncMutex` held by `lock`) until no other thread is writing.
  void waitForDrainer(std::unique_lock<std::mutex> &lock);
  /// Drop the fetched message which hasn't been completely written, if any.
  /// Returns true if there was one. Requires that no other thread is writing.
  bool dropPendingWrite();
  /// Invoke `onError` (if non-null) with `error`, swallowing any exception it
  /// throws.
  static void
  callErrorCallback(const std::shared_ptr<const WriteErrorCallback> &onError,
                    std::exception_ptr error);

  /// Protects the members below, except `pendingMsg`, which only the writing
  /// thread uses.
  std::mutex asyncMutex;
  /// Notified when `draining` is released.
  std::condition_variable drainDone;
  /// The callbacks registered at connect(). Shared so that they can be invoked
  /// without holding `asyncMutex` while disconnect() clears them.
  std::shared_ptr<const DataFetchCallback> fetchCallback;
  std::shared_ptr<const WriteErrorCallback> errorCallback;
  /// Incremented by each connect(), so the writer can tell that the port was
  /// reconnected while it was calling a callback.
  uint64_t connectionGeneration = 0;
  /// False once the DataFetchCallback returns nullptr, until
  /// notifyAsyncDataAvailable() is called.
  bool dataAvailable = false;
  /// True while a thread (`drainThread`) is writing fetched data.
  bool draining = false;
  std::thread::id drainThread;
  /// A fetched message which tryWrite() hasn't accepted yet. Once it has, any
  /// remaining frames of a windowed message are in `translationBuffer`.
  std::unique_ptr<SegmentedMessageData> pendingMsg;
};

/// Instantiated when a backend does not know how to create a write channel.
class UnknownWriteChannelPort : public WriteChannelPort {
public:
  UnknownWriteChannelPort(const Type *type, std::string errmsg)
      : WriteChannelPort(type), errmsg(errmsg) {}

  void connect(const ConnectOptions &options = {}) override {
    throw std::runtime_error(errmsg);
  }
  void connect(DataFetchCallback fetch, WriteErrorCallback onError,
               const ConnectOptions &options = {}) override {
    throw std::runtime_error(errmsg);
  }

protected:
  void writeImpl(const MessageData &) override {
    throw std::runtime_error(errmsg);
  }
  bool tryWriteImpl(const MessageData &) override {
    throw std::runtime_error(errmsg);
  }

  std::string errmsg;
};

/// A ChannelPort which reads data from the accelerator. It has two modes:
/// Callback and Polling which cannot be used at the same time. The mode is set
/// at connect() time. To change the mode, disconnect() and then connect()
/// again. When the port is disconnected, it will backpressure hardware.
class ReadChannelPort : public ChannelPort {

public:
  /// Primary callback API for raw reads.
  ///
  /// The message is passed by owning reference so the callee can consume it,
  /// move storage out of it, or leave it intact when returning `false` to
  /// request a retry with the same message object.
  using ReadCallback =
      std::function<bool(std::unique_ptr<SegmentedMessageData> &)>;

  /// Compatibility callback API for callers which want flattened message
  /// bytes instead of the owning segmented message object.
  using FlatReadCallback = std::function<bool(MessageData)>;

  ReadChannelPort(const Type *type)
      : ChannelPort(type), mode(Mode::Disconnected) {}

  /// Disconnect the channel. Warning: this method may block until all callbacks
  /// have returned. Do not call it anywhere which could be in a callback
  /// control path otherwise deadlock will occur.
  virtual void disconnect() override;
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

  virtual void connect(ReadCallback callback,
                       const ConnectOptions &options = {});

  /// Connect a compatibility callback which receives flattened `MessageData`
  /// objects. This adapts the primary segmented callback path.
  virtual void connect(FlatReadCallback callback,
                       const ConnectOptions &options = {});

  //===--------------------------------------------------------------------===//
  // Polling mode methods: To use futures or blocking reads, connect without any
  // arguments. You will then be able to use readAsync() or read().
  //===--------------------------------------------------------------------===//

  /// Default max data queue size set at connect time.
  static constexpr uint64_t DefaultMaxDataQueueMsgs = 32;

  /// Connect to the channel in polling mode.
  virtual void connect(const ConnectOptions &options = {}) override;

  /// Asynchronous polling read.
  ///
  /// Throws if the port is disconnected or currently connected in callback
  /// mode.
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
  void setMaxDataQueueMsgs(uint64_t maxMsgs) {
    maxDataQueueMsgs = maxMsgs;
    if (pollingState)
      pollingState->setMaxQueued(maxMsgs);
  }

protected:
  /// Invoke the currently registered callback.
  ///
  /// Handles synchronization race conditions with disconnects. Backends should
  /// use this helper instead of calling `callback` directly.
  bool invokeCallback(std::unique_ptr<SegmentedMessageData> &msg);

private:
  /// Backends should not call this directly. Use `invokeCallback()` instead,
  /// which handles synchronization with disconnects.
  ReadCallback callback;

protected:
  /// Indicates the current mode of the channel.
  enum Mode { Disconnected, Callback, Polling };
  volatile Mode mode;

  /// If a translated message has been assembled but not yet consumed, retain
  /// ownership here so retries present the same message object.
  std::unique_ptr<SegmentedMessageData> translatedMessage;

  /// Window translation support.
  std::vector<uint8_t> translationBuffer;
  /// Index of the next expected frame (for multi-frame windows).
  size_t nextFrameIndex = 0;
  /// For list fields: accumulated list data across frames.
  std::vector<uint8_t> listDataBuffer;
  /// Flag to track whether we're in the middle of accumulating list data.
  bool accumulatingListData = false;
  /// Reset translation state buffers and indices.
  void resetTranslationState();
  /// Translate incoming data if the port type is a window type. Returns true if
  /// the message has been completely received.
  bool translateIncoming(MessageData &data);

  //===--------------------------------------------------------------------===//
  // Polling mode members.
  //===--------------------------------------------------------------------===//

  uint64_t maxDataQueueMsgs = DefaultMaxDataQueueMsgs;
  std::optional<detail::PollingBuffer<MessageData>> pollingState;

  /// Synchronizes callback revocation during disconnect.
  std::mutex callbackMutex;
  std::condition_variable callbackCv;
  size_t activeCallbacks = 0;
};

/// Instantiated when a backend does not know how to create a read channel.
class UnknownReadChannelPort : public ReadChannelPort {
public:
  UnknownReadChannelPort(const Type *type, std::string errmsg)
      : ReadChannelPort(type), errmsg(errmsg) {}

  void connect(ReadCallback callback,
               const ConnectOptions &options = ConnectOptions()) override {
    throw std::runtime_error(errmsg);
  }
  void connect(FlatReadCallback callback,
               const ConnectOptions &options = ConnectOptions()) override {
    throw std::runtime_error(errmsg);
  }
  void connect(const ConnectOptions &options = ConnectOptions()) override {
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
