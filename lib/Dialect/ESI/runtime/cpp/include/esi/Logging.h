//===- Logging.h - ESI Runtime logging --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ESI runtime logging is very simple but very flexible. Rather than mandating a
// particular logging library, it allows users to hook into their existing
// logging system by implementing a simple interface.
//
//===----------------------------------------------------------------------===//
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef ESI_LOGGING_H
#define ESI_LOGGING_H

#include <any>
#include <functional>
#include <iosfwd>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace esi {

class Logger {
public:
  enum class Level {
    Trace, // Trace is even more detailed than debug and requires a compiler
           // flag to be enabled when the runtime is built. Allows clients to do
           // things like log every single message or interaction.
    Debug, // Information useful when trying to debug an application.
    Info,  // General information, like connecting to an accelerator.
    Warning, // May indicate a problem.
    Error,   // Many errors will be followed by exceptions which may get caught.
  };
  Logger(bool debugEnabled, bool traceEnabled)
      : debugEnabled(debugEnabled), traceEnabled(traceEnabled) {}
  virtual ~Logger() = default;
  bool getDebugEnabled() { return debugEnabled; }
  bool getTraceEnabled() { return traceEnabled; }

  /// Report a log message.
  /// Arguments:
  ///   level: The log level as defined by the 'Level' enum above.
  ///   subsystem: The subsystem that generated the log message.
  ///   msg: The log message.
  ///   details: Optional additional structured details to include in the log
  ///            message. If there are no details, this should be nullptr.
  virtual void
  log(Level level, const std::string &subsystem, const std::string &msg,
      const std::map<std::string, std::any> *details = nullptr) = 0;

  /// Report an error.
  virtual void error(const std::string &subsystem, const std::string &msg,
                     const std::map<std::string, std::any> *details = nullptr) {
    log(Level::Error, subsystem, msg, details);
  }
  /// Report a warning.
  virtual void
  warning(const std::string &subsystem, const std::string &msg,
          const std::map<std::string, std::any> *details = nullptr) {
    log(Level::Warning, subsystem, msg, details);
  }
  /// Report an informational message.
  virtual void info(const std::string &subsystem, const std::string &msg,
                    const std::map<std::string, std::any> *details = nullptr) {
    log(Level::Info, subsystem, msg, details);
  }

  /// Report a debug message. This is not virtual so that it can be inlined to
  /// minimize performance impact that debug messages have, allowing debug
  /// messages in Release builds.
  inline void debug(const std::string &subsystem, const std::string &msg,
                    const std::map<std::string, std::any> *details = nullptr) {
    if (debugEnabled)
      debugImpl(subsystem, msg, details);
  }
  /// Call the debug function callback only if debug is enabled then log a debug
  /// message. Allows users to run heavy weight debug message generation code
  /// only when debug is enabled, which in turns allows users to provide
  /// fully-featured debug messages in Release builds with minimal performance
  /// impact. Not virtual so that it can be inlined.
  inline void
  debug(std::function<
        void(std::string &subsystem, std::string &msg,
             std::unique_ptr<std::map<std::string, std::any>> &details)>
            debugFunc) {
    if (debugEnabled)
      debugImpl(debugFunc);
  }

  /// Log a trace message. If tracing is not enabled, this is a no-op. Since it
  /// is inlined, the compiler will hopefully optimize it away creating a
  /// zero-overhead call. This means that clients are free to go crazy with
  /// trace messages.
  inline void trace(const std::string &subsystem, const std::string &msg,
                    const std::map<std::string, std::any> *details = nullptr) {
#ifdef ESI_RUNTIME_TRACE
    if (traceEnabled)
      log(Level::Trace, subsystem, msg, details);
#endif
  }
  /// Log a trace message using a callback. Same as above, users can go hog-wild
  /// calling this.
  inline void
  trace(std::function<
        void(std::string &subsystem, std::string &msg,
             std::unique_ptr<std::map<std::string, std::any>> &details)>
            traceFunc) {
#ifdef ESI_RUNTIME_TRACE
    if (!traceEnabled)
      return;
    std::string subsystem;
    std::string msg;
    std::unique_ptr<std::map<std::string, std::any>> details = nullptr;
    traceFunc(subsystem, msg, details);
    log(Level::Trace, subsystem, msg, details.get());
#endif
  }

protected:
  /// Overrideable version of debug. Only gets called if debug is enabled.
  virtual void debugImpl(const std::string &subsystem, const std::string &msg,
                         const std::map<std::string, std::any> *details) {
    log(Level::Debug, subsystem, msg, details);
  }
  /// Overrideable version of debug. Only gets called if debug is enabled.
  virtual void
  debugImpl(std::function<
            void(std::string &subsystem, std::string &msg,
                 std::unique_ptr<std::map<std::string, std::any>> &details)>
                debugFunc) {
    if (!debugEnabled)
      return;
    std::string subsystem;
    std::string msg;
    std::unique_ptr<std::map<std::string, std::any>> details = nullptr;
    debugFunc(subsystem, msg, details);
    debugImpl(subsystem, msg, details.get());
  }

  /// Enable or disable debug messages.
  bool debugEnabled = false;

  /// Enable or disable trace messages.
  bool traceEnabled;
};

/// A thread-safe logger which calls functions implemented by subclasses. Only
/// protects the `log` method. If subclasses override other methods and need to
/// protect them, they need to do that themselves.
class TSLogger : public Logger {
public:
  using Logger::Logger;

  /// Grabs the lock and calls logImpl.
  void log(Level level, const std::string &subsystem, const std::string &msg,
           const std::map<std::string, std::any> *details) override final;

protected:
  /// Subclasses must implement this method to log messages.
  virtual void logImpl(Level level, const std::string &subsystem,
                       const std::string &msg,
                       const std::map<std::string, std::any> *details) = 0;

  /// Mutex to protect the stream from interleaved logging writes.
  std::mutex mutex;
};

/// A logger that writes to a C++ std::ostream.
class StreamLogger : public TSLogger {
public:
  /// Create a stream logger that logs to the given output stream and error
  /// output stream.
  StreamLogger(Level minLevel, std::ostream &out, std::ostream &error)
      : TSLogger(minLevel <= Level::Debug, minLevel <= Level::Trace),
        minLevel(minLevel), outStream(out), errorStream(error) {}
  /// Create a stream logger that logs to stdout, stderr.
  StreamLogger(Level minLevel);
  void logImpl(Level level, const std::string &subsystem,
               const std::string &msg,
               const std::map<std::string, std::any> *details) override;

private:
  /// The minimum log level to emit.
  Level minLevel;

  /// Everything except errors goes here.
  std::ostream &outStream;
  /// Just for errors.
  std::ostream &errorStream;
};

/// A logger that writes to the console. Includes color support.
class ConsoleLogger : public TSLogger {
public:
  /// Create a stream logger that logs to stdout, stderr.
  ConsoleLogger(Level minLevel);
  void logImpl(Level level, const std::string &subsystem,
               const std::string &msg,
               const std::map<std::string, std::any> *details) override;

private:
  /// The minimum log level to emit.
  Level minLevel;
};

/// A logger that does nothing.
class NullLogger : public Logger {
public:
  NullLogger() : Logger(false, false) {}
  void log(Level, const std::string &, const std::string &,
           const std::map<std::string, std::any> *) override {}
};

/// 'Stringify' a std::any. This is used to log std::any values by some loggers.
std::string toString(const std::any &a);

} // namespace esi

#endif // ESI_LOGGING_H
