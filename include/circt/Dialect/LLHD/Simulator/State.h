//===- State.h - Simulation state definition --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines structures used to keep track of the simulation state in the LLHD
// simulator.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LLHD_SIMULATOR_STATE_H
#define CIRCT_DIALECT_LLHD_SIMULATOR_STATE_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <map>
#include <queue>

namespace circt {
namespace llhd {
namespace sim {

/// The simulator's internal representation of time.
struct Time {
  /// Empty (zero) time constructor. All the time values are defaulted to 0.
  Time() = default;

  /// Construct with given time values.
  Time(uint64_t time, uint64_t delta, uint64_t eps)
      : time(time), delta(delta), eps(eps) {}

  /// Compare the time values in order of time, delta, eps.
  bool operator<(const Time &rhs) const;

  /// Return true if all the time values are equal.
  bool operator==(const Time &rhs) const;

  /// Add two time values.
  Time operator+(const Time &rhs) const;

  /// Return true if the time represents zero-time.
  bool isZero();

  /// Get the stored time in a printable format.
  std::string toString() const;

  uint64_t time;
  uint64_t delta;
  uint64_t eps;

private:
};

/// Detail structure that can be easily accessed by the lowered code.
struct SignalDetail {
  uint8_t *value;
  uint64_t offset;
  uint64_t instIndex;
  uint64_t globalIndex;
};

/// The simulator's internal representation of a signal.
struct Signal {
  /// Construct an "empty" signal.
  Signal(std::string name, std::string owner);

  /// Construct a signal with the given name, owner and initial value.
  Signal(std::string name, std::string owner, uint8_t *value, uint64_t size);

  /// Default move constructor.
  Signal(Signal &&) = default;

  /// Default signal destructor.
  ~Signal() = default;

  /// Returns true if the signals match in name, owner, size and value.
  bool operator==(const Signal &rhs) const;

  /// Returns true if the owner name is lexically smaller than rhs's owner, or
  /// the name is lexically smaller than rhs's name, in case they share the same
  /// owner.
  bool operator<(const Signal &rhs) const;

  /// Return the value of the signal in hexadecimal string format.
  std::string toHexString() const;

  /// Return the value of the i-th element of the signal in hexadecimal string
  /// format.
  std::string toHexString(unsigned) const;

  std::string name;
  std::string owner;
  // The list of instances this signal triggers.
  std::vector<unsigned> triggers;
  uint64_t size;
  std::unique_ptr<uint8_t> value;
  std::vector<std::pair<unsigned, unsigned>> elements;
};

/// The simulator's internal representation of one queue slot.
struct Slot {
  /// Create a new empty slot.
  Slot(Time time) : time(time) {}

  /// Returns true if the slot's time is smaller than the compared slot's time.
  bool operator<(const Slot &rhs) const;

  /// Returns true if the slot's time is greater than the compared slot's time.
  bool operator>(const Slot &rhs) const;

  /// Insert a change.
  void insertChange(int index, int bitOffset, uint8_t *bytes, unsigned width);

  /// Insert a scheduled process wakeup.
  void insertChange(unsigned inst);

  // A map from signal indexes to change buffers. Makes it easy to sort the
  // changes such that we can process one signal at a time.
  llvm::SmallVector<std::pair<unsigned, unsigned>, 32> changes;
  // Buffers for the signal changes.
  llvm::SmallVector<std::pair<unsigned, llvm::APInt>, 32> buffers;
  // The number of used change buffers in the slot.
  size_t changesSize = 0;

  // Processes with scheduled wakeup.
  llvm::SmallVector<unsigned, 4> scheduled;
  Time time;
  bool unused = false;
};

/// This is equivalent to and std::priorityQueue<Slot> ordered using the greater
/// operator, which adds an insertion method to add changes to a slot.
class UpdateQueue : public llvm::SmallVector<Slot, 8> {
  unsigned topSlot = 0;
  llvm::SmallVector<unsigned, 4> unused;

public:
  /// Check wheter a slot for the given time already exists. If that's the case,
  /// add the new change to it, else create a new slot and push it to the queue.
  void insertOrUpdate(Time time, int index, int bitOffset, uint8_t *bytes,
                      unsigned width);

  /// Check wheter a slot for the given time already exists. If that's the case,
  /// add the scheduled wakeup to it, else create a new slot and push it to the
  /// queue.
  void insertOrUpdate(Time time, unsigned inst);

  /// Return a reference to a slot with the given timestamp. If such a slot
  /// already exists, a reference to it will be returned. Otherwise a reference
  /// to a fresh slot is returned.
  Slot &getOrCreateSlot(Time time);

  /// Get a reference to the current top of the queue (the earliest event
  /// available).
  const Slot &top();

  /// Pop the current top of the queue. This marks the current top slot as
  /// unused and resets its internal structures such that they can be reused.
  void pop();

  unsigned events = 0;
};

/// State structure for process persistence across suspension.
struct ProcState {
  unsigned inst;
  int resume;
  bool *senses;
  uint8_t *resumeState;
};

/// The simulator internal representation of an instance.
struct Instance {
  Instance() = default;

  Instance(std::string name)
      : name(name), procState(nullptr), entityState(nullptr) {}

  // The instance name.
  std::string name;
  // The instance's hierarchical path.
  std::string path;
  // The instance's base unit.
  std::string unit;
  bool isEntity;
  size_t nArgs = 0;
  // The arguments and signals of this instance.
  llvm::SmallVector<SignalDetail, 0> sensitivityList;
  std::unique_ptr<ProcState> procState;
  std::unique_ptr<uint8_t> entityState;
  Time expectedWakeup;
  // A pointer to the base unit jitted function.
  void (*unitFPtr)(void **);
};

/// The simulator's state. It contains the current simulation time, signal
/// values and the event queue.
struct State {
  /// Construct a new empty (at 0 time) state.
  State() = default;

  /// State destructor, ensures all malloc'd regions stored in the state are
  /// correctly free'd.
  ~State();

  /// Pop the head of the queue and update the simulation time.
  Slot popQueue();

  /// Push a new scheduled wakeup event in the event queue.
  void pushQueue(Time time, unsigned inst);

  /// Find an instance in the instances list by name and return an
  /// iterator for it.
  llvm::SmallVectorTemplateCommon<Instance>::iterator
  getInstanceIterator(std::string instName);

  /// Add a new signal to the state. Returns the index of the new signal.
  int addSignal(std::string name, std::string owner);

  int addSignalData(int index, std::string owner, uint8_t *value,
                    uint64_t size);

  void addSignalElement(unsigned, unsigned, unsigned);

  /// Add a pointer to the process persistence state to a process instance.
  void addProcPtr(std::string name, ProcState *procStatePtr);

  /// Dump a signal to the out stream. One entry is added for every instance
  /// the signal appears in.
  void dumpSignal(llvm::raw_ostream &out, int index);

  /// Dump the instance layout. Used for testing purposes.
  void dumpLayout();

  /// Dump the instances each signal triggers. Used for testing purposes.
  void dumpSignalTriggers();

  Time time;
  std::string root;
  llvm::SmallVector<Instance, 0> instances;
  llvm::SmallVector<Signal, 0> signals;
  UpdateQueue queue;
};

} // namespace sim
} // namespace llhd
} // namespace circt

#endif // CIRCT_DIALECT_LLHD_SIMULATOR_STATE_H
