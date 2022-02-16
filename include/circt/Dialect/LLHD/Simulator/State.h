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
#include <regex>

namespace circt {
namespace llhd {
namespace sim {

class State;

/// The simulator's internal representation of time.
class Time {
public:
  /// All the time values are defaulted to 0.
  Time() : time(0), delta(0), eps(0) {}

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

  /// TODO: Use StringRef instead?
  /// Convert to human readable string
  std::string toString();

  uint64_t getTime() const { return time; }

private:
  uint64_t time;
  uint64_t delta;
  uint64_t eps;
};

/// Detail structure that can be easily accessed by the lowered code.
struct SignalDetail {
  uint8_t *value;
  uint64_t offset;
  uint64_t instIndex;
  uint64_t globalIndex;
};

/// The simulator's internal representation of a signal.
class Signal {
public:
  /// Construct an "empty" signal.
  Signal(std::string name, std::string owner)
    : name(name), owner(owner), size(0), value(nullptr) {}

  /// Construct a signal with the given name, owner and initial value.
  Signal(std::string name, std::string owner, uint8_t *value, uint64_t size)
    : name(name), owner(owner), size(size), value(value) {}

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

  bool isOwner(std::string rhs) const { return owner == rhs; };
  std::string getOwner() const { return owner; }

  bool isValidSigName() const {
    return std::regex_match(name, std::regex("(sig)?[0-9]*"));
  }

  std::string getName() const { return name; }

  uint64_t Size() const { return size; }

  uint8_t* Value() const { return value; }

  /// FIXME: We should return the iterator
  const std::vector<unsigned>& getTriggeredInstanceIndices () const {
    return instanceIndices;
  }

  void pushInstanceIndex(unsigned i) { instanceIndices.push_back(i); }

  bool hasElement() const { return elements.size() > 0; }

  size_t getElementSize() const { return elements.size(); }

  void pushElement(std::pair<unsigned, unsigned> val) {
    elements.push_back(val);
  }

  /// Update signal value and size
  void update(uint8_t* v, uint64_t s) {
    value = v;
    size = s;
  }

  /// Return the value of the signal in hexadecimal string format.
  std::string toHexString() const;

  /// Return the value of the i-th element of the signal in hexadecimal string
  /// format.
  std::string toHexString(unsigned) const;

private:
  std::string name;
  std::string owner;
  // The index to list of instances this signal triggers.
  std::vector<unsigned> instanceIndices;
  uint64_t size;
  uint8_t* value;
  std::vector<std::pair<unsigned, unsigned>> elements;
};

/// The simulator's internal representation of one queue slot.
class Slot {
public:
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

  Time getTime() const { return time; }

  uint64_t getTimeTime() const { return time.getTime(); }

  unsigned getNumChanges() const { return changesSize; }

  /// FIXME: Rename me
  std::pair<unsigned, unsigned> getChange(int index) const {
    return changes[index];
  }

  /// FIXME: Rename me
  std::pair<unsigned, llvm::APInt> getChangedBuffer(int index) const {
    return buffers[index];
  }

  bool isUsed() const { return !unused; }

  void sortChanges() {
    llvm::sort(changes.begin(), changes.begin() + changesSize);
  }

  /// Mark current slot to used
  void occupy(Time t) {
    unused = false;
    time = t;
  }

  void release() {
    unused = true;
    changesSize = 0;
    scheduled.clear();
    changes.clear();
    time = Time();
  }

  llvm::SmallVector<unsigned, 4> getScheduledWakups() const {
    return scheduled;
  }

private:
  // A map from signal indexes to change buffers. Makes it easy to sort the
  // changes such that we can process one signal at a time.
  llvm::SmallVector<std::pair<unsigned, unsigned>, 32> changes;
  // Buffers for the signal changes.
  llvm::SmallVector<std::pair<unsigned, llvm::APInt>, 32> buffers;
  // The number of used change buffers in the slot.
  unsigned changesSize = 0;

  // Processes with scheduled wakeup.
  llvm::SmallVector<unsigned, 4> scheduled;
  Time time;
  bool unused = false;
};

/// This is equivalent to std::priorityQueue<Slot> ordered using the greater
/// operator, which adds an insertion method to add changes to a slot.
class UpdateQueue : public llvm::SmallVector<Slot, 8> {
  unsigned topSlot = 0;
  llvm::SmallVector<unsigned, 4> unused;

public:
  unsigned numEvents = 0;

private:
  /// Return a reference to a slot with the given timestamp. If such a slot
  /// already exists, a reference to it will be returned. Otherwise a reference
  /// to a fresh slot is returned.
  Slot &getOrCreateSlot(Time time);

public:
  /// Check wheter a slot for the given time already exists. If that's the case,
  /// add the new change to it, else create a new slot and push it to the queue.
  void insertOrUpdate(Time time, int index, int bitOffset, uint8_t *bytes,
                      unsigned width);

  /// Check wheter a slot for the given time already exists. If that's the case,
  /// add the scheduled wakeup to it, else create a new slot and push it to the
  /// queue.
  void insertOrUpdate(Time time, unsigned inst);

  /// Get a reference to the current top of the queue (the earliest event
  /// available).
  const Slot &top();

  /// Pop the current top of the queue. This marks the current top slot as
  /// unused and resets its internal structures such that they can be reused.
  void pop();
};

/// State structure for process persistence across suspension.
struct ProcState {
  unsigned inst;
  int resume;
  bool *senses;
  uint8_t *resumeState;
};

/// The simulator internal representation of an instance.
class Instance {
public:
  enum InstType { Entity, Proc };

  Instance() : numArgs(0) {}

  Instance(std::string name)
      : name(name), numArgs(0), procState(nullptr), entityState(nullptr) {}

  Instance(std::string name, std::string path, std::string unit, int nArgs = 0)
      : name(name), path(path), unit(unit), numArgs(nArgs) {}

  ~Instance() { }

  const std::string getName() const { return name; }
  const std::string getPath() const { return path; }
  const std::string getBaseUnitName() const { return unit; }

  uint64_t getSensiSigIndex(int index) const {
    return sensitivityList[index + numArgs].globalIndex;
  }

  void updateSignalDetail(uint64_t index, uint8_t *value) {
    for (auto &I : sensitivityList) {
      if (I.globalIndex == index)
        I.value = value;
    }
  }

  void initSignalDetail(uint64_t globalIndex) {
    sensitivityList.push_back(
      SignalDetail({nullptr, 0, sensitivityList.size(), globalIndex}));
  }

  SignalDetail& getSignalDetail(unsigned index) {
    return sensitivityList[index];
  }

  void pushSignalDetail(SignalDetail SD) {
    sensitivityList.push_back(SD);
  }

  /// If current instance is waiting on given signal(index).
  bool isWaitingOnSignal(uint64_t sigIndex) const {
    auto IT = std::find_if(sensitivityList.begin(), sensitivityList.end(),
                           [sigIndex](const SignalDetail &sig) {
                             return sig.globalIndex == sigIndex;
                           });

    // procState->senses is updated by JIT functions generated by LLHDToLLVM.
    return sensitivityList.end() != IT &&
        procState->senses[IT - sensitivityList.begin()] == 0;
  }

  llvm::SmallVector<SignalDetail, 0>& getSensitivityList() {
    return sensitivityList;
  }

  void setType(InstType t) { type = t; }
  bool isEntity() const { return type == Entity; }
  bool isProc() const { return type == Proc; }

  void setProcState(ProcState *ps) { procState = ps; }
  void setEntityState(uint8_t *es) { entityState = es; }

  void setWakeupTime(Time t) { wakeupTime = t; }
  void invalidWakeupTime() { wakeupTime = Time(); }

  bool shouldWakeup(Time t) const { return t == wakeupTime; }

  void registerRunner(void(*fp)(void**)) { unitFPtr = fp; }

  // Simulate the instance with jitted function
  void run(void* state) {
    auto signalTable = sensitivityList.data();

    // Gather the instance arguments for unit invocation.
    llvm::SmallVector<void *, 3> args;
    if (isEntity())
      args.assign({state, &entityState, &signalTable});
    else
      args.assign({state, &procState, &signalTable});

    // Run the unit.
    (*unitFPtr)(args.data());
  }

  std::string dumpSensitivityList() const {
    std::string ret = "";
    for (auto I : sensitivityList)
      ret += std::to_string(I.globalIndex) + " ";
    return ret;
  }

private:
  // The instance name.
  std::string name;
  // The instance's hierarchical path.
  std::string path;
  // The instance's base unit name.
  std::string unit;
  InstType type;
  // Number of arguments of this instance
  size_t numArgs;
  // The arguments and signals of this instance.
  llvm::SmallVector<SignalDetail, 0> sensitivityList;
  ProcState* procState;
  uint8_t* entityState;
  Time wakeupTime;

private:
  // A pointer to the base unit jitted function.
  void (*unitFPtr)(void**);
};

/// The simulator's state. It contains the current simulation time, signal
/// values and the event queue.
class State {
public:
  /// Construct a new empty (at 0 time) state.
  State() = default;

  State(std::string r) : root(r) {}

  /// State destructor, ensures all malloc'd regions stored in the state are
  /// correctly free'd.
  ~State() {};

  /// Pop the head of the queue and update the simulation time.
  Slot popEvent() {
    assert(!events.empty() && "the event queue is empty");
    Slot top = events.top();
    events.pop();
    return top;
  }

  /// Push a new scheduled wakeup event in the event queue.
  void pushEvent(Time t, unsigned instIndex) {
    Time newTime = time + t;
    events.insertOrUpdate(newTime, instIndex);
    instances[instIndex].setWakeupTime(newTime);
  }

  /// Push a new scheduled wakeup event in the event queue.
  void pushEvent(Slot s) {
    events.push_back(s);
    ++events.numEvents;
  }

  unsigned hasEvents() const {
    return events.numEvents > 0;
  }

  /// Add a new signal to the state. Returns the index of the new signal.
  int addSignal(std::string name, std::string owner) {
    signals.push_back(Signal(name, owner));
    return signals.size() - 1;
  }

  /// TODO: This function should be part of signal class.
  int addSignalData(int index, std::string owner, uint8_t *value,
                    uint64_t size);

  /// TODO: This function should be part of signal class
  void addSignalElement(unsigned, unsigned, unsigned);

  size_t getInstanceSize() const { return instances.size(); }

  size_t getSignalSize() const { return signals.size(); }

  /// FIXME: We should return iterators
  const llvm::SmallVector<Signal, 0>& getSignals() const { return signals; }

  const Signal& getSignal(int index) const { return signals[index]; }

  Instance& getInstance(unsigned idx) {
    return instances[idx];
  }

  void pushInstance(Instance &i) { instances.push_back(std::move(i)); }

  void setInstanceEntityState(std::string owner, uint8_t *state) {
    auto II = findInstanceByName(owner);
    II->setEntityState(state);
  }

  /// Add a pointer to the process persistence state to a process instance.
  void setInstanceProcState(std::string owner, ProcState *state) {
    auto II = findInstanceByName(owner);
    state->inst = II - instances.begin();
    II->setProcState(state);
  }

  /// Spawn a new event.
  void spawnEvent(SignalDetail *SD, Time t, uint8_t *sigValue,
                  uint64_t sigWidth) {
    int bitOffset =
      (SD->value - signals[SD->globalIndex].Value()) * 8 + SD->offset;

    events.insertOrUpdate(time + t, SD->globalIndex,
                          bitOffset, sigValue, sigWidth);
  }

  /// Add triggers to signals
  void associateTrigger2Signal() {
    for (size_t i = 0, e = instances.size(); i < e; ++i) {
      auto &inst = instances[i];
      for (auto trigger : inst.getSensitivityList()) {
        signals[trigger.globalIndex].pushInstanceIndex(i);
      }
    }
  }

  // Update current simulation time
  void updateTime(Time t) { time = t; }

  std::string getRoot() const { return root; }

  Time getTime() const { return time; }

  /// Dump a signal to the out stream. One entry is added for every instance
  /// the signal appears in.
  void dumpSignal(llvm::raw_ostream &out, int index);

  /// Dump the instance layout. Used for testing purposes.
  void dumpLayout();

  /// Dump the instances each signal triggers. Used for testing purposes.
  void dumpSignalTriggers();

private:
  /// Find an instance in the instances list by name and return an
  /// iterator for it.
  llvm::SmallVectorTemplateCommon<Instance>::iterator
  findInstanceByName(std::string name);

private:
  Time time;
  std::string root;
  llvm::SmallVector<Instance, 0> instances;
  llvm::SmallVector<Signal, 0> signals;
  UpdateQueue events;
};

} // namespace sim
} // namespace llhd
} // namespace circt

#endif // CIRCT_DIALECT_LLHD_SIMULATOR_STATE_H
