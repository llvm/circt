//===- State.cpp - LLHD simulator state -------------------------*- C++ -*-===//
//
// This file implements the constructs used to keep track of the simulation
// state in the LLHD simulator.
//
//===----------------------------------------------------------------------===//

#include "State.h"

#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace llvm;
using namespace circt::llhd::sim;

//===----------------------------------------------------------------------===//
// Time
//===----------------------------------------------------------------------===//

bool Time::operator<(const Time &rhs) const {
  if (time < rhs.time)
    return true;
  if (time == rhs.time && delta < rhs.delta)
    return true;
  if (time == rhs.time && delta == rhs.delta && eps < rhs.eps)
    return true;
  return false;
}

bool Time::operator==(const Time &rhs) const {
  return (time == rhs.time && delta == rhs.delta && eps == rhs.eps);
}

Time Time::operator+(const Time &rhs) const {
  return Time(time + rhs.time, delta + rhs.delta, eps + rhs.eps);
}

bool Time::isZero() { return (time == 0 && delta == 0 && eps == 0); }

std::string Time::dump() {
  std::string ret;
  raw_string_ostream ss(ret);
  ss << time << "ns " << delta << "d " << eps << "e";
  return ss.str();
}

//===----------------------------------------------------------------------===//
// Signal
//===----------------------------------------------------------------------===//

Signal::Signal(std::string name, std::string owner)
    : name(name), owner(owner), size(0), value(nullptr) {}

Signal::Signal(std::string name, std::string owner, uint8_t *value,
               uint64_t size)
    : name(name), owner(owner), size(size), value(value) {}

bool Signal::operator==(const Signal &rhs) const {
  if (owner != rhs.owner || name != rhs.name || size != rhs.size)
    return false;
  return std::memcmp(value, rhs.value, size);
}

bool Signal::operator<(const Signal &rhs) const {
  if (owner < rhs.owner)
    return true;
  if (owner == rhs.owner && name < rhs.name)
    return true;
  return false;
}

std::string Signal::dump() {
  std::string ret;
  raw_string_ostream ss(ret);
  ss << "0x";
  for (int i = size - 1; i >= 0; --i) {
    ss << format_hex_no_prefix(static_cast<int>(value[i]), 2);
  }
  return ss.str();
}

//===----------------------------------------------------------------------===//
// Slot
//===----------------------------------------------------------------------===//

bool Slot::operator<(const Slot &rhs) const { return time < rhs.time; }

bool Slot::operator>(const Slot &rhs) const { return rhs.time < time; }

void Slot::insertChange(int index, int bitOffset, APInt &bytes) {
  changes[index].push_back(std::make_pair(bitOffset, bytes));
}

void Slot::insertChange(std::string inst) { scheduled.push_back(inst); }

//===----------------------------------------------------------------------===//
// UpdateQueue
//===----------------------------------------------------------------------===//
void UpdateQueue::insertOrUpdate(Time time, int index, int bitOffset,
                                 APInt &bytes) {
  for (size_t i = 0, e = c.size(); i < e; ++i) {
    if (time == c[i].time) {
      c[i].insertChange(index, bitOffset, bytes);
      return;
    }
  }
  Slot newSlot(time);
  newSlot.insertChange(index, bitOffset, bytes);
  push(newSlot);
}

void UpdateQueue::insertOrUpdate(Time time, std::string inst) {
  for (size_t i = 0, e = c.size(); i < e; ++i) {
    if (time == c[i].time) {
      c[i].insertChange(inst);
      return;
    }
  }
  Slot newSlot(time);
  newSlot.insertChange(inst);
  push(newSlot);
}

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

State::~State() {
  for (int i = 0, e = signals.size(); i < e; ++i)
    if (signals[i].value)
      std::free(signals[i].value);

  for (auto &entry : instances) {
    auto inst = entry.getValue();
    if (!inst.isEntity && inst.procState) {
      std::free(inst.procState->inst);
      std::free(inst.procState->senses);
      std::free(inst.procState);
    }
  }
}

Slot State::popQueue() {
  assert(!queue.empty() && "the event queue is empty");
  Slot pop = queue.top();
  queue.pop();
  return pop;
}

void State::pushQueue(Time t, int index, int bitOffset, APInt &bytes) {
  Time newTime = time + t;
  queue.insertOrUpdate(newTime, index, bitOffset, bytes);
}
void State::pushQueue(Time t, std::string inst) {
  Time newTime = time + t;
  queue.insertOrUpdate(newTime, inst);
}

int State::addSignal(std::string name, std::string owner) {
  signals.push_back(Signal(name, owner));
  return signals.size() - 1;
}

void State::addProcPtr(std::string name, ProcState *procStatePtr) {
  instances[name].procState = procStatePtr;
  // Copy string to owner name ptr.
  name.copy(instances[name].procState->inst, name.size());
  // Ensure the string is null-terminated.
  instances[name].procState->inst[name.size()] = '\0';
}

int State::addSignalData(int index, std::string owner, uint8_t *value,
                         uint64_t size) {
  auto inst = instances[owner];
  uint64_t globalIdx = inst.sensitivityList[index + inst.nArgs].globalIndex;

  // Add pointer and size to global signal table entry.
  signals[globalIdx].value = value;
  signals[globalIdx].size = size;

  // Add the value pointer to the signal detail struct for each instance this
  // signal appears in.
  for (auto inst : signals[globalIdx].triggers) {
    for (auto &detail : instances[inst].sensitivityList) {
      if (detail.globalIndex == globalIdx) {
        detail.value = value;
        break;
      }
    }
  }
  return globalIdx;
}

void State::dumpSignal(llvm::raw_ostream &out, int index) {
  auto &sig = signals[index];
  for (auto inst : sig.triggers) {
    std::string curr = inst, path = inst;
    while (instances[curr].name != sig.owner) {
      curr = instances[curr].parent;
      path = curr + "/" + path;
    }
    out << time.dump() << "  " << path << "/" << sig.name << "  " << sig.dump()
        << "\n";
  }
}

void State::dumpLayout() {
  llvm::errs() << "::------------------- Layout -------------------::\n";
  for (auto &inst : instances) {
    llvm::errs() << inst.getKey().str() << ":\n";
    llvm::errs() << "---parent: " << inst.getValue().parent << "\n";
    llvm::errs() << "---isEntity: " << inst.getValue().isEntity << "\n";
    llvm::errs() << "---sensitivity list: ";
    for (auto in : inst.getValue().sensitivityList) {
      llvm::errs() << in.globalIndex << " ";
    }
    llvm::errs() << "\n";
  }
  llvm::errs() << "::----------------------------------------------::\n";
}

void State::dumpSignalTriggers() {
  llvm::errs() << "::------------- Signal information -------------::\n";
  for (size_t i = 0, e = signals.size(); i < e; ++i) {
    llvm::errs() << signals[i].owner << "/" << signals[i].name << " triggers: ";
    for (auto trig : signals[i].triggers) {
      llvm::errs() << trig << " ";
    }
    llvm::errs() << "\n";
  }
  llvm::errs() << "::----------------------------------------------::\n";
}
