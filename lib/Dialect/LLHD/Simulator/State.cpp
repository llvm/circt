//===- State.cpp - LLHD simulator state -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the constructs used to keep track of the simulation
// state in the LLHD simulator.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LLHD/Simulator/State.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace llvm;
using namespace circt::llhd::sim;

//===----------------------------------------------------------------------===//
// Signal
//===----------------------------------------------------------------------===//
std::string Signal::toHexString() const {
  std::string ret;
  raw_string_ostream ss(ret);
  ss << "0x";
  for (int i = size - 1; i >= 0; --i) {
    ss << format_hex_no_prefix(static_cast<int>(value[i]), 2);
  }
  return ret;
}

std::string Signal::toHexString(unsigned elemIndex) const {
  assert(elements.size() > 0 && "the signal type has to be tuple or array!");
  auto elemSize = elements[elemIndex].second;
  auto ptr = value + elements[elemIndex].first;
  std::string ret;
  raw_string_ostream ss(ret);
  ss << "0x";
  for (int i = elemSize - 1; i >= 0; --i) {
    ss << format_hex_no_prefix(static_cast<int>(ptr[i]), 2);
  }
  return ret;
}
//===----------------------------------------------------------------------===//
// Slot
//===----------------------------------------------------------------------===//
void Slot::insertChange(int index, int bitOffset, uint8_t *bytes,
                        unsigned width) {
  auto obj = std::make_pair(
      bitOffset,
      APInt(width, makeArrayRef(reinterpret_cast<uint64_t *>(bytes),
                                llvm::divideCeil(width, 64))));

  // Map the signal index to the change buffer so we can retrieve
  // it after sorting.
  // Create a new change buffer if we don't have any unused one available for
  // reuse.
  if (changesSize >= changes.size()) {
    changes.push_back(std::make_pair(index, obj));
  } else {
    // Reuse the first available buffer.
    changes[changesSize] = std::make_pair(index, obj);
  }

  ++changesSize;
}

//===----------------------------------------------------------------------===//
// UpdateQueue
//===----------------------------------------------------------------------===//
void UpdateQueue::insertOrUpdate(Time time, int index, int bitOffset,
                                 uint8_t *bytes, unsigned width) {
  auto &slot = getOrCreateSlot(time);
  slot.insertChange(index, bitOffset, bytes, width);
}

void UpdateQueue::insertOrUpdate(Time time, unsigned inst) {
  auto &slot = getOrCreateSlot(time);
  slot.insertChange(inst);
}

Slot &UpdateQueue::getOrCreateSlot(Time time) {
  auto &top = begin()[topSlot];

  // Directly add to top slot.
  if (top.isUsed() && time == top.getTime()) {
    return top;
  }

  // We need to search through the queue for an existing slot only if we're
  // spawning an event later than the top slot. Adding to an existing slot
  // scheduled earlier than the top slot should never happens, as then it should
  // be the top.
  if (numEvents > 0 && top.getTime() < time) {
    for (size_t i = 0, e = size(); i < e; ++i) {
      if (time == begin()[i].getTime()) {
        return begin()[i];
      }
    }
  }

  // Spawn new event using an existing slot.
  if (!unused.empty()) {
    auto firstUnused = unused.pop_back_val();
    auto &newSlot = begin()[firstUnused];
    newSlot.occupy(time);

    // Update the top of the queue either if it is currently unused or the new
    // timestamp is earlier than it.
    if (!top.isUsed() || time < top.getTime())
      topSlot = firstUnused;

    ++numEvents;
    return newSlot;
  }

  // We do not have pre-allocated slots available, generate a new one.
  push_back(Slot(time));

  // Update the top of the queue either if it is currently unused or the new
  // timestamp is earlier than it.
  if (!top.isUsed() || time < top.getTime())
    topSlot = size() - 1;

  ++numEvents;
  return back();
}

const Slot &UpdateQueue::top() {
  assert(topSlot < size() && "top is pointing out of bounds!");

  // Sort the changes of the top slot such that all changes to the same signal
  // are in succession.
  auto &top = begin()[topSlot];
  top.sortChanges();
  return top;
}

void UpdateQueue::pop() {
  // Reset internal structures and decrease the event counter.
  auto &top = begin()[topSlot];
  top.release();
  --numEvents;

  // Add to unused slots list for easy retrieval.
  unused.push_back(topSlot);

  // Update the current top of the queue.
  topSlot = std::distance(
      begin(),
      std::min_element(begin(), end(), [](const auto &a, const auto &b) {
        // a is "smaller" than b if either a's timestamp is earlier than b's, or
        // b is unused (i.e. b has no actual meaning).
        return a.isUsed() && (a < b || !b.isUsed());
      }));
}

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

llvm::SmallVectorTemplateCommon<Instance>::iterator
State::findInstanceByName(std::string name) {
  auto II =
      std::find_if(instances.begin(), instances.end(),
                   [&](const auto &inst) { return name == inst.getName(); });

  assert(II != instances.end() && "instance does not exist!");

  return II;
}

int State::addSignalData(int index, std::string owner, uint8_t *value,
                         uint64_t size) {
  auto II = findInstanceByName(owner);

  uint64_t globalIdx = II->getSensiSigIndex(index);
  auto &sig = signals[globalIdx];

  // Add pointer and size to global signal table entry.
  sig.update(value, size);

  // Add the value pointer to the signal detail struct for each instance this
  // signal appears in.
  for (auto idx : signals[globalIdx].getTriggeredInstanceIndices()) {
    instances[idx].updateSignalDetail(globalIdx, sig.Value());
  }
  return globalIdx;
}

void State::addSignalElement(unsigned index, unsigned offset, unsigned size) {
  signals[index].pushElement(std::make_pair(offset, size));
}

void State::dumpSignal(llvm::raw_ostream &out, int index) {
  auto &sig = signals[index];
  for (auto instIdx : sig.getTriggeredInstanceIndices()) {
    out << time.toString() << "  " << instances[instIdx].getPath() << "/"
        << sig.getName() << "  " << sig.toHexString() << "\n";
  }
}

void State::dumpLayout() {
  llvm::errs() << "::------------------- Layout -------------------::\n";
  for (const auto &inst : instances) {
    llvm::errs() << inst.getName() << ":\n";
    llvm::errs() << "---path: " << inst.getPath() << "\n";
    llvm::errs() << "---isEntity: " << inst.isEntity() << "\n";
    llvm::errs() << "---sensitivity list: " << inst.dumpSensitivityList()
                 << "\n";
  }
  llvm::errs() << "::----------------------------------------------::\n";
}

void State::dumpSignalTriggers() {
  llvm::errs() << "::------------- Signal information -------------::\n";
  for (size_t i = 0, e = signals.size(); i < e; ++i) {
    llvm::errs() << signals[i].getOwner() << "/" << signals[i].getName()
                 << " triggers: ";
    for (auto trig : signals[i].getTriggeredInstanceIndices()) {
      llvm::errs() << trig << " ";
    }
    llvm::errs() << "\n";
  }
  llvm::errs() << "::----------------------------------------------::\n";
}
