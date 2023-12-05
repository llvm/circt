//===- FIRRTLConnectionGraph.h - Graph of connections in FIRRTL -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utilities for working with FIRRTL operations using LLVM
// graph utilties.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLConnectionGraph.h"

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"

using namespace circt;
using namespace firrtl;
using namespace detail;

//===----------------------------------------------------------------------===//
//
// FModuleOpIterator
//
//===----------------------------------------------------------------------===//

FModuleOpIterator::FModuleOpIterator(FModuleOp op) : op(op) {
  auto &operations = op.getBodyBlock()->getOperations();
  iterator = operations.begin();
  iteratorEnd = operations.end();
}

bool FModuleOpIterator::operator==(const FModuleOpIterator &other) const {
  if (iterator == iteratorEnd)
    return other.iterator == other.iteratorEnd;

  return iterator == other.iterator && iteratorEnd == other.iteratorEnd;
}

FModuleOpIterator &FModuleOpIterator::operator++() {
  assert(iterator != iteratorEnd &&
         "incrementing past the end of the iterator");
  ++iterator;
  return *this;
}

//===----------------------------------------------------------------------===//
//
// FConnectLikeIterator
//
//===----------------------------------------------------------------------===//

bool FConnectLikeIterator::operator==(const FConnectLikeIterator &other) const {
  if (visited)
    return other.visited;
  return op == other.op && !other.visited;
}

FIRRTLOperation &FConnectLikeIterator::operator*() {
  assert(!visited && "tried to dereference past the end");
  auto dest = op.getDest();
  FIRRTLOperation *result;
  if (auto blockArg = dyn_cast<BlockArgument>(dest)) {
    result = blockArg.getParentBlock()->getParentOp();
  } else {
    result = dest.getDefiningOp();
  }
  return *result;
}

FConnectLikeIterator &FConnectLikeIterator::operator++() {
  assert(!visited && "incrementing past the end of the iterator");
  visited = true;
  return *this;
}

//===----------------------------------------------------------------------===//
//
// ResultIterator
//
//===----------------------------------------------------------------------===//

ResultIterator::ResultIterator(FIRRTLOperation *op)
    : op(op), resultIndexEnd(op->getNumResults()) {
  if (resultIndex == resultIndexEnd)
    return;
  auto result = op->getResult(resultIndex);
  resultIterator = result.use_begin();
  resultIteratorEnd = result.use_end();
  fastforward();
}

void ResultIterator::fastforward() {
  while (resultIterator != resultIteratorEnd &&
         dyn_cast<FConnectLike>(resultIterator.getUser()) &&
         resultIterator.getOperand()->getOperandNumber() == 0)
    ++resultIterator;
  while (resultIterator == resultIteratorEnd && resultIndex != resultIndexEnd) {
    ++resultIndex;
    if (resultIndex == resultIndexEnd)
      break;
    auto result = op->getResult(resultIndex);
    resultIterator = result.use_begin();
    resultIteratorEnd = result.use_end();
    while (resultIterator != resultIteratorEnd &&
           dyn_cast<FConnectLike>(resultIterator.getUser()) &&
           resultIterator.getOperand()->getOperandNumber() == 0)
      ++resultIterator;
  }
}

bool ResultIterator::operator==(const ResultIterator &other) const {
  if (resultIndex == resultIndexEnd)
    return other.resultIndex == other.resultIndexEnd;

  return resultIterator == other.resultIterator &&
         resultIteratorEnd == other.resultIteratorEnd &&
         resultIndex == other.resultIndex &&
         resultIndexEnd == other.resultIndexEnd;
}

FIRRTLOperation &ResultIterator::operator*() const {
  assert(resultIndex != resultIndexEnd && "tried to dereference past the end");
  return *resultIterator->getOwner();
}

ResultIterator &ResultIterator::operator++() {
  assert(resultIndex != resultIndexEnd &&
         "incrementing past the end of the iterator");
  ++resultIterator;
  fastforward();
  return *this;
}

//===----------------------------------------------------------------------===//
//
// ConnectionIterator
//
//===----------------------------------------------------------------------===//

ConnectionIterator::ConnectionIterator(FIRRTLOperation *op, bool empty) {
  assert(op && "op should not be null");
  TypeSwitch<FIRRTLOperation *>(op)
      .Case<FModuleOp>([&](auto op) {
        iterator = empty ? FModuleOpIterator() : FModuleOpIterator(op);
      })
      .Case<FConnectLike>([&](auto op) {
        iterator = empty ? FConnectLikeIterator() : FConnectLikeIterator(op);
      })
      .Default([&](auto *op) {
        iterator = empty ? ResultIterator() : ResultIterator(op);
      });
  this->op = op;
}

bool ConnectionIterator::operator==(const ConnectionIterator &other) const {
  assert(iterator.index() == other.iterator.index() &&
         "comparing different iterator variants");
  switch (iterator.index()) {
  case 0:
    assert(false && "should be impossible");
    break;
  case 1:
    return std::get<FModuleOpIterator>(iterator) ==
           std::get<FModuleOpIterator>(other.iterator);
  case 2:
    return std::get<FConnectLikeIterator>(iterator) ==
           std::get<FConnectLikeIterator>(other.iterator);
  case 3:
    return std::get<ResultIterator>(iterator) ==
           std::get<ResultIterator>(other.iterator);
  }
  return false;
}

bool ConnectionIterator::operator!=(const ConnectionIterator &other) const {
  return !(*this == other);
}

FIRRTLOperation *ConnectionIterator::operator*() {
  FIRRTLOperation *result;
  switch (iterator.index()) {
  case 0:
    assert(false && "should be impossible");
    break;
  case 1:
    result = &(*std::get<FModuleOpIterator>(iterator));
    break;
  case 2:
    result = &(*std::get<FConnectLikeIterator>(iterator));
    break;
  case 3:
    result = &(*std::get<ResultIterator>(iterator));
    break;
  }
  return result;
}

ConnectionIterator &ConnectionIterator::operator++() {
  switch (iterator.index()) {
  case 0:
    assert(false && "should be impossible");
    break;
  case 1:
    ++std::get<FModuleOpIterator>(iterator);
    break;
  case 2:
    ++std::get<FConnectLikeIterator>(iterator);
    break;
  case 3:
    ++std::get<ResultIterator>(iterator);
    break;
  }
  return *this;
}

ConnectionIterator ConnectionIterator::operator++(int) {
  ConnectionIterator result(*this);
  ++(*this);
  return result;
}
