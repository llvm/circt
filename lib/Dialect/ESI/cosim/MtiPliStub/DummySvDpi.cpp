//===- DummySvDpi.cpp - Dummy stubs the svdpi.h -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dummy stubs for every function in svdpi.h. This produces an MtiPli.so for
// linking a DPI library in the absence of a simulator. Should dynamicall link
// with the MtiPli library supplied with the simulator at runtime. This shared
// object should not be distributed with CIRCT.
//
//===----------------------------------------------------------------------===//

#include "external/dpi/svdpi.h"
#undef NDEBUG
#include <cassert>

const char *svDpiVersion(void) {
  assert(false && "Linking error: should not ever execute.");
}
svBit svGetBitselBit(const svBitVecVal *s, int i) {
  assert(false && "Linking error: should not ever execute.");
}
svLogic svGetBitselLogic(const svLogicVecVal *s, int i) {
  assert(false && "Linking error: should not ever execute.");
}
void svPutBitselBit(svBitVecVal *d, int i, svBit s) {
  assert(false && "Linking error: should not ever execute.");
}
void svPutBitselLogic(svLogicVecVal *d, int i, svLogic s) {
  assert(false && "Linking error: should not ever execute.");
}
void svGetPartselBit(svBitVecVal *d, const svBitVecVal *s, int i, int w) {
  assert(false && "Linking error: should not ever execute.");
}
void svGetPartselLogic(svLogicVecVal *d, const svLogicVecVal *s, int i, int w) {
  assert(false && "Linking error: should not ever execute.");
}
void svPutPartselBit(svBitVecVal *d, const svBitVecVal s, int i, int w) {
  assert(false && "Linking error: should not ever execute.");
}
void svPutPartselLogic(svLogicVecVal *d, const svLogicVecVal s, int i, int w) {
  assert(false && "Linking error: should not ever execute.");
}
int svLeft(const svOpenArrayHandle h, int d) {
  assert(false && "Linking error: should not ever execute.");
}
int svRight(const svOpenArrayHandle h, int d) {
  assert(false && "Linking error: should not ever execute.");
}
int svLow(const svOpenArrayHandle h, int d) {
  assert(false && "Linking error: should not ever execute.");
}
int svHigh(const svOpenArrayHandle h, int d) {
  assert(false && "Linking error: should not ever execute.");
}
int svIncrement(const svOpenArrayHandle h, int d) {
  assert(false && "Linking error: should not ever execute.");
}
int svSize(const svOpenArrayHandle h, int d) {
  assert(false && "Linking error: should not ever execute.");
}
int svDimensions(const svOpenArrayHandle h) {
  assert(false && "Linking error: should not ever execute.");
}
void *svGetArrayPtr(const svOpenArrayHandle) {
  assert(false && "Linking error: should not ever execute.");
}
int svSizeOfArray(const svOpenArrayHandle) {
  assert(false && "Linking error: should not ever execute.");
}
void *svGetArrElemPtr(const svOpenArrayHandle, int indx1, ...) {
  assert(false && "Linking error: should not ever execute.");
}
void *svGetArrElemPtr1(const svOpenArrayHandle, int indx1) {
  assert(false && "Linking error: should not ever execute.");
}
void *svGetArrElemPtr2(const svOpenArrayHandle, int indx1, int indx2) {
  assert(false && "Linking error: should not ever execute.");
}
void *svGetArrElemPtr3(const svOpenArrayHandle, int indx1, int indx2,
                       int indx3) {
  assert(false && "Linking error: should not ever execute.");
}
void svPutBitArrElemVecVal(const svOpenArrayHandle d, const svBitVecVal *s,
                           int indx1, ...) {
  assert(false && "Linking error: should not ever execute.");
}
void svPutBitArrElem1VecVal(const svOpenArrayHandle d, const svBitVecVal *s,
                            int indx1) {
  assert(false && "Linking error: should not ever execute.");
}
void svPutBitArrElem2VecVal(const svOpenArrayHandle d, const svBitVecVal *s,
                            int indx1, int indx2) {
  assert(false && "Linking error: should not ever execute.");
}
void svPutBitArrElem3VecVal(const svOpenArrayHandle d, const svBitVecVal *s,
                            int indx1, int indx2, int indx3) {
  assert(false && "Linking error: should not ever execute.");
}
void svPutLogicArrElemVecVal(const svOpenArrayHandle d, const svLogicVecVal *s,
                             int indx1, ...) {
  assert(false && "Linking error: should not ever execute.");
}
void svPutLogicArrElem1VecVal(const svOpenArrayHandle d, const svLogicVecVal *s,
                              int indx1) {
  assert(false && "Linking error: should not ever execute.");
}
void svPutLogicArrElem2VecVal(const svOpenArrayHandle d, const svLogicVecVal *s,
                              int indx1, int indx2) {
  assert(false && "Linking error: should not ever execute.");
}
void svPutLogicArrElem3VecVal(const svOpenArrayHandle d, const svLogicVecVal *s,
                              int indx1, int indx2, int indx3) {
  assert(false && "Linking error: should not ever execute.");
}
void svGetBitArrElemVecVal(svBitVecVal *d, const svOpenArrayHandle s, int indx1,
                           ...) {
  assert(false && "Linking error: should not ever execute.");
}
void svGetBitArrElem1VecVal(svBitVecVal *d, const svOpenArrayHandle s,
                            int indx1) {
  assert(false && "Linking error: should not ever execute.");
}
void svGetBitArrElem2VecVal(svBitVecVal *d, const svOpenArrayHandle s,
                            int indx1, int indx2) {
  assert(false && "Linking error: should not ever execute.");
}
void svGetBitArrElem3VecVal(svBitVecVal *d, const svOpenArrayHandle s,
                            int indx1, int indx2, int indx3) {
  assert(false && "Linking error: should not ever execute.");
}
void svGetLogicArrElemVecVal(svLogicVecVal *d, const svOpenArrayHandle s,
                             int indx1, ...) {
  assert(false && "Linking error: should not ever execute.");
}
void svGetLogicArrElem1VecVal(svLogicVecVal *d, const svOpenArrayHandle s,
                              int indx1) {
  assert(false && "Linking error: should not ever execute.");
}
void svGetLogicArrElem2VecVal(svLogicVecVal *d, const svOpenArrayHandle s,
                              int indx1, int indx2) {
  assert(false && "Linking error: should not ever execute.");
}
void svGetLogicArrElem3VecVal(svLogicVecVal *d, const svOpenArrayHandle s,
                              int indx1, int indx2, int indx3) {
  assert(false && "Linking error: should not ever execute.");
}
svBit svGetBitArrElem(const svOpenArrayHandle s, int indx1, ...) {
  assert(false && "Linking error: should not ever execute.");
}
svBit svGetBitArrElem1(const svOpenArrayHandle s, int indx1) {
  assert(false && "Linking error: should not ever execute.");
}
svBit svGetBitArrElem2(const svOpenArrayHandle s, int indx1, int indx2) {
  assert(false && "Linking error: should not ever execute.");
}
svBit svGetBitArrElem3(const svOpenArrayHandle s, int indx1, int indx2,
                       int indx3) {
  assert(false && "Linking error: should not ever execute.");
}
svLogic svGetLogicArrElem(const svOpenArrayHandle s, int indx1, ...) {
  assert(false && "Linking error: should not ever execute.");
}
svLogic svGetLogicArrElem1(const svOpenArrayHandle s, int indx1) {
  assert(false && "Linking error: should not ever execute.");
}
svLogic svGetLogicArrElem2(const svOpenArrayHandle s, int indx1, int indx2) {
  assert(false && "Linking error: should not ever execute.");
}
svLogic svGetLogicArrElem3(const svOpenArrayHandle s, int indx1, int indx2,
                           int indx3) {
  assert(false && "Linking error: should not ever execute.");
}
void svPutLogicArrElem(const svOpenArrayHandle d, svLogic value, int indx1,
                       ...) {
  assert(false && "Linking error: should not ever execute.");
}
void svPutLogicArrElem1(const svOpenArrayHandle d, svLogic value, int indx1) {
  assert(false && "Linking error: should not ever execute.");
}
void svPutLogicArrElem2(const svOpenArrayHandle d, svLogic value, int indx1,
                        int indx2) {
  assert(false && "Linking error: should not ever execute.");
}
void svPutLogicArrElem3(const svOpenArrayHandle d, svLogic value, int indx1,
                        int indx2, int indx3) {
  assert(false && "Linking error: should not ever execute.");
}
void svPutBitArrElem(const svOpenArrayHandle d, svBit value, int indx1, ...) {
  assert(false && "Linking error: should not ever execute.");
}
void svPutBitArrElem1(const svOpenArrayHandle d, svBit value, int indx1) {
  assert(false && "Linking error: should not ever execute.");
}
void svPutBitArrElem2(const svOpenArrayHandle d, svBit value, int indx1,
                      int indx2) {
  assert(false && "Linking error: should not ever execute.");
}
void svPutBitArrElem3(const svOpenArrayHandle d, svBit value, int indx1,
                      int indx2, int indx3) {
  assert(false && "Linking error: should not ever execute.");
}
svScope svGetScope(void) {
  assert(false && "Linking error: should not ever execute.");
}
svScope svSetScope(const svScope scope) {
  assert(false && "Linking error: should not ever execute.");
}
const char *svGetNameFromScope(const svScope) {
  assert(false && "Linking error: should not ever execute.");
}
svScope svGetScopeFromName(const char *scopeName) {
  assert(false && "Linking error: should not ever execute.");
}
int svPutUserData(const svScope scope, void *userKey, void *userData) {
  assert(false && "Linking error: should not ever execute.");
}
void *svGetUserData(const svScope scope, void *userKey) {
  assert(false && "Linking error: should not ever execute.");
}
int svGetCallerInfo(const char **fileName, int *lineNumber) {
  assert(false && "Linking error: should not ever execute.");
}
int svIsDisabledState(void) {
  assert(false && "Linking error: should not ever execute.");
}
void svAckDisabledState(void) {
  assert(false && "Linking error: should not ever execute.");
}
