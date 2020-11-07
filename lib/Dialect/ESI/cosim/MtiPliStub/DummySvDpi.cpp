// -*- mode: C++; c-file-style: "cc-mode" -*-
//*************************************************************************
// Based on code from Verilator:
//
// Copyright 2009-2020 by Wilson Snyder. This program is free software; you can
// redistribute it and/or modify it under the terms of either the GNU
// Lesser General Public License Version 3 or the Perl Artistic License
// Version 2.0.
// SPDX-License-Identifier: LGPL-3.0-only OR Artistic-2.0
//
//=========================================================================
///
/// \file
/// \brief Stub functions purely for compilation so we don't have to have an RTL
/// simulator to link against
///
//=========================================================================

#define _VERILATED_DPI_CPP_

// On MSVC++ we need svdpi.h to declare exports, not imports
#define DPI_PROTOTYPES
#undef XXTERN
#define XXTERN DPI_EXTERN DPI_DLLESPEC
#undef EETERN
#define EETERN DPI_EXTERN DPI_DLLESPEC

#include "external/dpi/svdpi.h"

//======================================================================
//======================================================================
//======================================================================
// DPI ROUTINES

const char *svDpiVersion() { return "1800-2005"; }

//======================================================================
// Bit-select utility functions.

svBit svGetBitselBit(const svBitVecVal *sp, int bit) { return 0; }
svLogic svGetBitselLogic(const svLogicVecVal *sp, int bit) { return 0; }

void svPutBitselBit(svBitVecVal *dp, int bit, svBit s) {}
void svPutBitselLogic(svLogicVecVal *dp, int bit, svLogic s) {}

void svGetPartselBit(svBitVecVal *dp, const svBitVecVal *sp, int lsb,
                     int width) {}
void svGetPartselLogic(svLogicVecVal *dp, const svLogicVecVal *sp, int lsb,
                       int width) {}
void svPutPartselBit(svBitVecVal *dp, const svBitVecVal s, int lbit,
                     int width) {}

// cppcheck-suppress passedByValue
void svPutPartselLogic(svLogicVecVal *dp, const svLogicVecVal s, int lbit,
                       int width) {}

//======================================================================
// Open array querying functions

int svLeft(const svOpenArrayHandle h, int d) { return 0; }
int svRight(const svOpenArrayHandle h, int d) { return 0; }
int svLow(const svOpenArrayHandle h, int d) { return 0; }
int svHigh(const svOpenArrayHandle h, int d) { return 0; }
int svIncrement(const svOpenArrayHandle h, int d) { return 0; }
int svSize(const svOpenArrayHandle h, int d) { return 0; }
int svDimensions(const svOpenArrayHandle h) { return 0; }
/// Return pointer to open array data, or NULL if not in IEEE standard C layout
void *svGetArrayPtr(const svOpenArrayHandle h) { return nullptr; }
/// Return size of open array, or 0 if not in IEEE standard C layout
int svSizeOfArray(const svOpenArrayHandle h) { return 0; }

//======================================================================
// DPI accessors that simply call above functions

void *svGetArrElemPtr(const svOpenArrayHandle h, int indx1, ...) {
  return nullptr;
}
void *svGetArrElemPtr1(const svOpenArrayHandle h, int indx1) { return nullptr; }
void *svGetArrElemPtr2(const svOpenArrayHandle h, int indx1, int indx2) {
  return nullptr;
}
void *svGetArrElemPtr3(const svOpenArrayHandle h, int indx1, int indx2,
                       int indx3) {
  return nullptr;
}

void svPutBitArrElemVecVal(const svOpenArrayHandle d, const svBitVecVal *s,
                           int indx1, ...) {}
void svPutBitArrElem1VecVal(const svOpenArrayHandle d, const svBitVecVal *s,
                            int indx1) {}
void svPutBitArrElem2VecVal(const svOpenArrayHandle d, const svBitVecVal *s,
                            int indx1, int indx2) {}
void svPutBitArrElem3VecVal(const svOpenArrayHandle d, const svBitVecVal *s,
                            int indx1, int indx2, int indx3) {}
void svPutLogicArrElemVecVal(const svOpenArrayHandle d, const svLogicVecVal *s,
                             int indx1, ...) {}
void svPutLogicArrElem1VecVal(const svOpenArrayHandle d, const svLogicVecVal *s,
                              int indx1) {}
void svPutLogicArrElem2VecVal(const svOpenArrayHandle d, const svLogicVecVal *s,
                              int indx1, int indx2) {}
void svPutLogicArrElem3VecVal(const svOpenArrayHandle d, const svLogicVecVal *s,
                              int indx1, int indx2, int indx3) {}

//======================================================================
// From simulator storage into user space

void svGetBitArrElemVecVal(svBitVecVal *d, const svOpenArrayHandle s, int indx1,
                           ...) {}
void svGetBitArrElem1VecVal(svBitVecVal *d, const svOpenArrayHandle s,
                            int indx1) {}
void svGetBitArrElem2VecVal(svBitVecVal *d, const svOpenArrayHandle s,
                            int indx1, int indx2) {}
void svGetBitArrElem3VecVal(svBitVecVal *d, const svOpenArrayHandle s,
                            int indx1, int indx2, int indx3) {}
void svGetLogicArrElemVecVal(svLogicVecVal *d, const svOpenArrayHandle s,
                             int indx1, ...) {}
void svGetLogicArrElem1VecVal(svLogicVecVal *d, const svOpenArrayHandle s,
                              int indx1) {}
void svGetLogicArrElem2VecVal(svLogicVecVal *d, const svOpenArrayHandle s,
                              int indx1, int indx2) {}
void svGetLogicArrElem3VecVal(svLogicVecVal *d, const svOpenArrayHandle s,
                              int indx1, int indx2, int indx3) {}

svBit svGetBitArrElem(const svOpenArrayHandle s, int indx1, ...) { return 0; }

svBit svGetBitArrElem1(const svOpenArrayHandle s, int indx1) { return 0; }
svBit svGetBitArrElem2(const svOpenArrayHandle s, int indx1, int indx2) {
  return 0;
}
svBit svGetBitArrElem3(const svOpenArrayHandle s, int indx1, int indx2,
                       int indx3) {
  return 0;
}
svLogic svGetLogicArrElem(const svOpenArrayHandle s, int indx1, ...) {
  return 0;
}

svLogic svGetLogicArrElem1(const svOpenArrayHandle s, int indx1) { return 0; }
svLogic svGetLogicArrElem2(const svOpenArrayHandle s, int indx1, int indx2) {
  return 0;
}
svLogic svGetLogicArrElem3(const svOpenArrayHandle s, int indx1, int indx2,
                           int indx3) {
  return 0;
}

void svPutBitArrElem(const svOpenArrayHandle d, svBit value, int indx1, ...) {}

void svPutBitArrElem1(const svOpenArrayHandle d, svBit value, int indx1) {}
void svPutBitArrElem2(const svOpenArrayHandle d, svBit value, int indx1,
                      int indx2) {}
void svPutBitArrElem3(const svOpenArrayHandle d, svBit value, int indx1,
                      int indx2, int indx3) {}
void svPutLogicArrElem(const svOpenArrayHandle d, svLogic value, int indx1,
                       ...) {}

void svPutLogicArrElem1(const svOpenArrayHandle d, svLogic value, int indx1) {}
void svPutLogicArrElem2(const svOpenArrayHandle d, svLogic value, int indx1,
                        int indx2) {}
void svPutLogicArrElem3(const svOpenArrayHandle d, svLogic value, int indx1,
                        int indx2, int indx3) {}

//======================================================================
// Functions for working with DPI context

svScope svGetScope() { return nullptr; }

svScope svSetScope(const svScope scope) { return nullptr; }

const char *svGetNameFromScope(const svScope scope) { return nullptr; }

svScope svGetScopeFromName(const char *scopeName) { return nullptr; }

int svPutUserData(const svScope scope, void *userKey, void *userData) {
  return 0;
}

void *svGetUserData(const svScope scope, void *userKey) { return 0; }

int svGetCallerInfo(const char **fileNamepp, int *lineNumberp) { return false; }

//======================================================================
// Disables

int svIsDisabledState() { return 0; }

void svAckDisabledState() {}
