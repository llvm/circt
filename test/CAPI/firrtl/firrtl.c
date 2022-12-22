/*===- firrtl.c - Simple test of C APIs -----------------------------------===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/* RUN: circt-capi-firrtl-test 2>&1 | FileCheck %s
 */

#include <stdio.h>

#include "circt-c/Dialect/FIRRTL.h"

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

#define EXPECT(c)                                                              \
  if (!(c)) {                                                                  \
    return __LINE__;                                                           \
  }

#define IF_ERR_RET(code)                                                       \
  do {                                                                         \
    int errCode_ = (code);                                                     \
    if (errCode_ != 0) {                                                       \
      return errCode_;                                                         \
    }                                                                          \
  } while (false);

#define EXPECT_EXPORT(expected)                                                \
  do {                                                                         \
    FirrtlStringRef str = firrtlExportFirrtl(ctx);                             \
    if (!firrtlStringRefEqual(str,                                             \
                              firrtlCreateStringRefFromCString(expected))) {   \
      printf("\n\n\n\n\n>> FAILED: %s (%d)\n-- expected --\n%s\n----\n-- "     \
             "actual --\n%.*s\n----\n\n\n\n\n",                                \
             __FUNCTION__, __LINE__, expected, (int)str.length, str.data);     \
      return __LINE__;                                                         \
    }                                                                          \
    firrtlDestroyString(ctx, str);                                             \
  } while (false)

#define MK_STR firrtlCreateStringRefFromCString

int runOnce(int (*fn)(FirrtlContext ctx)) {
  FirrtlContext ctx = firrtlCreateContext();
  int errCode = fn(ctx);
  firrtlDestroyContext(ctx);
  return errCode;
}

int testGenHeader(FirrtlContext ctx) {
  firrtlVisitCircuit(ctx, MK_STR("cIrCuIt"));
  firrtlVisitModule(ctx, MK_STR("cIrCuIt"));

  EXPECT_EXPORT("circuit cIrCuIt :\n  module cIrCuIt :\n\n");
  return 0;
}

int testGenPorts(FirrtlContext ctx) {
  FirrtlType tyUInt8 = {
      .kind = FIRRTL_TYPE_KIND_UINT,
      .u = {.uint = {.width = 8}},
  };
  FirrtlType tySInt8 = {
      .kind = FIRRTL_TYPE_KIND_SINT,
      .u = {.sint = {.width = 8}},
  };

  FirrtlType nestedVecElement = {.kind = FIRRTL_TYPE_KIND_VECTOR,
                                 .u.vector = {
                                     .type = &tyUInt8,
                                     .count = 5,
                                 }};

  FirrtlTypeBundleField nestedFieldElement[] = {
      {.flip = false, .name = MK_STR("nestedField1"), .type = &tyUInt8},
      {.flip = true, .name = MK_STR("nestedField2"), .type = &tySInt8},
  };

  FirrtlType nestedFieldElementTy = {
      .kind = FIRRTL_TYPE_KIND_BUNDLE,
      .u.bundle = {
          .fields = nestedFieldElement,
          .count = ARRAY_SIZE(nestedFieldElement),
      }};

  FirrtlTypeBundleField subFields[] = {
      {.flip = false, .name = MK_STR("field1"), .type = &tyUInt8},
      {.flip = true, .name = MK_STR("field2"), .type = &tySInt8},
      {.flip = false, .name = MK_STR("field3"), .type = &nestedVecElement},
      {.flip = true, .name = MK_STR("field4"), .type = &nestedFieldElementTy},
  };

  FirrtlType portCases[] = {
      tyUInt8,
      {
          .kind = FIRRTL_TYPE_KIND_UINT,
          .u = {.uint = {.width = 16}},
      },
      tySInt8,
      {
          .kind = FIRRTL_TYPE_KIND_SINT,
          .u = {.sint = {.width = 16}},
      },
      {
          .kind = FIRRTL_TYPE_KIND_CLOCK,
          .u = {.clock = {}},
      },
      {
          .kind = FIRRTL_TYPE_KIND_RESET,
          .u = {.reset = {}},
      },
      {
          .kind = FIRRTL_TYPE_KIND_ASYNC_RESET,
          .u = {.asyncReset = {}},
      },
      {
          .kind = FIRRTL_TYPE_KIND_ANALOG,
          .u = {.analog = {.width = 8}},
      },
      {
          .kind = FIRRTL_TYPE_KIND_ANALOG,
          .u = {.analog = {.width = 16}},
      },
      {
          .kind = FIRRTL_TYPE_KIND_VECTOR,
          .u = {.vector = {.type = &portCases[0], 3}},
      },
      {
          .kind = FIRRTL_TYPE_KIND_VECTOR,
          .u = {.vector = {.type = &portCases[1], 6}},
      },
      {
          .kind = FIRRTL_TYPE_KIND_VECTOR,
          .u = {.vector = {.type = &nestedVecElement, 10}},
      },
      {
          .kind = FIRRTL_TYPE_KIND_BUNDLE,
          .u = {.bundle = {.fields = subFields, ARRAY_SIZE(subFields)}},
      },
  };

  firrtlVisitCircuit(ctx, MK_STR("PortTest"));
  firrtlVisitModule(ctx, MK_STR("PortTest"));

  for (unsigned int d = 0; d < 2; d++) {
    FirrtlPortDirection dir;
    const char *baseName;
    if (d == 0) {
      dir = FIRRTL_PORT_DIRECTION_INPUT;
      baseName = "inPort";
    } else {
      dir = FIRRTL_PORT_DIRECTION_OUTPUT;
      baseName = "outPort";
    }
    for (unsigned int i = 0; i < ARRAY_SIZE(portCases); i++) {
      char portName[64] = {};
      sprintf(portName, "%s%d", baseName, i);
      firrtlVisitPort(ctx, MK_STR(portName), dir, &portCases[i]);
    }
  }

  EXPECT_EXPORT("circuit PortTest :\n\
  module PortTest :\n\
    input inPort0 : UInt<8>\n\
    input inPort1 : UInt<16>\n\
    input inPort2 : SInt<8>\n\
    input inPort3 : SInt<16>\n\
    input inPort4 : Clock\n\
    input inPort5 : Reset\n\
    input inPort6 : AsyncReset\n\
    input inPort7 : Analog<8>\n\
    input inPort8 : Analog<16>\n\
    input inPort9 : UInt<8>[3]\n\
    input inPort10 : UInt<16>[6]\n\
    input inPort11 : UInt<8>[5][10]\n\
    input inPort12 : { field1 : UInt<8>, flip field2 : SInt<8>, field3 : UInt<8>[5], flip field4 : { nestedField1 : UInt<8>, flip nestedField2 : SInt<8> } }\n\
    output outPort0 : UInt<8>\n\
    output outPort1 : UInt<16>\n\
    output outPort2 : SInt<8>\n\
    output outPort3 : SInt<16>\n\
    output outPort4 : Clock\n\
    output outPort5 : Reset\n\
    output outPort6 : AsyncReset\n\
    output outPort7 : Analog<8>\n\
    output outPort8 : Analog<16>\n\
    output outPort9 : UInt<8>[3]\n\
    output outPort10 : UInt<16>[6]\n\
    output outPort11 : UInt<8>[5][10]\n\
    output outPort12 : { field1 : UInt<8>, flip field2 : SInt<8>, field3 : UInt<8>[5], flip field4 : { nestedField1 : UInt<8>, flip nestedField2 : SInt<8> } }\n\n");

  return 0;
}

int testGenerated() {
  IF_ERR_RET(runOnce(&testGenHeader));
  IF_ERR_RET(runOnce(&testGenPorts));

  return 0;
}

void errorHandler(FirrtlStringRef message, void *userData) {
  *(bool *)userData = true;
}

int testErrNoCircuit(FirrtlContext ctx) {
  bool triggered = false;
  firrtlSetErrorHandler(ctx, errorHandler, &triggered);

  EXPECT(triggered == false);
  firrtlVisitModule(ctx, MK_STR("NoCircuit"));
  EXPECT(triggered == true);
  EXPECT_EXPORT("");

  return 0;
}

int testErrNoModule(FirrtlContext ctx) {
  bool triggered = false;
  firrtlSetErrorHandler(ctx, errorHandler, &triggered);

  firrtlVisitCircuit(ctx, MK_STR("NoModule"));
  EXPECT(triggered == false);

  FirrtlType tyUInt8 = {
      .kind = FIRRTL_TYPE_KIND_UINT,
      .u = {.uint = {.width = 8}},
  };
  firrtlVisitPort(ctx, MK_STR("port"), FIRRTL_PORT_DIRECTION_INPUT, &tyUInt8);
  EXPECT(triggered == true);
  EXPECT_EXPORT("circuit NoModule :\n");

  return 0;
}

int testErrNoHeader(FirrtlContext ctx) {
  bool triggered = false;
  firrtlSetErrorHandler(ctx, errorHandler, &triggered);

  EXPECT(triggered == false);

  FirrtlType tyUInt8 = {
      .kind = FIRRTL_TYPE_KIND_UINT,
      .u = {.uint = {.width = 8}},
  };
  firrtlVisitPort(ctx, MK_STR("port"), FIRRTL_PORT_DIRECTION_INPUT, &tyUInt8);
  EXPECT(triggered == true);
  EXPECT_EXPORT("");

  return 0;
}

int testExpectedError() {
  IF_ERR_RET(runOnce(&testErrNoCircuit));
  IF_ERR_RET(runOnce(&testErrNoModule));
  IF_ERR_RET(runOnce(&testErrNoHeader));

  return 0;
}

int main() {
  fprintf(stderr, "@generated\n");
  int errCode = testGenerated();
  fprintf(stderr, "%d\n", errCode);

  fprintf(stderr, "@expectedError\n");
  errCode = testExpectedError();
  fprintf(stderr, "%d\n", errCode);

  // clang-format off
  // CHECK-LABEL: @generated
  // CHECK: 0
  // CHECK-LABEL: @expectedError
  // CHECK: 0
  // clang-format on

  return 0;
}
