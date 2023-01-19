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
#define MK_REF_EXPR_INLINE(expr)                                               \
  {                                                                            \
    .kind = FIRRTL_EXPR_KIND_REF, .u = {.ref = {.value = MK_STR(expr)} }       \
  }

void errorHandler(FirrtlStringRef message, void *userData) {
  ++(*(size_t *)userData);
}

int runOnce(int (*fn)(FirrtlContext ctx, size_t *errCount)) {
  FirrtlContext ctx = firrtlCreateContext();
  size_t errCount = 0;
  firrtlSetErrorHandler(ctx, errorHandler, &errCount);
  int errCode = fn(ctx, &errCount);
  firrtlDestroyContext(ctx);
  return errCode;
}

int testGenHeader(FirrtlContext ctx, size_t *errCount) {
  firrtlVisitCircuit(ctx, MK_STR("cIrCuIt"));
  firrtlVisitModule(ctx, MK_STR("cIrCuIt"));

  EXPECT_EXPORT("circuit cIrCuIt :\n  module cIrCuIt :\n\n");
  EXPECT(*errCount == 0);
  return 0;
}

int testGenPorts(FirrtlContext ctx, size_t *errCount) {
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

  EXPECT(*errCount == 0);
  return 0;
}

int testGenExtModule(FirrtlContext ctx, size_t *errCount) {
  FirrtlType tyUInt8 = {
      .kind = FIRRTL_TYPE_KIND_UINT,
      .u = {.uint = {.width = 8}},
  };
  FirrtlType tySInt8 = {
      .kind = FIRRTL_TYPE_KIND_SINT,
      .u = {.sint = {.width = 8}},
  };

  firrtlVisitCircuit(ctx, MK_STR("Meow"));
  firrtlVisitModule(ctx, MK_STR("Meow"));
  firrtlVisitExtModule(ctx, MK_STR("ExtMeow"), MK_STR(""));
  firrtlVisitExtModule(ctx, MK_STR("ExtMeow2"), MK_STR("ExtMeow2"));
  firrtlVisitPort(ctx, MK_STR("inPort"), FIRRTL_PORT_DIRECTION_INPUT, &tyUInt8);
  firrtlVisitPort(ctx, MK_STR("outPort"), FIRRTL_PORT_DIRECTION_OUTPUT,
                  &tySInt8);

  FirrtlParameter parameters[] = {
      {.kind = FIRRTL_PARAMETER_KIND_INT,
       .u =
           {
               .int_ = {.value = 12},
           }},
      {.kind = FIRRTL_PARAMETER_KIND_DOUBLE,
       .u =
           {
               .double_ = {.value = 34.56},
           }},
      {.kind = FIRRTL_PARAMETER_KIND_STRING,
       .u =
           {
               .string = {.value = MK_STR("string")},
           }},
      {.kind = FIRRTL_PARAMETER_KIND_RAW,
       .u =
           {
               .string = {.value = MK_STR("raw")},
           }},
  };

  for (unsigned int i = 0; i < ARRAY_SIZE(parameters); i++) {
    char paramName[64] = {};
    sprintf(paramName, "param%d", i);
    firrtlVisitParameter(ctx, MK_STR(paramName), &parameters[i]);
  }

  EXPECT_EXPORT("circuit Meow :\n\
  module Meow :\n\n\
  extmodule ExtMeow :\n\n\
  extmodule ExtMeow2 :\n\
    input inPort : UInt<8>\n\
    output outPort : SInt<8>\n\
    defname = ExtMeow2\n\
    parameter param0 = 0\n\
    parameter param1 = 34.560000000000002\n\
    parameter param2 = \"string\"\n\
    parameter param3 = \"raw\"\n\n");

  EXPECT(*errCount == 0);

  // duplicate test
  firrtlVisitParameter(ctx, MK_STR("param0"), &parameters[0]);

  EXPECT(*errCount == 1);

  firrtlVisitExtModule(ctx, MK_STR("ExtMeow3"), MK_STR("ExtMeow3"));
  firrtlVisitParameter(ctx, MK_STR("param0"), &parameters[0]);

  EXPECT(*errCount == 1);

  EXPECT_EXPORT("circuit Meow :\n\
  module Meow :\n\n\
  extmodule ExtMeow :\n\n\
  extmodule ExtMeow2 :\n\
    input inPort : UInt<8>\n\
    output outPort : SInt<8>\n\
    defname = ExtMeow2\n\
    parameter param0 = 0\n\
    parameter param1 = 34.560000000000002\n\
    parameter param2 = \"string\"\n\
    parameter param3 = \"raw\"\n\n\
  extmodule ExtMeow3 :\n\
    defname = ExtMeow3\n\
    parameter param0 = 0\n\n");
  EXPECT(*errCount == 1);

  return 0;
}

int testGenStatement(FirrtlContext ctx, size_t *errCount) {
  firrtlVisitCircuit(ctx, MK_STR("StatementTest"));
  firrtlVisitModule(ctx, MK_STR("StatementTest"));

  FirrtlType analogs[3];
  for (unsigned int i = 0; i < ARRAY_SIZE(analogs); i++) {
    FirrtlType analog = {.kind = FIRRTL_TYPE_KIND_ANALOG,
                         .u = {.analog = {.width = 8}}};
    char analogName[64] = {};
    sprintf(analogName, "analog%d", i + 1);
    firrtlVisitPort(ctx, MK_STR(analogName), FIRRTL_PORT_DIRECTION_INPUT,
                    &analog);
    analogs[i] = analog;
  }

  FirrtlType uint1 = {.kind = FIRRTL_TYPE_KIND_UINT,
                      .u = {.uint = {.width = 1}}};
  FirrtlType uint8s[2];
  for (unsigned int i = 0; i < ARRAY_SIZE(uint8s); i++) {
    FirrtlType port = {.kind = FIRRTL_TYPE_KIND_UINT,
                       .u = {.uint = {.width = 8}}};
    char name[64] = {};
    sprintf(name, "uintPort%d", i + 1);
    firrtlVisitPort(ctx, MK_STR(name), FIRRTL_PORT_DIRECTION_INPUT, &port);
    uint8s[i] = port;
  }

  FirrtlTypeBundleField analogFields[] = {
      {.name = MK_STR("field1"), .flip = false, .type = &analogs[0]},
      {.name = MK_STR("field2"), .flip = true, .type = &analogs[1]},
      {.name = MK_STR("field3"), .flip = false, .type = &analogs[2]},
      {.name = MK_STR("field4"), .flip = true, .type = &uint8s[0]},
  };
  FirrtlType analogBundle = {
      .kind = FIRRTL_TYPE_KIND_BUNDLE,
      .u = {.bundle = {.fields = analogFields,
                       .count = ARRAY_SIZE(analogFields)}}};
  firrtlVisitPort(ctx, MK_STR("analogBundle"), FIRRTL_PORT_DIRECTION_INPUT,
                  &analogBundle);

  FirrtlTypeBundleField uintFields[] = {
      {.name = MK_STR("field1"), .flip = false, .type = &uint8s[0]},
      {.name = MK_STR("field2"), .flip = true, .type = &uint8s[0]},
      {.name = MK_STR("field3"), .flip = false, .type = &uint8s[0]},
      {.name = MK_STR("field4"), .flip = true, .type = &uint8s[0]},
  };
  FirrtlType uintBundle = {
      .kind = FIRRTL_TYPE_KIND_BUNDLE,
      .u = {.bundle = {.fields = uintFields, .count = ARRAY_SIZE(uintFields)}}};
  firrtlVisitPort(ctx, MK_STR("uintIOBundle"), FIRRTL_PORT_DIRECTION_INPUT,
                  &uintBundle);
  for (unsigned int i = 0; i < ARRAY_SIZE(uintFields); i++) {
    uintFields[i].flip = false;
  }
  firrtlVisitPort(ctx, MK_STR("uintInputBundle"), FIRRTL_PORT_DIRECTION_INPUT,
                  &uintBundle);
  firrtlVisitPort(ctx, MK_STR("uintOutputBundle"), FIRRTL_PORT_DIRECTION_OUTPUT,
                  &uintBundle);
  uintBundle.u.bundle.fields[3].flip = true;
  uintBundle.u.bundle.count = 2;
  firrtlVisitPort(ctx, MK_STR("uintInputConnectBundle"),
                  FIRRTL_PORT_DIRECTION_INPUT, &uintBundle);

  FirrtlType clockPort = {.kind = FIRRTL_TYPE_KIND_CLOCK, .u = {.clock = {}}};
  firrtlVisitPort(ctx, MK_STR("clock"), FIRRTL_PORT_DIRECTION_INPUT,
                  &clockPort);

  FirrtlStatementAttachOperand attachOperands[] = {
      {.expr = MK_REF_EXPR_INLINE("analog1")},
      {.expr = MK_REF_EXPR_INLINE("analog2")},
      {.expr = MK_REF_EXPR_INLINE("analog3")},
  };
  FirrtlStatementAttachOperand attachBundleOperand = {
      .expr = MK_REF_EXPR_INLINE("analogBundle")};
  FirrtlStatementAttachOperand attachBundleOperands[] = {
      {.expr = MK_REF_EXPR_INLINE("analogBundle.field1")},
      {.expr = MK_REF_EXPR_INLINE("analogBundle.field2")},
      {.expr = MK_REF_EXPR_INLINE("analogBundle.field3")},
  };

  FirrtlType tyUInt32 = {
      .kind = FIRRTL_TYPE_KIND_UINT,
      .u = {.uint = {.width = 32}},
  };

  FirrtlPrimArg argsAdd[] = {
      {.kind = FIRRTL_PRIM_ARG_KIND_EXPR,
       .u = {.expr = {.value = MK_REF_EXPR_INLINE("uintPort1")}}},
      {.kind = FIRRTL_PRIM_ARG_KIND_EXPR,
       .u = {.expr = {.value = MK_REF_EXPR_INLINE("uintPort2")}}},
  };
  FirrtlPrim primAdd = {
      .op = FIRRTL_PRIM_OP_ADD,
      .args = argsAdd,
      .argsCount = ARRAY_SIZE(argsAdd),
  };
  FirrtlPrimArg argsSubField[] = {
      {.kind = FIRRTL_PRIM_ARG_KIND_EXPR,
       .u = {.expr = {.value = MK_REF_EXPR_INLINE("uintPort1")}}},
      {.kind = FIRRTL_PRIM_ARG_KIND_EXPR,
       .u = {.expr = {.value = MK_REF_EXPR_INLINE("analogBundle.field4")}}},
  };
  FirrtlPrim primSubField = {
      .op = FIRRTL_PRIM_OP_SUB,
      .args = argsSubField,
      .argsCount = ARRAY_SIZE(argsSubField),
  };

  FirrtlExpr printfOperands[] = {{.kind = FIRRTL_EXPR_KIND_UINT,
                                  .u = {.uint = {.value = 123, .width = 8}}}};
  FirrtlStringRef optName = MK_STR("optName");

  FirrtlStatement statements[] = {
      {.kind = FIRRTL_STATEMENT_KIND_ATTACH,
       .u = {.attach = {.operands = attachOperands,
                        .count = ARRAY_SIZE(attachOperands)}}},
      {.kind = FIRRTL_STATEMENT_KIND_ATTACH,
       .u = {.attach = {.operands = &attachBundleOperand, .count = 1}}},
      {.kind = FIRRTL_STATEMENT_KIND_ATTACH,
       .u = {.attach = {.operands = attachBundleOperands,
                        .count = ARRAY_SIZE(attachBundleOperands)}}},
      {.kind = FIRRTL_STATEMENT_KIND_SEQ_MEMORY,
       .u = {.seqMem = {.name = MK_STR("seqMem"),
                        .type = {.kind = FIRRTL_TYPE_KIND_VECTOR,
                                 .u = {.vector = {.type = &tyUInt32,
                                                  .count = 1024}}},
                        .readUnderWrite = FIRRTL_READ_UNDER_WRITE_UNDEFINED}}},
      {.kind = FIRRTL_STATEMENT_KIND_NODE,
       .u = {.node =
                 {
                     .name = MK_STR("node1"),
                     .expr = {.kind = FIRRTL_EXPR_KIND_PRIM,
                              .u = {.prim = {.value = &primAdd}}},
                 }}},
      {.kind = FIRRTL_STATEMENT_KIND_NODE,
       .u = {.node =
                 {
                     .name = MK_STR("node2"),
                     .expr = {.kind = FIRRTL_EXPR_KIND_PRIM,
                              .u = {.prim = {.value = &primSubField}}},
                 }}},
      {.kind = FIRRTL_STATEMENT_KIND_WIRE,
       .u = {.wire =
                 {
                     .name = MK_STR("wire1"),
                     .type = uint1,
                 }}},
      {.kind = FIRRTL_STATEMENT_KIND_WIRE,
       .u = {.wire =
                 {
                     .name = MK_STR("wireEnable"),
                     .type = uint1,
                 }}},
      {.kind = FIRRTL_STATEMENT_KIND_INVALID,
       .u = {.invalid =
                 {
                     .ref = {.value = MK_STR("wire1")},
                 }}},
      // Invalidate bundle
      {.kind = FIRRTL_STATEMENT_KIND_INVALID,
       .u = {.invalid =
                 {
                     .ref = {.value = MK_STR("uintIOBundle")},
                 }}},
      {.kind = FIRRTL_STATEMENT_KIND_INVALID,
       .u = {.invalid =
                 {
                     .ref = {.value = MK_STR("uintInputBundle")},
                 }}},
      {.kind = FIRRTL_STATEMENT_KIND_INVALID,
       .u = {.invalid =
                 {
                     .ref = {.value = MK_STR("uintOutputBundle")},
                 }}},
      {.kind = FIRRTL_STATEMENT_KIND_INVALID,
       .u = {.invalid =
                 {
                     .ref = {.value = MK_STR("analogBundle.field4")},
                 }}},
      {.kind = FIRRTL_STATEMENT_KIND_WHEN_BEGIN,
       .u = {.whenBegin =
                 {
                     .condition = MK_REF_EXPR_INLINE("wire1"),
                 }}},
      {.kind = FIRRTL_STATEMENT_KIND_WIRE,
       .u = {.wire =
                 {
                     .name = MK_STR("wire2"),
                     .type = uint1,
                 }}},
      {.kind = FIRRTL_STATEMENT_KIND_ELSE, .u = {.else_ = {}}},
      {.kind = FIRRTL_STATEMENT_KIND_WHEN_BEGIN,
       .u = {.whenBegin =
                 {
                     .condition = MK_REF_EXPR_INLINE("wire1"),
                 }}},
      {.kind = FIRRTL_STATEMENT_KIND_WIRE,
       .u = {.wire =
                 {
                     .name = MK_STR("wire3"),
                     .type = uint1,
                 }}},
      {.kind = FIRRTL_STATEMENT_KIND_ELSE, .u = {.else_ = {}}},
      {.kind = FIRRTL_STATEMENT_KIND_WHEN_BEGIN,
       .u = {.whenBegin =
                 {
                     .condition = MK_REF_EXPR_INLINE("wire1"),
                 }}},
      {.kind = FIRRTL_STATEMENT_KIND_WIRE,
       .u = {.wire =
                 {
                     .name = MK_STR("wire4"),
                     .type = uint1,
                 }}},
      {.kind = FIRRTL_STATEMENT_KIND_ELSE, .u = {.else_ = {}}},
      {.kind = FIRRTL_STATEMENT_KIND_WIRE,
       .u = {.wire =
                 {
                     .name = MK_STR("wire5"),
                     .type = uint1,
                 }}},
      {.kind = FIRRTL_STATEMENT_KIND_WHEN_END, .u = {.whenEnd = {}}},
      {.kind = FIRRTL_STATEMENT_KIND_WIRE,
       .u = {.wire =
                 {
                     .name = MK_STR("wire6"),
                     .type = uint1,
                 }}},
      {.kind = FIRRTL_STATEMENT_KIND_WHEN_END, .u = {.whenEnd = {}}},
      {.kind = FIRRTL_STATEMENT_KIND_WHEN_END, .u = {.whenEnd = {}}},
      {.kind = FIRRTL_STATEMENT_KIND_CONNECT,
       .u = {.connect = {.left = {.value = MK_STR("uintOutputBundle.field1")},
                         .right = {.value = MK_STR("uintInputBundle.field1")},
                         .isPartial = false}}},
      {.kind = FIRRTL_STATEMENT_KIND_CONNECT,
       .u = {.connect = {.left = {.value = MK_STR("uintOutputBundle")},
                         .right = {.value = MK_STR("uintInputBundle")},
                         .isPartial = true}}},
      {.kind = FIRRTL_STATEMENT_KIND_CONNECT,
       .u = {.connect = {.left = {.value = MK_STR("uintOutputBundle")},
                         .right = {.value = MK_STR("uintInputConnectBundle")},
                         .isPartial = true}}},
      {.kind = FIRRTL_STATEMENT_KIND_MEM_PORT,
       .u = {.memPort = {.direction = FIRRTL_MEM_DIRECTION_INFER,
                         .name = MK_STR("mpInfer"),
                         .memName = {.value = MK_STR("seqMem")},
                         .memIndex = MK_REF_EXPR_INLINE("uintPort1"),
                         .clock = MK_REF_EXPR_INLINE("clock")}}},
      {.kind = FIRRTL_STATEMENT_KIND_MEM_PORT,
       .u = {.memPort = {.direction = FIRRTL_MEM_DIRECTION_READ_WRITE,
                         .name = MK_STR("mpRW"),
                         .memName = {.value = MK_STR("seqMem")},
                         .memIndex = MK_REF_EXPR_INLINE("uintPort1"),
                         .clock = MK_REF_EXPR_INLINE("clock")}}},
      {.kind = FIRRTL_STATEMENT_KIND_PRINTF,
       .u = {.printf = {.clock = MK_REF_EXPR_INLINE("clock"),
                        .condition = MK_REF_EXPR_INLINE("wire1"),
                        .format = MK_STR("test %d"),
                        .operands = printfOperands,
                        .operandsCount = ARRAY_SIZE(printfOperands),
                        .name = NULL}}},
      {.kind = FIRRTL_STATEMENT_KIND_PRINTF,
       .u = {.printf = {.clock = MK_REF_EXPR_INLINE("clock"),
                        .condition = MK_REF_EXPR_INLINE("wire1"),
                        .format = MK_STR("test2 %d"),
                        .operands = printfOperands,
                        .operandsCount = ARRAY_SIZE(printfOperands),
                        .name = &optName}}},
      {.kind = FIRRTL_STATEMENT_KIND_SKIP, .u = {.skip = {}}},
      {.kind = FIRRTL_STATEMENT_KIND_STOP,
       .u = {.stop = {.clock = MK_REF_EXPR_INLINE("clock"),
                      .condition = MK_REF_EXPR_INLINE("wire1"),
                      .exitCode = 123,
                      .name = NULL}}},
      {.kind = FIRRTL_STATEMENT_KIND_STOP,
       .u = {.stop = {.clock = MK_REF_EXPR_INLINE("clock"),
                      .condition = MK_REF_EXPR_INLINE("wire1"),
                      .exitCode = 123,
                      .name = &optName}}},
      {.kind = FIRRTL_STATEMENT_KIND_ASSERT,
       .u = {.assert = {.clock = MK_REF_EXPR_INLINE("clock"),
                        .predicate = MK_REF_EXPR_INLINE("wire1"),
                        .enable = MK_REF_EXPR_INLINE("wireEnable"),
                        .message = MK_STR("assertion failed message"),
                        .name = NULL}}},
      {.kind = FIRRTL_STATEMENT_KIND_ASSERT,
       .u = {.assert = {.clock = MK_REF_EXPR_INLINE("clock"),
                        .predicate = MK_REF_EXPR_INLINE("wire1"),
                        .enable = MK_REF_EXPR_INLINE("wireEnable"),
                        .message = MK_STR("assertion failed message"),
                        .name = &optName}}},
      {.kind = FIRRTL_STATEMENT_KIND_ASSUME,
       .u = {.assume = {.clock = MK_REF_EXPR_INLINE("clock"),
                        .predicate = MK_REF_EXPR_INLINE("wire1"),
                        .enable = MK_REF_EXPR_INLINE("wireEnable"),
                        .message = MK_STR("assertion failed message"),
                        .name = NULL}}},
      {.kind = FIRRTL_STATEMENT_KIND_ASSUME,
       .u = {.assume = {.clock = MK_REF_EXPR_INLINE("clock"),
                        .predicate = MK_REF_EXPR_INLINE("wire1"),
                        .enable = MK_REF_EXPR_INLINE("wireEnable"),
                        .message = MK_STR("assertion failed message"),
                        .name = &optName}}},
  };

  for (unsigned int i = 0; i < ARRAY_SIZE(statements); i++) {
    firrtlVisitStatement(ctx, &statements[i]);
  }

  EXPECT(*errCount == 0);
  EXPECT_EXPORT("circuit StatementTest :\n\
  module StatementTest :\n\
    input analog1 : Analog<8>\n\
    input analog2 : Analog<8>\n\
    input analog3 : Analog<8>\n\
    input uintPort1 : UInt<8>\n\
    input uintPort2 : UInt<8>\n\
    input analogBundle : { field1 : Analog<8>, flip field2 : Analog<8>, field3 : Analog<8>, flip field4 : UInt<8> }\n\
    input uintIOBundle : { field1 : UInt<8>, flip field2 : UInt<8>, field3 : UInt<8>, flip field4 : UInt<8> }\n\
    input uintInputBundle : { field1 : UInt<8>, field2 : UInt<8>, field3 : UInt<8>, field4 : UInt<8> }\n\
    output uintOutputBundle : { field1 : UInt<8>, field2 : UInt<8>, field3 : UInt<8>, field4 : UInt<8> }\n\
    input uintInputConnectBundle : { field1 : UInt<8>, field2 : UInt<8> }\n\
    input clock : Clock\n\
\n\
    attach(analog1, analog2, analog3)\n\
    attach(analogBundle)\n\
    attach(analogBundle.field1, analogBundle.field2, analogBundle.field3)\n\
    smem seqMem : UInt<32>[1024] undefined\n\
    node node1 = add(uintPort1, uintPort2)\n\
    node node2 = sub(uintPort1, analogBundle.field4)\n\
    wire wire1 : UInt<1>\n\
    wire wireEnable : UInt<1>\n\
    wire1 is invalid\n\
    uintIOBundle.field2 is invalid\n\
    uintIOBundle.field4 is invalid\n\
    uintOutputBundle is invalid\n\
    analogBundle.field4 is invalid\n\
    when wire1 :\n\
      wire wire2 : UInt<1>\n\
    else when wire1 :\n\
      wire wire3 : UInt<1>\n\
    else :\n\
      when wire1 :\n\
        wire wire4 : UInt<1>\n\
      else :\n\
        wire wire5 : UInt<1>\n\
      wire wire6 : UInt<1>\n\
    uintOutputBundle.field1 <= uintInputBundle.field1\n\
    uintOutputBundle <= uintInputBundle\n\
    uintOutputBundle.field1 <= uintInputConnectBundle.field1\n\
    uintOutputBundle.field2 <= uintInputConnectBundle.field2\n\
    infer mport mpInfer = seqMem[uintPort1], clock\n\
    rdwr mport mpRW = seqMem[uintPort1], clock\n\
    printf(clock, wire1, \"test %d\", UInt<8>(1))\n\
    printf(clock, wire1, \"test2 %d\", UInt<8>(1)) : optName\n\
    skip\n\
    stop(clock, wire1, 123)\n\
    stop(clock, wire1, 123) : optName\n\
    assert(clock, wire1, wireEnable, \"assertion failed message\")\n\
    assert(clock, wire1, wireEnable, \"assertion failed message\") : optName\n\
    assume(clock, wire1, wireEnable, \"assertion failed message\")\n\
    assume(clock, wire1, wireEnable, \"assertion failed message\") : optName\n\n");
  EXPECT(*errCount == 0);

  return 0;
}

int testGenerated() {
  IF_ERR_RET(runOnce(&testGenHeader));
  IF_ERR_RET(runOnce(&testGenPorts));
  IF_ERR_RET(runOnce(&testGenExtModule));
  IF_ERR_RET(runOnce(&testGenStatement));

  return 0;
}

int testErrNoCircuit(FirrtlContext ctx, size_t *errCount) {
  EXPECT(*errCount == 0);
  firrtlVisitModule(ctx, MK_STR("NoCircuit"));
  EXPECT(*errCount == 1);
  EXPECT_EXPORT("");
  EXPECT(*errCount == 1);

  return 0;
}

int testErrNoModule(FirrtlContext ctx, size_t *errCount) {
  firrtlVisitCircuit(ctx, MK_STR("NoModule"));
  EXPECT(*errCount == 0);

  FirrtlType tyUInt8 = {
      .kind = FIRRTL_TYPE_KIND_UINT,
      .u = {.uint = {.width = 8}},
  };
  firrtlVisitPort(ctx, MK_STR("port"), FIRRTL_PORT_DIRECTION_INPUT, &tyUInt8);
  EXPECT(*errCount == 1);
  EXPECT_EXPORT("circuit NoModule :\n");
  EXPECT(*errCount == 1);

  return 0;
}

int testErrNoHeader(FirrtlContext ctx, size_t *errCount) {
  EXPECT(*errCount == 0);

  FirrtlType tyUInt8 = {
      .kind = FIRRTL_TYPE_KIND_UINT,
      .u = {.uint = {.width = 8}},
  };
  firrtlVisitPort(ctx, MK_STR("port"), FIRRTL_PORT_DIRECTION_INPUT, &tyUInt8);
  EXPECT(*errCount == 1);
  EXPECT_EXPORT("");
  EXPECT(*errCount == 1);

  return 0;
}

int testErrDuplicatePortName(FirrtlContext ctx, size_t *errCount) {
  firrtlVisitCircuit(ctx, MK_STR("DupPortName"));
  firrtlVisitModule(ctx, MK_STR("DupPortName"));

  FirrtlType tyUInt8 = {
      .kind = FIRRTL_TYPE_KIND_UINT,
      .u = {.uint = {.width = 8}},
  };
  firrtlVisitPort(ctx, MK_STR("port"), FIRRTL_PORT_DIRECTION_INPUT, &tyUInt8);
  EXPECT(*errCount == 0);

  firrtlVisitPort(ctx, MK_STR("port"), FIRRTL_PORT_DIRECTION_INPUT, &tyUInt8);
  EXPECT(*errCount == 1);
  EXPECT_EXPORT("circuit DupPortName :\n\
  module DupPortName :\n\
    input port : UInt<8>\n\n");

  // same port name in a new module
  firrtlVisitModule(ctx, MK_STR("AnotherMod"));
  EXPECT(*errCount == 1);
  firrtlVisitPort(ctx, MK_STR("port"), FIRRTL_PORT_DIRECTION_INPUT, &tyUInt8);
  EXPECT(*errCount == 1);
  firrtlVisitPort(ctx, MK_STR("port"), FIRRTL_PORT_DIRECTION_INPUT, &tyUInt8);
  EXPECT(*errCount == 2);
  EXPECT_EXPORT("circuit DupPortName :\n\
  module DupPortName :\n\
    input port : UInt<8>\n\
\n\
  module AnotherMod :\n\
    input port : UInt<8>\n\n");
  EXPECT(*errCount == 2);

  return 0;
}

int testErrStmtAttach(FirrtlContext ctx, size_t *errCount) {
  firrtlVisitCircuit(ctx, MK_STR("StmtAttach"));
  firrtlVisitModule(ctx, MK_STR("StmtAttach"));

  FirrtlType analog = {.kind = FIRRTL_TYPE_KIND_ANALOG,
                       .u = {.analog = {.width = 8}}};
  FirrtlTypeBundleField analogFields[] = {
      {.name = MK_STR("field"), .flip = false, .type = &analog},
  };
  FirrtlType analogBundle = {
      .kind = FIRRTL_TYPE_KIND_BUNDLE,
      .u = {.bundle = {.fields = analogFields,
                       .count = ARRAY_SIZE(analogFields)}}};
  firrtlVisitPort(ctx, MK_STR("analogPort"), FIRRTL_PORT_DIRECTION_INPUT,
                  &analog);
  firrtlVisitPort(ctx, MK_STR("analogBundle"), FIRRTL_PORT_DIRECTION_INPUT,
                  &analogBundle);

  EXPECT(*errCount == 0);
  size_t expectedErrCount = 0;

  {
    FirrtlStatementAttachOperand attachOperand = {
        .expr = MK_REF_EXPR_INLINE("nonExistPort")};
    FirrtlStatement statement = {
        .kind = FIRRTL_STATEMENT_KIND_ATTACH,
        .u = {.attach = {.operands = &attachOperand, .count = 1}}};
    firrtlVisitStatement(ctx, &statement);

    EXPECT(*errCount == ++expectedErrCount);
  }

  {
    FirrtlStatementAttachOperand attachOperand = {
        .expr = MK_REF_EXPR_INLINE("analogBundle.nonExistField")};
    FirrtlStatement statement = {
        .kind = FIRRTL_STATEMENT_KIND_ATTACH,
        .u = {.attach = {.operands = &attachOperand, .count = 1}}};
    firrtlVisitStatement(ctx, &statement);

    EXPECT(*errCount == ++expectedErrCount);
  }

  EXPECT_EXPORT("circuit StmtAttach :\n\
  module StmtAttach :\n\
    input analogPort : Analog<8>\n\
    input analogBundle : { field : Analog<8> }\n\n");
  EXPECT(*errCount == expectedErrCount);

  return 0;
}

int testErrStmtNode(FirrtlContext ctx, size_t *errCount) {
  firrtlVisitCircuit(ctx, MK_STR("StmtNode"));
  firrtlVisitModule(ctx, MK_STR("StmtNode"));

  FirrtlType tyUInt8 = {
      .kind = FIRRTL_TYPE_KIND_UINT,
      .u = {.uint = {.width = 8}},
  };
  FirrtlType tySInt8 = {
      .kind = FIRRTL_TYPE_KIND_SINT,
      .u = {.uint = {.width = 8}},
  };
  firrtlVisitPort(ctx, MK_STR("u8Port1"), FIRRTL_PORT_DIRECTION_INPUT,
                  &tyUInt8);
  firrtlVisitPort(ctx, MK_STR("u8Port2"), FIRRTL_PORT_DIRECTION_INPUT,
                  &tyUInt8);
  firrtlVisitPort(ctx, MK_STR("u8Port3"), FIRRTL_PORT_DIRECTION_INPUT,
                  &tyUInt8);
  firrtlVisitPort(ctx, MK_STR("s8Port3"), FIRRTL_PORT_DIRECTION_INPUT,
                  &tySInt8);

  EXPECT(*errCount == 0);
  size_t expectedErrCount = 0;

  // Args count mismatch
  {
    FirrtlPrimArg args[] = {
        {.kind = FIRRTL_PRIM_ARG_KIND_EXPR,
         .u = {.expr = {.value = MK_REF_EXPR_INLINE("u8Port1")}}},
        {.kind = FIRRTL_PRIM_ARG_KIND_EXPR,
         .u = {.expr = {.value = MK_REF_EXPR_INLINE("u8Port2")}}},
        {.kind = FIRRTL_PRIM_ARG_KIND_EXPR,
         .u = {.expr = {.value = MK_REF_EXPR_INLINE("u8Port3")}}},
    };
    FirrtlPrim prim = {
        .op = FIRRTL_PRIM_OP_ADD,
        .args = args,
        .argsCount = ARRAY_SIZE(args),
    };
    FirrtlStatement statement = {
        .kind = FIRRTL_STATEMENT_KIND_NODE,
        .u = {.node = {
                  .name = MK_STR("node1"),
                  .expr = {.kind = FIRRTL_EXPR_KIND_PRIM,
                           .u = {.prim = {.value = &prim}}},
              }}};
    firrtlVisitStatement(ctx, &statement);

    EXPECT(*errCount >= ++expectedErrCount);
    expectedErrCount = *errCount;
  }

  // Type mismatch
  {
    FirrtlPrimArg args[] = {
        {.kind = FIRRTL_PRIM_ARG_KIND_EXPR,
         .u = {.expr = {.value = MK_REF_EXPR_INLINE("u8Port1")}}},
        {.kind = FIRRTL_PRIM_ARG_KIND_EXPR,
         .u = {.expr = {.value = MK_REF_EXPR_INLINE("s8Port2")}}},
    };
    FirrtlPrim prim = {
        .op = FIRRTL_PRIM_OP_ADD,
        .args = args,
        .argsCount = ARRAY_SIZE(args),
    };
    FirrtlStatement statement = {
        .kind = FIRRTL_STATEMENT_KIND_NODE,
        .u = {.node = {
                  .name = MK_STR("node2"),
                  .expr = {.kind = FIRRTL_EXPR_KIND_PRIM,
                           .u = {.prim = {.value = &prim}}},
              }}};
    firrtlVisitStatement(ctx, &statement);

    EXPECT(*errCount >= ++expectedErrCount);
    expectedErrCount = *errCount;
  }

  EXPECT_EXPORT("circuit StmtNode :\n\
  module StmtNode :\n\
    input u8Port1 : UInt<8>\n\
    input u8Port2 : UInt<8>\n\
    input u8Port3 : UInt<8>\n\
    input s8Port3 : SInt<8>\n\n");
  EXPECT(*errCount == expectedErrCount);

  return 0;
}

int testErrStmtWhen(FirrtlContext ctx, size_t *errCount) {
  firrtlVisitCircuit(ctx, MK_STR("StmtWhen"));
  firrtlVisitModule(ctx, MK_STR("StmtWhen"));

  FirrtlType tyUInt1 = {
      .kind = FIRRTL_TYPE_KIND_UINT,
      .u = {.uint = {.width = 1}},
  };

  FirrtlStatement wire = {.kind = FIRRTL_STATEMENT_KIND_WIRE,
                          .u = {.wire = {
                                    .name = MK_STR("wire1"),
                                    .type = tyUInt1,
                                }}};
  firrtlVisitStatement(ctx, &wire);

  EXPECT(*errCount == 0);
  size_t expectedErrCount = 0;

  FirrtlStatement whenBegin = {
      .kind = FIRRTL_STATEMENT_KIND_WHEN_BEGIN,
      .u = {.whenBegin = {
                .condition = MK_REF_EXPR_INLINE("wire1"),
            }}};

  // No WhenEnd
  {
    firrtlVisitStatement(ctx, &whenBegin);

    EXPECT(*errCount == expectedErrCount);
    EXPECT_EXPORT("");
    EXPECT(*errCount == ++expectedErrCount);
  }

  // Reference to items inside when block
  {
    FirrtlStatement stmts[] = {
        {.kind = FIRRTL_STATEMENT_KIND_WIRE,
         .u = {.wire =
                   {
                       .name = MK_STR("wire2"),
                       .type = tyUInt1,
                   }}},
        {.kind = FIRRTL_STATEMENT_KIND_ELSE, .u = {.else_ = {}}},
        {.kind = FIRRTL_STATEMENT_KIND_WIRE,
         .u = {.wire =
                   {
                       .name = MK_STR("wire3"),
                       .type = tyUInt1,
                   }}},
        {.kind = FIRRTL_STATEMENT_KIND_WHEN_END, .u = {.whenEnd = {}}},
    };
    for (unsigned int i = 0; i < ARRAY_SIZE(stmts); i++) {
      firrtlVisitStatement(ctx, &stmts[i]);
    }

    EXPECT(*errCount == expectedErrCount);

    FirrtlStatement ref = {.kind = FIRRTL_STATEMENT_KIND_INVALID,
                           .u = {.invalid = {
                                     .ref = {.value = MK_STR("wire1")},
                                 }}};
    firrtlVisitStatement(ctx, &ref);
    EXPECT(*errCount == expectedErrCount);

    ref.u.invalid.ref.value = MK_STR("wire2");
    firrtlVisitStatement(ctx, &ref);
    EXPECT(*errCount == ++expectedErrCount);

    ref.u.invalid.ref.value = MK_STR("wire3");
    firrtlVisitStatement(ctx, &ref);
    EXPECT(*errCount == ++expectedErrCount);
  }

  // Multiple `else` regions
  {
    firrtlVisitStatement(ctx, &whenBegin);

    FirrtlStatement stmts[] = {
        {.kind = FIRRTL_STATEMENT_KIND_WIRE,
         .u = {.wire =
                   {
                       .name = MK_STR("wire2"),
                       .type = tyUInt1,
                   }}},
        {.kind = FIRRTL_STATEMENT_KIND_ELSE, .u = {.else_ = {}}},
        {.kind = FIRRTL_STATEMENT_KIND_WIRE,
         .u = {.wire =
                   {
                       .name = MK_STR("wire3"),
                       .type = tyUInt1,
                   }}},
    };
    for (unsigned int i = 0; i < ARRAY_SIZE(stmts); i++) {
      firrtlVisitStatement(ctx, &stmts[i]);
    }

    EXPECT(*errCount == expectedErrCount);

    FirrtlStatement else_ = {.kind = FIRRTL_STATEMENT_KIND_ELSE,
                             .u = {.else_ = {}}};
    firrtlVisitStatement(ctx, &else_);
    EXPECT(*errCount == ++expectedErrCount);

    FirrtlStatement whenEnd = {.kind = FIRRTL_STATEMENT_KIND_WHEN_END,
                               .u = {.whenEnd = {}}};
    firrtlVisitStatement(ctx, &whenEnd);
  }

  EXPECT_EXPORT("circuit StmtWhen :\n\
  module StmtWhen :\n\
    wire wire1 : UInt<1>\n\
    when wire1 :\n\
      wire wire2 : UInt<1>\n\
    else :\n\
      wire wire3 : UInt<1>\n\
    wire1 is invalid\n\
    when wire1 :\n\
      wire wire2 : UInt<1>\n\
    else :\n\
      wire wire3 : UInt<1>\n\n");
  EXPECT(*errCount == expectedErrCount);

  return 0;
}

int testErrStmtConnect(FirrtlContext ctx, size_t *errCount) {
  firrtlVisitCircuit(ctx, MK_STR("StmtConnect"));
  firrtlVisitModule(ctx, MK_STR("StmtConnect"));

  FirrtlType tyUInt8 = {
      .kind = FIRRTL_TYPE_KIND_UINT,
      .u = {.uint = {.width = 8}},
  };

  FirrtlType tySInt8 = {
      .kind = FIRRTL_TYPE_KIND_SINT,
      .u = {.uint = {.width = 8}},
  };

  EXPECT(*errCount == 0);
  size_t expectedErrCount = 0;

  // Type mismatch
  {
    firrtlVisitPort(ctx, MK_STR("uiIn8"), FIRRTL_PORT_DIRECTION_INPUT,
                    &tyUInt8);
    firrtlVisitPort(ctx, MK_STR("siOut8"), FIRRTL_PORT_DIRECTION_OUTPUT,
                    &tySInt8);

    FirrtlStatement connect = {
        .kind = FIRRTL_STATEMENT_KIND_CONNECT,
        .u = {.connect = {.left = {.value = MK_STR("siOut8")},
                          .right = {.value = MK_STR("uiIn8")},
                          .isPartial = false}}};

    EXPECT(*errCount == expectedErrCount);
    firrtlVisitStatement(ctx, &connect);
    EXPECT(*errCount == ++expectedErrCount);
  }

  EXPECT_EXPORT("circuit StmtConnect :\n\
  module StmtConnect :\n\
    input uiIn8 : UInt<8>\n\
    output siOut8 : SInt<8>\n\n");
  EXPECT(*errCount == expectedErrCount);

  return 0;
}

int testExpectedError() {
  IF_ERR_RET(runOnce(&testErrNoCircuit));
  IF_ERR_RET(runOnce(&testErrNoModule));
  IF_ERR_RET(runOnce(&testErrNoHeader));
  IF_ERR_RET(runOnce(&testErrDuplicatePortName));
  IF_ERR_RET(runOnce(&testErrStmtAttach));
  IF_ERR_RET(runOnce(&testErrStmtNode));
  IF_ERR_RET(runOnce(&testErrStmtWhen));
  IF_ERR_RET(runOnce(&testErrStmtConnect));

  return 0;
}

int main() {
  fprintf(stderr, "@generated\n");
  int errCode = testGenerated();
  fprintf(stderr, "generated{%d}\n", errCode);

  fprintf(stderr, "@expectedError\n");
  errCode = testExpectedError();
  fprintf(stderr, "expectedError{%d}\n", errCode);

  // clang-format off
  // CHECK-LABEL: @generated
  // CHECK: generated{0}
  // CHECK-LABEL: @expectedError
  // CHECK: expectedError{0}
  // clang-format on

  return 0;
}
