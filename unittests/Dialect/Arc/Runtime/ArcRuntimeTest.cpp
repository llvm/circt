#include "gtest/gtest.h"

#define ARC_RUNTIME_JITBIND_FNDECL
#include "circt/Dialect/Arc/Runtime/ArcRuntime.h"
#include "circt/Dialect/Arc/Runtime/IRInterface.h"
#include "circt/Dialect/Arc/Runtime/JITBind.h"
#include "circt/Dialect/Arc/Runtime/String.h"

struct TestImpl {
  uint64_t foo = 0x123456;
};

TEST(ArcRuntimeTest, InstanceLifecycle) {
  const uint64_t bogusStateBytes = 2048;
  auto bogusModel = ArcRuntimeModelInfo();
  bogusModel.apiVersion = ARC_RUNTIME_API_VERSION;
  bogusModel.modelName = "bogus";
  bogusModel.numStateBytes = bogusStateBytes;
  ArcState *state0 = arcRuntimeAllocateInstance(&bogusModel, nullptr);
  ArcState *state1 = arcRuntimeAllocateInstance(&bogusModel, "foo;bar");
  EXPECT_NE(state0, nullptr);
  EXPECT_NE(state1, nullptr);
  EXPECT_NE(state0, state1);
  EXPECT_EQ(state0->magic, ARC_RUNTIME_MAGIC);
  EXPECT_NE(state0->impl, nullptr);
  EXPECT_EQ((intptr_t)(state0->modelState) % 16, 0);
  EXPECT_EQ(state1->magic, ARC_RUNTIME_MAGIC);
  EXPECT_NE(state1->impl, nullptr);
  EXPECT_EQ((intptr_t)(state1->modelState) % 16, 0);
  bool notZero = false;
  for (uint64_t i = 0; i < bogusStateBytes; ++i)
    notZero |= (state0->modelState[i] != 0);
  EXPECT_FALSE(notZero);
  for (uint64_t i = 0; i < bogusStateBytes; ++i)
    notZero |= (state1->modelState[i] != 0);
  EXPECT_FALSE(notZero);

  arcRuntimeOnInitialized(state1);
  arcRuntimeOnInitialized(state0);

  for (auto i = 0; i < 24; ++i)
    arcRuntimeOnEval(state0);

  arcRuntimeDeleteInstance(state0);
  arcRuntimeDeleteInstance(state1);
}

TEST(ArcRuntimeTest, DieOnWrongAPIVersion) {
  auto bogusModel = ArcRuntimeModelInfo();
  bogusModel.apiVersion = ARC_RUNTIME_API_VERSION + 1;
  bogusModel.modelName = "bogus";
  bogusModel.numStateBytes = 8;
  EXPECT_DEATH(arcRuntimeAllocateInstance(&bogusModel, nullptr), "");
}

TEST(ArcRuntimeTest, GetStateFromModelState) {
  auto impl = TestImpl();
  auto state = ArcState();
  state.impl = static_cast<void *>(&impl);
  state.magic = ARC_RUNTIME_MAGIC;
  EXPECT_EQ(arcRuntimeGetStateFromModelState(state.modelState, 0), &state);
  EXPECT_EQ(arcRuntimeGetStateFromModelState(&state.modelState[123], 123),
            &state);
}

TEST(ArcRuntimeTest, DieOnWrongMagic) {
  auto impl = TestImpl();
  auto state = ArcState();
  state.impl = static_cast<void *>(&impl);
  state.magic = ~ARC_RUNTIME_MAGIC;
  EXPECT_DEATH(arcRuntimeGetStateFromModelState(state.modelState, 0), "");
}

TEST(ArcRuntimeTest, GetAPIVersion) {
  EXPECT_EQ(arcRuntimeGetAPIVersion(), ARC_RUNTIME_API_VERSION);
}

TEST(ArcRuntimeTest, InstanceLifecycleIR) {
  auto &api = circt::arc::runtime::getArcRuntimeAPICallbacks();
  const uint64_t bogusStateBytes = 2048;
  auto bogusModel = ArcRuntimeModelInfo();
  bogusModel.apiVersion = ARC_RUNTIME_API_VERSION;
  bogusModel.modelName = "bogus";
  bogusModel.numStateBytes = bogusStateBytes;
  uint8_t *modelState = api.fnAllocInstance(&bogusModel, nullptr);
  EXPECT_NE(modelState, nullptr);
  ArcState *state = arcRuntimeGetStateFromModelState(modelState, 0);
  EXPECT_EQ(state->magic, ARC_RUNTIME_MAGIC);
  EXPECT_NE(state->impl, nullptr);
  EXPECT_EQ(&state->modelState[0], modelState);
  EXPECT_EQ((intptr_t)(state->modelState) % 16, 0);
  bool notZero = false;
  for (uint64_t i = 0; i < bogusStateBytes; ++i)
    notZero |= (state->modelState[i] != 0);
  EXPECT_FALSE(notZero);
  api.fnOnInitialized(modelState);
  for (auto i = 0; i < 128; ++i)
    api.fnOnEval(modelState);
  api.fnDeleteInstance(modelState);
}

TEST(ArcRuntimeTest, DynamicStringOperations) {
  using namespace circt::arc::runtime;
  auto makeAndVerifyString = [](const char *text, size_t expectedSize) {
    DynamicString str{};
    arcRuntimeIR_stringInit(&str, text);
    EXPECT_STREQ(str.data, text);
    EXPECT_EQ(str.size, expectedSize);
    return str;
  };

  auto str1 = makeAndVerifyString("Hello", 5);
  auto str2 = makeAndVerifyString("World!", 6);

  DynamicString outStr{};
  arcRuntimeIR_stringConcat(&outStr, &str1, &str2, nullptr);
  EXPECT_STREQ(outStr.data, "HelloWorld!");
  EXPECT_EQ(outStr.size, 11);

  delete[] str1.data;
  delete[] str2.data;
  delete[] outStr.data;
}