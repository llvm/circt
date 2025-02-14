// RUN: circt-test --verify-diagnostics --split-input-file %s

// expected-error @below {{`ignore` attribute of test "Foo" must be a boolean}}
verif.formal @Foo {ignore = "hello"} {}

// -----
// expected-error @below {{`require_runners` attribute of test "Foo" must be an array}}
verif.formal @Foo {require_runners = "hello"} {}

// -----
// expected-error @below {{`exclude_runners` attribute of test "Foo" must be an array}}
verif.formal @Foo {exclude_runners = "hello"} {}

// -----
// expected-error @below {{element of `require_runners` array of test "Foo" must be a string}}
verif.formal @Foo {require_runners = [42]} {}

// -----
// expected-error @below {{element of `exclude_runners` array of test "Foo" must be a string}}
verif.formal @Foo {exclude_runners = [42]} {}
