# Implementation Plan: Add Rvalue String Indexing Support to ImportVerilog

## Overview

Add support for rvalue (read-only) string indexing in the ImportVerilog conversion, enabling expressions like `byte c = foo[idx]` where `foo` is a string. This will resolve 126 failing test cases with the error "unsupported expression: element select into string".

**Goals:**
- Support reading individual characters from strings using indexing
- Handle both constant and dynamic indices
- Convert indices to the required TwoValuedI32 type
- Use the existing `moore::StringGetCOp` operation
- Only support rvalue operations (reading), NOT lvalue operations (writing)

**Success Criteria:**
- All 126 failing test cases pass
- String indexing works with constant and dynamic indices
- Proper error messages for unsupported lvalue string indexing
- Code follows existing patterns in the codebase

---

## Prerequisites

**Required Knowledge:**
- `moore::StringGetCOp` signature: `StringGetCOp(StringType str, TwoValuedI32 index) -> TwoValuedI8`
- `TwoValuedI32` = `!moore.i32` (32-bit two-valued integer)
- `TwoValuedI8` = `!moore.i8` (8-bit two-valued integer, represents a character)
- `StringType` = `!moore.string` (SystemVerilog string type)

**Dependencies:**
- No new operations needed - `StringGetCOp` already exists
- No new type definitions needed
- No migrations required

**Environment:**
- File to modify: `lib/Conversion/ImportVerilog/Expressions.cpp`
- Specifically: `ElementSelectExpression::visit` method (lines 182-297)

---

## Implementation Steps

### Step 1: Add StringType to Supported Types Check

**Location:** `lib/Conversion/ImportVerilog/Expressions.cpp`, lines 193-198

**Current Code:**
```cpp
if (!isa<moore::IntType, moore::ArrayType, moore::UnpackedArrayType,
         moore::QueueType, moore::AssocArrayType>(derefType)) {
  mlir::emitError(loc) << "unsupported expression: element select into "
                       << expr.value().type->toString() << "\n";
  return {};
}
```

**Change:**
Add `moore::StringType` to the list of supported types:
```cpp
if (!isa<moore::IntType, moore::ArrayType, moore::UnpackedArrayType,
         moore::QueueType, moore::AssocArrayType, moore::StringType>(derefType)) {
  mlir::emitError(loc) << "unsupported expression: element select into "
                       << expr.value().type->toString() << "\n";
  return {};
}
```

**Rationale:** This allows string types to pass the initial type check and proceed to the string-specific handling logic.

---

### Step 2: Add Lvalue Check for Strings

**Location:** After the AssocArrayType handling block (after line 222), before the constant index handling

**Add New Code Block:**
```cpp
// Handle string indexing (rvalue only)
if (isa<moore::StringType>(derefType)) {
  // String indexing is only supported for rvalue (reading characters)
  if (isLvalue) {
    mlir::emitError(loc) << "lvalue string indexing is not supported; "
                         << "strings are immutable and cannot be indexed for assignment";
    return {};
  }
  
  // Convert the index to an rvalue
  auto index = context.convertRvalueExpression(expr.selector());
  if (!index)
    return {};
  
  // Convert index to TwoValuedI32 if needed
  auto i32Type = moore::IntType::getInt(context.getContext(), 32);
  index = context.materializeConversion(i32Type, index, 
                                       expr.selector().type->isSigned(), loc);
  if (!index)
    return {};
  
  // Create the StringGetCOp operation
  // Result type is TwoValuedI8 (byte/character)
  auto resultType = moore::IntType::getInt(context.getContext(), 8);
  return moore::StringGetCOp::create(builder, loc, resultType, value, index);
}
```

**Rationale:**
- Strings in SystemVerilog are immutable, so lvalue indexing (assignment) is not supported
- The index must be converted to `TwoValuedI32` as required by `StringGetCOp`
- The result is `TwoValuedI8` representing a single character
- This handles both constant and dynamic indices uniformly

**Key Implementation Details:**
1. **Placement:** Insert this block after the AssocArrayType handling (line 222) and before the `resultType` calculation (line 224)
2. **Early return:** The block returns early, so it won't interfere with other type handling
3. **Type conversion:** Uses `materializeConversion` to handle any index type (int, logic, etc.)
4. **Signedness:** Respects the signedness of the index expression from Slang AST

---

### Step 3: Handle Type Conversion Edge Cases

**Considerations:**

The `materializeConversion` function (lines 2626-2810) already handles:
- Integer width conversions (truncation, zero-extension, sign-extension)
- Domain conversions (two-valued ↔ four-valued)
- Type casting between different integer types

**No changes needed** to `materializeConversion` because:
- It already handles conversion from any `IntType` to `TwoValuedI32`
- It properly handles signedness during conversion
- It works with both constant and dynamic values

**Verification:**
- Review lines 2639-2673 to confirm integer type conversion logic
- The conversion path: `expr.selector()` → `convertRvalueExpression` → `materializeConversion` → `TwoValuedI32`

---

### Step 4: Verify Result Type Handling

**Result Type:** `TwoValuedI8` (8-bit two-valued integer)

**Verification Points:**
1. The result type matches what `StringGetCOp` expects (see line 3076 in MooreOps.td)
2. The result type matches the Slang AST type from `expr.type` (should be `byte` or equivalent)
3. No additional conversion needed - the result is already the correct type

**Note:** The `type` variable at line 183 contains the expected result type from Slang. For string indexing, this should be a byte type (8-bit integer). The implementation creates `TwoValuedI8` directly, which should match.

---

## File Changes Summary

### Modified Files

**`lib/Conversion/ImportVerilog/Expressions.cpp`**
- **Lines 193-198:** Add `moore::StringType` to the type check
- **After line 222:** Insert new string indexing handling block (~25 lines)

### No New Files Created

### No Files Deleted

---

## Testing Strategy

### Unit Tests

**Existing Test Infrastructure:**
- The 126 failing tests already exist and test string indexing
- These tests should pass after the implementation

**Test Coverage:**
1. **Constant index:** `byte c = str[0];`
2. **Dynamic index:** `byte c = str[i];`
3. **Index type conversion:** Various index types (int, logic, etc.)
4. **Lvalue rejection:** `str[0] = 'x';` should produce an error
5. **Edge cases:** Empty strings, out-of-bounds indices (runtime behavior)

### Integration Tests

**Manual Testing Steps:**
1. Run the failing test suite: `ninja -C build check-circt`
2. Verify all 126 string indexing tests pass
3. Check that error messages are clear for unsupported lvalue operations
4. Verify no regressions in other tests

### Expected Test Results

**Before Implementation:**
- 126 tests fail with "unsupported expression: element select into string"

**After Implementation:**
- All 126 tests pass
- No new test failures
- Clear error message for lvalue string indexing attempts

---

## Rollback Plan

### How to Revert Changes

**Simple Rollback:**
1. The changes are localized to one file: `Expressions.cpp`
2. Revert the two modifications:
   - Remove `moore::StringType` from the type check
   - Remove the string indexing handling block

**Git Command:**
```bash
git checkout HEAD -- lib/Conversion/ImportVerilog/Expressions.cpp
```

### No Data Migration Rollback Needed

This is a pure code change with no data migrations or schema changes.

---

## Edge Cases and Error Handling

### Edge Case 1: Lvalue String Indexing

**Scenario:** User attempts to assign to a string index: `str[0] = 'x';`

**Handling:**
```cpp
if (isLvalue) {
  mlir::emitError(loc) << "lvalue string indexing is not supported; "
                       << "strings are immutable and cannot be indexed for assignment";
  return {};
}
```

**Error Message:** Clear explanation that strings are immutable

---

### Edge Case 2: Index Type Conversion Failure

**Scenario:** Index cannot be converted to `TwoValuedI32`

**Handling:**
```cpp
index = context.materializeConversion(i32Type, index,
                                     expr.selector().type->isSigned(), loc);
if (!index)
  return {};
```

**Error Message:** Propagated from `materializeConversion` (existing error handling)

---

### Edge Case 3: Out-of-Bounds Index

**Scenario:** Index is outside the string bounds: `str[100]` when string has 10 characters

**Handling:**
- This is a **runtime** error, not a compile-time error
- The `StringGetCOp` operation will be created successfully
- Runtime behavior follows SystemVerilog semantics (returns 0 or undefined)
- No compile-time checking needed

---

### Edge Case 4: Negative Index

**Scenario:** Index is negative: `str[-1]`

**Handling:**
- Treated as a very large unsigned index (two's complement)
- Runtime behavior follows SystemVerilog semantics
- No special compile-time handling needed

---

### Edge Case 5: Four-Valued Index

**Scenario:** Index contains X or Z bits: `str[4'bxxxx]`

**Handling:**
- `materializeConversion` will convert four-valued to two-valued
- X and Z bits are converted to 0 (SystemVerilog default)
- The conversion is automatic via `LogicToIntOp`

---

## Code Patterns and Style

### Follow Existing Patterns

**Pattern 1: Type-Specific Handling**
- Similar to `AssocArrayType` handling (lines 201-222)
- Early return after handling the specific type
- Clear error messages for unsupported operations

**Pattern 2: Index Conversion**
- Similar to queue indexing (lines 258-275)
- Use `convertRvalueExpression` for the index
- Use `materializeConversion` for type conversion

**Pattern 3: Operation Creation**
- Use `::create` static method pattern
- Pass `builder`, `loc`, `resultType`, and operands
- Return the created operation's result

### Code Style Guidelines

1. **Indentation:** 2 spaces (consistent with existing code)
2. **Comments:** Clear, concise explanations for non-obvious logic
3. **Error Messages:** Descriptive and actionable
4. **Variable Names:** Follow existing conventions (`index`, `resultType`, etc.)
5. **Line Length:** Keep under 80 characters where possible

---

## Estimated Effort

### Complexity Assessment: **Low**

**Justification:**
- Small, localized change (one file, ~30 lines)
- Uses existing operations and infrastructure
- Clear requirements and well-defined scope
- No new operations or types needed

### Time Estimate: **2-4 hours**

**Breakdown:**
- Implementation: 1 hour
- Testing: 1 hour
- Code review and refinement: 1-2 hours

### Risk Assessment: **Low**

**Risks:**
- Minimal risk of breaking existing functionality
- Changes are isolated to string indexing path
- Existing tests provide good coverage

---

## Additional Considerations

### Performance

**No Performance Impact:**
- The implementation adds one type check (`isa<moore::StringType>`)
- The string indexing path is only taken for string types
- No impact on other type handling paths

### Maintainability

**High Maintainability:**
- Code follows existing patterns
- Clear separation of concerns
- Well-documented with comments
- Easy to extend for future string operations

### Future Extensions

**Potential Future Work:**
- String slicing: `str[3:0]` (range selection)
- String methods: `str.len()`, `str.substr()`, etc.
- String comparison operations
- String formatting operations

**Note:** These are out of scope for this implementation but could follow similar patterns.

---

## References

### Key Files

1. **`lib/Conversion/ImportVerilog/Expressions.cpp`** - Main implementation file
2. **`include/circt/Dialect/Moore/MooreOps.td`** - StringGetCOp definition (line 3070)
3. **`include/circt/Dialect/Moore/MooreTypes.td`** - Type definitions (TwoValuedI32, TwoValuedI8, StringType)

### Key Operations

- **`moore::StringGetCOp`** - Get character from string at index
- **`moore::IntType::getInt(context, width)`** - Create two-valued integer type
- **`context.materializeConversion(type, value, isSigned, loc)`** - Type conversion

### Key Concepts

- **Rvalue vs Lvalue:** Read-only vs writable expressions
- **Two-valued vs Four-valued:** Bit vs logic (2-state vs 4-state)
- **Domain conversion:** Converting between two-valued and four-valued types

---

## Summary

This implementation plan provides a complete roadmap for adding rvalue string indexing support to the ImportVerilog conversion. The changes are minimal, well-scoped, and follow existing patterns in the codebase. The implementation should resolve all 126 failing test cases while maintaining code quality and consistency.

**Next Steps:**
1. Review this plan with the team
2. Implement the changes as described
3. Run tests to verify correctness
4. Submit for code review

