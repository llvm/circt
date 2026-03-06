# TODO: Fall-through Blocks - Blocks That Aren't Wait Targets

## Severity: Medium

## The Problem

Consider this process:

```mlir
llhd.process -> i32 {
  %cond = ... some condition ...
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %val1 = ... compute something ...
  cf.br ^bb3(%val1 : i32)
^bb2:
  %val2 = ... compute something else ...
  cf.br ^bb3(%val2 : i32)
^bb3(%arg: i32):
  llhd.wait yield (%arg : i32), delay %time, ^bb4
^bb4:
  llhd.halt %arg : i32
}
```

In this example:
- **^bb1 and ^bb2** are "fall-through" blocks - they're not wait targets
- They just do computation and branch to ^bb3
- **^bb3 and ^bb4** are wait targets (they have wait/halt)

## Current Implementation

Location: `lib/Dialect/Arc/Transforms/LowerProcesses.cpp:284-292`

```cpp
// Clone operations from process blocks into switch blocks
for (Block &processBlock : processOp.getBody()) {
  // Find the corresponding switch block (if this block has an index)
  auto it = analysis.blockIndices.find(&processBlock);
  if (it == analysis.blockIndices.end()) {
    // This block doesn't have an index, so it's not a resumption point
    // We'll handle it when we clone the blocks that branch to it
    continue;
  }
  // ...
}
```

We **skip blocks that don't have an index** (fall-through blocks). The comment says "We'll handle it when we clone the blocks that branch to it" but **we don't actually do that**!

## The Issue

If we have a process like the example above, blocks ^bb1 and ^bb2 would be **completely ignored**. Their operations wouldn't be cloned anywhere, and the control flow would be broken.

## What Needs to Happen

There are several approaches:

### Approach 1: Inline Fall-through Blocks

When we encounter a branch to a fall-through block, inline that block's operations:

```cpp
// When cloning operations
for (Operation &op : processBlock.without_terminator()) {
  if (auto brOp = dyn_cast<cf::BranchOp>(&op)) {
    Block *dest = brOp.getDest();
    if (analysis.blockIndices.find(dest) == analysis.blockIndices.end()) {
      // Destination is a fall-through block - inline it
      inlineFallThroughBlock(dest, bodyBuilder, mapping);
      continue;
    }
  }
  bodyBuilder.clone(op, mapping);
}

void inlineFallThroughBlock(Block *block, OpBuilder &builder, IRMapping &mapping) {
  // Clone all operations from the fall-through block
  for (Operation &op : block->without_terminator()) {
    builder.clone(op, mapping);
  }
  
  // Handle the terminator
  auto *term = block->getTerminator();
  if (auto brOp = dyn_cast<cf::BranchOp>(term)) {
    // Continue inlining if destination is also fall-through
    Block *dest = brOp.getDest();
    if (analysis.blockIndices.find(dest) == analysis.blockIndices.end()) {
      inlineFallThroughBlock(dest, builder, mapping);
    } else {
      // Destination is indexed - create branch to it
      builder.clone(*term, mapping);
    }
  } else {
    builder.clone(*term, mapping);
  }
}
```

### Approach 2: Recursively Clone Reachable Blocks

Starting from each indexed block, recursively clone all reachable blocks until we hit another indexed block:

```cpp
void cloneBlockAndSuccessors(Block *block, OpBuilder &builder, 
                              IRMapping &mapping,
                              const DenseMap<Block*, unsigned> &indices) {
  for (Operation &op : block->without_terminator()) {
    builder.clone(op, mapping);
  }
  
  auto *term = block->getTerminator();
  for (Block *succ : term->getSuccessors()) {
    if (indices.find(succ) == indices.end()) {
      // Fall-through block - clone it recursively
      cloneBlockAndSuccessors(succ, builder, mapping, indices);
    }
  }
  
  // Clone terminator (might need special handling)
  builder.clone(*term, mapping);
}
```

### Approach 3: Reject Processes with Fall-through Blocks (Simplest)

Just reject them for now:

```cpp
// During analysis
for (Block &block : processOp.getBody()) {
  if (analysis.blockIndices.find(&block) == analysis.blockIndices.end()) {
    // This is a fall-through block
    if (&block != analysis.entryBlock) {
      return processOp.emitError(
          "processes with fall-through blocks are not yet supported");
    }
  }
}
```

## Implementation Steps (Approach 1 - Inlining)

1. **Identify fall-through blocks during analysis**:
   ```cpp
   SmallVector<Block *> fallThroughBlocks;
   for (Block &block : processOp.getBody()) {
     if (analysis.blockIndices.find(&block) == analysis.blockIndices.end()) {
       fallThroughBlocks.push_back(&block);
     }
   }
   ```

2. **Implement inlining function**:
   - Handle block arguments (need to map them)
   - Handle nested fall-through blocks (recursion)
   - Handle cycles (detect and error)

3. **Update cloning logic**:
   - When cloning a branch, check if destination is fall-through
   - If so, inline instead of creating branch

4. **Handle block arguments**:
   ```cpp
   // When inlining a block with arguments
   for (unsigned i = 0; i < block->getNumArguments(); ++i) {
     BlockArgument arg = block->getArgument(i);
     Value operand = branchOp.getOperand(i);
     mapping.map(arg, mapping.lookupOrDefault(operand));
   }
   ```

5. **Add tests**:
   - Simple fall-through (one intermediate block)
   - Multiple fall-throughs in sequence
   - Conditional branches to fall-throughs
   - Fall-throughs with block arguments

## Current Behavior

If you try to lower a process with fall-through blocks, it will:
1. **Silently skip** the fall-through blocks during cloning
2. **Produce invalid IR** because branches to those blocks will be broken
3. **Likely fail verification** or produce incorrect results

## Example That Would Fail

```mlir
llhd.process -> i32 {
  %cond = hw.constant true
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:  // Fall-through block - would be skipped!
  %c10 = hw.constant 10 : i32
  cf.br ^bb3(%c10 : i32)
^bb2:  // Fall-through block - would be skipped!
  %c20 = hw.constant 20 : i32
  cf.br ^bb3(%c20 : i32)
^bb3(%val: i32):  // Wait target - would be cloned
  llhd.halt %val : i32
}
```

This would produce broken IR because:
- ^bb1 and ^bb2 wouldn't exist in the lowered code
- The entry block would try to branch to non-existent blocks
- The block argument %val wouldn't be properly mapped

## Why It's Not Implemented

1. **Complexity**: Proper handling requires careful control flow analysis
2. **Block arguments**: Fall-through blocks often have block arguments, which adds another layer of complexity
3. **Testing**: Would need comprehensive tests with various control flow patterns
4. **Common case**: Many simple processes don't have fall-through blocks

## Impact

Without this fix:
- Processes with complex control flow are silently miscompiled
- No error is produced, leading to incorrect behavior
- Users must use only simple linear control flow

## Workaround

Use simple linear control flow without intermediate blocks:
- Put all computation in blocks that are wait targets
- Avoid conditional branches to intermediate blocks

## Effort Estimate

**Medium** - Requires control flow analysis and careful handling of block arguments. The inlining approach is conceptually straightforward but has edge cases to handle.

