# "CIRT" / Circuit IR and Tools

This is an experimental repository, applying the MLIR/LLVM approach to building
modular tools for Verilog, focusing on FIRRTL.

"CIRT" stands for "Circuit IR and Tools" or perhaps "CIRCuiT + 
IntermediateRepresenTation + Toolbox" (hat tip to Aliaksei Chapyzhenka).  The
T can be further expanded as Tool, Translator, Team, Tech, Target, Tree, Type,
...  This name can still be changed if a better one is suggested.  :-)

## Setting this up

These are the commands Chris used to set this up on a Mac:

1) Install Dependences of LLVM/MLIR according to [the
  instructions](https://mlir.llvm.org/getting_started/), including cmake and ninja. 

2) Check out LLVM and CIRT repo's:

```
$ cd ~/Projects
$ git clone git@github.com:llvm/llvm-project.git
$ git clone git@github.com:sifive/cirt.git cirt
```

3) HACK: Add symlink because I can't figure out how to get
   `LLVM_EXTERNAL_CIRT_SOURCE_DIR` to work with cmake (I'd love help with
   this!):

```
$ cd ~/Projects/llvm-project
$ ln -s ../cirt cirt
```

4) Configure the build to build MLIR and CIRT using a command like this
   (replace `/Users/chrisl` with the paths you want to use):

```
$ cd ~/Projects/llvm-project
$ mkdir build
$ cd build
$ cmake -G Ninja ../llvm  -DLLVM_EXTERNAL_PROJECTS="cirt" -DLLVM_EXTERNAL_CIRT_SOURCE_DIR=/Users/chrisl/Projects/cirt   -DLLVM_ENABLE_PROJECTS="mlir;cirt" -DLLVM_TARGETS_TO_BUILD="X86;RISCV"  -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=DEBUG
```

The `-DCMAKE_BUILD_TYPE=DEBUG` flag enables debug information, which makes the
whole tree compile slower, but allows you to step through code into the LLVM
and MLIR frameworks.

To get something that runs fast, use `-DCMAKE_BUILD_TYPE=Release` or
`-DCMAKE_BUILD_TYPE=RelWithDebInfo` if you want to go fast and optionally if
you want debug info to go with it.  `RELEASE` mode makes a very large difference
in performance.

5) Build MLIR and run MLIR tests as a smoketest - this isn't needed, but is
reasonably fast and a good sanity check:

```
$ ninja check-mlir
```

6) Build CIRT and run CIRT tests:

```
$ ninja check-cirt
```

7) Optionally configure your environment:

It is useful to add the .../llvm-project/build/bin directory to the end
of your PATH, allowing you to use the tools like cirt-opt in a natural way on
the command line.  Similarly, you need to be in the build directory to invoke
ninja, which is super annoying.  You might find a bash/zsh alias like this to
be useful:

```bash
build() {
  (cd $HOME/Projects/llvm-project/build/; ninja $1 $2 $3)
}
```

This allows you to invoke `build check-cirt` from any directory and have it do
the right thing.

## Submitting changes to CIRT

The project is small so there is no formal process yet. Just talk to Chris, or
create a PR.

## Submitting changes to LLVM / MLIR

This project depends on MLIR and LLVM, and it is occasionally useful to improve them.
To get set up for this:

1) Follow the "[How to Contribute](https://mlir.llvm.org/getting_started/Contributing/)" instructions, and install the right tools, e.g. 'arcanist' and `clang-format`.
 2) Get an LLVM Phabricator account
 3) [Ask for LLVM commit access](https://llvm.org/docs/DeveloperPolicy.html#obtaining-commit-access), the barrier is low.

### Submitting a patch

The patch flow goes like this:

1) Check out the LLVM mono repo (as described above).
2) Make changes to your codebase.
3) Stage your changes with `git add`.
4) Run clang-format to tidy up the details of the patch with `git clang-format origin/master` 
5) Create a [patch in Phabricator](https://llvm.org/docs/Phabricator.html) with `arc diff`.
6) Iterate on review, changing your code and sending another patch with `arc diff`.

When your review converges and your patch is approved, do the following:

1) commit your changes with `git commit`
2) rebase your changes with `git pull --rebase`
3) retest your patch with `ninja check-mlir`  (or whatever other target makes sense)
4) push your changes with `git push`

