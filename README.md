# clattner-test / spt

This is a test repository, experimenting with an approach of building a modular set of tools
for verilog and other SiFive related projects.

SPT is just a placeholder, "SiFive Platform Tools", better suggestion welcome :-)


## Setting this up

These are the commands Chris used to set this up on a Mac:

1) Install Dependences of LLVM/MLIR according to [the
  instructions](https://mlir.llvm.org/getting_started/), including cmake and ninja. 

2) Check out LLVM and SPT repo's:

```
$ cd ~/Projects
$ git clone git@github.com:llvm/llvm-project.git
$ git clone git@github.com:sifive/clattner-experimental.git spt
```

3) HACK: Add symlink because I can't figure out how to get `LLVM_EXTERNAL_SPT_SOURCE_DIR` to work with cmake:

```
$ cd ~/Projects/llvm-project
$ ln -s ../spt spt
```

4) Configure the build to build MLIR and SPT (MLIR is probably not necessary, but it builds 
reasonably fast and is good to provide a sanity check that things are working): 

```
$ cd ~/Projects/llvm-project
$ mkdir build
$ cd build
$ cmake -G Ninja ../llvm  -DLLVM_EXTERNAL_PROJECTS="spt" -DLLVM_EXTERNAL_SPT_SOURCE_DIR=/Users/chrisl/Projects/spt   -DLLVM_ENABLE_PROJECTS="mlir;spt" -DLLVM_TARGETS_TO_BUILD="X86" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON
```

5) Build MLIR and run MLIR tests as a smoketest:

```
$ ninja check-mlir
```

6) Build SPT and run SPT tests:

```
$ ninja check-spt
```
