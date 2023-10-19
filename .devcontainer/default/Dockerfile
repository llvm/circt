FROM --platform=$BUILDPLATFORM ubuntu:rolling

# The following line is needed to set the $TARGETARCH variable to amd64 or arm64 depending on the target architecture. If you need alternative formulations (x86_64 or aarch64) use $(arch) in a RUN command.
ARG TARGETARCH

# This can be changed to "ci" to remove developer tools and reduce image size. Currently this only saves about 200MB of space
ARG CONTAINER_MODE=dev

# The version of clang to use
ARG CLANG_VERSION=17

# Setup container environment
ENV \
    # Change locale to suppress some warnings
    LC_ALL=C \
    #
    # Set VSCode as default editor
    EDITOR="code --wait" \
    GIT_EDITOR="code --wait"

# Setup container
RUN \
    #
    # Disable apt interactive prompts for this RUN command
    export DEBIAN_FRONTEND="noninteractive" && \
    #
    # Get release information (used to determine the correct URL for the LLVM apt repository)
    . /etc/lsb-release && \
    #
    # Get package list
    apt-get update && \
    #
    # Add LLVM/Clang repository so we can install newer versions of clang
    apt-get install -y software-properties-common gnupg curl && \
    curl -L https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    apt-add-repository "deb http://apt.llvm.org/$DISTRIB_CODENAME/ llvm-toolchain-$DISTRIB_CODENAME-$CLANG_VERSION main" && \
    # 
    # Update again now that we've added the LLVM repository
    apt-get update && \
    # 
    # Install required packages
    apt-get install -y \
      #
      # The following packages are used to build CIRCT
      ccache \
      cmake \
      make \
      ninja-build \
      clang-$CLANG_VERSION \
      clang-tools-$CLANG_VERSION \
      clang-format-$CLANG_VERSION \
      clang-tidy-$CLANG_VERSION \
      clangd-$CLANG_VERSION \
      lld-$CLANG_VERSION \
      lldb-$CLANG_VERSION \
      jq \
      #
      # ShellCheck is useful for validating bash scripts
      shellcheck \
      #
      # Verilator is used to run generated verilog
      verilator \
    && \
    #
    # Remove GCC (installed as a dependency of CMake) and the MSVC version of clang to reduce CMake confusion
    apt remove -y gcc && \
    apt autoremove -y && \
    rm /usr/bin/clang-cl-$CLANG_VERSION && \
    #
    # Create suffix-free links to clang binaries
    for FILE in $(find /usr/bin -name "*-$CLANG_VERSION"); do ln -s ${FILE} ${FILE%-$CLANG_VERSION}; done && \
    #
    # Verilator hardcodes "g++" as the C++ compiler, so we symlink clang++ to g++ to avoid having gcc installed
    cd /usr/bin && ln -s clang++-$CLANG_VERSION g++ && \
    #
    # Based on the container mode, either remove unnecessary packages or install additional developer-oriented packages
    case "$CONTAINER_MODE" in \
      "ci") \
        #
        # Remove packages we only needed for this step
        apt-add-repository -r "deb http://apt.llvm.org/$UBUNTU_VERSION_NAME/ llvm-toolchain-$UBUNTU_VERSION_NAME-$CLANG_VERSION main" && \
        apt remove -y software-properties-common gnupg curl && \
        #
        # Clean up apt to reduce image size
        apt autoremove -y && \
        rm -rf /var/lib/apt/lists/* \
        ;; \
      "dev") \
        # Install additional development tools
        apt-get install -y git \
        ;; \
      *) \
        echo "Invalid CONTAINER_MODE: $CONTAINER_MODE" && \
        exit 1 \
        ;; \
    esac
