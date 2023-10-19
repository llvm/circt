## Using VSCode devcontainers

VSCode supports creating development environments inside of a docker container. With some light configuration, we can get a consistent development environment that is robust to changes in your local environment and also doesn't require you to install and manage dependencies on your machine.

### Setup

1. Install Docker, a tool for managing containerized VMs: https://www.docker.com
  - If you have Docker installed, make sure it is updated to the most recent version (there is a "Check for Updates" option in the application UI).
  - If installing Docker on macOS for the first time, I recommend selecting the "Advanced" installation option, specifying a "User" installation and disabling the two  options below. This makes it so your Docker installation does not require root privileges for anything, making updates more seamless. This will require you telling VSCode where the `docker` executable ended up by changing the "dev.containers.dockerPath" setting `"/Users/<your-user-name>/.docker/bin/docker”`. You should make sure this executable exists by executing `"/Users/<your-user-name>/.docker/bin/docker --version”`, if you are not on macOS the `docker` executable might be at a different path.
2. Install Visual Studio Code and the Remote Containers extensions: https://code.visualstudio.com/docs/devcontainers/tutorial
3. Configure Docker by opening up the Docker application and navigating to "Settings"
  - Recommended settings for macOS (some of these are defaults):
    - General:
      - "Choose file sharing implementation for your containers": VirtioFS (better IO performance)
    - Resources:
      - CPUs: Allow docker to use most or all of your CPUs
      - Memory: Allow docker to use most or all of your memory
4. Open up this repository in VSCode
  - VSCode has an "Install 'code' command in PATH" command which installs a helpful tool to open VSCode from the commandline (`code <path-to-this-repo>`)
5. Select "Reopen in Container" in the notification that pops up
  - If there is no popup you can manually open the dev container by selecting "Dev Containers: Rebuild and Reopen in Container" from the command palette (Command-Shift-P on macOS)
  - Occasionally, after pulling from git, VSCode may prompt you to rebuild the container if the container definition has changed
6. Wait for the container to be built
7. Launch "CMake: Configure" from the command pallette.
  - The first time you do this, you will need to select a CMake Kit. There should only be one set up in the devcontainer (called "Clang <version>").
8. Build the default "CMake: build" task (Command-Shift-B on macOS)
  - The first time you do this, you may need to select "CMake: build" from a menu that pops up. This is due to an issue where VSCode does not seem to import "tasks.json" tasks until a task has been run. Subsequent invocations will use the default build task specified in "tasks.json"

### Usage

VSCode's capabilities are steadily evolving, but here are a few things you can do:
- From the command palette (Command-Shift-P on macOS)
  - `CMake: Configure` (Configures the project)
  - `CMake: Build Target` (allows you to select a target to build)
  - `Terminal: Create New Terminal` (opens a terminal in the container environment)
- From a C++ source file (requires `CMake: Configure` to have run)
  - `clangd` integration with jump-to-source and autocomplete
- From an MLIR file (requires the `circt-lsp-server` CMake target to be built)
  - MLIR language server support for CIRCT dialects
- Specify additional extensions to be installed in the devcontainer in `VSCode Settings > Extensions > Dev Containers > Default Extensions`
