# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\cmake-build-debug"

# Utility rule file for glut.

# Include the progress variables for this target.
include CMakeFiles/glut.dir/progress.make

CMakeFiles/glut: CMakeFiles/glut-complete


CMakeFiles/glut-complete: glut-prefix/src/glut-stamp/glut-install
CMakeFiles/glut-complete: glut-prefix/src/glut-stamp/glut-mkdir
CMakeFiles/glut-complete: glut-prefix/src/glut-stamp/glut-download
CMakeFiles/glut-complete: glut-prefix/src/glut-stamp/glut-update
CMakeFiles/glut-complete: glut-prefix/src/glut-stamp/glut-patch
CMakeFiles/glut-complete: glut-prefix/src/glut-stamp/glut-configure
CMakeFiles/glut-complete: glut-prefix/src/glut-stamp/glut-build
CMakeFiles/glut-complete: glut-prefix/src/glut-stamp/glut-install
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir="D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\cmake-build-debug\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Completed 'glut'"
	"C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -E make_directory "D:/Documents/Mark Karpati stuff/BME/04 felev/Grafika/GrafikaHF02CLion/cmake-build-debug/CMakeFiles"
	"C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -E touch "D:/Documents/Mark Karpati stuff/BME/04 felev/Grafika/GrafikaHF02CLion/cmake-build-debug/CMakeFiles/glut-complete"
	"C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -E touch "D:/Documents/Mark Karpati stuff/BME/04 felev/Grafika/GrafikaHF02CLion/cmake-build-debug/glut-prefix/src/glut-stamp/glut-done"

glut-prefix/src/glut-stamp/glut-install: glut-prefix/src/glut-stamp/glut-build
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir="D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\cmake-build-debug\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "No install step for 'glut'"
	cd /d "D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\cmake-build-debug\glut-prefix\src\glut-build" && "C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -E echo_append
	cd /d "D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\cmake-build-debug\glut-prefix\src\glut-build" && "C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -E touch "D:/Documents/Mark Karpati stuff/BME/04 felev/Grafika/GrafikaHF02CLion/cmake-build-debug/glut-prefix/src/glut-stamp/glut-install"

glut-prefix/src/glut-stamp/glut-mkdir:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir="D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\cmake-build-debug\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Creating directories for 'glut'"
	"C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -E make_directory "D:/Documents/Mark Karpati stuff/BME/04 felev/Grafika/GrafikaHF02CLion/src/freeglut"
	"C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -E make_directory "D:/Documents/Mark Karpati stuff/BME/04 felev/Grafika/GrafikaHF02CLion/cmake-build-debug/glut-prefix/src/glut-build"
	"C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -E make_directory "D:/Documents/Mark Karpati stuff/BME/04 felev/Grafika/GrafikaHF02CLion/cmake-build-debug/glut-prefix"
	"C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -E make_directory "D:/Documents/Mark Karpati stuff/BME/04 felev/Grafika/GrafikaHF02CLion/cmake-build-debug/glut-prefix/tmp"
	"C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -E make_directory "D:/Documents/Mark Karpati stuff/BME/04 felev/Grafika/GrafikaHF02CLion/cmake-build-debug/glut-prefix/src/glut-stamp"
	"C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -E make_directory "D:/Documents/Mark Karpati stuff/BME/04 felev/Grafika/GrafikaHF02CLion/cmake-build-debug/glut-prefix/src"
	"C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -E make_directory "D:/Documents/Mark Karpati stuff/BME/04 felev/Grafika/GrafikaHF02CLion/cmake-build-debug/glut-prefix/src/glut-stamp"
	"C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -E touch "D:/Documents/Mark Karpati stuff/BME/04 felev/Grafika/GrafikaHF02CLion/cmake-build-debug/glut-prefix/src/glut-stamp/glut-mkdir"

glut-prefix/src/glut-stamp/glut-download: glut-prefix/src/glut-stamp/glut-urlinfo.txt
glut-prefix/src/glut-stamp/glut-download: glut-prefix/src/glut-stamp/glut-mkdir
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir="D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\cmake-build-debug\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_4) "Performing download step (download, verify and extract) for 'glut'"
	cd /d "D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\src" && "C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -P "D:/Documents/Mark Karpati stuff/BME/04 felev/Grafika/GrafikaHF02CLion/cmake-build-debug/glut-prefix/src/glut-stamp/download-glut.cmake"
	cd /d "D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\src" && "C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -P "D:/Documents/Mark Karpati stuff/BME/04 felev/Grafika/GrafikaHF02CLion/cmake-build-debug/glut-prefix/src/glut-stamp/verify-glut.cmake"
	cd /d "D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\src" && "C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -P "D:/Documents/Mark Karpati stuff/BME/04 felev/Grafika/GrafikaHF02CLion/cmake-build-debug/glut-prefix/src/glut-stamp/extract-glut.cmake"
	cd /d "D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\src" && "C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -E touch "D:/Documents/Mark Karpati stuff/BME/04 felev/Grafika/GrafikaHF02CLion/cmake-build-debug/glut-prefix/src/glut-stamp/glut-download"

glut-prefix/src/glut-stamp/glut-update: glut-prefix/src/glut-stamp/glut-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir="D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\cmake-build-debug\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_5) "No update step for 'glut'"
	"C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -E echo_append
	"C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -E touch "D:/Documents/Mark Karpati stuff/BME/04 felev/Grafika/GrafikaHF02CLion/cmake-build-debug/glut-prefix/src/glut-stamp/glut-update"

glut-prefix/src/glut-stamp/glut-patch: glut-prefix/src/glut-stamp/glut-update
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir="D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\cmake-build-debug\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_6) "No patch step for 'glut'"
	"C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -E echo_append
	"C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -E touch "D:/Documents/Mark Karpati stuff/BME/04 felev/Grafika/GrafikaHF02CLion/cmake-build-debug/glut-prefix/src/glut-stamp/glut-patch"

glut-prefix/src/glut-stamp/glut-configure: glut-prefix/tmp/glut-cfgcmd.txt
glut-prefix/src/glut-stamp/glut-configure: glut-prefix/src/glut-stamp/glut-patch
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir="D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\cmake-build-debug\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_7) "No configure step for 'glut'"
	cd /d "D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\cmake-build-debug\glut-prefix\src\glut-build" && "C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -E echo_append
	cd /d "D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\cmake-build-debug\glut-prefix\src\glut-build" && "C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -E touch "D:/Documents/Mark Karpati stuff/BME/04 felev/Grafika/GrafikaHF02CLion/cmake-build-debug/glut-prefix/src/glut-stamp/glut-configure"

glut-prefix/src/glut-stamp/glut-build: glut-prefix/src/glut-stamp/glut-configure
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir="D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\cmake-build-debug\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_8) "No build step for 'glut'"
	cd /d "D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\cmake-build-debug\glut-prefix\src\glut-build" && "C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -E echo_append
	cd /d "D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\cmake-build-debug\glut-prefix\src\glut-build" && "C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -E touch "D:/Documents/Mark Karpati stuff/BME/04 felev/Grafika/GrafikaHF02CLion/cmake-build-debug/glut-prefix/src/glut-stamp/glut-build"

glut: CMakeFiles/glut
glut: CMakeFiles/glut-complete
glut: glut-prefix/src/glut-stamp/glut-build
glut: glut-prefix/src/glut-stamp/glut-configure
glut: glut-prefix/src/glut-stamp/glut-download
glut: glut-prefix/src/glut-stamp/glut-install
glut: glut-prefix/src/glut-stamp/glut-mkdir
glut: glut-prefix/src/glut-stamp/glut-patch
glut: glut-prefix/src/glut-stamp/glut-update
glut: CMakeFiles/glut.dir/build.make

.PHONY : glut

# Rule to build all files generated by this target.
CMakeFiles/glut.dir/build: glut

.PHONY : CMakeFiles/glut.dir/build

CMakeFiles/glut.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\glut.dir\cmake_clean.cmake
.PHONY : CMakeFiles/glut.dir/clean

CMakeFiles/glut.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" "D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion" "D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion" "D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\cmake-build-debug" "D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\cmake-build-debug" "D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\cmake-build-debug\CMakeFiles\glut.dir\DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/glut.dir/depend

