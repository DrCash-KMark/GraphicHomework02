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

# Include any dependencies generated for this target.
include CMakeFiles/program.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/program.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/program.dir/flags.make

CMakeFiles/program.dir/src/framework.cpp.obj: CMakeFiles/program.dir/flags.make
CMakeFiles/program.dir/src/framework.cpp.obj: CMakeFiles/program.dir/includes_CXX.rsp
CMakeFiles/program.dir/src/framework.cpp.obj: ../src/framework.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\cmake-build-debug\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/program.dir/src/framework.cpp.obj"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\program.dir\src\framework.cpp.obj -c "D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\src\framework.cpp"

CMakeFiles/program.dir/src/framework.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/program.dir/src/framework.cpp.i"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\src\framework.cpp" > CMakeFiles\program.dir\src\framework.cpp.i

CMakeFiles/program.dir/src/framework.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/program.dir/src/framework.cpp.s"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\src\framework.cpp" -o CMakeFiles\program.dir\src\framework.cpp.s

CMakeFiles/program.dir/src/Skeleton.cpp.obj: CMakeFiles/program.dir/flags.make
CMakeFiles/program.dir/src/Skeleton.cpp.obj: CMakeFiles/program.dir/includes_CXX.rsp
CMakeFiles/program.dir/src/Skeleton.cpp.obj: ../src/Skeleton.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\cmake-build-debug\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/program.dir/src/Skeleton.cpp.obj"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\program.dir\src\Skeleton.cpp.obj -c "D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\src\Skeleton.cpp"

CMakeFiles/program.dir/src/Skeleton.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/program.dir/src/Skeleton.cpp.i"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\src\Skeleton.cpp" > CMakeFiles\program.dir\src\Skeleton.cpp.i

CMakeFiles/program.dir/src/Skeleton.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/program.dir/src/Skeleton.cpp.s"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\src\Skeleton.cpp" -o CMakeFiles\program.dir\src\Skeleton.cpp.s

# Object files for target program
program_OBJECTS = \
"CMakeFiles/program.dir/src/framework.cpp.obj" \
"CMakeFiles/program.dir/src/Skeleton.cpp.obj"

# External object files for target program
program_EXTERNAL_OBJECTS =

program.exe: CMakeFiles/program.dir/src/framework.cpp.obj
program.exe: CMakeFiles/program.dir/src/Skeleton.cpp.obj
program.exe: CMakeFiles/program.dir/build.make
program.exe: CMakeFiles/program.dir/linklibs.rsp
program.exe: CMakeFiles/program.dir/objects1.rsp
program.exe: CMakeFiles/program.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\cmake-build-debug\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable program.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\program.dir\link.txt --verbose=$(VERBOSE)
	"C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -E copy "D:/Documents/Mark Karpati stuff/BME/04 felev/Grafika/GrafikaHF02CLion/src/freeglut/bin/x64/freeglut.dll" "D:/Documents/Mark Karpati stuff/BME/04 felev/Grafika/GrafikaHF02CLion/cmake-build-debug/freeglut.dll"
	"C:\Program Files\JetBrains\CLion 2020.3.2\bin\cmake\win\bin\cmake.exe" -E copy "D:/Documents/Mark Karpati stuff/BME/04 felev/Grafika/GrafikaHF02CLion/src/glew/bin/Release/x64/glew32.dll" "D:/Documents/Mark Karpati stuff/BME/04 felev/Grafika/GrafikaHF02CLion/cmake-build-debug/glew32.dll"

# Rule to build all files generated by this target.
CMakeFiles/program.dir/build: program.exe

.PHONY : CMakeFiles/program.dir/build

CMakeFiles/program.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\program.dir\cmake_clean.cmake
.PHONY : CMakeFiles/program.dir/clean

CMakeFiles/program.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" "D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion" "D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion" "D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\cmake-build-debug" "D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\cmake-build-debug" "D:\Documents\Mark Karpati stuff\BME\04 felev\Grafika\GrafikaHF02CLion\cmake-build-debug\CMakeFiles\program.dir\DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/program.dir/depend

