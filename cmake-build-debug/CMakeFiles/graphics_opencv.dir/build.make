# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/xbb1973/Documents/code/workdir/lab_digital_image

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/xbb1973/Documents/code/workdir/lab_digital_image/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/graphics_opencv.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/graphics_opencv.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/graphics_opencv.dir/flags.make

CMakeFiles/graphics_opencv.dir/graphics.cpp.o: CMakeFiles/graphics_opencv.dir/flags.make
CMakeFiles/graphics_opencv.dir/graphics.cpp.o: ../graphics.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/xbb1973/Documents/code/workdir/lab_digital_image/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/graphics_opencv.dir/graphics.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/graphics_opencv.dir/graphics.cpp.o -c /Users/xbb1973/Documents/code/workdir/lab_digital_image/graphics.cpp

CMakeFiles/graphics_opencv.dir/graphics.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/graphics_opencv.dir/graphics.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/xbb1973/Documents/code/workdir/lab_digital_image/graphics.cpp > CMakeFiles/graphics_opencv.dir/graphics.cpp.i

CMakeFiles/graphics_opencv.dir/graphics.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/graphics_opencv.dir/graphics.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/xbb1973/Documents/code/workdir/lab_digital_image/graphics.cpp -o CMakeFiles/graphics_opencv.dir/graphics.cpp.s

# Object files for target graphics_opencv
graphics_opencv_OBJECTS = \
"CMakeFiles/graphics_opencv.dir/graphics.cpp.o"

# External object files for target graphics_opencv
graphics_opencv_EXTERNAL_OBJECTS =

graphics_opencv: CMakeFiles/graphics_opencv.dir/graphics.cpp.o
graphics_opencv: CMakeFiles/graphics_opencv.dir/build.make
graphics_opencv: /usr/local/lib/libopencv_gapi.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_stitching.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_aruco.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_bgsegm.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_bioinspired.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_ccalib.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_dnn_objdetect.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_dnn_superres.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_dpm.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_face.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_freetype.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_fuzzy.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_hfs.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_img_hash.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_line_descriptor.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_quality.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_reg.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_rgbd.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_saliency.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_sfm.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_stereo.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_structured_light.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_superres.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_surface_matching.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_tracking.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_videostab.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_xfeatures2d.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_xobjdetect.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_xphoto.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_highgui.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_shape.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_datasets.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_plot.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_text.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_dnn.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_ml.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_phase_unwrapping.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_optflow.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_ximgproc.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_video.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_videoio.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_imgcodecs.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_objdetect.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_calib3d.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_features2d.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_flann.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_photo.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_imgproc.4.2.0.dylib
graphics_opencv: /usr/local/lib/libopencv_core.4.2.0.dylib
graphics_opencv: CMakeFiles/graphics_opencv.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/xbb1973/Documents/code/workdir/lab_digital_image/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable graphics_opencv"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/graphics_opencv.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/graphics_opencv.dir/build: graphics_opencv

.PHONY : CMakeFiles/graphics_opencv.dir/build

CMakeFiles/graphics_opencv.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/graphics_opencv.dir/cmake_clean.cmake
.PHONY : CMakeFiles/graphics_opencv.dir/clean

CMakeFiles/graphics_opencv.dir/depend:
	cd /Users/xbb1973/Documents/code/workdir/lab_digital_image/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/xbb1973/Documents/code/workdir/lab_digital_image /Users/xbb1973/Documents/code/workdir/lab_digital_image /Users/xbb1973/Documents/code/workdir/lab_digital_image/cmake-build-debug /Users/xbb1973/Documents/code/workdir/lab_digital_image/cmake-build-debug /Users/xbb1973/Documents/code/workdir/lab_digital_image/cmake-build-debug/CMakeFiles/graphics_opencv.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/graphics_opencv.dir/depend
