cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

project(LargeScaleMapping)

SET( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin) 

set(HDRS_DBOW2
  DBoW2/BowVector.h
  DBoW2/FORB.h 
  DBoW2/FClass.h       
  DBoW2/FeatureVector.h
  DBoW2/ScoringObject.h   
  DBoW2/TemplatedVocabulary.h)
set(SRCS_DBOW2
  DBoW2/BowVector.cpp
  DBoW2/FORB.cpp      
  DBoW2/FeatureVector.cpp
  DBoW2/ScoringObject.cpp)

find_package(OpenCV REQUIRED)

include_directories(
	${OpenCV_INCLUDE_DIRS} 
	${G2O_INCLUDE_DIR} 
	${CSPARSE_INCLUDE_DIR} 
	${GTSAM_INCLUDE_DIR}
	${DBoW2_INCLUDE_DIR}
	"/usr/include/eigen3/"
)

add_executable(
	visualOdometry ${PROJECT_SOURCE_DIR}/visualOdometry/VisualOdometry.cpp
)
add_executable(
	bundleAdjust ${PROJECT_SOURCE_DIR}/visualOdometry/bundleAdjust.cpp
)

add_executable(
	BoWtest ${PROJECT_SOURCE_DIR}/visualOdometry/bagOfWordsDetector.cpp
)



target_link_libraries(
	visualOdometry ${OpenCV_LIBS}
)
target_link_libraries(
	bundleAdjust ${OpenCV_LIBS} ${G2O_LIBS} ${CSPARSE_LIBS} g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
)
target_link_libraries(
	BoWtest ${OpenCV_LIBS} ${DBoW2_LIBS}  DBoW2
)
target_include_directories(
	BoWtest PUBLIC ${DBoW2_INCLUDE_DIR}
)