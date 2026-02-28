cmake_minimum_required(VERSION 3.16) # minimum cmake version
project(sentiment_cpp) # project name

set(CMAKE_CXX_STANDARD 17) # use c++17
set(CMAKE_CXX_STANDARD_REQUIRED ON) # enforce standard

include_directories(include) # headers in cpp/include

add_library(core
  src/tsv_reader.cpp
  src/text_clean.cpp
  src/tokenizer.cpp
  src/ngrams.cpp
  src/hashing.cpp
  src/sparse_vec.cpp
  src/dataset.cpp
  src/logreg_model.cpp
  src/metrics.cpp
  src/sgd_trainer.cpp
  src/model_io.cpp
  src/csv_write.cpp
  src/json_write.cpp
)

add_executable(train_cpp app/train_main.cpp) # trainer entrypoint
target_link_libraries(train_cpp core) # link core

add_executable(predict_cpp app/predict_main.cpp) # predictor entrypoint
target_link_libraries(predict_cpp core) # link core
