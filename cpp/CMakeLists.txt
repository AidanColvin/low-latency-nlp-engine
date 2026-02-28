cmake_minimum_required(VERSION 3.16) # cmake version
project(sentiment_cpp) # project name

set(CMAKE_CXX_STANDARD 17) # use c++17
set(CMAKE_CXX_STANDARD_REQUIRED ON) # enforce standard

include_directories(include) # header path

add_library(core
  src/args.cpp
  src/tsv_reader.cpp
  src/text_clean.cpp
  src/tokenizer.cpp
  src/ngrams.cpp
  src/hashing.cpp
  src/dataset.cpp
  src/metrics.cpp
  src/logreg_model.cpp
  src/sgd_trainer.cpp
  src/model_io.cpp
  src/predict.cpp
  src/csv_write.cpp
  src/json_write.cpp
)

add_executable(train_cpp app/train_main.cpp) # trainer binary
target_link_libraries(train_cpp core) # link core

add_executable(predict_cpp app/predict_main.cpp) # predictor binary
target_link_libraries(predict_cpp core) # link core