#pragma once
#include "logreg_model.h" // model
#include "dataset.h" // dataset

float accuracy_dataset(const LogRegModel& m, const Dataset& ds, float thr); // accuracy for labeled dataset