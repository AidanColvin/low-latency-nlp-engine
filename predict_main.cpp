// load model, featurize test, write submission.csv
#include "tsv_reader.h" // read test
#include "dataset.h" // build test dataset
#include "model_io.h" // load model
#include "logreg_model.h" // predict
#include "csv_write.h" // write submission
#include <iostream> // cout
#include <stdexcept> // runtime_error
#include <vector> // vector

int main(int argc, char** argv) {
  try {
    if (argc < 4) throw std::runtime_error("usage: predict_cpp <test_tsv> <model_bin> <out_csv>"); // args
    std::string test_path = argv[1]; // test.tsv
    std::string model_path = argv[2]; // model.bin
    std::string out_csv = argv[3]; // submission.csv

    int dim = 1 << 20; // must match training dim
    int n_min = 1; int n_max = 2; // must match training ngram range
    float thr = 0.5f; // must match threshold

    auto model = load_model_bin(model_path); // load trained model
    auto rows = read_test_tsv(test_path); // load test rows

    Dataset test_ds = build_test_dataset(rows, dim, n_min, n_max); // hashed features

    std::vector<int> labels; labels.reserve(test_ds.x.size()); // output labels
    for (int i = 0; i < (int)test_ds.x.size(); i++) {
      float p = predict_proba(model, test_ds.x[i].idx, test_ds.x[i].val); // proba
      labels.push_back(predict_label(p, thr)); // label
    }

    write_submission_csv(out_csv, test_ds.id, labels); // id,label csv
    std::cout << "wrote " << labels.size() << " predictions to " << out_csv << "\n"; // log
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n"; // error
    return 1;
  }
}