import os, csv, pytest

TRAIN = "data/raw/train.tsv"
TEST  = "data/raw/test.tsv"
SAMPLE_SUB = "data/raw/sample-submission.csv"

class TestDataFiles:
    def test_raw_dir_exists(self):
        assert os.path.isdir("data/raw")
    def test_train_exists(self):
        assert os.path.isfile(TRAIN), "data/raw/train.tsv missing"
    def test_test_exists(self):
        assert os.path.isfile(TEST), "data/raw/test.tsv missing"
    def test_sample_submission_exists(self):
        assert os.path.isfile(SAMPLE_SUB)
    def test_train_not_empty(self):
        assert os.path.getsize(TRAIN) > 0
    def test_test_not_empty(self):
        assert os.path.getsize(TEST) > 0

class TestTrainFormat:
    def _rows(self):
        with open(TRAIN, newline="", encoding="utf-8") as f:
            return list(csv.reader(f, delimiter="\t"))
    def test_has_rows(self):
        assert len(self._rows()) >= 2
    def test_two_columns(self):
        for i, row in enumerate(self._rows()[1:], 2):
            assert len(row) >= 2, f"row {i} has only {len(row)} col(s)"
    def test_no_blank_rows(self):
        blank = [i+1 for i,r in enumerate(self._rows()) if all(c.strip()==""for c in r)]
        assert blank == [], f"Blank rows: {blank}"
    def test_labels_are_binary(self):
        rows = self._rows()
        header = [c.lower() for c in rows[0]]
        # find which column is the numeric label (0 or 1)
        label_col = None
        for i, h in enumerate(header):
            if h in ("label","sentiment","target","class","y","polarity"):
                label_col = i
                break
        if label_col is None:
            # try to find a column that only has 0/1 values
            for i in range(len(rows[0])):
                vals = {r[i].strip() for r in rows[1:] if len(r) > i}
                if vals.issubset({"0","1"}):
                    label_col = i
                    break
        if label_col is None:
            pytest.skip("Could not identify a binary label column — check TSV format")
        labels = {rows[i][label_col].strip() for i in range(1, len(rows)) if len(rows[i]) > label_col}
        assert labels.issubset({"0","1"}), f"Label column {label_col} has non-binary values: {list(labels)[:5]}"
    def test_text_not_empty(self):
        rows = self._rows()
        start = 1 if rows[0][0].lower() in ("text","review","sentence","comment") else 0
        empty = [i+start+1 for i,r in enumerate(rows[start:]) if r and r[0].strip()==""]
        assert empty == [], f"Empty text at rows: {empty[:5]}"

class TestTestFormat:
    def _rows(self):
        with open(TEST, newline="", encoding="utf-8") as f:
            return list(csv.reader(f, delimiter="\t"))
    def test_has_rows(self):
        assert len(self._rows()) >= 2
    def test_has_columns(self):
        for i,row in enumerate(self._rows()[1:], 2):
            assert len(row) >= 1, f"row {i} empty"
