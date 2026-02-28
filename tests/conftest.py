import pytest

@pytest.fixture
def sample_texts():
    return [
        "I absolutely love this product, it works great!",
        "Terrible experience, would not recommend to anyone.",
        "It was okay, nothing special but not bad either.",
        "Best purchase I have ever made, highly recommend!",
        "Completely broken on arrival, very disappointed.",
    ]

@pytest.fixture
def sample_labels():
    return [1, 0, 1, 1, 0]

@pytest.fixture
def sample_pairs(sample_texts, sample_labels):
    return list(zip(sample_texts, sample_labels))
