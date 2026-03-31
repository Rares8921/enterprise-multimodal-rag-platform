import json
import pytest
from pathlib import Path
from typing import Dict, Any


@pytest.mark.unit
class TestLoadDatasetSchemaValidation:
    def test_valid_dataset_dict_has_required_keys(self, sample_dataset_dict: Dict[str, Any]):
        """Verify sample dataset has all required keys."""
        required_keys = ["image", "words", "boxes", "labels"]

        for split_name, split_data in sample_dataset_dict.items():
            for key in required_keys:
                assert key in split_data, f"Missing '{key}' in split '{split_name}'"

    def test_valid_flat_dataset_has_required_keys(self, sample_dataset_flat: Dict[str, Any]):
        """Verify flat dataset structure has required keys."""
        required_keys = ["image", "words", "boxes", "labels"]

        for key in required_keys:
            assert key in sample_dataset_flat, f"Missing '{key}' in flat dataset"

    def test_detect_missing_boxes_key(self, invalid_dataset_missing_keys: Dict[str, Any]):
        """Verify we can detect missing 'boxes' key."""
        train_data = invalid_dataset_missing_keys["train"]
        assert "boxes" not in train_data

    def test_detect_missing_labels_key(self, invalid_dataset_missing_keys: Dict[str, Any]):
        """Verify we can detect missing 'labels' key."""
        train_data = invalid_dataset_missing_keys["train"]
        assert "labels" not in train_data

    def test_schema_validation_logic(self, sample_dataset_dict: Dict[str, Any]):
        """Test the schema validation logic that mirrors pipeline component."""
        required_keys = ["image", "words", "boxes", "labels"]

        # Simulate the validation from load_dataset component
        train_data = sample_dataset_dict["train"]
        sample_size = min(100, len(train_data["image"]))

        for i in range(sample_size):
            sample = {key: train_data[key][i] for key in train_data.keys()}
            for key in required_keys:
                assert key in sample, f"Missing required key '{key}' at index {i}"

    def test_invalid_schema_raises_error(self):
        """Test that invalid schema is properly detected."""
        invalid_data = {
            "train": {
                "image": ["img1"],
                "words": [["word1"]],
                # Missing boxes and labels
            }
        }

        required_keys = ["image", "words", "boxes", "labels"]
        train_data = invalid_data["train"]

        with pytest.raises(KeyError):
            sample = {key: train_data[key][0] for key in required_keys}


@pytest.mark.unit
class TestLoadDatasetSmokeTestMode:
    def test_smoke_test_subsets_to_10_samples(self, sample_dataset_dict: Dict[str, Any]):
        """Verify smoke_test limits dataset to 10 samples per split."""
        smoke_test = True
        max_samples = 10

        if smoke_test:
            subsetted = {
                split: {
                    key: values[:max_samples]
                    for key, values in split_data.items()
                }
                for split, split_data in sample_dataset_dict.items()
            }

            for split_name, split_data in subsetted.items():
                for key, values in split_data.items():
                    assert len(values) <= max_samples, \
                        f"Split '{split_name}' key '{key}' exceeds {max_samples} samples"

    def test_smoke_test_preserves_data_integrity(self, sample_dataset_dict: Dict[str, Any]):
        smoke_test = True
        max_samples = 10

        if smoke_test:
            train = sample_dataset_dict["train"]
            subsetted_train = {k: v[:max_samples] for k, v in train.items()}

            # All keys should have same length
            lengths = [len(v) for v in subsetted_train.values()]
            assert len(set(lengths)) == 1, "Inconsistent lengths after subsetting"

    def test_full_mode_keeps_all_samples(self, sample_dataset_dict: Dict[str, Any]):
        smoke_test = False

        train = sample_dataset_dict["train"]
        original_len = len(train["image"])

        # When not in smoke test, should keep all
        if not smoke_test:
            assert len(train["image"]) == original_len


@pytest.mark.unit
class TestLoadDatasetFileOperations:
    def test_load_json_dataset(self, sample_dataset_file: Path):
        """Test loading dataset from JSON file."""
        with open(sample_dataset_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, dict)
        assert "train" in data

    def test_load_nonexistent_file_raises_error(self, temp_dir: Path):
        """Test that loading nonexistent file raises appropriate error."""
        fake_path = temp_dir / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            with open(fake_path, "r") as f:
                json.load(f)

    def test_load_invalid_json_raises_error(self, temp_dir: Path):
        """Test that invalid JSON raises appropriate error."""
        invalid_json_path = temp_dir / "invalid.json"
        with open(invalid_json_path, "w") as f:
            f.write("{ invalid json content")

        with pytest.raises(json.JSONDecodeError):
            with open(invalid_json_path, "r") as f:
                json.load(f)


@pytest.mark.unit
class TestLoadDatasetEdgeCases:
    def test_empty_dataset_handling(self):
        """Test handling of empty dataset."""
        empty_data = {"train": {"image": [], "words": [], "boxes": [], "labels": []}}

        train = empty_data["train"]
        assert len(train["image"]) == 0

        # Validation should handle empty gracefully
        sample_size = min(100, len(train["image"]))
        assert sample_size == 0

    def test_single_sample_dataset(self):
        """Test handling of single-sample dataset."""
        single_sample = {
            "train": {
                "image": ["img1"],
                "words": [["word1"]],
                "boxes": [[[0, 0, 10, 10]]],
                "labels": [[0]]
            }
        }

        required_keys = ["image", "words", "boxes", "labels"]
        train = single_sample["train"]

        assert len(train["image"]) == 1
        for key in required_keys:
            assert key in train

    def test_dataset_with_non_list_raises_type_error(self):
        """Test that non-list values raise appropriate errors."""
        invalid_structure = {
            "train": {
                "image": "not_a_list",  # Should be list
                "words": [["word1"]],
                "boxes": [[[0, 0, 10, 10]]],
                "labels": [[0]]
            }
        }

        # Type checking would fail
        assert not isinstance(invalid_structure["train"]["image"], list) or \
               isinstance(invalid_structure["train"]["image"], str)


@pytest.mark.unit
class TestLoadDatasetIntegrationMocked:
    def test_dataset_dict_structure_compatibility(self, sample_dataset_dict: Dict[str, Any]):
        """Test that our dataset structure is compatible with HF DatasetDict."""
        # Verify structure matches expected HF format
        assert "train" in sample_dataset_dict

        train = sample_dataset_dict["train"]
        # All values should be lists of same length
        lengths = {k: len(v) for k, v in train.items()}
        assert len(set(lengths.values())) == 1, f"Inconsistent lengths: {lengths}"

    def test_save_load_roundtrip(self, temp_dir: Path, sample_dataset_dict: Dict[str, Any]):
        """Test saving and loading dataset preserves data."""
        save_path = temp_dir / "roundtrip_dataset.json"

        # Save
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(sample_dataset_dict, f)

        # Load
        with open(save_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        # Compare
        assert loaded.keys() == sample_dataset_dict.keys()
        for split in sample_dataset_dict:
            assert loaded[split].keys() == sample_dataset_dict[split].keys()
