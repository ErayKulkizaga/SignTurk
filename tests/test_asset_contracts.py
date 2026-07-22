import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class ModelAssetContractTests(unittest.TestCase):
    def _assert_asset_bundle(self, directory: str, expected_classes: int) -> None:
        asset_dir = REPO_ROOT / directory
        config = json.loads((asset_dir / "demo_config.json").read_text(encoding="utf-8"))
        labels = json.loads((asset_dir / "label_map.json").read_text(encoding="utf-8"))
        normalization = json.loads(
            (asset_dir / "norm_stats.json").read_text(encoding="utf-8")
        )

        self.assertEqual(config["num_classes"], expected_classes)
        self.assertEqual(len(labels), expected_classes)
        self.assertEqual(
            sorted(int(class_id) for class_id in labels),
            list(range(expected_classes)),
        )
        self.assertEqual(len(normalization["mean"]), config["feat_dim"])
        self.assertEqual(len(normalization["std"]), config["feat_dim"])
        self.assertTrue((asset_dir / config["model_file"]).is_file())

        original_ids = [entry["original_class_id"] for entry in labels.values()]
        self.assertEqual(len(original_ids), len(set(original_ids)))
        self.assertTrue(all(entry.get("TR") and entry.get("EN") for entry in labels.values()))

    def test_live_model_bundle(self) -> None:
        self._assert_asset_bundle("demo_assets_179", expected_classes=179)

    def test_legacy_model_bundle(self) -> None:
        self._assert_asset_bundle("model_assets", expected_classes=184)

    def test_avatar_landmark_vocabulary(self) -> None:
        landmark_files = list((REPO_ROOT / "dataset" / "landmarks").glob("*.json"))
        self.assertEqual(len(landmark_files), 226)

    def test_frontend_avatar_path_is_case_safe(self) -> None:
        avatar_path = REPO_ROOT / "frontend" / "avatar.glb"
        avatar_page = (REPO_ROOT / "frontend" / "avatar3d.html").read_text(
            encoding="utf-8"
        )
        self.assertTrue(avatar_path.is_file())
        self.assertIn("/frontend/avatar.glb", avatar_page)


if __name__ == "__main__":
    unittest.main()
