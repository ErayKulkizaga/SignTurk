import json
import hashlib
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
        model_path = asset_dir / config["model_file"]
        manifest = json.loads((REPO_ROOT / "model-assets.json").read_text(encoding="utf-8"))
        manifest_entry = next(
            asset for asset in manifest["assets"] if asset["path"] == model_path.relative_to(REPO_ROOT).as_posix()
        )
        if model_path.is_file():
            digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
            self.assertEqual(model_path.stat().st_size, manifest_entry["bytes"])
            self.assertEqual(digest, manifest_entry["sha256"])

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

    def test_model_manifest_is_complete_and_immutable(self) -> None:
        manifest = json.loads((REPO_ROOT / "model-assets.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(len(manifest["assets"]), 8)
        paths = [asset["path"] for asset in manifest["assets"]]
        self.assertEqual(len(paths), len(set(paths)))
        for asset in manifest["assets"]:
            self.assertTrue(asset["url"].startswith("https://"))
            self.assertGreater(asset["bytes"], 1_000_000)
            self.assertEqual(len(asset["sha256"]), 64)

    def test_research_ensemble_contract(self) -> None:
        root = REPO_ROOT / "research" / "model_226"
        config = json.loads((root / "config" / "backend_config.json").read_text(encoding="utf-8"))
        labels = json.loads((root / "config" / "labels" / "label_map.json").read_text(encoding="utf-8"))
        metrics = json.loads((root / "evaluation" / "four_stream_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(config["project"], "SignTurk")
        self.assertEqual(config["num_classes"], 226)
        self.assertEqual(len(labels), 226)
        self.assertEqual(set(config["streams"]), set(config["weights"]))
        self.assertAlmostEqual(sum(config["weights"].values()), 1.0, places=5)
        self.assertEqual(metrics["samples"], 3742)
        self.assertEqual(metrics["correct_top_1"], 3524)
        self.assertAlmostEqual(metrics["metrics"]["top_1_accuracy"], 3524 / 3742)

    def test_archived_notebooks_have_no_execution_output(self) -> None:
        notebook_dir = REPO_ROOT / "research" / "model_226" / "notebooks"
        for path in notebook_dir.glob("*.ipynb"):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for cell in notebook["cells"]:
                self.assertIsNone(cell.get("execution_count"))
                self.assertFalse(cell.get("outputs", []))


if __name__ == "__main__":
    unittest.main()
