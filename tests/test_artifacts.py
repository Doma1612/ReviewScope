import numpy as np

from reviewscope_ml.pipelines.artifacts import load_run, run_is_complete, save_run


class TestArtifactRoundtrip:
    def test_save_load_roundtrip(self, small_run, tmp_path):
        run_dir = tmp_path / small_run.run_name
        save_run(run_dir, small_run)
        assert run_is_complete(run_dir)

        loaded = load_run(run_dir)
        assert loaded.run_name == small_run.run_name
        assert loaded.doc_ids == small_run.doc_ids
        assert np.array_equal(loaded.labels, small_run.labels)
        assert np.allclose(loaded.coords_2d, small_run.coords_2d)
        assert np.allclose(loaded.coords_3d, small_run.coords_3d)
        assert loaded.cluster_ids == [0, 1]
        assert loaded.clusters[0].label == "label 0"
        assert loaded.clusters[0].word_frequencies == {"alpha": 3}
        assert loaded.micro_labels is None

    def test_micro_labels_roundtrip(self, two_stage_run, tmp_path):
        run_dir = tmp_path / "ts"
        save_run(run_dir, two_stage_run)
        loaded = load_run(run_dir)
        assert np.array_equal(loaded.micro_labels, two_stage_run.micro_labels)
        assert loaded.clusters[0].micro_cluster_ids == [10, 11]

    def test_incomplete_dir_detected(self, small_run, tmp_path):
        run_dir = tmp_path / "incomplete"
        save_run(run_dir, small_run)
        (run_dir / "clusters.json").unlink()
        assert not run_is_complete(run_dir)
