from reviewscope_ml.runtime.gpu import GpuClaim, GpuStatus, pick_freest_gpu, pick_idle_gpus


def gpu(index, used_mb, util=0):
    return GpuStatus(
        index=index, name="TITAN X", memory_total_mb=12_288,
        memory_used_mb=used_mb, utilization_pct=util,
    )


class TestPickIdleGpus:
    # The usual box state: other groups on 0 and 1, 2 and 3 idle.
    BOX = [gpu(0, 5_000), gpu(1, 3_200), gpu(2, 143), gpu(3, 200)]

    def test_busy_devices_never_claimed(self):
        assert set(pick_idle_gpus(self.BOX, max_gpus=None)) == {2, 3}

    def test_all_idle_claims_everything(self):
        idle_box = [gpu(i, 150) for i in range(4)]
        assert len(pick_idle_gpus(idle_box, max_gpus=None)) == 4

    def test_max_gpus_caps_the_claim(self):
        idle_box = [gpu(i, 150) for i in range(4)]
        assert len(pick_idle_gpus(idle_box, max_gpus=2)) == 2

    def test_emptiest_first(self):
        assert pick_idle_gpus(self.BOX, max_gpus=None)[0] == 2  # 143 < 200 used

    def test_utilization_counts_as_busy(self):
        box = [gpu(0, 150, util=80), gpu(1, 150)]
        assert pick_idle_gpus(box, max_gpus=None) == [1]

    def test_full_box_returns_empty(self):
        box = [gpu(0, 11_000), gpu(1, 9_000)]
        assert pick_idle_gpus(box, max_gpus=None) == []

    def test_freest_single_matches(self):
        assert pick_freest_gpu(self.BOX) == 2


class TestGpuClaim:
    def test_multi_claim_properties(self):
        c = GpuClaim([2, 3], "test")
        assert c.device == "cuda"
        assert c.gpu_id == 2  # first device, single-GPU compat

    def test_empty_claim_is_cpu(self):
        c = GpuClaim([], "fallback")
        assert c.device == "cpu"
        assert c.gpu_id is None
