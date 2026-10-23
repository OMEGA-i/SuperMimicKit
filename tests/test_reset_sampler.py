import torch

from supermimic.data import ResetSampler


def test_bins_respect_clip_boundaries_and_mixture():
    sampler = ResetSampler(torch.tensor([42, 143]))
    assert sampler.clips.tolist() == [0, 1, 1, 1]
    ids, times = sampler.sample(2000)
    assert ((times >= 0) & (times < sampler.valid[ids])).all()
    assert torch.equal(sampler.clips[sampler.bin_ids(ids, times)], ids)
    torch.testing.assert_close(sampler.probabilities(), torch.full((4,), 0.25))
    # A high-return bin becomes less likely; repeated visits are averaged.
    sampler.update(torch.tensor([0, 0, 1]), torch.tensor([9., 7., 0.]))
    torch.testing.assert_close(sampler.returns, torch.tensor([.8, 0., 0., 0.]))
    p = sampler.probabilities()
    assert p[0] < p[1] and (p >= 0.5 / 4).all()
    torch.testing.assert_close(p.sum(), torch.tensor(1.))


def test_historical_difficulty_and_resume():
    sampler = ResetSampler(torch.tensor([142]))
    for _ in range(20):
        sampler.update(torch.tensor([0, 1]), torch.tensor([0., 10.]))
    # Once current returns equalize, historical difficulty still matters.
    sampler.returns.fill_(5.)
    assert sampler.probabilities()[0] > sampler.probabilities()[1]
    restored = ResetSampler(torch.tensor([142]))
    restored.load_state_dict(sampler.state_dict())
    torch.manual_seed(9)
    expected = sampler.sample(100)
    torch.manual_seed(9)
    actual = restored.sample(100)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
