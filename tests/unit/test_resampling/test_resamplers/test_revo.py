import numpy as np
import pickle
from wepy.runners.mock import MockState
from wepy.resampling.distances.mock import MockDistance
from wepy.resampling.resamplers.revo import REVOResampler, REVOResamplerFactory, _ImageWrapper
from wepy.walker import Walker

def test__ImageWrapper():

    wrapped = _ImageWrapper(MockDistance().image)

    assert wrapped(MockState(1)) == MockState(1)

    # check that it is pickleable for sending to subprocesses
    pickle.loads(pickle.dumps(wrapped))

class Test_REVOResamplerFactory:

    resampler = REVOResamplerFactory(
        distance_metric=MockDistance(),
        merge_dist=1,
        char_dist=1,
    )

class Test_REVOResampler:

    def test___init__(self):

        resampler = REVOResampler(
            merge_dist=1.0,
            char_dist=1.0,
            dist_exponent=3,
            distance=MockDistance(),
            weights=True,
            merge_alg="pairs",
            pmin=0.1,
            pmax=0.4,
            seed=1,
            num_proc=1,
        )

        assert resampler.seed == 1
        assert np.isclose(resampler.lpmin, np.log(0.1 / 100))

        assert REVOResampler(
            merge_dist=1.0,
            char_dist=1.0,
            dist_exponent=3,
            distance=MockDistance(),
            weights=True,
            merge_alg="pairs",
            pmin=0.1,
            pmax=0.4,
            seed=None,
            num_proc=1,
        ).seed is None


    def test__novelty(self):

        resampler = REVOResampler(
            merge_dist=1.0,
            char_dist=1.0,
            dist_exponent=3,
            distance=MockDistance(),
            weights=False,
            merge_alg="pairs",
            pmin=0.1,
            pmax=0.4,
            seed=1,
            num_proc=1,
        )

        # UGLY,TOREV: there shouldn't be the possibility of negatives
        # of these values but current code accepts them.
        assert resampler._novelty(-1, 1) == 0.
        assert resampler._novelty(0.1, -1) == 0.
        assert resampler._novelty(0., 0) == 0.
        assert resampler._novelty(0.1, 0) == 0.
        assert resampler._novelty(0., 1) == 0.

        assert resampler._novelty(0.1, 1) == 1.

        # with weights
        resampler = REVOResampler(
            merge_dist=1.0,
            char_dist=1.0,
            dist_exponent=3,
            distance=MockDistance(),
            weights=True,
            merge_alg="pairs",
            pmin=0.1,
            pmax=0.4,
            seed=1,
            num_proc=1,
        )

        assert resampler._novelty(.4, 1) > 0.
        assert resampler._novelty(.1, 1) > 0.

        assert np.isclose(resampler._novelty(.4, 1000), 0.)

    def test__calc_variation(self):

        resampler = REVOResampler(
            merge_dist=1.0,
            char_dist=1.0,
            dist_exponent=3,
            distance=MockDistance(),
            weights=False,
            merge_alg="pairs",
            pmin=0.1,
            pmax=0.4,
            seed=1,
            num_proc=1,
        )

        variation, walker_variations = resampler._calc_variation(
            [0.4, 0.1],
            [1, 1],
            [
                [0., 1.,],
                [1., 0.,]
            ],
        )

    def test__calc_variation_loss(self):

        resampler = REVOResampler(
            merge_dist=1.0,
            char_dist=1.0,
            dist_exponent=3,
            distance=MockDistance(),
            weights=False,
            merge_alg="pairs",
            pmin=0.1,
            pmax=0.4,
            seed=1,
            num_proc=1,
        )

        # only one option
        assert resampler._calc_variation_loss(
            [0.1, 0.4, 0.3],
            [0.01, 0.02, 0.03],
            [
                (0, 1),
            ],
        ) == (0, 1)

        resampler._calc_variation_loss(
            [0.1, 0.4, 0.3],
            [0.01, 0.02, 0.03],
            [
                (0, 1),
                (1, 2),
            ],
        )

        # if no suitable pairs are found returns None
        assert resampler._calc_variation_loss(
            [0.1, 0.4, 0.3],
            [0.01, 0.02, 0.03],
            [],
        ) is None

    def test__find_eligible_merge_pairs(self):

        resampler = REVOResampler(
            merge_dist=2.0,
            char_dist=1.0,
            dist_exponent=3,
            distance=MockDistance(),
            weights=False,
            merge_alg="pairs",
            pmin=0.1,
            pmax=0.9,
            seed=1,
            num_proc=1,
        )

        # TODO: figure out some combinations of outputs that generates
        # some eligible pairs
        assert resampler._find_eligible_merge_pairs(
            [0.1, 0.4, 0.3],
            [
                [0., 1., 2.],
                [1., 0., 1.5],
                [2., 1.5, 0.],
            ],
            2,
            [2, 2, 2],
        ) == []

    def test_decide(self):

        resampler = REVOResampler(
            merge_dist=2.0,
            char_dist=1.0,
            dist_exponent=3,
            distance=MockDistance(),
            weights=False,
            merge_alg="pairs",
            pmin=0.1,
            pmax=0.9,
            seed=1,
            num_proc=1,
        )

        recs, variation = resampler.decide(
            [0.1, 0.4, 0.3],
            [1, 1, 1],
            [
                [0., 1., 2.],
                [1., 0., 1.5],
                [2., 1.5, 0.],
            ],
        )

    def test__all_to_all_distance(self):

        resampler = REVOResampler(
            merge_dist=2.0,
            char_dist=1.0,
            dist_exponent=3,
            distance=MockDistance(),
            weights=False,
            merge_alg="pairs",
            pmin=0.1,
            pmax=0.9,
            seed=1,
            num_proc=1,
        )

        assert resampler._all_to_all_distance(
            [
                Walker(
                    MockState(0),
                    0.1,
                ),
                Walker(
                    MockState(2),
                    0.1,
                ),
            ]
        ) == (
            [
                [0., 2.],
                [2., 0.],
            ],
            [
                MockState(0),
                MockState(2)
            ]
        )

        # test with pool
        resampler = REVOResampler(
            merge_dist=2.0,
            char_dist=1.0,
            dist_exponent=3,
            distance=MockDistance(),
            weights=False,
            merge_alg="pairs",
            pmin=0.1,
            pmax=0.9,
            seed=1,
            num_proc=2,
        )

        assert resampler._all_to_all_distance(
            [
                Walker(
                    MockState(0),
                    0.1,
                ),
                Walker(
                    MockState(2),
                    0.1,
                ),
            ]
        ) == (
            [
                [0., 2.],
                [2., 0.],
            ],
            [
                MockState(0),
                MockState(2)
            ]
        )

    def test_resample(self):

        resampler = REVOResampler(
            merge_dist=1.0,
            char_dist=1.0,
            dist_exponent=3,
            distance=MockDistance(),
            weights=True,
            merge_alg="pairs",
            pmin=0.1,
            pmax=0.4,
            seed=1,
            num_proc=1,
        )

        resampled_walkers, resampling_data, resampler_data = resampler.resample([
                Walker(
                    MockState(1),
                    0.1,
                ),
                Walker(
                    MockState(2),
                    0.1,
                ),
            ])

        assert len(resampled_walkers) == 2
