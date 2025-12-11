# Third Party Library
import pytest

# First Party Library
from wepy.resampling.resamplers.resampler import ResamplerABC, ResamplerError


class Test_ResamplerABC:

    def test___init__(self):

        ResamplerABC()

        ResamplerABC(
            min_num_walkers=None,
            max_num_walkers=None,
        )

        ResamplerABC(
            min_num_walkers=3,
            max_num_walkers=3,
        )

        with pytest.raises(ResamplerError):
            ResamplerABC(
                min_num_walkers=4,
                max_num_walkers=3,
            )

        with pytest.raises(ResamplerError):
            ResamplerABC(min_num_walkers=0)
