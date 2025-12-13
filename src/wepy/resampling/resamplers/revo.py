# Standard Library
import itertools as it
import logging
import multiprocessing as mp
import random as rand
import time
from typing import Callable, Generic, Literal, TypedDict, TypeVar, Annotated

# Third Party Library
import attrs
import numpy as np
from numpy.typing import NDArray

# First Party Library
from wepy.typing import Shape
from wepy.resampling.decisions.clone_merge import CloneMergeDecisionRecord
from wepy.resampling.distances.base import Distance
from wepy.resampling.resamplers.clone_merge import CloneMergeResampler, CloneMergeResamplingRecord
from wepy.util.multiprocessing import proc_pool_worker_setup, queue_listener_context
from wepy.walker import Walker, WalkerState
from wepy.util.attrs import AttrsMappingMixin

logger = logging.getLogger(__name__)

DistanceMetric_ = TypeVar("DistanceMetric_", bound=Distance)
DistanceImage_ = TypeVar("DistanceImage_")
WalkerState_ = TypeVar("WalkerState_", bound=WalkerState)

MergeAlgorithm = Literal["pairs", "greedy"]


class REVOResamplerError(Exception):
    pass


class _ImageWrapper(Generic[WalkerState_, DistanceImage_]):
    """Wrapper callable to inject a few log messages to image
    computation.

    Useful for if the image function does not log anything and this
    will guarantee some logs are generated which is useful for
    troubleshooting process pool problem.

    """

    def __init__(self, image_func: Callable[[WalkerState_], DistanceImage_]) -> None:
        self.image_func = image_func

    def __call__(self, state: WalkerState_) -> DistanceImage_:
        logger.info("Starting image computation")
        result = self.image_func(state)
        logger.info("Finished image computation")
        return result

@attrs.define
class REVOResamplerResamplerRecord(AttrsMappingMixin):
    distance_matrix: Annotated[
        NDArray[np.float32],
        Shape((Ellipsis, Ellipsis,))
    ]
    variation: Annotated[
        NDArray[np.float32],
        Shape((1,)),
    ]


class REVOResampler(
    CloneMergeResampler,
    Generic[DistanceMetric_, DistanceImage_, WalkerState_],
):
    r"""Resampler implementing the REVO algorithm.

    You can find more detailed information in the paper "REVO:
    Resampling of ensembles by variation optimization" but
    briefly:

    REVO is a Weighted Ensemble based enhanced sampling algorithm
    which uses cloning and merging to create ensembles of diverse
    trajectories without defining any regions. It instead optimizes a
    measure of “variation” that depends on the pairwise distances
    between the walkers and their weights.


    REVO solves this optimization problem using a greedy algorithm
    which at each step selects best walkers for resampling operations
    (cloning and merging) in order to maximize the "trajectory
    variation".

    The trajectory variation is defined as


    .. math::
        V = \sum_{i} V_i = \sum_i \sum_{j}(\frac{d_{ij}}{d_0})
        ^{\alpha}\phi_i\phi_j

    where

    :math:`V_i` : the trajectory variation value of walker `i`

    :math:`d_{ij}` : the distance between walker i and j
    according the distance metric

    :math:`\alpha` : modulates the influence of the distances in the
    variation calculation

    :math:`d_0` : the characteristic distance and is used to make the
    equation unitless.

    :math:`\phi` : is a non-negative function which is a measure of
    the relative importance of the walker and is referred to as a
    "novelty function".  Here it is a function of a walker's weight.

    Furthermore REVO needs the following parameters:

       pmin: the minimum statistical weight. REVO does not clone
       walkers with a weight less than pmin.

       pmax: The maximum statistical weight. It prevents the
       accumulation of too much weight in one walker.

       merge_dist: This is the merge-distance threshold. The distance
        between merged walkers should be less than this value.

    The resample function, called during every cycle, takes the
    ensemble of walkers and performs the follow steps:

       - Calculate the pairwise all-to-all distance matrix using the distance metric
       - Decides which walkers should be merged or cloned
       - Applies the cloning and merging decisions to get the resampled walkers
       - Creates the resampling data that includes
       - distance_matrix : the calculated all-to-all distance matrix

           - n_walkers : the number of walkers. number of walkers is
             kept constant thought the resampling.

           - variation : the final value of trajectory variation

           - images : the images of walkers that is defined by the distance object

           - image_shape : the shape of the image

    The algorithm saves the records of cloning and merging
    information in resampling data.

    Only the net clones and merges are recorded in the resampling records.

    """

    distance_metric: DistanceMetric_
    merge_dist: float
    char_dist: float
    dist_exponent: int
    weights: bool
    merge_alg: MergeAlgorithm
    pmin: float
    pmax: float
    seed: int | None
    num_proc: int
    lpmin: float

    RESAMPLING_FIELDS = CloneMergeResampler.RESAMPLING_FIELDS
    RESAMPLING_SHAPES = CloneMergeResampler.RESAMPLING_SHAPES
    RESAMPLING_DTYPES = CloneMergeResampler.RESAMPLING_DTYPES

    RESAMPLING_RECORD_FIELDS = CloneMergeResampler.RESAMPLING_RECORD_FIELDS

    RESAMPLER_FIELDS = CloneMergeResampler.RESAMPLER_FIELDS + ("distance_matrix", "variation",)

    RESAMPLER_SHAPES = CloneMergeResampler.RESAMPLER_SHAPES + (Ellipsis, (1,))
    RESAMPLER_DTYPES = CloneMergeResampler.RESAMPLER_DTYPES + (float, float,)

    # fields that can be used for a table like representation
    RESAMPLER_RECORD_FIELDS = CloneMergeResampler.RESAMPLER_RECORD_FIELDS + (
        "variation",
    )

    def __init__(
        self,
        merge_dist: float,
        char_dist: float,
        distance: DistanceMetric_,
        weights: bool,
        merge_alg: MergeAlgorithm,
        pmin: float,
        pmax: float,
        dist_exponent: int,
        seed: int | None,
        num_proc: int = 1,
    ) -> None:
        """Constructor for the REVO Resampler.

        Parameters
        ----------
        dist_exponent : int
          The distance exponent that modifies distance and weight novelty
          relative to each other in the variation equation.

        merge_dist : float
            The merge distance threshold. Units should be the same as
            the distance metric.

        char_dist : float
            The characteristic distance value. It is calculated by
            running a single dynamic cycle and then calculating the
            average distance between all walkers. Units should be the
            same as the distance metric.

        distance : object implementing Distance
            The distance metric to compare walkers.

        weights : bool
            Turns off or on the weight novelty in
            calculating the variation equation. When weight is
            False, the value of the novelty function is set to 1 for all
            walkers.

        merge_alg : string
            Indication of which algorithm is used to find pairs to merge.

            'pairs' (default) indicates that a list of all suitable pairs is generated,
            and the pair that minimizes the expected variation loss is chosen.

            'greedy' indicates that first the lowest variation walker is
            selected, and this is attempted to be merged with the closest
            suitable walker

        init_state : WalkerState object
            Used for automatically determining the state image shape.

        seed : None or int, optional
            The random seed. If None, the system (random) one will be used.

        num_proc : int, optional
            The number of processors used to calculate the walker images
            and all-to-all distances.

        """

        # call the init methods in the CloneMergeResampler
        # superclass. We set the min and max number of walkers to be
        # constant
        super().__init__(
            pmin=pmin,
            pmax=pmax,
            min_num_walkers=Ellipsis,
            max_num_walkers=Ellipsis,
        )

        # ln(probability_min)
        self.lpmin = np.log(self.pmin / 100)
        self.dist_exponent = dist_exponent

        self.merge_dist = merge_dist
        self.merge_alg = merge_alg

        # the distance metric
        self.distance = distance

        # the characteristic distance, char_dist
        self.char_dist = char_dist

        # setting the random seed
        self.seed = seed
        if seed is not None:
            rand.seed(seed)

        # setting the weights parameter
        self.weights = weights

        # setting the number of processors
        self.num_proc = num_proc

    def _novelty(self, walker_weight: float, num_walker_copy: int) -> float:
        """Calculates the novelty function value.

        Parameters
        ----------
        walker_weight : float
            The weight of the walker.

        num_walker_copy : int
          The number of copies of the walker.

        Returns
        -------
        novelty : float
        The calcualted value of novelty for the given walker.

        """

        novelty = 0.0
        if walker_weight > 0 and num_walker_copy > 0:
            if self.weights:
                novelty = np.log(walker_weight / num_walker_copy) - self.lpmin
            else:
                novelty = 1.0

        if novelty < 0:
            novelty = 0.0

        return novelty

    def _calc_variation(
        self,
        walker_weights: list[float],
        num_walker_copies: list[int],
        distance_matrix: list[list[float]],
    ) -> tuple[float, list[float]]:
        """Calculates the variation value.

        Parameters
        ----------
        walker_weights : list of float
            The weights of all walkers. The sum of all weights should be 1.0.

        num_walker_copies : list of int
            The number of copies of each walker.
            0 means the walker does not exist anymore.
            1 means there is one of the this walker.
            >1 means it should be cloned to this number of walkers.

        distance_matrix : list of arraylike of shape (num_walkers)

        Returns
        -------
        variation : float
           The calculated variation value.

        walker_variations : arraylike of shape (num_walkers)
           The Vi value of each walker.

        """

        num_walkers = len(walker_weights)

        # set the novelty values
        walker_novelties = np.array(
            [
                self._novelty(walker_weights[i], num_walker_copies[i])
                for i in range(num_walkers)
            ]
        )

        # the value to be optimized
        variation: float = 0.0

        # the walker variation values (Vi values)
        walker_variations = np.zeros(num_walkers)

        # calculate the variation and walker variation values
        for i in range(num_walkers - 1):
            if num_walker_copies[i] > 0:
                for j in range(i + 1, num_walkers):
                    if num_walker_copies[j] > 0:
                        partial_variation = (
                            (
                                (distance_matrix[i][j] / self.char_dist)
                                ** self.dist_exponent
                            )
                            * walker_novelties[i]
                            * walker_novelties[j]
                        )

                        variation += (
                            partial_variation
                            * num_walker_copies[i]
                            * num_walker_copies[j]
                        )
                        walker_variations[i] += partial_variation * num_walker_copies[j]
                        walker_variations[j] += partial_variation * num_walker_copies[i]

        return variation, walker_variations

    def _calc_variation_loss(
        self,
        walker_variation: list[float],
        weights: list[float],
        eligible_pairs: list[tuple[int, int]],
    ) -> tuple[int, int] | None:
        """Calculates the loss to variation through merging of eligible walkers.

        Parameters
        ----------
        walker_variations : arraylike of shape (num_walkers)
           The Vi value of each walker.

        weights : list of float
            The weights of all walkers. The sum of all weights should be 1.0.

        eligible_pairs : list of tuples
            Pairs of walker indexes that meet the criteria for merging.

        Returns
        -------
        variation_loss_list : tuple or None
            A tuple of the walker merge pair indicies that meet the criteria
            for merging and minimize variation loss. If none is found returns None
        """

        v_loss_min = np.inf

        min_loss_pair: tuple[int, int] | None = None

        for pair in eligible_pairs:
            walker_i = pair[0]
            walker_j = pair[1]

            wt_i = weights[walker_i]
            wt_j = weights[walker_j]

            v_i = walker_variation[walker_i]
            v_j = walker_variation[walker_j]

            v_loss = (wt_j * v_i + wt_i * v_j) / (wt_i + wt_j)

            if v_loss < v_loss_min:
                min_loss_pair = pair

                v_loss_min = v_loss

        return min_loss_pair

    def _find_eligible_merge_pairs(
        self,
        weights: list[float],
        distance_matrix: list[list[float]],
        max_var_idx: int,
        num_walker_copies: list[int],
    ) -> list[tuple[int, int]]:
        """Find pairs of walkers that are eligible to be merged.

        Parameters
        ----------
        weights : list of float
            The weights of all walkers. The sum of all weights should be 1.0.

        distance_matrix : list of arraylike of shape (num_walkers)
            The distance between every walker according to the distance metric.

        max_var_idx : int
            The index of the walker that had the highest walker variance
            and is a candidate for cloning.

        num_walker_copies : list of int
             0 means the walker does not exist anymore.
             1 means there is one of the this walker.
             >1 means it should be cloned to this number of walkers.

        Returns
        -------
        eligible_pairs : list of tuples
            Pairs of walker indexes that meet the criteria for merging.

        """

        eligible_pairs = []

        for i in range(len(weights) - 1):
            for j in range(i + 1, len(weights)):
                if i != max_var_idx and j != max_var_idx:
                    if num_walker_copies[i] == 1 and num_walker_copies[j] == 1:
                        if weights[i] + weights[j] < self.pmax:
                            if distance_matrix[i][j] < self.merge_dist:
                                eligible_pairs.append((i, j))

        return eligible_pairs

    def decide(
        self,
        walker_weights: list[float],
        num_walker_copies: list[int],
        distance_matrix: list[list[float]],
    ) -> tuple[
        list[CloneMergeDecisionRecord],
        float,
    ]:
        """Optimize the trajectory variation by making decisions for resampling.

        Parameters
        ----------
        walker_weights : list of float
            The weights of all walkers. The sum of all weights should be 1.0.

        num_walker_copies : list of int
            The number of copies of each walker.
            0 means the walker is not exists anymore.
            1 means there is one of the this walker.
            >1 means it should be cloned to this number of walkers.

        distance_matrix : list of arraylike of shape (num_walkers)

        Returns
        -------
        resampling_data : list of dict of str: value
            The resampling records resulting from the decisions.
        variation : float
            The optimized value of the trajectory variation.

        """
        num_walkers = len(walker_weights)

        variations = []
        merge_groups = [[] for i in range(num_walkers)]
        walker_clone_nums = [0 for i in range(num_walkers)]

        # make copy of walkers properties
        new_walker_weights = walker_weights.copy()
        new_num_walker_copies = num_walker_copies.copy()

        # calculate the initial variation which will be optimized
        variation, walker_variations = self._calc_variation(
            walker_weights, new_num_walker_copies, distance_matrix
        )
        variations.append(variation)

        # maximize the variance through cloning and merging
        logger.info(f"Starting variance optimization: {variation}")

        _count = 1
        productive = True
        while productive:
            _log = False
            if _count == 1 or _count % 10 == 0:
                _log = True
                logger.info(f"Optimization iteration: {_count}")
            _count += 1
            productive = False
            # find min and max walker_variationss, alter new_amp

            # initialize to None, we may not find one of each
            min_idx = None
            max_idx = None

            # selects a walker with minimum walker_variations and a walker with
            # maximum walker_variations walker (distance to other walkers) will be
            # tagged for cloning (stored in maxwind), except if it is
            # already a keep merge target
            max_tups = []
            for i, value in enumerate(walker_variations):
                # 1. must have an amp >=1 which gives the number of clones to be made of it
                # 2. clones for the given amplitude must not be smaller than the minimum probability
                # 3. must not already be a keep merge target
                if (
                    (new_num_walker_copies[i] >= 1)
                    and (
                        new_walker_weights[i] / (new_num_walker_copies[i] + 1)
                        > self.pmin
                    )
                    and (len(merge_groups[i]) == 0)
                ):
                    max_tups.append((value, i))

            if len(max_tups) > 0:
                max_value, max_idx = max(max_tups)

            maybe_merge_pair: tuple[int, int] | None = None
            if self.merge_alg == "pairs":
                pot_merge_pairs = self._find_eligible_merge_pairs(
                    new_walker_weights, distance_matrix, max_idx, new_num_walker_copies
                )
                maybe_merge_pair = self._calc_variation_loss(
                    walker_variations, new_walker_weights, pot_merge_pairs
                )
            elif self.merge_alg == "greedy":
                # walker with the lowest walker_variations (distance to other walkers)
                # will be tagged for merging (stored in min_idx)
                min_tups = [
                    (value, i)
                    for i, value in enumerate(walker_variations)
                    if new_num_walker_copies[i] == 1
                    and (new_walker_weights[i] < self.pmax)
                ]

                if len(min_tups) > 0:
                    min_value, min_idx = min(min_tups)

                # does min_idx have an eligible merging partner?
                closewalk = None
                condition_list = np.array([i is not None for i in [min_idx, max_idx]])
                if condition_list.all() and min_idx != max_idx:
                    # get the walkers that aren't the minimum and the max
                    # walker_variations walkers, as candidates for merging
                    closewalks = set(range(num_walkers)).difference([min_idx, max_idx])

                    # remove those walkers that if they were merged with
                    # the min walker_variations walker would violate the pmax
                    closewalks = [
                        idx
                        for idx in closewalks
                        if (new_num_walker_copies[idx] == 1)
                        and (
                            new_walker_weights[idx] + new_walker_weights[min_idx]
                            < self.pmax
                        )
                    ]

                    # if there are any walkers left, get the distances of
                    # the close walkers to the min walker_variations walker if that
                    # distance is less than the maximum merge distance
                    if len(closewalks) > 0:
                        closewalks_dists = [
                            (distance_matrix[min_idx][i], i)
                            for i in closewalks
                            if distance_matrix[min_idx][i] < (self.merge_dist)
                        ]

                    # if any were found set this as the closewalk
                    if len(closewalks_dists) > 0:
                        closedist, closewalk = min(closewalks_dists)
                        maybe_merge_pair = (min_idx, closewalk)

            else:
                raise ValueError(f"Unrecognized value for merge_alg: {self.merge_alg}")

            # did we find a suitable pair to merge?
            if maybe_merge_pair is not None:
                min_idx = maybe_merge_pair[0]
                closewalk = maybe_merge_pair[1]

                # change new_amp
                tempsum = new_walker_weights[min_idx] + new_walker_weights[closewalk]
                new_num_walker_copies[min_idx] = new_walker_weights[min_idx] / tempsum
                new_num_walker_copies[closewalk] = (
                    new_walker_weights[closewalk] / tempsum
                )
                new_num_walker_copies[max_idx] += 1

                # re-determine variation function, and walker_variations values
                new_variation, walker_variations = self._calc_variation(
                    new_walker_weights, new_num_walker_copies, distance_matrix
                )

                if new_variation > variation:
                    variations.append(new_variation)

                    if _log:
                        logger.info(f"Variance move to {new_variation} accepted")

                    productive = True
                    variation = new_variation

                    # make a decision on which walker to keep
                    # (min_idx, or closewalk), equivalent to:
                    # `random.choices([closewalk, min_idx],
                    #                 weights=[new_walker_weights[closewalk], new_walker_weights[min_idx])`
                    r = rand.uniform(
                        0.0, new_walker_weights[closewalk] + new_walker_weights[min_idx]
                    )

                    # keeps closewalk and gets rid of min_idx
                    if r < new_walker_weights[closewalk]:
                        keep_idx = closewalk
                        squash_idx = min_idx

                    # keep min_idx, get rid of closewalk
                    else:
                        keep_idx = min_idx
                        squash_idx = closewalk

                    # update weight
                    new_walker_weights[keep_idx] += new_walker_weights[squash_idx]
                    new_walker_weights[squash_idx] = 0.0

                    # update new_num_walker_copies
                    new_num_walker_copies[squash_idx] = 0
                    new_num_walker_copies[keep_idx] = 1

                    # add the squash index to the merge group
                    merge_groups[keep_idx].append(squash_idx)

                    # add the indices of the walkers that were already
                    # in the merge group that was just squashed
                    merge_groups[keep_idx].extend(merge_groups[squash_idx])

                    # reset the merge group that was just squashed to empty
                    merge_groups[squash_idx] = []

                    # increase the number of clones that the cloned
                    # walker has
                    walker_clone_nums[max_idx] += 1

                    # new variation for starting new stage
                    new_variation, walker_variations = self._calc_variation(
                        new_walker_weights, new_num_walker_copies, distance_matrix
                    )
                    variations.append(new_variation)

                    if _log:
                        logger.info("variance after selection: {}".format(new_variation))

                # if not productive
                else:
                    new_num_walker_copies[min_idx] = 1
                    new_num_walker_copies[closewalk] = 1
                    new_num_walker_copies[max_idx] -= 1

        final_variation = variations[-1]
        logger.info(f"Finished optimization: {final_variation}")

        logger.info("Assigning clones")
        decision_records = self.assign_clones(merge_groups, walker_clone_nums)


        return decision_records, final_variation

    def _all_to_all_distance(
        self,
        walkers: list[Walker[WalkerState_]],
    ) -> tuple[
        list[list[float]],
        list[DistanceImage_],
    ]:
        """Calculate the pairwise all-to-all distances between walkers.

        Parameters
        ----------
        walkers : list of walkers


        Returns
        -------
        distance_matrix : list of arraylike of shape (num_walkers)

        images : list of image obeject

        """
        # initialize an all-to-all matrix, with 0.0 for self distances
        dist_mat = [[0.0 for _ in range(len(walkers))] for _ in range(len(walkers))]

        logger.info("Starting calculation of walker images")
        start_time = time.time()

        # make images for all the walker states for us to compute distances on
        if self.num_proc > 1:
            logger.info(
                f"Multiple processes requested ({self.num_proc}) will run in Pool."
            )

            _distance_image = _ImageWrapper(self.distance.image)

            # NOTE: Must use spawn here, otherwise there are problems
            # with deadlocking in the sub-processes
            mp_ctx = mp.get_context(method="spawn")

            # TODO: This should be part of some setup phase
            logger.info("Starting multiprocessing.Pool")
            with (
                queue_listener_context(mp_ctx) as log_queue,
                mp_ctx.Pool(
                    self.num_proc,
                    initializer=proc_pool_worker_setup,
                    initargs=(log_queue,),
                    # Set some upper bound so that it gets cleaned up
                    # in case of leaks
                    maxtasksperchild=4,
                ) as pool,
            ):

                logger.info(
                    f"Running parallel map calculation on {len(walkers)} walkers"
                )
                images = pool.map(
                    _distance_image,
                    [walker.state for walker in walkers],
                )
                logger.info("Finished running parallel map calculation")
                logger.info("Shutting down Pool")

            logger.info("Pool shutdown complete")

        else:
            logger.info("Calculating images without parallelism")
            images = []
            for walker in walkers:
                image = self.distance.image(walker.state)
                images.append(image)

        end_time = time.time()

        _image_time = end_time - start_time

        logger.info(f"Calculating walker state images took: {_image_time} s")

        logger.info("Calculating image distances")
        start_time = time.time()

        # get the combinations of indices for all walker pairs
        for i, j in it.combinations(range(len(images)), 2):
            # calculate the distance between the two walkers
            dist = self.distance.image_distance(images[i], images[j])

            # save this in the matrix in both spots
            dist_mat[i][j] = dist
            dist_mat[j][i] = dist

        end_time = time.time()

        _dist_time = end_time - start_time

        logger.info(f"Calculating image distances took: {_dist_time} s")

        return dist_mat, images

    def resample(
        self,
        walkers: list[Walker[WalkerState_]],
    ) -> tuple[
        list[Walker[WalkerState_]],
        list[CloneMergeResamplingRecord],
        list[REVOResamplerResamplerRecord],
    ]:
        """Resamples walkers based on REVO algorithm

        Parameters
        ----------
        walkers : list of walkers


        Returns
        -------
        resampled_walkers : list of resampled_walkers

        resampling_data : list of dict of str: value
            The resampling records resulting from the decisions.

        resampler_data :list of dict of str: value
            The resampler records resulting from the resampler actions.

        """

        # initialize the parameters
        num_walkers = len(walkers)
        walker_weights = [walker.weight for walker in walkers]

        # Needs to be floats to do partial amps during second variation calculations.
        num_walker_copies = np.ones(num_walkers)

        # calculate distance matrix
        logger.info("Calculating walker distances")
        distance_matrix, images = self._all_to_all_distance(walkers)
        logger.info("Finished calculating distances")

        logger.info("Distance_matrix: ")
        logger.info("\n{}".format(str(np.array(distance_matrix))))

        # determine cloning and merging actions to be performed, by
        # maximizing the variation, i.e. the Decider
        logger.info("Making resampling decisions")
        decision_records, variation = self.decide(
            walker_weights, num_walker_copies, distance_matrix
        )
        logger.info("Finished resampling decisions")

        # actually do the cloning and merging of the walkers
        resampled_walkers = self.DECISION.action(walkers, [decision_records])

        ## Generate the full resampling records

        # because there is only one step in resampling here we just
        # add another field for the step as 0 and add the walker index
        # to its record as well
        resampling_records = []
        for walker_idx, decision_record in enumerate(decision_records):
            # UGLY: we need to wrap the field data into the shape
            # declared in the CloneMergeResampler, see other notes on
            # why
            resampling_record = CloneMergeResamplingRecord(
                # The decision record fields are simple, so we wrap
                # them here as well
                decision_id=np.array([[decision_record.decision_id]]),
                target_idxs=np.array([[
                    np.array(decision_record.target_idxs),
                ]]),
                step_idx=np.array([[0]]),
                walker_idx=np.array([[walker_idx]])
            )
            resampling_records.append(resampling_record)

        # flatten the distance matrix and give the number of walkers
        # as well for the resampler data, there is just one per cycle
        resampler_records = [
            REVOResamplerResamplerRecord(
                distance_matrix=np.ravel(np.array(distance_matrix)),
                variation=np.array([[variation]]),
            )
        ]

        return resampled_walkers, resampling_records, resampler_records


@attrs.define
class REVOResamplerFactory(Generic[DistanceMetric_]):

    distance_metric: DistanceMetric_
    merge_dist: float
    char_dist: float
    dist_exponent: int = 4
    weights: bool = True
    merge_alg: MergeAlgorithm = "pairs"
    pmin: float = 1e-12
    pmax: float = 0.1
    seed: int | None = None

    def __call__(
        self,
        num_cores: int,
    ) -> REVOResampler:

        return REVOResampler(
            distance=self.distance_metric,
            merge_dist=self.merge_dist,
            char_dist=self.char_dist,
            dist_exponent=self.dist_exponent,
            weights=self.weights,
            merge_alg=self.merge_alg,
            pmin=self.pmin,
            pmax=self.pmax,
            seed=self.seed,
            num_proc=num_cores,
        )
