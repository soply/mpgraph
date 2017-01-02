import unittest
import numpy as np

from sklearn import linear_model

from mp_utils import calc_B_y_beta
from problem_factory.synthetic_random_data import \
    create_specific_problem_data_from_problem
from tiling import wrapper_create_tiling

# Run characteristics for comparison. Can be changed!
problem = {
    'identifier': "testing",
    'beta_min': 1e-6,
    'beta_max': 100.0,
    'upper_bound_tilingcreation': 15,
    'n_measurements': 350,
    'n_features': 1250,
    'sparsity_level': 8,
    'smallest_signal': 1.5,
    'largest_signal': 2.0,
    'noise_type_signal': 'linf_bounded',
    'noise_lev_signal': 0.3,
    'noise_type_measurements': 'gaussian',
    'noise_lev_measurements': 0.0,
    'random_seed' : 1,
    'betas_to_test' : np.linspace(2 * 1e-6, 0.95 * 100.0, 10)
}

class CompareToScipyLARSTestCase(unittest.TestCase):
    """ Test class implementing a test that compares the tiling results with the
    scipy implementation of the lars algorithm. Concretely, we use the problem
    setup defined in the beginning of this file, and we create the support
    tiling for this problem.
    Afterwards, we run the scipy-lars algorithm for each single beta in the list
    problem['betas_to_test'], and we compare for each of these beta's the
    support-paths and the related sign-patterns. If we spot a difference in the
    results, the tests fail. Note that we do not compare the alpha's that
    are related to these different supports and sign patterns.

    The test characteristics (ie. the used example) can be altered by varying
    the problem characteristics above. Note that we run both algorithm for
    problem['upper_bound_tilingcreation'] iterations. """

    def setUp(self):
        """ Set up for the tests by calculating the tiling and the lars-path
        for some distinct beta's given in problem['betas_to_test']. """
        tiling_options = {
            "verbose" : 0,
            "mode" : "LARS",
            "print_summary" : True
        }
        self.problem = dict(problem.items() + \
                            {"tiling_options" : tiling_options}.items())
        # Create problem data
        np.random.seed(problem["random_seed"])
        random_state = np.random.get_state()
        self.problem["random_state"] = random_state
        # Creating problem data
        A, y, u_real, v_real = create_specific_problem_data_from_problem(
            self.problem)
        # Run tiling creation
        self.tiling = wrapper_create_tiling(A, y, self.problem["beta_min"],
                                        self.problem["beta_max"],
                                        self.problem["upper_bound_tilingcreation"],
                                        options=self.problem["tiling_options"])
        self.scipy_supports = []
        self.scipy_sign_patterns = []
        for i, beta in enumerate(self.problem["betas_to_test"]):
            B_beta, y_beta = calc_B_y_beta(A, y, self.tiling.svdU,
                                           self.tiling.svdS, beta)
            # Scipy implementation scales the the data by 1/n_samples -> we need
            # to compensate for that
            B_beta = np.sqrt(y_beta.shape[0]) * B_beta
            y_beta = np.sqrt(y_beta.shape[0]) * y_beta
            alphas, active, coefs = linear_model.lars_path(B_beta, y_beta,
                                    method = 'lar',
                                    max_iter = self.problem["upper_bound_tilingcreation"])
            self.scipy_sign_patterns.append(np.sign(coefs))
            self.scipy_supports.append(coefs.astype("bool"))

    def test_support_equality(self):
        """ Tests the support and sign pattern equality of the scipy
        implementation and our tiling implementation.

        Passes if all calculated supports and sign patterns are equal."""
        for i, beta in enumerate(self.problem["betas_to_test"]):
            tiling_supports, tiling_sign_patterns = \
                self.tiling.find_supportpath_to_beta(beta)
            for j in range(self.problem["upper_bound_tilingcreation"]):
                scipy_sup = np.where(self.scipy_supports[i][:,j])[0]
                self.assertTrue(np.array_equal(scipy_sup,
                    tiling_supports[j]),
                    'Wrong support {0}, {1}, {2}'.format(
                    tiling_supports[j], scipy_sup, beta))
                idx = self.scipy_sign_patterns[i][:,j] != 0
                scipy_sign_pattern = self.scipy_sign_patterns[i][idx, j]
                self.assertTrue(np.array_equal(scipy_sign_pattern,
                    tiling_sign_patterns[j]),
                    'Wrong sign pattern {0}, {1}, {2}'.format(
                    scipy_sign_pattern, tiling_sign_patterns[j], beta))
