'''Bayesian Subspace models.'''


import math
import torch

from .bayesmodel import BayesianParameter
from .bayesmodel import BayesianParameterSet
from .bayesmodel import BayesianModel
from .bayesmodel import BayesianModelSet
from .ppca import kl_div_std_norm
from .normalset import NormalSetElement
from ..expfamilyprior import GammaPrior
from ..expfamilyprior import NormalIsotropicCovariancePrior
from ..expfamilyprior import MatrixNormalPrior
from ..expfamilyprior import NormalFullCovariancePrior
from ..utils import make_symposdef


########################################################################
# Probabilistic Linear Discriminant Analysis (PLDA)
########################################################################

class PLDASet(BayesianModelSet):
    '''Set of Normal distribution for the Probabilistic Linear
    Discriminant Analysis (PLDA) model. When combined with a
    :any:`Mixture` it is a PLDA.

    Attributes:
        mean (``torch.Tensor``): Expected mean.
        precision (``torch.Tensor[1]``): Expected precision (scalar).
        noise_subspace (``torch.Tensor[noise_subspace_dim, data_dim]``):
            Mean of the matrix defining the "noise" subspace.
            (within-class variations).
        class_subspace (``torch.Tensor[class_subspace_dim, data_dim]``):
            Mean of the matrix defining the "class" subspace.
            (across-class variations).
        class_means: (``torch.Tensor[]``)

    '''

    @staticmethod
    def create(model_conf, mean, variance, create_model_handle):
        dtype, device = mean.dtype, mean.device
        n_classes = model_conf['size']
        dim_noise_subspace = model_conf['dim_noise_subspace']
        dim_class_subspace = model_conf['dim_class_subspace']
        noise_std = model_conf['noise_std']
        prior_strength = model_conf['prior_strength']

        # Precision.
        shape = torch.tensor([prior_strength], dtype=dtype, device=device)
        rate = torch.tensor([prior_strength * variance.sum()], dtype=dtype,
                            device=device)
        prior_prec = GammaPrior(shape, rate)
        posterior_prec = GammaPrior(shape, rate)

        # Global mean.
        mean_variance = torch.tensor([1. / float(prior_strength)], dtype=dtype,
                                    device=device)
        prior_mean = NormalIsotropicCovariancePrior(mean, mean_variance)
        posterior_mean = NormalIsotropicCovariancePrior(mean, mean_variance)

        # Noise subspace.
        noise_subspace = torch.zeros(dim_noise_subspace, len(mean), dtype=dtype,
                                    device=device)
        cov = torch.eye(noise_subspace.size(0), dtype=dtype, device=device)
        cov /= prior_strength
        prior_noise_subspace = MatrixNormalPrior(noise_subspace, cov)
        rand_init = noise_subspace + noise_std * torch.randn(*noise_subspace.size(),
                                                            dtype=dtype,
                                                            device=device)
        posterior_noise_subspace = MatrixNormalPrior(rand_init, cov)

        # Class subspace.
        class_subspace = torch.eye(dim_class_subspace, len(mean), dtype=dtype,
                                    device=device)
        cov = torch.eye(class_subspace.size(0), dtype=dtype, device=device)
        cov /= prior_strength
        prior_class_subspace = MatrixNormalPrior(class_subspace, cov)
        rand_init = class_subspace + noise_std * torch.randn(*class_subspace.size(),
                                                            dtype=dtype,
                                                            device=device)
        posterior_class_subspace = MatrixNormalPrior(rand_init, cov)

        # cov = same as class subspace.
        class_mean_priors, class_mean_posteriors = [], []
        class_means_p = torch.eye(n_classes, dim_class_subspace, dtype=dtype,
                                    device=device)
        for mean_i in class_means_p:
            class_mean_priors.append(NormalFullCovariancePrior(mean_i, cov))
            class_mean_posteriors.append(NormalFullCovariancePrior(
                mean_i + noise_std * torch.randn(mean_i.shape[0], dtype=dtype,
                                                device=device),
                cov)
            )

        return PLDASet(prior_mean, posterior_mean, prior_prec, posterior_prec,
                       prior_noise_subspace, posterior_noise_subspace,
                       prior_class_subspace, posterior_class_subspace,
                       class_mean_priors, class_mean_posteriors)

    def __init__(self, prior_mean, posterior_mean, prior_prec, posterior_prec,
                 prior_noise_subspace, posterior_noise_subspace,
                 prior_class_subspace, posterior_class_subspace,
                 prior_means, posterior_means):
        '''
        Args:
            prior_mean (``NormalIsotropicCovariancePrior``): Prior over
                the global mean.
            posterior_mean (``NormalIsotropicCovariancePrior``):
                Posterior over the global mean.
            prior_prec (``GammaPrior``): Prior over the global precision.
            posterior_prec (``GammaPrior``): Posterior over the global
                precision.
            prior_noise_subspace (``MatrixNormalPrior``): Prior over
                the subpace.
            posterior_noise_subspace (``MatrixNormalPrior``): Posterior
                over the noise subspace.
            prior_class_subspace (``MatrixNormalPrior``): Prior over
                the class subspace.
            posterior_class_subspace (``MatrixNormalPrior``): Posterior
                over the class subspace.
            prior_means (``list``): List of prior distributions over
                the class' means.
            posterior_means (``list``): List of posterior distributions
                over the class' means.

        '''
        super().__init__()
        self.mean_param = BayesianParameter(prior_mean, posterior_mean)
        self.precision_param = BayesianParameter(prior_prec, posterior_prec)
        self.noise_subspace_param = BayesianParameter(prior_noise_subspace,
                                                      posterior_noise_subspace)
        self.class_subspace_param = BayesianParameter(prior_class_subspace,
                                                      posterior_class_subspace)
        self.class_mean_params = BayesianParameterSet([
            BayesianParameter(prior, posterior)
            for prior, posterior in zip(prior_means, posterior_means)
        ])
        self._subspace1_dim, self._data_dim = self.noise_subspace.size()
        self._subspace2_dim, _ = self.class_subspace.size()

    @property
    def mean(self):
        _, mean = self.mean_param.expected_value(concatenated=False)
        return mean

    @property
    def precision(self):
        return self.precision_param.expected_value()[1]

    @property
    def noise_subspace(self):
        _, exp_value2 = \
            self.noise_subspace_param.expected_value(concatenated=False)
        return exp_value2

    @property
    def class_subspace(self):
        _, exp_value2 =  \
            self.class_subspace_param.expected_value(concatenated=False)
        return exp_value2

    @property
    def class_means(self):
        means = []
        for mean_param in self.class_mean_params:
            _, mean = mean_param.expected_value(concatenated=False)
            means.append(mean)
        return torch.stack(means)

    def _get_expectation(self):
        log_prec, prec = self.precision_param.expected_value(concatenated=False)
        noise_s_quad, noise_s_mean = \
            self.noise_subspace_param.expected_value(concatenated=False)
        class_s_quad, class_s_mean = \
            self.class_subspace_param.expected_value(concatenated=False)
        m_quad, m_mean = self.mean_param.expected_value(concatenated=False)
        class_mean_mean, class_mean_quad = [], []
        for mean_param in self.class_mean_params:
            mean_quad, mean = mean_param.expected_value(concatenated=False)
            class_mean_mean.append(mean)
            class_mean_quad.append(mean_quad)
        return {
            'log_prec': log_prec, 'prec': prec,
            'm_quad':m_quad, 'm_mean': m_mean,
            'noise_s_quad':noise_s_quad, 'noise_s_mean': noise_s_mean,
            'class_s_quad':class_s_quad, 'class_s_mean': class_s_mean,
            'class_mean_mean': torch.stack(class_mean_mean),
            'class_mean_quad': torch.stack(class_mean_quad)
        }

    def _precompute(self, s_stats):
        '''Pre-compute intermediary results for the expected
        log-likelihood/expected natural parameters.

        '''
        # Expected value of the parameters.
        self.cache.update(self._get_expectation())

        # Estimate the posterior distribution of the latent variables.
        self.latent_posterior(s_stats)

        # Load the necessary value from the cache.
        noise_means = self.cache['noise_means']
        global_mean = self.cache['m_mean']
        class_s_mean = self.cache['class_s_mean']
        class_mean_mean = self.cache['class_mean_mean']
        l_quads = self.cache['l_quads']
        class_s_quad = self.cache['class_s_quad']
        class_mean_quad = self.cache['class_mean_quad']
        noise_s_quad = self.cache['noise_s_quad'].view(-1)
        m_quad = self.cache['m_quad']
        class_means = class_mean_mean @ class_s_mean
        class_quads = class_mean_quad.view(len(self), -1) @ \
            class_s_quad.view(-1)

        # Separate the s. statistics.
        data_quad, data = s_stats[:, 0], s_stats[:, 1:]
        npoints = len(data)

        # Intermediary computations to compute the exp. llh for each
        # components.
        means = noise_means + class_means.view(len(self), 1, -1) + \
            global_mean.view(1, 1, -1)
        lnorm = l_quads @ noise_s_quad.view(-1)
        lnorm += (class_quads).view(len(self), 1) + m_quad
        lnorm += 2 * torch.sum(
            noise_means * (class_means.view(len(self), 1, -1) + \
            global_mean.view(1, 1, -1)), dim=-1)
        lnorm += 2 * (class_means @ global_mean).view(-1, 1)

        deltas = -2 * torch.sum(means.detach() * data.view(1, npoints, -1), dim=-1)
        deltas += data_quad.view(1, -1)
        deltas += lnorm.detach()

        # We store values necessary for the accumulation and to compute
        # the expected value of the natural parameters.
        self.cache['deltas'] = deltas
        self.cache['means'] = means
        self.cache['lnorm'] = lnorm
        self.cache['npoints'] = npoints

    def latent_posterior(self, stats):
        # Extract the portion of the s. statistics we need.
        data = stats[:, 1:].detach()
        length = len(stats)

        prec = self.cache['prec']
        noise_s_quad = self.cache['noise_s_quad']
        noise_s_mean = self.cache['noise_s_mean']
        m_mean = self.cache['m_mean']
        class_means = self.cache['class_mean_mean'] @ self.cache['class_s_mean']

        # The covariance matrix is the same for all the latent variables.
        l_cov = torch.inverse(
            torch.eye(self._subspace1_dim, dtype=stats.dtype,
                      device=stats.device) + prec * noise_s_quad)

        # Compute the means conditioned on the class.
        data_mean = data.view(length, 1, -1) - class_means
        data_mean -= m_mean.view(1, 1, -1)
        l_means = prec *  data_mean @ noise_s_mean.t() @ l_cov

        # Intermediary computation that will be necessary to compute the
        # elbo / accumuate the statistics.
        self.cache['noise_means'] = torch.stack([
            l_means[:, i, :] @ noise_s_mean
            for i in range(len(self))
        ])
        l_quad = torch.stack([
            l_cov + l_means[:, i, :, None] * l_means[:, i, None, :]
            for i in range(len(self))
        ])
        self.cache['l_quads'] = l_quad.view(len(self), len(stats), -1)
        self.cache['l_means'] = l_means
        self.cache['l_cov'] = l_cov

        self.cache['l_kl_divs'] = torch.stack([
            kl_div_std_norm(l_means[:, i, :], l_cov)
            for i in range(len(self))
        ])

        return l_means, l_cov

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        return [
            [self.precision_param, self.mean_param],
            [self.noise_subspace_param, *self.class_mean_params],
            [self.class_subspace_param],
        ]

    def sufficient_statistics(self, data):
        stats = torch.cat([torch.sum(data ** 2, dim=1).view(-1, 1), data],
                          dim=-1)

        # To avoid repeating many computation we compute and store in
        # the cache all the necessary intermediate results.
        self._precompute(stats)

        return stats

    def forward(self, s_stats):
        # Load the necessary value from the cache.
        prec = self.cache['prec']
        log_prec = self.cache['log_prec']
        deltas = self.cache['deltas']
        exp_llhs = -.5 * (prec * deltas - self._data_dim * log_prec + \
            self._data_dim * math.log(2 * math.pi))

        return exp_llhs.t()

    def accumulate(self, s_stats, parent_msg=None):
        if parent_msg is None:
            raise ValueError('"parent_msg" should not be None')
        resps = parent_msg.detach()

        dtype = s_stats.dtype
        device = s_stats.device

        # Separate the s. statistics.
        _, data = s_stats[:, 0].detach(), s_stats[:, 1:].detach()

        # Load cached values and clear the cache.
        deltas = self.cache['deltas'].detach()
        l_means = self.cache['l_means'].detach()
        l_quads = self.cache['l_quads'].detach()
        prec = self.cache['prec'].detach()
        noise_means = self.cache['noise_means'].detach()
        m_mean = self.cache['m_mean'].detach()
        class_s_mean = self.cache['class_s_mean'].detach()
        class_s_quad = self.cache['class_s_quad'].detach()
        class_mean_mean = self.cache['class_mean_mean'].detach()
        class_mean_quad = self.cache['class_mean_quad'].detach()
        class_means = class_mean_mean @ class_s_mean
        self.clear_cache()

        data_noise_class_mean = data - (noise_means + class_means[:, None, :])
        data_noise_class_mean = (resps.t()[:, :, None] * data_noise_class_mean).sum(dim=0)
        acc_mean = torch.sum(data_noise_class_mean, dim=0)
        del data_noise_class_mean

        acc_noise_s_stats1 = (resps.t()[:, :, None] * l_quads).sum(dim=0)
        acc_noise_s_stats1 = acc_noise_s_stats1.sum(dim=0)
        acc_noise_s_stats1 = acc_noise_s_stats1.view(self._subspace1_dim,
                                                     self._subspace1_dim)
        acc_noise_s_stats1 = make_symposdef(acc_noise_s_stats1)
        data_class_means = (class_means + m_mean[None, :])
        data_class_means = resps.t()[:, :, None] * (data[None, :, :] - \
            data_class_means[:, None, :])
        acc_noise_s_stats2 = torch.stack([
            l_means[:, i, :].t() @ data_class_means[i]
            for i in range(len(self))
        ]).sum(dim=0).view(-1)
        del data_class_means

        acc_class_s_stats1 = \
            (resps @ class_mean_quad.view(len(self), -1)).sum(dim=0)
        acc_class_s_stats1 = acc_class_s_stats1.view(self._subspace2_dim,
                                                     self._subspace2_dim)
        acc_class_s_stats1 = make_symposdef(acc_class_s_stats1)
        data_noise_means = (data - noise_means - m_mean)
        acc_means = (resps.t()[:, :, None] * data_noise_means).sum(dim=1)
        acc_class_s_stats2 = acc_means[:, :, None] * class_mean_mean[:, None, :]
        acc_class_s_stats2 = acc_class_s_stats2.sum(dim=0).t().contiguous().view(-1)

        acc_stats = {
            self.precision_param: torch.cat([
                .5 * torch.tensor(len(s_stats) * self._data_dim, dtype=dtype,
                                  device=device).view(1),
                -.5 * torch.sum(resps.t() * deltas).view(1)
            ]),
            self.mean_param: torch.cat([
                - .5 * torch.tensor(len(s_stats) * prec, dtype=dtype,
                                    device=device),
                prec * acc_mean
            ]),
            self.noise_subspace_param:  torch.cat([
                - .5 * prec * acc_noise_s_stats1.view(-1),
                prec * acc_noise_s_stats2
            ]),
            self.class_subspace_param:  torch.cat([
                - .5 * prec * acc_class_s_stats1.view(-1),
                prec * acc_class_s_stats2
            ])
        }

        # Accumulate the statistics for the class means.
        class_mean_acc_stats1 = class_s_quad
        class_mean_acc_stats1 = make_symposdef(class_mean_acc_stats1)
        w_data_noise_means = resps.t()[:, :, None] * data_noise_means
        class_mean_acc_stats2 = \
            (class_s_mean @ w_data_noise_means.sum(dim=1).t()).t()
        for i, mean_param in enumerate(self.class_mean_params):
            class_mean_acc_stats = {
                mean_param: torch.cat([
                    -.5 * prec * resps[:, i].sum() * class_mean_acc_stats1.view(-1),
                    prec * class_mean_acc_stats2[i]
                ])
            }
            acc_stats.update(class_mean_acc_stats)

        return acc_stats

    ####################################################################
    # BayesianModelSet interface.
    ####################################################################

    def __getitem__(self, key):
        _, class_s_mean = self.class_subspace_param.expected_value(concatenated=False)
        _, class_mean = self.class_mean_params[key].expected_value(concatenated=False)
        mean = self.mean + class_s_mean.t() @ class_mean
        _, noise_s_mean = self.noise_subspace_param.expected_value(concatenated=False)
        cov = noise_s_mean.t() @ noise_s_mean
        cov += torch.eye(self._data_dim, dtype=cov.dtype,
                         device=cov.device) / self.precision
        return NormalSetElement(mean=mean, cov=cov)

    def __len__(self):
        return len(self.class_mean_params)

    def local_kl_div_posterior_prior(self, parent_msg=None):
        if parent_msg is None:
            raise ValueError('"parent_msg" should not be None')
        resps = parent_msg
        return torch.sum(resps * self.cache['l_kl_divs'].t(), dim=-1).detach()


########################################################################
# Probabilistic Linear Discriminant Analysis (PLDA) with a marginalized
# noise subspace.
########################################################################

class MarginalPLDASet(BayesianModelSet):
    '''Set of Normal distribution for the Probabilistic Linear
    Discriminant Analysis (PLDA) model with a marginalized noise
    subspace. This corresponds to a set of Normal densities with shared
    covariances whose means are constrain to lie in a subspace.

    Attributes:
        base_normal (:any:`Normal`): Normal model which is "wrapped"
            to make this normal set.
        class_subspace (``torch.Tensor[class_subspace_dim, data_dim]``):
            Mean of the matrix defining the "class" subspace.
            (across-class variations).
        class_means: (``torch.Tensor[]``)

    '''

    @staticmethod
    def create(model_conf, mean, variance, create_model_handle):
        dtype, device = mean.dtype, mean.device
        normal = create_model_handle(model_conf['base_normal'], mean, variance,
                                     create_model_handle)
        n_classes = model_conf['size']
        dim_class_subspace = model_conf['dim_class_subspace']
        noise_std = model_conf['noise_std']
        prior_strength = model_conf['prior_strength']

        # Class subspace.
        class_subspace = torch.eye(dim_class_subspace, len(mean), dtype=dtype,
                                    device=device)
        cov = torch.eye(class_subspace.size(0), dtype=dtype, device=device)
        cov /= prior_strength
        prior_class_subspace = MatrixNormalPrior(class_subspace, cov)
        rand_init = class_subspace + noise_std * torch.randn(*class_subspace.size(),
                                                            dtype=dtype,
                                                            device=device)
        posterior_class_subspace = MatrixNormalPrior(rand_init, cov)

        # cov = same as class subspace.
        class_mean_priors, class_mean_posteriors = [], []
        class_means_p = torch.eye(n_classes, dim_class_subspace, dtype=dtype,
                                    device=device)
        for mean_i in class_means_p:
            class_mean_priors.append(NormalFullCovariancePrior(mean_i, cov))
            class_mean_posteriors.append(NormalFullCovariancePrior(
                mean_i + noise_std * torch.randn(mean_i.shape[0], dtype=dtype,
                                                device=device),
                cov)
            )

        return MarginalPLDASet(normal, prior_class_subspace,
                               posterior_class_subspace,
                               class_mean_priors, class_mean_posteriors)


    def __init__(self, normal, prior_class_subspace, posterior_class_subspace,
                 prior_means, posterior_means):
        '''
        Args:
            normal (:any:`Normal`): Normal model which will be "wrapped"
                to make this normal set.
            prior_class_subspace (``MatrixNormalPrior``): Prior over
                the class subspace.
            posterior_class_subspace (``MatrixNormalPrior``): Posterior
                over the class subspace.
            prior_means (``list``): List of prior distributions over
                the class' means.
            posterior_means (``list``): List of posterior distributions
                over the class' means.

        '''
        super().__init__()
        self.base_normal = normal
        self.class_subspace_param = BayesianParameter(prior_class_subspace,
                                                      posterior_class_subspace)
        self.class_mean_params = BayesianParameterSet([
            BayesianParameter(prior, posterior)
            for prior, posterior in zip(prior_means, posterior_means)
        ])
        self._subspace_dim = self.class_subspace.shape[0]

    @property
    def class_subspace(self):
        _, exp_value2 =  \
            self.class_subspace_param.expected_value(concatenated=False)
        return exp_value2

    @property
    def class_means(self):
        means = []
        for mean_param in self.class_mean_params:
            _, mean = mean_param.expected_value(concatenated=False)
            means.append(mean)
        return torch.stack(means)

    def _get_expectation(self):
        global_mean = self.base_normal.mean
        global_precision = torch.inverse(self.base_normal.cov)

        class_s_quad, class_s_mean = \
            self.class_subspace_param.expected_value(concatenated=False)

        # "class_s_quad" correspond to:
        #   E[MM^T] = q * U + E[M] E[M^T]
        # (where 'q' is the dimension of the subspace) w.r.t to the
        # matrix normal prior. However, we need to compute the following
        # expectation:
        #   E[MAM^T] = U tr(A) + E[M] A E[M^T]
        subspace_cov = (class_s_quad - class_s_mean @ class_s_mean.t())
        subspace_cov /= self._subspace_dim
        class_s_quad_prec = subspace_cov * torch.trace(global_precision)
        class_s_quad_prec += class_s_mean @ global_precision @ class_s_mean.t()

        class_mean_mean, class_mean_quad = [], []
        for mean_param in self.class_mean_params:
            mean_quad, mean = mean_param.expected_value(concatenated=False)
            class_mean_mean.append(mean)
            class_mean_quad.append(mean_quad)
        return {
            'mean': global_mean,
            'precision': global_precision,
            'class_s_quad_prec': class_s_quad_prec,
            'class_s_mean': class_s_mean,
            'class_mean_mean': torch.stack(class_mean_mean),
            'class_mean_quad': torch.stack(class_mean_quad),
            'subspace_cov': subspace_cov
        }

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        return self.base_normal.mean_field_factorization() + [
            [*self.class_mean_params],
            [self.class_subspace_param]
        ]

    def sufficient_statistics(self, data):
        return data

    def forward(self, data):
        exp_params = self._get_expectation()
        global_mean = exp_params['mean']
        global_precision = exp_params['precision']
        c_s_mean = exp_params['class_s_mean']
        c_s_quad_prec = exp_params['class_s_quad_prec']
        c_m_mean = exp_params['class_mean_mean']
        c_m_quad = exp_params['class_mean_quad']

        # Projection of the mean onto the observed space.
        c_means = c_m_mean @ c_s_mean

        # Log-likelihood of the base normal (common for all the
        # components).
        normal_stats = self.base_normal.sufficient_statistics(data)
        normal_llh = self.base_normal(normal_stats)
        self.cache['normal_stats'] = normal_stats

        # Component specific computations.
        prec_X_mean = (data - global_mean) @ global_precision
        exp_llh = normal_llh[:, None] + prec_X_mean @ c_means.t()
        exp_llh += -.5 * \
            (c_m_quad.view(len(self), -1) @ c_s_quad_prec.view(-1))[None]

        # Store intermediate results for the accumulation of the s.
        # statistics.
        self.cache['c_means'] = c_means
        self.cache['prec_X_mean'] = prec_X_mean
        self.cache['exp_params'] = exp_params

        return exp_llh


    def accumulate(self, data, parent_msg=None):
        if parent_msg is None:
            raise ValueError('"parent_msg" should not be None')
        resps = parent_msg.detach()

        c_means = self.cache['c_means']
        c_s_mean = self.cache['exp_params']['class_s_mean']
        c_s_quad_prec = self.cache['exp_params']['class_s_quad_prec']
        prec_X_mean = self.cache['prec_X_mean']
        c_m_quad = self.cache['exp_params']['class_mean_quad']
        s_cov = self.cache['exp_params']['subspace_cov']

        # Expected value w.r.t. of the matrix normal posterior of the
        # quadratic term:
        #   E[M^T v_z v_z^T M]
        exp_m_quad = torch.sum(resps[:, :, None, None] * c_m_quad, dim=1)
        idxs = torch.eye(len(c_m_quad)).view(-1).long()
        trace = (s_cov * exp_m_quad).view(len(data), -1)[:, idxs].sum(dim=-1)
        identity = torch.eye(data.shape[1], dtype=data.dtype, device=data.device)
        quad_term = trace[:, None, None] * identity[None] + \
            c_s_mean.t()[None] @ exp_m_quad @ c_s_mean

        # E[(M^T v_z) x^T] + E[x^T (M^T v_z)]
        exp_means = resps @ c_means
        mean_term = exp_means[:, :, None] * data[:, None, :]
        mean_term += data[:, :, None] * exp_means[:, None, :]

        # xx^T
        data_quad = (data[:, :, None] * data[:, None, :])

        # Sufficient statistics for the base normal model.
        n_stats = torch.cat([
            (data_quad - mean_term + quad_term).view(len(data), -1),
            data - exp_means,
            torch.ones(data.size(0), 2, dtype=data.dtype, device=data.device),
        ], dim=-1)
        acc_stats = self.base_normal.accumulate(n_stats)

        acc_class_s_stats1 = \
            (resps @ class_mean_quad.view(len(self), -1)).sum(dim=0)
        acc_class_s_stats1 = acc_class_s_stats1.view(self._subspace2_dim,
                                                     self._subspace2_dim)
        acc_class_s_stats1 = make_symposdef(acc_class_s_stats1)
        data_noise_means = (data - noise_means - m_mean)
        acc_means = (resps.t()[:, :, None] * data_noise_means).sum(dim=1)
        acc_class_s_stats2 = acc_means[:, :, None] * class_mean_mean[:, None, :]
        acc_class_s_stats2 = acc_class_s_stats2.sum(dim=0).t().contiguous().view(-1)
        self.class_subspace_param:  torch.cat([
                - .5 * prec * acc_class_s_stats1.view(-1),
                prec * acc_class_s_stats2
            ])

        # Accumulate the statistics for the class means.
        class_mean_acc_stats1 = c_s_quad_prec
        class_mean_acc_stats1 = make_symposdef(class_mean_acc_stats1)
        class_mean_acc_stats2 = resps.t()  @ prec_X_mean @ c_s_mean.t()
        for i, mean_param in enumerate(self.class_mean_params):
            class_mean_acc_stats = {
                mean_param: torch.cat([
                    -.5 * resps[:, i].sum() * class_mean_acc_stats1.view(-1),
                    class_mean_acc_stats2[i]
                ])
            }
            acc_stats.update(class_mean_acc_stats)

        return acc_stats


    ####################################################################
    # BayesianModelSet interface.
    ####################################################################

    def __getitem__(self, key):
        pass

    def __len__(self):
        return len(self.class_mean_params)


__all__ = ['PLDASet', 'MarginalPLDASet']
