import math
from typing import Any, Dict, Sequence, Optional, cast, Hashable
from coba.environments import Context, Action
from coba.learners.primitives import Learner, Probs, Info

import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, LassoCV

from econml.dml import LinearDML


class IGWBanditLearner(Learner):

    def __init__(self, num_actions: int, context_dim: int,
                do_feature_selection:   bool  = True,
                estimation_constant:    float = 2,
                estimation_rate:        float = 1.,
                confidence_parameter:   float = 0.95) -> None:

        # model parameters
        self.K:     int               = num_actions
        self.d:     int               = context_dim
        self.model: Dict[Action, Any] = {}

        # history
        self._X: np.typing.NDArray[np.float] = np.empty(shape=(0, self.d), dtype=np.float)
        self._A: np.typing.NDArray[np.int]   = np.empty(shape=(0,), dtype=np.int)
        self._Y: np.typing.NDArray[np.float] = np.empty(shape=(0,), dtype=np.float)

        # time record
        self._t:     int     = 1
        self._epoch: int     = 1

        # exploration parameters
        self._estimation_constant:  float = estimation_constant
        self._estimation_rate:      float = estimation_rate
        self._delta: float = 1. - confidence_parameter
        self._gamma: float = 1.

        # lasso vs standard regression
        self._do_feature_selection = do_feature_selection

    @property
    def params(self) -> Dict[str, Any]:
        return {"family": "inverse_gap_weighting"}

    def predict(self, context: Context, actions: Sequence[Action]) -> Probs:

        # uniform policy if not enough actions have been sampled
        if len(self.model) < self.K:
            return (1. / self.K) * np.ones(self.K)

        yhat = np.ones(self.K)
        context = np.array(context).reshape(1,-1)
        for action in range(self.K):
            yhat[action] = self.model[action].predict(context)

        # if self._t > 1000:
        #     self._gamma = 5000
        probs = 1. / (self.K + self._gamma * (np.max(yhat) - yhat))
        probs[np.argmax(yhat)] = 0.
        probs[np.argmax(yhat)] = 1. - np.sum(probs)

        return probs

    def learn(self, context: Context, action: Action, reward: float, probability: float, info: Info) -> None:
        if isinstance(action, (list, tuple)):
            action = np.argmax(action)

        self._X = np.vstack([self._X, context])
        self._A = np.hstack([self._A, action])
        self._Y = np.hstack([self._Y, reward])

        # if (self._t == np.power(2, self._epoch)):
        if (self._t % 100 == 0):
            self._update()
            self._epoch += 1

        self._t += 1

    def _update(self) -> None:
        # Train model
        self.model = {}
        for action in range(self.K):
            idx = self._A == action
            X_a = self._X[idx]
            Y_a = self._Y[idx]

            if (X_a.shape[0] < 5):
                return

            if self._do_feature_selection:
                # Choose Lasso penalty
                model_cv = LassoCV()
                model_cv.fit(X_a, Y_a)

                # Fit Lasso model
                model_a = Lasso(alpha=model_cv.alpha_)
                model_a.fit(X_a, Y_a)
                self.model[action] = model_a
            else:
                model_a = LinearRegression()
                model_a.fit(X_a, Y_a)
                self.model[action] = model_a

        # Set exploration parameter
        if self._do_feature_selection:
            pseudodim = 0
            for a in range(self.K):
                pseudodim += np.count_nonzero(self.model[a].coef_)
            pseudodim = pseudodim * np.log(self.d)
        else:
            pseudodim = self.K * self.d
        excess_risk_bound = self._estimation_constant * pseudodim * np.log(math.pow(self._epoch, 2) / self._delta) / math.pow(self._X.shape[0], self._estimation_rate)
        self._gamma = np.sqrt(self.K / (excess_risk_bound))


class SemiparametricIGWBanditLearner(Learner):

    def __init__(self, num_actions: int, context_dim: int,
                do_feature_selection:   bool = False,
                estimation_constant:    float = 2,
                estimation_rate:        float = 1.,
                confidence_parameter:   float = 0.95) -> None:

        # model parameters
        self.K:     int = num_actions
        self.d:     int = context_dim
        self.model: Any = None

        # history
        self._X: np.typing.NDArray[np.float] = np.empty(shape=(0, self.d), dtype=np.float)
        self._A: np.typing.NDArray[np.int]   = np.empty(shape=(0,), dtype=np.int)
        self._Y: np.typing.NDArray[np.float] = np.empty(shape=(0,), dtype=np.float)
        self._P: np.typing.NDArray[np.float] = np.empty(shape=(0, self.K), dtype=np.float)

        # time record
        self._t:     int     = 1
        self._epoch: int     = 1

        # exploration parameters
        self._estimation_constant:  float = estimation_constant
        self._estimation_rate:      float = estimation_rate
        self._delta: float = 1. - confidence_parameter
        self._gamma: float = 1.

        self._do_feature_selection = do_feature_selection

    @property
    def params(self) -> Dict[str, Any]:
        return {"family": "semiparametric_inverse_gap_weighting"}

    def predict(self, context: Context, actions: Sequence[Action]) -> Probs:

        # uniform policy if not enough actions have been sampled
        if not self.model:
            return (1. / self.K) * np.ones(self.K)

        context = np.array(context).reshape(1,-1)
        theta_hat = np.atleast_1d(np.squeeze(self.model.const_marginal_effect(X=context)))
        theta_hat = np.concatenate(([0.], theta_hat))
        probs = 1. / (self.K + self._gamma * (np.max(theta_hat) - theta_hat))
        probs[np.argmax(theta_hat)] = 0.
        probs[np.argmax(theta_hat)] = 1. - np.sum(probs)

        return probs

    def learn(self, context: Context, action: Action, reward: float, probability: float, info: Info) -> None:
        if isinstance(action, (list, tuple)):
            action = np.argmax(action)

        # append to history
        self._X = np.vstack([self._X, context])
        self._A = np.hstack([self._A, action])
        self._Y = np.hstack([self._Y, reward])
        probs = self.predict(context, [k for k in range(self.K)])
        self._P = np.vstack([self._P, probs])

        # update model
        # if (self._t == np.power(2, self._epoch)):
        if (self._t % 100 == 0):
            if self._t > 2*self.K:
                self._update()
            self._epoch += 1

        self._t += 1

    def _update(self) -> None:
        # Train model
        self.model = LinearDML(model_y=LinearRegression(),
                                model_t=PropensityModel(self._P),
                                discrete_treatment=True,
                                linear_first_stages=True,
                                cv=1)

        self.model.fit(self._Y, self._A, X=self._X)

        pseudodim = self.K * self.d
        excess_risk_bound = self._estimation_constant * pseudodim * np.log(math.pow(self._epoch, 2) / self._delta) / math.pow(self._X.shape[0], self._estimation_rate)
        self._gamma = np.sqrt(self.K / (excess_risk_bound))


class MeanModel():
    def __init__(self):
        pass

    def fit(self, *args, **kwargs):
        pass

    def predict(self, X):
        return np.zeros(X.shape[0])

class PropensityModel():
    def __init__(self, probs):
        self.probs = probs

    def fit(self, *args, **kwargs):
        pass

    def predict_proba(self, X):
        return self.probs


class EpsilonGreedyLearner(Learner):

    def __init__(self, K, eps):
        self.K = K
        self.eps = eps
        self.means = np.zeros(K)
        self.counts = np.zeros(K)

    @property
    def params(self) -> Dict[str, Any]:
        return {"family": "epsilon_greedy"}

    def learn(self, context: Context, action: Action, reward: float, probability: float, info: Info) -> None:
        if isinstance(action, (list, tuple)):
            action = np.argmax(action)
        
        alpha = 1. / (self.counts[action] + 1)
        self.means[action] = (1. - alpha) * self.means[action] + alpha * reward
        self.counts[action] = self.counts[action] + 1

    def predict(self, context: Context, actions: Sequence[Action]) -> Probs:
        probs = np.ones(self.K) * self.eps / self.K
        probs[np.argmax(self.means)] += 1. - self.eps
        return probs


# class LassoEpsilonGreedy(Learner):

#     def __init__(self, K, p, eps):
#         self.K = K
#         self.p = p
#         self.eps = eps

#         self.xs = np.empty(shape=(0, p), dtype=float)
#         self.ws = np.empty(shape=(0,), dtype=int)
#         self.yobs = np.empty(shape=(0,), dtype=float)
#         self.models = None

#     @property
#     def params(self) -> Dict[str, Any]:
#         return {"family": "epsilon_greedy"}

#     def learn(self, context: Context, action: Action, reward: float, probability: float, info: Info) -> None:
#         if not isinstance(action, int):
#             action = np.argmax(action)

#         # append to history
#         self._X = np.vstack([self._X, context])
#         self._A = np.hstack([self._A, action])
#         self._Y = np.hstack([self._Y, reward])


#     def update(self, xs, ws, yobs, *args, **kwargs):
#         self.xs = np.row_stack([self.xs, xs])
#         self.ws = np.hstack([self.ws, ws])
#         self.yobs = np.hstack([self.yobs, yobs])

#         self.models = []
#         for action in range(self.K):
#             X_a = self._X
#             xs_w = xs[ws==w]
#             yobs_w = yobs[ws==w]
#             model = LassoCV(cv=5)
#             model.fit(xs[ws == w], yobs[ws == w])
#             self.models.append(model)
#         return self

#     def predict(self, context: Context, actions: Sequence[Action]) -> Probs:
#         if not self.model:
#             return (1. / self.K) * np.ones(self.K)

#         context = np.array(context).reshape(1,-1)
#         yhat = np.zeros(self.K)
#         for action in range(self.K):
#             yhat[action] = self.models[action].predict(context)

#         probs = np.ones(self.K) * self.eps / self.K
#         probs[np.argmax(yhat)] += 1. - self.eps

#         return probs