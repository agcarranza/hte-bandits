import math
import numpy as np

from typing import Any, Dict, Sequence, Hashable
from coba.environments import Context, Action
from coba.learners.primitives import Learner, Probs, Info
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from econml.dml import LinearDML, SparseLinearDML


class IGWBanditLearner(Learner):

    def __init__(self,
                epoch_schedule:         int   = 0,
                do_feature_selection:   bool  = False,
                tuning_parameter:       float = 2,
                estimation_rate:        float = 1.,
                confidence_parameter:   float = 0.95
                ) -> None:

        # model parameters
        self.model: Dict[Hashable, Any] = {}

        # history
        self._X: np.typing.NDArray[np.float] = []
        self._A: np.typing.NDArray[np.float] = []
        self._Y: np.typing.NDArray[np.float] = []

        # time record
        self._t:     int     = 1
        self._epoch: int     = 1

        # exploration parameters
        self._epoch_schedule:   int   = epoch_schedule
        self._tuning_parameter: float = tuning_parameter
        self._estimation_rate:  float = estimation_rate
        self._delta:            float = 1. - confidence_parameter
        self._gamma:            float = 1.

        # lasso vs standard regression
        self._do_feature_selection = do_feature_selection

    @property
    def params(self) -> Dict[str, Any]:
        return {"family": "inverse_gap_weighting"}

    def predict(self, context: Context, actions: Sequence[Action]) -> Probs:
        valid_contexts = []
        for c in context:
            if isinstance(c, (float, int)):
                valid_contexts.append(c)
        context = valid_contexts

        min_uniform_samples = len(actions) * sum([1./n for n in range(1, len(actions)+1)]) + 10 # coupon collector

        if len(self.model) < len(actions) or self._t <= min_uniform_samples:
            probs = np.ones(len(actions)) / len(actions)
        else:
            yhat = np.ones(len(actions))
            context = np.array(context).reshape(1,-1)
            for action in actions:
                yhat[np.argmax(action)] = self.model[action].predict(context)

            probs = 1. / (len(actions) + self._gamma * (np.max(yhat) - yhat))
            probs[np.argmax(yhat)] = 0.
            probs[np.argmax(yhat)] = 1. - np.sum(probs)

        if self._epoch_schedule == 0:
            update_condition = self._t == np.power(2, self._epoch)
        else:
            update_condition = self._t % self._epoch_schedule == 0
        
        if update_condition:
            self._epoch += 1
            if self._t > min_uniform_samples:
                self._update(actions)
        self._t += 1
        return probs

    def learn(self, context: Context, action: Action, reward: float, probability: float, info: Info) -> None:
        valid_contexts = []
        for c in context:
            if isinstance(c, (float, int)):
                valid_contexts.append(c)
        context = valid_contexts

        if len(self._X) == 0:
            self._X = np.array(context, dtype=np.float).reshape(1,-1)
            self._A = np.array(action,  dtype=np.float).reshape(1,-1)
            self._Y = np.atleast_1d(np.array(reward,  dtype=np.float))
        else:
            self._X = np.vstack([self._X, context])
            self._A = np.vstack([self._A, action])
            self._Y = np.hstack([self._Y, reward])

    def _update(self, actions: Sequence[Action]) -> None:
        assert len(self._X) > 0
        min_samples_per_action = 5

        # Train model
        self.model = {}
        for action in actions:
            idx = np.all(self._A == action, axis=1)
            X_a = self._X[idx]
            Y_a = self._Y[idx]

            if (X_a.shape[0] < min_samples_per_action):
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
            for action in actions:
                pseudodim += np.count_nonzero(self.model[action].coef_)
            pseudodim = pseudodim * np.log(self._X.shape[1])
        else:
            pseudodim = self._A.shape[1] * self._X.shape[1]
        excess_risk_bound = self._tuning_parameter * pseudodim * np.log(math.pow(self._epoch, 2) / self._delta) / math.pow(self._X.shape[0], self._estimation_rate)
        self._gamma = np.sqrt(self._A.shape[1] / excess_risk_bound)


class SemiparametricIGWBanditLearner(Learner):

    def __init__(self,
                epoch_schedule:         int   = 0,
                do_feature_selection:   bool  = False,
                tuning_parameter:       float = 2,
                estimation_rate:        float = 1.,
                confidence_parameter:   float = 0.95
                ) -> None:

        # model parameters
        self.model: Any = None

        # history
        self._X: np.typing.NDArray[np.float] = []
        self._A: np.typing.NDArray[np.float] = []
        self._Y: np.typing.NDArray[np.float] = []
        self._P: np.typing.NDArray[np.float] = []

        # time record
        self._t:     int     = 1
        self._epoch: int     = 1

        # exploration parameters
        self._epoch_schedule:   int   = epoch_schedule
        self._tuning_parameter: float = tuning_parameter
        self._estimation_rate:  float = estimation_rate
        self._delta:            float = 1. - confidence_parameter
        self._gamma:            float = 1.

        self._do_feature_selection = do_feature_selection

    @property
    def params(self) -> Dict[str, Any]:
        return {"family": "semiparametric_inverse_gap_weighting"}

    def predict(self, context: Context, actions: Sequence[Action]) -> Probs:
        
        valid_contexts = []
        for c in context:
            if isinstance(c, (float, int)):
                valid_contexts.append(c)
        context = valid_contexts

        # uniform policy if not enough actions have been sampled
        min_uniform_samples = len(actions) * sum([1./n for n in range(1, len(actions)+1)]) + 10 # coupon collector
        if not self.model or self._t <= min_uniform_samples:
            probs = np.ones(len(actions)) / len(actions)
        else:
            context = np.array(context).reshape(1,-1)
            theta_hat = np.atleast_1d(np.squeeze(self.model.const_marginal_effect(X=context)))
            theta_hat = np.concatenate(([0.], theta_hat))
            probs = 1. / (len(actions) + self._gamma * (np.max(theta_hat) - theta_hat))
            probs[np.argmax(theta_hat)] = 0.
            probs[np.argmax(theta_hat)] = 1. - np.sum(probs)

        if self._epoch_schedule == 0:
            update_condition = self._t == np.power(2, self._epoch)
        else:
            update_condition = self._t % self._epoch_schedule == 0
        
        if update_condition:
            self._epoch += 1
            if self._t > min_uniform_samples:
                self._update()
        self._t += 1

        # append policy to policy history
        if len(self._P) == 0:
            self._P = np.array(probs, dtype=np.float).reshape(1,-1)
        else:
            self._P = np.vstack([self._P, probs])

        return probs

    def learn(self, context: Context, action: Action, reward: float, probability: float, info: Info) -> None:
        valid_contexts = []
        for c in context:
            if isinstance(c, (float, int)):
                valid_contexts.append(c)
        context = valid_contexts

        if len(self._X) == 0:
            self._X = np.array(context, dtype=np.float).reshape(1,-1)
            self._A = np.array(action,  dtype=np.float).reshape(1,-1)
            self._Y = np.atleast_1d(np.array(reward,  dtype=np.float))
        else:
            self._X = np.vstack([self._X, context])
            self._A = np.vstack([self._A, action])
            self._Y = np.hstack([self._Y, reward])

    def _update(self) -> None:
        assert len(self._X) > 0

        if self._do_feature_selection:
            self.model = SparseLinearDML(model_y=Lasso(),
                                        model_t=PropensityModel(self._P),
                                        discrete_treatment=True,
                                        linear_first_stages=False,
                                        cv=1)
        else:
            self.model = LinearDML(model_y=LinearRegression(),
                                    model_t=PropensityModel(self._P),
                                    discrete_treatment=True,
                                    linear_first_stages=False,
                                    cv=1)

        self.model.fit(Y=self._Y, T=np.argmax(self._A, axis=1), X=self._X)

        if self._do_feature_selection:
            pseudodim = np.count_nonzero(self.model.coef_) * np.log(self._X.shape[1])
        else:
            pseudodim = self._A.shape[1] * self._X.shape[1]

        excess_risk_bound = self._tuning_parameter * pseudodim * np.log(math.pow(self._epoch, 2) / self._delta) / math.pow(self._X.shape[0], self._estimation_rate)
        self._gamma = np.sqrt(self._A.shape[1] / excess_risk_bound)
        


class ZeroMeanModel():
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