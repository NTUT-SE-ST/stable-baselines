import pytest
import os
import warnings

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import  SubprocVecEnv, DummyVecEnv
from stable_baselines import PPO2, A2C, ACER, ACKTR
from stable_baselines.common.action_mask_env import DiscreteActionMaskEnv, MultiDiscreteActionMaskEnv, MultiDiscreteUnbalancedActionMaskEnv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

@pytest.mark.slow
@pytest.mark.parametrize('vec_env', [SubprocVecEnv, DummyVecEnv])
@pytest.mark.parametrize('policy', [MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy])
@pytest.mark.parametrize('env_class', [DiscreteActionMaskEnv, MultiDiscreteActionMaskEnv, MultiDiscreteUnbalancedActionMaskEnv])
def test_action_mask_learn_ppo2(vec_env, policy, env_class):
    env = vec_env([lambda: env_class() for i in range(2)])

    model = PPO2(policy, env, verbose=0, nminibatches=2)
    model.learn(total_timesteps=500)
    env.close()

@pytest.mark.slow
@pytest.mark.parametrize('vec_env', [SubprocVecEnv, DummyVecEnv])
@pytest.mark.parametrize('policy', [MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy])
@pytest.mark.parametrize('env_class', [DiscreteActionMaskEnv, MultiDiscreteActionMaskEnv, MultiDiscreteUnbalancedActionMaskEnv])
def test_action_mask_run_ppo2(vec_env, policy, env_class):
    env = vec_env([lambda: env_class()])

    model = PPO2(policy, env, verbose=0, nminibatches=1)

    obs, done, action_masks = env.reset(), [False], []
    while not done[0]:
        action, _states = model.predict(obs, action_mask=action_masks)
        obs, _, done, infos = env.step(action)

        action_masks.clear()
        for info in infos:
            env_action_mask = info.get('action_mask')
            action_masks.append(env_action_mask)

    env.close()

@pytest.mark.slow
@pytest.mark.parametrize('vec_env', [SubprocVecEnv, DummyVecEnv])
@pytest.mark.parametrize('policy', [MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy])
@pytest.mark.parametrize('env_class', [DiscreteActionMaskEnv, MultiDiscreteActionMaskEnv, MultiDiscreteUnbalancedActionMaskEnv])
def test_action_mask_learn_a2c(vec_env, policy, env_class):
    env = vec_env([lambda: env_class() for i in range(2)])

    model = A2C(policy, env, verbose=0)
    model.learn(total_timesteps=500)
    env.close()

@pytest.mark.slow
@pytest.mark.parametrize('vec_env', [SubprocVecEnv, DummyVecEnv])
@pytest.mark.parametrize('policy', [MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy])
@pytest.mark.parametrize('env_class', [DiscreteActionMaskEnv, MultiDiscreteActionMaskEnv, MultiDiscreteUnbalancedActionMaskEnv])
def test_action_mask_run_a2c(vec_env, policy, env_class):
    env = vec_env([lambda: env_class()])

    model = A2C(policy, env, verbose=0)

    obs, done, action_masks = env.reset(), [False], []
    while not done[0]:
        action, _states = model.predict(obs, action_mask=action_masks)
        obs, _, done, infos = env.step(action)

        action_masks.clear()
        for info in infos:
            env_action_mask = info.get('action_mask')
            action_masks.append(env_action_mask)

    env.close()

@pytest.mark.slow
@pytest.mark.parametrize('vec_env', [SubprocVecEnv, DummyVecEnv])
@pytest.mark.parametrize('policy', [MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy])
@pytest.mark.parametrize('env_class', [DiscreteActionMaskEnv])
def test_action_mask_learn_acer(vec_env, policy, env_class):
    env = vec_env([lambda: env_class() for i in range(2)])

    model = ACER(policy, env, verbose=0)
    model.learn(total_timesteps=500)
    env.close()

@pytest.mark.slow
@pytest.mark.parametrize('vec_env', [SubprocVecEnv, DummyVecEnv])
@pytest.mark.parametrize('policy', [MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy])
@pytest.mark.parametrize('env_class', [DiscreteActionMaskEnv])
def test_action_mask_run_acer(vec_env, policy, env_class):
    env = vec_env([lambda: env_class()])

    model = ACER(policy, env, verbose=0)

    obs, done, action_masks = env.reset(), [False], []
    while not done[0]:
        action, _states = model.predict(obs, action_mask=action_masks)
        obs, _, done, infos = env.step(action)

        action_masks.clear()
        for info in infos:
            env_action_mask = info.get('action_mask')
            action_masks.append(env_action_mask)

    env.close()

@pytest.mark.slow
@pytest.mark.parametrize('vec_env', [SubprocVecEnv, DummyVecEnv])
@pytest.mark.parametrize('policy', [MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy])
@pytest.mark.parametrize('env_class', [DiscreteActionMaskEnv])
def test_action_mask_learn_acktr(vec_env, policy, env_class):
    env = vec_env([lambda: env_class() for i in range(2)])

    model = ACKTR(policy, env, verbose=0)
    model.learn(total_timesteps=500)
    env.close()

@pytest.mark.slow
@pytest.mark.parametrize('vec_env', [SubprocVecEnv, DummyVecEnv])
@pytest.mark.parametrize('policy', [MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy])
@pytest.mark.parametrize('env_class', [DiscreteActionMaskEnv])
def test_action_mask_run_acktr(vec_env, policy, env_class):
    env = vec_env([lambda: env_class()])

    model = ACKTR(policy, env, verbose=0)

    obs, done, action_masks = env.reset(), [False], []
    while not done[0]:
        action, _states = model.predict(obs, action_mask=action_masks)
        obs, _, done, infos = env.step(action)

        action_masks.clear()
        for info in infos:
            env_action_mask = info.get('action_mask')
            action_masks.append(env_action_mask)

    env.close()
