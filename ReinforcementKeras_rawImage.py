import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import multiprocessing as mp
import pickle
import os
from sys import platform
from datetime import datetime
from gym.envs.registration import register

from SimulatorDriverClass import SimulatorDriver
from ProcessedImageEnvironmentClass import ProcessedImageEnvironment
from RawImageEnvironmentClass import RawImageEnvironment
from FixPolicyEnvironmentClass import FixPolicyEnvironment

# Reference: https://keras.io/examples/rl/actor_critic_cartpole/
register(
    id='FixPolicy-v0',
    entry_point='FixPolicyEnvironmentClass:FixPolicyEnvironment'
)
register(
    id='ProcessedImage-v0',
    entry_point='ProcessedImageEnvironmentClass:ProcessedImageEnvironment'
)
# register(
#     id='RawImage-v0',
#     entry_point='RawImageEnvironmentClass:RawImageEnvironment'
# )

if __name__ == '__main__':
    if platform == 'linux' or platform == 'linux2':
        mp.set_start_method('spawn')
    mp.freeze_support()

    # gamma = 0.99  # Discount factor for past rewards
    max_steps_per_episode = int(1e8)
    # env = gym.make("CartPole-v0")  # Create the environment
    # env.seed(seed)
    # env = FixPolicyEnvironment()
    env = RawImageEnvironment()
    eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0


    # MARK: - Model Configuration

    # num_inputs = env.observation_space.shape[0]
    # num_actions = env.action_space.n
    # inputs = layers.Input(shape=(num_inputs,), name='input')
    # common1 = layers.Dense(128, activation="relu", name='common1')(inputs)
    # common2 = layers.Dense(128, activation="relu", name='common2')(common1)
    # action = layers.Dense(num_actions, activation="softmax", name='action')(common2)

    image_inputs = layers.Input(shape=env.observation_spec['image']['shape'], dtype=np.float, name='image')
    image_layer = kr.layers.Conv2D(filters=64, kernel_size=4, strides=(2, 2), activation='relu', name='image_conv1')(image_inputs)
    image_layer = kr.layers.MaxPooling2D(pool_size=(4,4))(image_layer)
    image_layer = kr.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu', name='image_conv2')(image_layer)
    image_layer = kr.layers.MaxPooling2D(pool_size=(4,4))(image_layer)
    image_layer = kr.layers.Conv2D(filters=128, kernel_size=3, strides=(1, 1), activation='relu', name='image_conv3')(image_layer)
    image_layer = kr.layers.MaxPooling2D(pool_size=(2,2))(image_layer)
    image_layer = kr.layers.Flatten(name='flattened')(image_layer)
    # image_layer = kr.layers.Dense(256, activation='relu', name='image_dense1')(image_layer)
    image_layer = kr.layers.Dense(64, activation='relu', name='image_dense2')(image_layer)
    image_layer = kr.layers.Dense(1, activation='tanh', name='image_dense3')(image_layer)

    subPara_inputs = layers.Input(shape=env.observation_spec['subPara']['shape'], dtype=np.float, name='subPara')

    common = layers.concatenate([image_dense2, subPara_inputs])
    common = layers.Dense(128, activation="relu", name='common_dense1')(common)
    common = layers.Dense(128, activation="relu", name='common_dense1')(common)
    num_actions = env.action_spec['shape'][0]
    action = layers.Dense(num_actions, activation="softmax", name='common_output')(common)


    # if exitst, load previous model
    model = None
    modelDirPath = './savedModels/ReinforcementKeras_rawImage'
    if not os.path.isdir(modelDirPath):
        os.mkdir(modelDirPath)
    modelPath = f'{modelDirPath}/model.h5'
    if os.path.exists(modelPath):
        model = keras.models.load_model(modelPath)
    else:
        model = keras.Model(inputs=[image_inputs, subPara_inputs], outputs=action)
        # if no previous model, copy weights from DLAgent
        dlModelPath = '../DLAgents/continuousModel_rawImage.h5'
        if os.path.exists(dlModelPath):
            dlModel = keras.models.load_model(dlModelPath)
            for layerIndex in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
                model.layers[layerIndex].set_weights(dlModel.layers[layerIndex].get_weights())

    recordsPath = f"{modelDirPath}/record.pickle"
    records = []
    running_reward = 0.0
    if os.path.exists(recordsPath):
        with open(recordsPath, "rb") as file:
            records = pickle.load(file)
            running_reward = records[-1][2]


    # MARK: - Training

    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    huber_loss = keras.losses.Huber()
    action_probs_history = []
    rewards_history = []
    discounted_rewards_history = []
    correct_action_history = []
    gradient_descent_count = 0
    gamma = 0.95
    episodes_amount_needed_for_one_decent = 5
    entropy_beta = 1.0

    while running_reward < 2000:  # Run until solved
        episode_count_in_single_descent = 1
        with tf.GradientTape() as tape:
            while episode_count_in_single_descent <= episodes_amount_needed_for_one_decent:
                state_old = env.reset()
                state_new = None
                episode_reward = 0
                done = False
                while not done:
                    # env.render(); Adding this line would show the attempts
                    # of the agent in a pop up window.
                    # state_old = tf.convert_to_tensor(state_old)
                    _state = []
                    for state_element in state_old:
                        temp = tf.convert_to_tensor(state_element)
                        temp = tf.expand_dims(temp, axis=0)
                        _state.append(temp)
                    state_old = _state
                    # print(state_old)
                    # https://www.tensorflow.org/api_docs/python/tf/expand_dims
                    # state_old = tf.expand_dims(state_old, axis=0)

                    # Predict action probabilities and estimated future rewards
                    # from environment state
                    action_probs = model(state_old)

                    # Sample action from action probability distribution
                    action = np.random.choice(num_actions, p=np.squeeze(action_probs))
                    # action_probs_history.append(tf.math.log(action_probs[0, action]))
                    action_probs_history.append(action_probs[0, action])

                    # Apply the sampled action in our environment
                    state_new, reward, done, myAction = env.step(action)
                    # correct_action_history.append(myAction)
                    print(f"observation:")
                    print(state_old[0])
                    print(f"action_prob: {action_probs[0]}, action = {action}, reward = {reward}")
                    rewards_history.append(reward)
                    episode_reward += reward
                    state_old = state_new

                    if done:
                        env.simulatorDriver.backToMenu()
                        # calculate discounted rewards
                        discounted_sum = 0
                        for r in rewards_history[::-1]:
                            discounted_sum = r + gamma * discounted_sum
                            discounted_rewards_history.insert(0, discounted_sum)
                        # break
                        episode_count_in_single_descent += 1
                        # Update running reward to check condition for solving
                        running_reward = (1-gamma) * episode_reward + gamma * running_reward

            # correct_action_history = np.array(correct_action_history)
            # print(f"correct actions probabilities: {np.sum(correct_action_history == 0)/correct_action_history.shape[0]}, {np.sum(correct_action_history == 1)/correct_action_history.shape[0]}, {np.sum(correct_action_history == 2)/correct_action_history.shape[0]}, {np.sum(correct_action_history == 3)/correct_action_history.shape[0]}")

            # Calculate expected value from rewards
            # - At each timestep what was the total reward received after that timestep
            # - Rewards in the past are discounted by multiplying them with gamma
            # - These are the labels for our critic
            # returns = []
            # discounted_sum = 0
            # for r in rewards_history[::-1]:
            #     discounted_sum = r + gamma * discounted_sum
            #     returns.insert(0, discounted_sum)

            # Normalize
            # returns = np.array(returns)
            # returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            # returns = returns.tolist()
            discounted_rewards_history = np.array(discounted_rewards_history)
            discounted_rewards_history = (discounted_rewards_history - np.mean(discounted_rewards_history)) / (np.std(discounted_rewards_history) + eps)
            discounted_rewards_history = discounted_rewards_history.tolist()

            # Calculating loss values to update our network
            # history = zip(action_probs_history, critic_value_history, returns)
            history = zip(action_probs_history, discounted_rewards_history)
            actor_losses = []
            for prob, ret in history:
                # At this point in history, the critic estimated that we would get a
                # total reward = `value` in the future. We took an action with log probability
                # of `log_prob` and ended up recieving a total reward = `ret`.
                # The actor must be updated so that it predicts an action that leads to
                # high rewards (compared to critic's estimate) with high probability.
                log_prob = tf.math.log(prob)
                policy_loss = -log_prob * ret
                entropy_loss = ( prob * log_prob) * entropy_beta
                # print(f"policy_loss: {policy_loss}")
                # print(f"entropy_loss: {entropy_loss}")
                actor_losses.append(policy_loss + entropy_loss)  # actor loss

            # Backpropagation
            loss_value = sum(actor_losses)/len(actor_losses)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            records.append([gradient_descent_count, episode_reward, running_reward, loss_value])

            # Clear the loss and reward history
            action_probs_history.clear()
            rewards_history.clear()
            discounted_rewards_history.clear()
            correct_action_history = []


        # Log details
        gradient_descent_count += 1
        if gradient_descent_count % 1 == 0:
            model.save(modelPath)
            with open(recordsPath, "wb") as file:
                pickle.dump(records, file)
            # print(f"episode {gradient_descent_count}: reward = {episode_reward}")
            # print("running reward: {:.2f} at episode {}".format(running_reward, gradient_descent_count))
