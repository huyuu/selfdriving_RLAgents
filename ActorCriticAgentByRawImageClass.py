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

from SimulatorDriverClass import SimulatorDriver
from RawImageEnvironmentClass import RawImageEnvironment

# Reference: https://keras.io/examples/rl/actor_critic_cartpole/


if __name__ == '__main__':
    if platform == 'linux' or platform == 'linux2':
        mp.set_start_method('spawn')
    mp.freeze_support()

    # gamma = 0.99  # Discount factor for past rewards
    max_steps_per_episode = int(1e8)
    # env = gym.make("CartPole-v0")  # Create the environment
    # env.seed(seed)
    env = RawImageEnvironment()
    eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0


    # MARK: - Model Configuration


    # num_inputs = 4
    # num_actions = 4
    # inputs = layers.Input(shape=(num_inputs,))
    # common = layers.Dense(16, activation="relu")(inputs)
    # common = layers.Dense(16, activation="relu")(common)
    # action = layers.Dense(num_actions, activation="softmax")(common)
    # critic = layers.Dense(1)(common)

    # model = None
    # modelPath = './model.h5'
    # if os.path.exists(modelPath):
    #     model = keras.models.load_model(modelPath)
    # model = keras.Model(inputs=inputs, outputs=[action, critic])


    num_actions = 4
    imageInputs = layers.Input(shape=env.observation_spec['image']['shape'], name='image')
    imageLayers = layers.Conv2D(filters=32, kernel_size=8, strides=(4, 4), activation='relu')(imageInputs)
    imageLayers = layers.Conv2D(filters=32, kernel_size=8, strides=(4, 4), activation='relu')(imageLayers)
    imageLayers = layers.Flatten()(imageLayers)
    imageLayers = layers.Dense(16, activation='relu')(imageLayers)
    imageLayers = layers.Dense(8, activation='relu')(imageLayers)

    subParaInputs = layers.Input(shape=env.observation_spec['subPara']['shape'], name='subPara')
    subParaLayers = layers.Dense(8, activation='relu')(subParaInputs)
    subParaLayers = layers.Dense(8, activation='relu')(subParaInputs)

    common = layers.concatenate([imageLayers, subParaLayers])

    action = layers.Dense(16, activation="relu")(common)
    action = layers.Dense(num_actions, activation="softmax")(action)

    critic = layers.Dense(16, activation='relu')(common)
    critic = layers.Dense(1, activation='relu')(critic)

    model = None
    modelPath = './modelRawImage.h5'
    if os.path.exists(modelPath):
        model = keras.models.load_model(modelPath)
    model = keras.Model(inputs=[imageInputs, subParaInputs], outputs=[action, critic])


    # MARK: - Training

    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    huber_loss = keras.losses.Huber()
    action_probs_history = []
    critic_value_history = []
    rewards_history = []
    running_reward = 0
    episode_count = 0
    records = []
    gamma = 0.95
    epsilon_decay_episode = 200
    epsilon = 1.0
    epsilon_decay_rate = (epsilon - 0.05) / 200

    while running_reward < 2000:  # Run until solved
        state = env.reset()
        episode_reward = 0
        with tf.GradientTape() as tape:
            for timestep in range(1, max_steps_per_episode):
                # env.render(); Adding this line would show the attempts
                # of the agent in a pop up window.

                state_after = []
                for state_element in state:
                    temp = tf.convert_to_tensor(state_element)
                    temp = tf.expand_dims(temp, axis=0)
                    state_after.append(temp)
                # state = tf.convert_to_tensor(state)
                # https://www.tensorflow.org/api_docs/python/tf/expand_dims
                # state = tf.expand_dims(state, axis=0)

                # Predict action probabilities and estimated future rewards
                # from environment state
                action_probs, critic_value = model(state_after)

                critic_value_history.append(critic_value[0, 0])

                # Sample action from action probability distribution
                action = np.random.choice(num_actions, p=np.squeeze(action_probs))
                action_probs_history.append(tf.math.log(action_probs[0, action]))

                # Apply the sampled action in our environment
                state, reward, done = env.step(action)
                print(f"{datetime.now()}: ", action_probs, critic_value, reward)
                rewards_history.append(reward)
                episode_reward += reward

                if done:
                    env.simulatorDriver.backToMenu()
                    break

            # Update running reward to check condition for solving
            running_reward = (1-gamma) * episode_reward + gamma * running_reward

            records.append([episode_count, episode_reward, running_reward])
            with open("records.pickle", "wb") as file:
                pickle.dump(records, file)

            # Calculate expected value from rewards
            # - At each timestep what was the total reward received after that timestep
            # - Rewards in the past are discounted by multiplying them with gamma
            # - These are the labels for our critic
            returns = []
            discounted_sum = 0
            for r in rewards_history[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)

            # Normalize
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()

            # Calculating loss values to update our network
            history = zip(action_probs_history, critic_value_history, returns)
            actor_losses = []
            critic_losses = []
            for log_prob, value, ret in history:
                # At this point in history, the critic estimated that we would get a
                # total reward = `value` in the future. We took an action with log probability
                # of `log_prob` and ended up recieving a total reward = `ret`.
                # The actor must be updated so that it predicts an action that leads to
                # high rewards (compared to critic's estimate) with high probability.
                diff = ret - value
                actor_losses.append(-log_prob * diff)  # actor loss

                # The critic must be updated so that it predicts a better estimate of
                # the future rewards.
                critic_losses.append(
                    huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                )

            # Backpropagation
            if timestep != 1:
                loss_value = sum(actor_losses) + sum(critic_losses)
                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                # Clear the loss and reward history
                action_probs_history.clear()
                critic_value_history.clear()
                rewards_history.clear()

        # Log details
        episode_count += 1
        if episode_count % 10 == 0:
            model.save(modelPath)
            print(f"episode {episode_count}: reward = {episode_reward}")
            # print("running reward: {:.2f} at episode {}".format(running_reward, episode_count))
