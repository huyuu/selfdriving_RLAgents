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
from ProcessedImageEnvironmentClass import ProcessedImageEnvironment
from FixPolicyEnvironmentClass import FixPolicyEnvironment

# Reference: https://keras.io/examples/rl/actor_critic_cartpole/


if __name__ == '__main__':
    if platform == 'linux' or platform == 'linux2':
        mp.set_start_method('spawn')
    mp.freeze_support()

    # gamma = 0.99  # Discount factor for past rewards
    max_steps_per_episode = int(1e8)
    # env = gym.make("CartPole-v0")  # Create the environment
    # env.seed(seed)
    env = FixPolicyEnvironment()
    eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0


    # MARK: - Model Configuration

    num_inputs = 4
    num_actions = 4
    inputs = layers.Input(shape=(num_inputs,))
    common2 = layers.Dense(128, activation="relu")(inputs)
    action = layers.Dense(num_actions, activation="softmax")(common2)
    critic = layers.Dense(1)(common2)


    # num_actions = 3
    # inputs = layers.Input(shape=(1, 160, 320, 3))
    # common = layers.Conv2D(filters=32, kernel_size=8, strides=(4, 4), activation='relu')(inputs)
    # common = layers.Conv2D(filters=32, kernel_size=8, strides=(4, 4), activation='relu')(common)
    # common = layers.Conv2D(filters=32, kernel_size=8, strides=(4, 4), activation='relu')(common)
    # common = layers.flatten()(common)
    # common = layers.Dense(64)(common)
    # common = layers.Dense(8)(common)
    #
    # action = layers.Dense(16, activation="relu")(common)
    # action = layers.Dense(num_actions, activation="softmax")(action)
    #
    # critic = layers.Dense(8)(common)
    # critic = layers.Dense(1)(critic)

    model = None
    if os.path.exists('./model.h5'):
        model = keras.models.load_model('./model.h5')
    model = keras.Model(inputs=inputs, outputs=[action, critic])


    # MARK: - Training

    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    huber_loss = keras.losses.Huber()
    action_probs_history = []
    critic_value_history = []
    rewards_history = []
    discounted_rewards_history = []
    running_reward = 0
    gradient_descent_count = 0
    records = []
    gamma = 0.1
    episodes_amount_needed_for_one_decent = 3

    while running_reward < 2000:  # Run until solved
        episode_count_in_single_descent = 1
        with tf.GradientTape() as tape:
            while episode_count_in_single_descent <= episodes_amount_needed_for_one_decent:
                state = env.reset()
                episode_reward = 0
                done = False
                while not done:
                    # env.render(); Adding this line would show the attempts
                    # of the agent in a pop up window.

                    state = tf.convert_to_tensor(state)
                    # https://www.tensorflow.org/api_docs/python/tf/expand_dims
                    state = tf.expand_dims(state, axis=0)

                    # Predict action probabilities and estimated future rewards
                    # from environment state
                    action_probs, critic_value = model(state)
                    critic_value_history.append(critic_value[0, 0])

                    # Sample action from action probability distribution
                    action = np.random.choice(num_actions, p=np.squeeze(action_probs))
                    action_probs_history.append(tf.math.log(action_probs[0, action]))

                    # Apply the sampled action in our environment
                    state, reward, done, myAction = env.step(action)
                    print(f"{datetime.now()}: center-d: {state[0]}, speed: {state[3]}, myAction = {myAction}, reward = {reward}", action_probs[0], critic_value[0, 0])
                    rewards_history.append(reward)
                    episode_reward += reward

                    if done:
                        # env.simulatorDriver.backToMenu()
                        # calculate discounted rewards
                        discounted_sum = 0
                        for r in rewards_history[::-1]:
                            discounted_sum = r + gamma * discounted_sum
                            discounted_rewards_history.insert(0, discounted_sum)
                        # break
                episode_count_in_single_descent += 1
            # Update running reward to check condition for solving
            running_reward = (1-gamma) * episode_reward + gamma * running_reward
            records.append([gradient_descent_count, episode_reward, running_reward])
            with open("records.pickle", "wb") as file:
                pickle.dump(records, file)

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
            # discounted_rewards_history = np.array(discounted_rewards_history)
            # discounted_rewards_history = (discounted_rewards_history - np.mean(discounted_rewards_history)) / (np.std(discounted_rewards_history) + eps)
            # discounted_rewards_history = discounted_rewards_history.tolist()

            # Calculating loss values to update our network
            # history = zip(action_probs_history, critic_value_history, returns)
            history = zip(action_probs_history, critic_value_history, discounted_rewards_history)
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
            loss_value = sum(actor_losses) + sum(critic_losses)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Clear the loss and reward history
            action_probs_history.clear()
            critic_value_history.clear()
            rewards_history.clear()
            discounted_rewards_history.clear()


        # Log details
        gradient_descent_count += 1
        if gradient_descent_count % 10 == 0:
            model.save('./model.h5')
            print(f"episode {gradient_descent_count}: reward = {episode_reward}")
            # print("running reward: {:.2f} at episode {}".format(running_reward, gradient_descent_count))
