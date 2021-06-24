import tensorflow as tf
from tensorflow import keras
import numpy as np
import copy

from FixPolicyEnvironmentClass import FixPolicyEnvironment



def pretrainModel(model, env):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()],
    )

    for data_set_count in range(3):
        xs, ys = getExperience(env)
        x_train = xs[:int(len(xs)*0.8)]
        y_train = ys[:int(len(ys)*0.8)]
        x_test = xs[-int(len(xs)*0.8):]
        y_test = ys[-int(len(ys)*0.8):]

        # print(xs[:10, :])
        # print(ys[:10, :])
        model.fit(
            x_train,
            y_train,
            batch_size=16,
            epochs=5,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            validation_data=(x_test, y_test),
        )

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    x_test, y_test = getExperience(env)
    results = model.evaluate(x_test, y_test, batch_size=128)
    print("test loss, test acc:", results)
    # y_predict = model.predict(x_test[:20, :])
    # for i in range(20):
    #     print(f"x_       : {x_test[i, :]}")
    #     print(f"y_predict: {y_predict[i, :]}")
    #     print(f"y_correct: {y_test[i, :]}")
    #     print("")



def getExperience(env, episods=10):
    episode_count = 1
    observation_history = []
    label_history = []
    action_candidates = [0, 0, 0, 0]
    while episode_count <= 10:
        observation_old = env.reset()
        isDone = False
        while not isDone:
            observation_new, reward, isDone, correct_action = env.step(action=0)
            observation_history.append(observation_old.tolist())
            new_label = copy.deepcopy(action_candidates)
            new_label[correct_action] = 1
            label_history.append(new_label)
            observation_old = observation_new
        episode_count += 1
    return np.array(observation_history), np.array(label_history)
