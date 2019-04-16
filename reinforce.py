import sys
import gym
import pylab
import numpy as np
import datetime
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

EPISODES = 1000

class REINFORCEAgent:
    def __init__(self, state_size, action_size):

        self.render = False
        self.load_model = False

        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.hidden1, self.hidden2 = 24, 24

        self.model = self.build_model()

        self.states, self.actions, self.rewards = [], [], []

        if self.load_model:
            self.model.load_weights("./models/cartpole_reinforce.h5")

    def build_model(self):
        model = Sequential()
        model.add(Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(self.hidden2, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform'))
        model.summary()
        model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=self.learning_rate))
        return model

    def get_action(self, state):
        policy = self.model.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

    def train_model(self):
        episode_length = len(self.states)

        discounted_rewards = self.discount_rewards(self.rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        update_inputs = np.zeros((episode_length, self.state_size))
        advantages = np.zeros((episode_length, self.action_size))

        for i in range(episode_length):
            update_inputs[i] = self.states[i]
            advantages[i][self.actions[i]] = discounted_rewards[i]

        self.model.fit(update_inputs, advantages, epochs=1, verbose=0)
        self.states, self.actions, self.rewards = [], [], []

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = REINFORCEAgent(state_size, action_size)

    scores, episodes= [], []
    acc =np.zeros((3,EPISODES))
    for e in range(EPISODES):
        starttime=datetime.datetime.now()
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done or score == 499 else -100

            agent.append_sample(state, action, reward)

            score += reward
            state = next_state

            if done:
                agent.train_model()

                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                endtime=datetime.datetime.now()
                timetake=str(endtime-starttime)
                print("timetake",timetake)
                timetake=timetake[5:]
                #print("timetake",timetake)
                #time[0, e] = float(timetake)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./graphs/cartpole_reinforce.png")
                print("episode:", e, "  score:", score, "time: ", timetake)
                acc[0,e] = e
                acc[1,e] = score
                acc[2,e] = timetake
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    np.savetxt("Time_reinforce.csv", acc, delimiter=',')
                    sys.exit()

        if e % 50 == 0:
            agent.model.save_weights("./models/cartpole_reinforce.h5")