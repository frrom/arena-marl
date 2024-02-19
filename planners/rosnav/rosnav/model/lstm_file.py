#%% LSTM network
import numpy as np
import torch
import stable_baselines3
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

class CustomLSTMPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=64, n_hidden=64, reuse=False, **_kwargs):
        super(CustomLSTMPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, n_hidden, reuse, layer_norm=True, feature_extraction="mlp")

        with self.sess.graph.as_default():
            self.lstm = torch.nn.LSTM(input_size=self.feature_dim, hidden_size=n_lstm, num_layers=1)
            self.critic_linear = torch.nn.Linear(n_lstm, 1)
            self.actor_linear = torch.nn.Linear(n_lstm, ac_space.shape[0])

    def forward(self, x):
        x = self.feature_extraction(x)
        x, _ = self.lstm(x.unsqueeze(0))
        x = x.squeeze(0)
        value = self.critic_linear(x)
        action = self.actor_linear(x)
        return value, action

# Create a sample dataset
data = np.random.rand(100, 10)
labels = np.random.randint(0, 2, 100)

# Define the model
model = stable_baselines3.PPO(CustomLSTMPolicy, DummyVecEnv([lambda: data]), verbose=1, tensorboard_log="./tensorboard/")

# Train the model on the sample data
model.learn(total_timesteps=1000, callback=CheckpointCallback(save_freq=1000, save_path="./models/"))

# Predict the classes of the test data
predictions = model.predict(data)

#%% second approach

import gym
import torch
import stable_baselines3
from stable_baselines3.common.policies import LstmPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

# Define the custom LSTM network architecture
class CustomLSTM(torch.nn.Module):
    def __init__(self, observation_space, action_space):
        super(CustomLSTM, self).__init__()

        self.lstm = torch.nn.LSTM(input_size=observation_space.shape[0],
                                  hidden_size=32,
                                  num_layers=2,
                                  batch_first=True)

        self.fc = torch.nn.Linear(in_features=32,
                                  out_features=action_space.n)

    def forward(self, x, hidden_state=None):
        x, hidden_state = self.lstm(x, hidden_state)
        x = self.fc(x[:, -1, :])
        return x, hidden_state

# Define the custom policy class, using the custom LSTM network
class CustomPolicy(LstmPolicy):
    def __init__(self, observation_space, action_space, *args, **kwargs):
        super(CustomPolicy, self).__init__(observation_space, action_space,
                                           model_class=CustomLSTM,
                                           *args, **kwargs)

# Create an environment to use for training
env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])

# Train the custom policy
model = CustomPolicy(env.observation_space, env.action_space)
model.learn(total_timesteps=10000, log_interval=100)


#%% PI controller

import control
import numpy as np

# Define the motor parameters and state-space model
R = 0.1 # Resistance (ohms)
L = 0.01 # Inductance (henries)
Kt = 0.01 # Torque constant (Nm/A)
Kb = 0.01 # Back-EMF constant (V/(rad/s))
J = 0.01 # Inertia (kg m^2)

# Define the state-space matrices
A = np.array([[-R/L, -Kb/L], [Kt/J, -Kt*Kb/J/L]])
B = np.array([[1/L], [0]])
C = np.array([[0, 1]])
D = np.array([[0]])

# Create the state-space model
sys = control.ss(A, B, C, D)

# Define the desired steady-state response
des_resp = control.tf(1, [1, 0])

# Define the disturbance
disturbance = np.ones(100) # Constant disturbance

# Design the PI controller
Kp = 1.0 # Proportional gain
Ki = 0.1 # Integral gain
controller = control.tf([Kp, Ki], [1, 0])

# Close the loop around the system and controller
cl_sys = control.feedback(controller*sys, 1)

# Simulate the closed-loop response to a step input
T, Y, X = control.forced_response(cl_sys, disturbance, T=np.linspace(0, 10, 100))

#%% optimization model-based

import numpy as np
import scipy
import scipy.optimize as optimize
import matplotlib.pyplot as plt

# Define state space model
A = np.array([[0, 1], [-1, -2]])
B = np.array([[0], [1]])
C = np.array([[1, 0]])
D = np.array([[0]])
sys = (A, B, C, D)

# Define target performance parameters
omega = 1.0 # Resonant frequency
zeta = 0.7 # Damping ratio

# Define PI controller
def pi_controller(Kp, Ki, x, t, omega, zeta):
    # Define error signal
    e = np.sin(omega*t) - x[0]
    # Define control signal
    u = Kp*e + Ki*np.trapz(e, t)
    # Return control signal
    return u

# Define objective function for optimization
def obj_fun(x, *args):
    Kp, Ki = x
    t, omega, zeta, sys = args
    A, B, C, D = sys
    x0 = np.array([0, 0])
    # Simulate response of system with PI controller
    x = scipy.integrate.odeint(lambda x, t: A @ x + B * pi_controller(Kp, Ki, x, t, omega, zeta), x0, t)
    # Calculate objective value as sum of squared errors
    obj_val = np.sum((np.sin(omega*t) - x[:,0])**2)
    # Return objective value
    return obj_val

# Optimize PI controller gains
x0 = np.array([0.1, 0.1]) # Initial guess for gains
bounds = [(0, None), (0, None)] # Bounds for optimization
t = np.linspace(0, 10, 1000) # Time vector for simulation
args = (t, omega, zeta, sys)
res = optimize.minimize(obj_fun, x0, args=args, bounds=bounds, method='L-BFGS-B')

# Plot response of optimized PI controller
Kp, Ki = res.x
x = scipy.integrate.odeint(lambda x, t: A @ x + B * pi_controller(Kp, Ki, x, t, omega, zeta), x0, t)
plt.plot(t, x[:,0], label='Position')
plt.plot(t, np.sin(omega*t), label='Reference')
plt.legend()
plt.show()

#%% optimization frequency response analysis

import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from scipy.signal import bode, TransferFunction

# Define state space model
A = np.array([[0, 1], [-1, -2]])
B = np.array([[0], [1]])
C = np.array([[1, 0]])
D = np.array([[0]])
sys = (A, B, C, D)

# Define target performance parameters
omega = 1.0 # Resonant frequency
zeta = 0.7 # Damping ratio

# Define PI controller
def pi_controller(Kp, Ki, x, t, omega, zeta):
    # Define error signal
    e = np.sin(omega*t) - x[0]
    # Define control signal
    u = Kp*e + Ki*np.trapz(e, t)
    # Return control signal
    return u

# Define objective function for optimization
def obj_fun(x, *args):
    Kp, Ki = x
    omega, zeta, sys = args
    A, B, C, D = sys
    # Define transfer function for system with PI controller
    num = [Kp + Ki*omega, Kp*omega]
    den = [1, 2*zeta*omega, omega**2]
    T = TransferFunction(num, den)
    w, mag, phase = bode(T, omega)
    # Calculate objective value as difference between desired and actual damping ratio
    obj_val = np.abs(zeta - mag[0]/20*np.log10(np.sqrt(2)))
    # Return objective value
    return obj_val

# Optimize PI controller gains
x0 = np.array([0.1, 0.1]) # Initial guess for gains
bounds = [(0, None), (0, None)] # Bounds for optimization
args = (omega, zeta, sys)
res = optimize.minimize(obj_fun, x0, args=args, bounds=bounds, method='L-BFGS-B')

# Plot frequency response of optimized PI controller
Kp, Ki = res.x
num = [Kp + Ki*omega, Kp*omega]
den = [1, 2*zeta*omega, omega**2]
T = TransferFunction(num, den)
w, mag, phase = bode(T, omega)
plt.figure()
plt.semilogx(w, mag)
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Magnitude (dB)')
plt.figure()
plt.semilogx(w, phase)
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Phase (deg)')
plt.show()


#%% matlab code

% Define the motor parameters and state-space model
R = 0.1; % Resistance (ohms)
L = 0.01; % Inductance (henries)
Kt = 0.01; % Torque constant (Nm/A)
Kb = 0.01; % Back-EMF constant (V/(rad/s))
J = 0.01; % Inertia (kg m^2)

% Define the state-space matrices
A = [-R/L, -Kb/L; Kt/J, -Kt*Kb/J/L];
B = [1/L; 0];
C = [0, 1];
D = 0;

% Create the state-space model
sys = ss(A, B, C, D);

% Define the desired steady-state response
des_resp = tf(1, [1, 0]);

% Define the disturbance
disturbance = ones(100, 1); % Constant disturbance

% Design the PI controller
Kp = 1.0; % Proportional gain
Ki = 0.1; % Integral gain
controller = tf([Kp, Ki], [1, 0]);

% Close the loop around the system and controller
cl_sys = feedback(controller*sys, 1);

% Simulate the closed-loop response to a step input
t = linspace(0, 10, 100);
[y, t, x] = lsim(cl_sys, disturbance, t);

