import os
import cv2
import numpy
import typing
import pygame
import tensorflow

import gymnasium as gym
from gymnasium import spaces

import view_image
import view_regression
import image_noise_addition

ObsType = typing.TypeVar("ObsType")
ActType = typing.TypeVar("ActType")

class EnvAsimov1DNoise(gym.Env):
    # Overrides Env.metadata
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config, render_mode="human"):
        self.config = config
        self.noise_type = config["noise"]["choice"]
        self.render_mode = render_mode

        # Overrides Env.action_space
        self.action_space = spaces.Dict({
            "noise_variance_compensation": spaces.Box(low=-self.config["noise"]["settings"][self.noise_type]["max_variance"], high=self.config["noise"]["settings"][self.noise_type]["max_variance"], shape=(1,), dtype=numpy.float32),
        })

        # Overrides Env.observation_space
        self.observation_space = spaces.Dict({
            "noise_certainty": spaces.Box(low=0, high=2*self.config["noise"]["settings"][self.noise_type]["max_variance"], shape=(1,), dtype=numpy.float32),
            "deltas_noise_certainty": spaces.Box(low=-2*self.config["noise"]["settings"][self.noise_type]["max_variance"], high=2 * self.config["noise"]["settings"][self.noise_type]["max_variance"], shape=(self.config["rl"]["observation_delta_length"],), dtype=numpy.float32),
            "noise_variance_compensation": spaces.Box(low=-self.config["noise"]["settings"][self.noise_type]["max_variance"], high=self.config["noise"]["settings"][self.noise_type]["max_variance"], shape=(1,), dtype=numpy.float32),
            "deltas_noise_variance_compensation": spaces.Box(low=-2*self.config["noise"]["settings"][self.noise_type]["max_variance"], high=2 * self.config["noise"]["settings"][self.noise_type]["max_variance"], shape=(self.config["rl"]["observation_delta_length"],), dtype=numpy.float32),
        })

        # Further initializations
        noise_type_settings = config["noise"]["settings"][self.noise_type]
        self.noise_classifier_variances = [round(variance, 1) for variance in numpy.linspace(0.0, noise_type_settings["max_variance"], noise_type_settings["num_of_variances"])]
        noise_classifier_path = os.path.join(self.config["sl"]["storage_dir"], self.noise_type, self.config["data"]["colormode"], "models", config["sl"]["model"], "model.h5")
        self.noise_classifier = tensorflow.keras.models.load_model(noise_classifier_path)
        original_data_path = os.path.join(config["data"]["original_dir"], "test")
        self.image_files = [os.path.join(original_data_path, f) for f in os.listdir(original_data_path) if not f.startswith(".")]
        self.viewerImage = None
        self.viewerNoise = None
        self.seed()

    def enable_visualizations(self):
        self.viewerImage = view_image.ViewImage()
        self.viewerNoise = view_regression.ViewRegression(self.noise_type, self.config["noise"]["settings"][self.noise_type]["max_variance"], pow(self.config["noise"]["settings"][self.noise_type]["done_threshold"], 2))

    # Overrides method Env.seed(self, seed=None)
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    # Overrides abstract method Env.step(self, action)
    def step(self, action: ActType) -> typing.Tuple[ObsType, float, bool, dict]:
        # Update state
        previous_noise_certainty = self.noise_certainty
        previous_noise_variance_compensation = self.noise_variance_compensation
        self.noise_variance_compensation = self.noise_variance_multiplier * action["noise_variance_compensation"]
        if self.config["rl"]["relative_actions"]:
            self.noise_variance_compensation += previous_noise_variance_compensation
        self.noise_variance_compensation = min(max(self.noise_variance_compensation, -self.config["noise"]["settings"][self.noise_type]["max_variance"]), self.config["noise"]["settings"][self.noise_type]["max_variance"])
        self._compute_image_with_noise()

        # Compute return values
        self.deltas_noise_certainty.pop(0)
        self.deltas_noise_certainty.append(self.noise_certainty - previous_noise_certainty)
        self.deltas_noise_variance_compensation.pop(0)
        self.deltas_noise_variance_compensation.append(self.noise_variance_compensation - previous_noise_variance_compensation)
        observation = self._observed_state()
        reward = 0. - pow(abs(self.noise_variance_offset + self.noise_variance_compensation), 0.5)
        done = bool(reward > -self.config["noise"]["settings"][self.noise_type]["done_threshold"])
        info = {}

        if done:
            self.done_count += 1
        done = bool(self.done_count >= self.config["rl"]["done_count_threshold"])

        return observation, reward, done, False, info

    # Overrides abstract method Env.reset(self)
    def reset(self, seed = None, options=None) -> ObsType:
        # Select image and degree of misalignment
        if self.config["data"]["colormode"] == "grayscale":
            image = cv2.imread(self.image_files[self.np_random.integers(0, len(self.image_files)-1)], cv2.IMREAD_GRAYSCALE)
        elif self.config["data"]["colormode"] == "rgb":
            image = cv2.imread(self.image_files[self.np_random.integers(0, len(self.image_files)-1)], flags=cv2.IMREAD_COLOR)
        self.base_image = cv2.resize(image, (self.config["data"]["shape"]["height"], self.config["data"]["shape"]["width"]), interpolation = cv2.INTER_AREA)
        self.noise_variance_offset = self.np_random.uniform(low=-self.config["noise"]["settings"][self.noise_type]["max_variance"], high=self.config["noise"]["settings"][self.noise_type]["max_variance"])
        self.noise_variance_multiplier = self.np_random.uniform(low=self.config["rl"]["min_variance_multiplier"], high=self.config["rl"]["max_variance_multiplier"])
        print("Reset to random offset: " + str(self.noise_variance_offset) + " / multiplier: " + str(self.noise_variance_multiplier))
        if self.viewerImage:
            self.viewerNoise.reset(-self.noise_variance_offset)
        self.done_count = 0

        self.deltas_noise_certainty = []
        self.deltas_noise_variance_compensation = []

        self.noise_variance_compensation = self.np_random.uniform(low=-self.config["noise"]["settings"][self.noise_type]["max_variance"], high=self.config["noise"]["settings"][self.noise_type]["max_variance"])
        self._compute_image_with_noise()

        for i in range(self.config["rl"]["observation_delta_length"]):
            previous_noise_certainty = self.noise_certainty
            previous_noise_variance_compensation = self.noise_variance_compensation
            self._compute_image_with_noise()
            self.deltas_noise_certainty.append(self.noise_certainty - previous_noise_certainty)
            self.deltas_noise_variance_compensation.append(self.noise_variance_compensation - previous_noise_variance_compensation)

        # Compute return value
        observation = self._observed_state()
        return observation, {}

    def _observed_state(self):
        return {
            "noise_certainty": numpy.array([self.noise_certainty], numpy.float32),
            "deltas_noise_certainty": numpy.array(self.deltas_noise_certainty, numpy.float32),
            "noise_variance_compensation": numpy.array([self.noise_variance_compensation], numpy.float32),
            "deltas_noise_variance_compensation": numpy.array(self.deltas_noise_variance_compensation, numpy.float32),
        }

    def _compute_image_with_noise(self):
        # Compute noise parameters
        self.noise_variance = abs(self.noise_variance_offset + self.noise_variance_compensation)  # abs to avoid complex numbers (square root of negative number)

        summed_noise_certainty = 0
        for iteration in range(self.config["rl"]["evaluate_average_count"]):
            # Apply noise parameters
            if self.noise_type == "blur":
                self.image_with_noise = image_noise_addition.Noise.blur(self.base_image, self.noise_variance)
            elif self.noise_type == "gaussian":
                if self.config["data"]["colormode"] == "grayscale":
                    self.image_with_noise = image_noise_addition.Noise.gauss(self.base_image, 0, self.noise_variance)
                else:
                    self.image_with_noise = image_noise_addition.Noise.gauss_multicolor(self.base_image, 0, self.noise_variance)

            # Apply classifier
            img_array = tensorflow.keras.preprocessing.image.img_to_array(self.image_with_noise)
            img_array = tensorflow.expand_dims(img_array, 0)
            prediction = self.noise_classifier.predict(img_array)[0]

            # Interpret classifier
            noise_certainty = 0
            for index in range(len(self.noise_classifier_variances)):
                noise_certainty += self.noise_classifier_variances[index] * prediction[index]
            noise_certainty = min(max(noise_certainty, 0), 2 * self.config["noise"]["settings"][self.noise_type]["max_variance"])
            summed_noise_certainty += noise_certainty

        self.noise_certainty = summed_noise_certainty / self.config["rl"]["evaluate_average_count"]

    # Overrides abstract method Env.render(self, mode="human")
    def render(self, mode="human"):
        if mode == "human":
            # Render to the current display or terminal and return nothing.
            if self.viewerImage:
                self.viewerImage.apply(cv2.cvtColor(self.image_with_noise, cv2.COLOR_GRAY2RGB))
                self.viewerNoise.apply(self.noise_variance_compensation, self.noise_certainty)
            return None
        if mode == "rgb_array":
            # Return an numpy.ndarray with shape (x, y, 3), representing RGB values for an x-by-y pixel image.
            print("Noise Variance (offset + compensation): " + str(self.noise_variance) + " / Certainty: " + str(self.noise_certainty))
            return cv2.cvtColor(self.image_with_noise, cv2.COLOR_GRAY2RGB)
        if mode == "ansi":
            # Return a string (str) or StringIO.StringIO containing a terminal-style text representation.
            return str(self.image_with_noise)
        raise NotImplementedError

    # Used by gym.utils.play
    def get_keys_to_action(self):
        return {
            ():                              {"noise_variance_compensation": 0},
            (pygame.K_LEFT,):                {"noise_variance_compensation": -0.05},
            (pygame.K_RIGHT,):               {"noise_variance_compensation": +0.05},
            (pygame.K_LEFT, pygame.K_RIGHT): {"noise_variance_compensation": 0},
        }
    
# Wrapper to deal with the limited support for Dict action_space in stable-baselines3
class TupleDictWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

        # Overrides Wrapper.action_space
        self.action_space = gym.spaces.Box(
            low=numpy.array([
                    env.action_space.spaces["noise_variance_compensation"].low[0],
                ], dtype=numpy.float32),
            high=numpy.array([
                    env.action_space.spaces["noise_variance_compensation"].high[0],
                ], dtype=numpy.float32),
            dtype=numpy.float32
        )

    @staticmethod
    def action_tuple_to_dict(t: numpy.ndarray) -> ActType:
        return {
            "noise_variance_compensation": t[0],
        }

    @staticmethod
    def action_dict_to_tuple(d: ActType) -> numpy.ndarray:
        return numpy.array([
            d["noise_variance_compensation"],
        ], numpy.float32)

    # Overrides abstract method Wrapper.step(self, action)
    def step(self, action_tuple: numpy.ndarray) -> typing.Tuple[numpy.ndarray, float, bool, dict]:
        action_dict = self.action_tuple_to_dict(action_tuple)
        return self.env.step(action_dict)

    # Used by gym.utils.play
    def get_keys_to_action(self):
        return {k: self.action_dict_to_tuple(v) for k, v in self.env.unwrapped.get_keys_to_action().items()}