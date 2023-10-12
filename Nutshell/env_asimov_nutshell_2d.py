import os
import cv2
import numpy
import typing
import pygame
import tensorflow

import gymnasium as gym
from gymnasium import spaces

import view_2D
import view_image
import view_regression
import image_noise_addition

ObsType = typing.TypeVar("ObsType")
ActType = typing.TypeVar("ActType")


class EnvAsimov2DNoise(gym.Env):
    # Overrides Env.metadata
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config, render_mode="human"):
        self.config = config
        self.render_mode = render_mode

        # Overrides Env.action_space
        self.action_space = spaces.Dict({
            "gaussian_variance_compensation": spaces.Box(low=-self.config["noise"]["settings"]["gaussian"]["max_variance"], high=self.config["noise"]["settings"]["gaussian"]["max_variance"], shape=(1,), dtype=numpy.float32),
            "blur_variance_compensation":     spaces.Box(low=-self.config["noise"]["settings"]["blur"]["max_variance"], high=self.config["noise"]["settings"]["blur"]["max_variance"], shape=(1,), dtype=numpy.float32),
        })

        # Overrides Env.observation_space
        self.observation_space = spaces.Dict({
            "gaussian_certainty": spaces.Box(low=0, high=2*self.config["noise"]["settings"]["gaussian"]["max_variance"], shape=(1,), dtype=numpy.float32),
            "deltas_gaussian_certainty": spaces.Box(low=-2*self.config["noise"]["settings"]["gaussian"]["max_variance"], high=2*self.config["noise"]["settings"]["gaussian"]["max_variance"], shape=(self.config["rl"]["observation_delta_length"],), dtype=numpy.float32),
            "gaussian_variance_compensation": spaces.Box(low=-self.config["noise"]["settings"]["gaussian"]["max_variance"], high=self.config["noise"]["settings"]["gaussian"]["max_variance"], shape=(1,), dtype=numpy.float32),
            "deltas_gaussian_variance_compensation": spaces.Box(low=-2*self.config["noise"]["settings"]["gaussian"]["max_variance"], high=2*self.config["noise"]["settings"]["gaussian"]["max_variance"], shape=(self.config["rl"]["observation_delta_length"],), dtype=numpy.float32),
            "blur_certainty": spaces.Box(low=0, high=2*self.config["noise"]["settings"]["blur"]["max_variance"], shape=(1,), dtype=numpy.float32),
            "deltas_blur_certainty": spaces.Box(low=-2*self.config["noise"]["settings"]["blur"]["max_variance"], high=2*self.config["noise"]["settings"]["blur"]["max_variance"], shape=(self.config["rl"]["observation_delta_length"],), dtype=numpy.float32),
            "blur_variance_compensation": spaces.Box(low=-self.config["noise"]["settings"]["blur"]["max_variance"], high=self.config["noise"]["settings"]["blur"]["max_variance"], shape=(1,), dtype=numpy.float32),
            "deltas_blur_variance_compensation": spaces.Box(low=-2*self.config["noise"]["settings"]["blur"]["max_variance"], high=2*self.config["noise"]["settings"]["blur"]["max_variance"], shape=(self.config["rl"]["observation_delta_length"],), dtype=numpy.float32),
        })

        # Further initializations 
        noise_type_settings = config["noise"]["settings"]
        self.noise_classifier_variances = [round(variance, 1) for variance in numpy.linspace(0.0, noise_type_settings["gaussian"]["max_variance"], noise_type_settings["gaussian"]["num_of_variances"])]
        noise_classifier_path = os.path.join(self.config["sl"]["storage_dir"], "gaussian", self.config["data"]["colormode"], "models", config["sl"]["model"], "model.h5")
        self.noise_classifier = tensorflow.keras.models.load_model(noise_classifier_path)
        
        self.blur_classifier_variances = [round(variance, 1) for variance in numpy.linspace(0.0, noise_type_settings["blur"]["max_variance"], noise_type_settings["blur"]["num_of_variances"])]
        blur_classifier_path = os.path.join(self.config["sl"]["storage_dir"], "blur", self.config["data"]["colormode"], "models", config["sl"]["model"], "model.h5")
        self.blur_classifier = tensorflow.keras.models.load_model(blur_classifier_path)
        
        original_data_path = os.path.join(config["data"]["original_dir"], "test")
        self.image_files = [os.path.join(original_data_path, f) for f in os.listdir(original_data_path) if not f.startswith(".")]
        self.viewerImage = None
        self.viewerGaussian = None
        self.viewerBlur = None
        self.viewer2D = None
        self.seed()

    def enable_visualizations(self):
        self.viewerImage = view_image.ViewImage()
        self.viewerGaussian = view_regression.ViewRegression("Gaussian", self.config["noise"]["settings"]["gaussian"]["max_variance"], pow(self.config["noise"]["settings"]["gaussian"]["done_threshold"], 2))
        self.viewerBlur = view_regression.ViewRegression("Blur", self.config["noise"]["settings"]["blur"]["max_variance"], pow(self.config["noise"]["settings"]["blur"]["done_threshold"], 2))
        self.viewer2D = view_2D.View2D("Gaussian/Blur",
                                       "Gaussian", self.config["noise"]["settings"]["gaussian"]["max_variance"], pow(self.config["noise"]["settings"]["gaussian"]["done_threshold"], 2),
                                       "Blur", self.config["noise"]["settings"]["blur"]["max_variance"], pow(self.config["noise"]["settings"]["blur"]["done_threshold"], 2))

    # Overrides method Env.seed(self, seed=None)
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    # Overrides abstract method Env.step(self, action)
    def step(self, action: ActType) -> typing.Tuple[ObsType, float, bool, dict]:
        # Update state
        previous_gaussian_certainty = self.gaussian_certainty
        previous_gaussian_variance_compensation = self.gaussian_variance_compensation
        self.gaussian_variance_compensation = self.gaussian_variance_multiplier * action["gaussian_variance_compensation"]
        if self.config["rl"]["relative_actions"]:
            self.gaussian_variance_compensation += previous_gaussian_variance_compensation
        self.gaussian_variance_compensation = min(max(self.gaussian_variance_compensation, -self.config["noise"]["settings"]["gaussian"]["max_variance"]), self.config["noise"]["settings"]["gaussian"]["max_variance"])

        previous_blur_certainty = self.blur_certainty
        previous_blur_variance_compensation = self.blur_variance_compensation
        self.blur_variance_compensation = self.blur_variance_multiplier * action["blur_variance_compensation"]
        if self.config["rl"]["relative_actions"]:
            self.blur_variance_compensation += previous_blur_variance_compensation
        self.blur_variance_compensation = min(max(self.blur_variance_compensation, -self.config["noise"]["settings"]["blur"]["max_variance"]), self.config["noise"]["settings"]["blur"]["max_variance"])

        self._compute_image_with_noise()

        # Compute return values
        self.deltas_gaussian_certainty.pop(0)
        self.deltas_gaussian_certainty.append(self.gaussian_certainty - previous_gaussian_certainty)
        self.deltas_gaussian_variance_compensation.pop(0)
        self.deltas_gaussian_variance_compensation.append(self.gaussian_variance_compensation - previous_gaussian_variance_compensation)

        self.deltas_blur_certainty.pop(0)
        self.deltas_blur_certainty.append(self.blur_certainty - previous_blur_certainty)
        self.deltas_blur_variance_compensation.pop(0)
        self.deltas_blur_variance_compensation.append(self.blur_variance_compensation - previous_blur_variance_compensation)

        observation = self._observed_state()
        gaussian_reward = 0. - pow(abs(self.gaussian_variance_offset + self.gaussian_variance_compensation), 0.5)
        blur_reward = 0. - pow(abs(self.blur_variance_offset + self.blur_variance_compensation), 0.5)
        reward = gaussian_reward + blur_reward
        done = bool(gaussian_reward > -self.config["noise"]["settings"]["gaussian"]["done_threshold"]) and bool(blur_reward > -self.config["noise"]["settings"]["blur"]["done_threshold"])
        info = {}

        if done:
            self.done_count += 1
        done = bool(self.done_count >= self.config["rl"]["done_count_threshold"])

        return observation, reward, done, False, info

    # Overrides abstract method Env.reset(self)
    def reset(self, seed = None, options = None) -> ObsType:
        # Select image and degree of misalignment
        if self.config["data"]["colormode"] == "grayscale":
            image = cv2.imread(self.image_files[self.np_random.integers(0, len(self.image_files)-1)], cv2.IMREAD_GRAYSCALE)
        elif self.config["data"]["colormode"] == "rgb":
            image = cv2.imread(self.image_files[self.np_random.integers(0, len(self.image_files)-1)], flags=cv2.IMREAD_COLOR)
        self.base_image = cv2.resize(image, (self.config["data"]["shape"]["height"], self.config["data"]["shape"]["width"]), interpolation = cv2.INTER_AREA)
        self.gaussian_variance_offset = self.np_random.uniform(low=-self.config["noise"]["settings"]["gaussian"]["max_variance"], high=self.config["noise"]["settings"]["gaussian"]["max_variance"])
        self.gaussian_variance_multiplier = self.np_random.uniform(low=self.config["rl"]["min_variance_multiplier"], high=self.config["rl"]["max_variance_multiplier"])
        self.blur_variance_offset = self.np_random.uniform(low=-self.config["noise"]["settings"]["blur"]["max_variance"], high=self.config["noise"]["settings"]["blur"]["max_variance"])
        self.blur_variance_multiplier = self.np_random.uniform(low=self.config["rl"]["min_variance_multiplier"], high=self.config["rl"]["max_variance_multiplier"])
        print("[Noise] Reset to random offset: " + str(self.gaussian_variance_offset) + " / multiplier: " + str(self.gaussian_variance_multiplier))
        print("[Blur] Reset to random offset: " + str(self.blur_variance_offset) + " / multiplier: " + str(self.blur_variance_multiplier))
        if self.viewerImage:
            self.viewerGaussian.reset(-self.gaussian_variance_offset)
            self.viewerBlur.reset(-self.blur_variance_offset)
            self.viewer2D.reset(-self.gaussian_variance_offset, -self.blur_variance_offset)
        self.done_count = 0

        self.deltas_gaussian_certainty = []
        self.deltas_gaussian_variance_compensation = []

        self.deltas_blur_certainty = []
        self.deltas_blur_variance_compensation = []

        self.gaussian_variance_compensation = self.np_random.uniform(low=-self.config["noise"]["settings"]["gaussian"]["max_variance"], high=self.config["noise"]["settings"]["gaussian"]["max_variance"])
        self.blur_variance_compensation = self.np_random.uniform(low=-self.config["noise"]["settings"]["blur"]["max_variance"], high=self.config["noise"]["settings"]["blur"]["max_variance"])
        self._compute_image_with_noise()

        for i in range(self.config["rl"]["observation_delta_length"]):
            previous_gaussian_certainty = self.gaussian_certainty
            previous_gaussian_variance_compensation = self.gaussian_variance_compensation
            previous_blur_certainty = self.blur_certainty
            previous_blur_variance_compensation = self.blur_variance_compensation
            self._compute_image_with_noise()
            self.deltas_gaussian_certainty.append(self.gaussian_certainty - previous_gaussian_certainty)
            self.deltas_gaussian_variance_compensation.append(self.gaussian_variance_compensation - previous_gaussian_variance_compensation)
            self.deltas_blur_certainty.append(self.blur_certainty - previous_blur_certainty)
            self.deltas_blur_variance_compensation.append(self.blur_variance_compensation - previous_blur_variance_compensation)

        # Compute return value
        observation = self._observed_state()
        return observation, {}

    def _observed_state(self):
        return {
            "gaussian_certainty": numpy.array([self.gaussian_certainty], numpy.float32),
            "deltas_gaussian_certainty": numpy.array(self.deltas_gaussian_certainty, numpy.float32),
            "gaussian_variance_compensation": numpy.array([self.gaussian_variance_compensation], numpy.float32),
            "deltas_gaussian_variance_compensation": numpy.array(self.deltas_gaussian_variance_compensation, numpy.float32),
            "blur_certainty": numpy.array([self.blur_certainty], numpy.float32),
            "deltas_blur_certainty": numpy.array(self.deltas_blur_certainty, numpy.float32),
            "blur_variance_compensation": numpy.array([self.blur_variance_compensation], numpy.float32),
            "deltas_blur_variance_compensation": numpy.array(self.deltas_blur_variance_compensation, numpy.float32),
        }

    def _compute_image_with_noise(self):
        # Compute noise parameters
        self.gaussian_variance = abs(self.gaussian_variance_offset + self.gaussian_variance_compensation)  # abs to avoid complex numbers (square root of negative number)
        self.blur_variance = abs(self.blur_variance_offset + self.blur_variance_compensation)  # abs to avoid complex numbers (square root of negative number)

        summed_gaussian_certainty = 0
        summed_blur_certainty = 0
        for iteration in range(self.config["rl"]["evaluate_average_count"]):
            # Apply noise parameters
            self.image_with_noise = image_noise_addition.Noise.blur(self.base_image, self.blur_variance)
            if self.config["data"]["colormode"] == "grayscale":
                self.image_with_noise = image_noise_addition.Noise.gauss(self.image_with_noise, 0, self.gaussian_variance)
            else:
                self.image_with_noise = image_noise_addition.Noise.gauss_multicolor(self.image_with_noise, 0, self.gaussian_variance)

            # Apply classifiers
            img_array = tensorflow.keras.preprocessing.image.img_to_array(self.image_with_noise)
            img_array = tensorflow.expand_dims(img_array, 0)
            noise_prediction = self.noise_classifier.predict(img_array)[0]
            blur_prediction = self.blur_classifier.predict(img_array)[0]

            # Interpret classifier
            gaussian_certainty = 0
            for index in range(len(self.noise_classifier_variances)):
                gaussian_certainty += self.noise_classifier_variances[index] * noise_prediction[index]
            gaussian_certainty = min(max(gaussian_certainty, 0), 2 * self.config["noise"]["settings"]["gaussian"]["max_variance"])
            summed_gaussian_certainty += gaussian_certainty

            blur_certainty = 0
            for index in range(len(self.blur_classifier_variances)):
                blur_certainty += self.blur_classifier_variances[index] * blur_prediction[index]
            blur_certainty = min(max(blur_certainty, 0), 2 * self.config["noise"]["settings"]["blur"]["max_variance"])
            summed_blur_certainty += blur_certainty

        self.gaussian_certainty = summed_gaussian_certainty / self.config["rl"]["evaluate_average_count"]
        self.blur_certainty = summed_blur_certainty / self.config["rl"]["evaluate_average_count"]

    # Overrides abstract method Env.render(self, mode="human")
    def render(self, mode="human"):
        if mode == "human":
            # Render to the current display or terminal and return nothing.
            if self.viewerImage:
                self.viewerImage.apply(cv2.cvtColor(self.image_with_noise, cv2.COLOR_GRAY2RGB))
                self.viewerGaussian.apply(self.gaussian_variance_compensation, self.gaussian_certainty)
                self.viewerBlur.apply(self.blur_variance_compensation, self.blur_certainty)
                self.viewer2D.apply(self.gaussian_variance_compensation, self.blur_variance_compensation)
            return None
        if mode == "rgb_array":
            # Return an numpy.ndarray with shape (x, y, 3), representing RGB values for an x-by-y pixel image.
            print("[Noise] Variance (offset + compensation): " + str(self.gaussian_variance) + " / Certainty: " + str(self.gaussian_certainty))
            print("[Blur] Variance (offset + compensation): " + str(self.blur_variance) + " / Certainty: " + str(self.blur_certainty))
            return cv2.cvtColor(self.image_with_noise, cv2.COLOR_GRAY2RGB)
        if mode == "ansi":
            # Return a string (str) or StringIO.StringIO containing a terminal-style text representation.
            return str(self.image_with_noise)
        raise NotImplementedError

    # Used by gym.utils.play
    def get_keys_to_action(self):
        return {
            ():                {"gaussian_variance_compensation": 0,
                                "blur_variance_compensation": 0},
            (pygame.K_LEFT,):  {"gaussian_variance_compensation": -0.05,
                                "blur_variance_compensation": 0},
            (pygame.K_RIGHT,): {"gaussian_variance_compensation": +0.05,
                                "blur_variance_compensation": 0},
            (pygame.K_DOWN,):  {"gaussian_variance_compensation": 0,
                                "blur_variance_compensation": -0.05},
            (pygame.K_UP,):    {"gaussian_variance_compensation": 0,
                                "blur_variance_compensation": +0.05},
        }


# Wrapper to deal with the limited support for Dict action_space in stable-baselines3
class TupleDictWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

        # Overrides Wrapper.action_space
        self.action_space = spaces.Box(
            low=numpy.array([
                    env.action_space.spaces["gaussian_variance_compensation"].low[0],
                    env.action_space.spaces["blur_variance_compensation"].low[0],
                ], dtype=numpy.float32),
            high=numpy.array([
                    env.action_space.spaces["gaussian_variance_compensation"].high[0],
                    env.action_space.spaces["blur_variance_compensation"].high[0],
                ], dtype=numpy.float32),
            dtype=numpy.float32
        )

    @staticmethod
    def action_tuple_to_dict(t: numpy.ndarray) -> ActType:
        return {
            "gaussian_variance_compensation": t[0],
            "blur_variance_compensation": t[1],
        }

    @staticmethod
    def action_dict_to_tuple(d: ActType) -> numpy.ndarray:
        return numpy.array([
            d["gaussian_variance_compensation"],
            d["blur_variance_compensation"],
        ], numpy.float32)

    # Overrides abstract method Wrapper.step(self, action)
    def step(self, action_tuple: numpy.ndarray) -> typing.Tuple[numpy.ndarray, float, bool, dict]:
        action_dict = self.action_tuple_to_dict(action_tuple)
        return self.env.step(action_dict)

    # Used by gym.utils.play
    def get_keys_to_action(self):
        return {k: self.action_dict_to_tuple(v) for k, v in self.env.get_keys_to_action().items()}
