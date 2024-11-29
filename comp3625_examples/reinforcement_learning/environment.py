from gymnasium import Env
import random

from gymnasium.core import RenderFrame


class TwoChoiceTaskWithContext(Env[int, int]):
    """
    A super-simple task, in which the agent is shown a context cue & must choose to press button 0 or 1.
    The agent must learn that when the context cue is "on", the button 0 has a reward probability of 75%
    compared to 25% for button 1. When the cue is "off", the probabilities are reversed.
    """

    def __init__(self):
        self.cue = 1
        self.p = [0.25, 0.75]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[int, dict]:
        """ reset the environment. The complex signature is for compatibility with the Gym API"""
        self.cue = 1
        return self.cue, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        """
        execute a choice in the environment
        :param action: action number to perform (button 0 or 1)
        :return: new state, reward, and default values for terminated, truncated, and info. The last three are present
        only for compatibility with the Gym API
        """

        # compute reward
        reward_probability = abs(self.cue - self.p[action])
        reward = 1 if random.random() < reward_probability else -1

        # randomize the context signal for next time
        self.cue = random.randint(0, 1)

        # return results
        return self.cue, reward, False, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """this method present only for compatibility with the Gym API"""
        pass
