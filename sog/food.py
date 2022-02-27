"""Module for logic and data relevant to food items
"""
import enum
import random

# food and hunger evolution should be closely connected
MAX_HUNGER = 1
class Nutrition(enum.Enum):
    LOW = MAX_HUNGER / 10
    MID = MAX_HUNGER / 5
    HIGH = MAX_HUNGER / 2
    @staticmethod
    def randomize():
        return random.choice([Nutrition.LOW, Nutrition.MID, Nutrition.HIGH])

    @staticmethod
    def get_max():
        return Nutrition.HIGH

class Food:
    """A food item
    """

    def __init__(self, nutrition=None):
        if nutrition is None:
            nutrition = Nutrition.randomize()
        self.nutrition = nutrition

    def get_reward_value(self):
        return self.nutrition.value

    def get_nutrition_value(self):
        return self.nutrition.value

    def __str__(self):
        return "nutr=" + str(self.nutrition)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, o):
        return self.nutrition == o.nutrition

    def __hash__(self):
        return hash(self.nutrition)
