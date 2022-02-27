import math
import random
import numpy as np
import enum


class WorldTile:
    """Gridworld cell type enumeration
    """
    EMPTY = 0
    AGENT = 1
    FOOD = 2

    NUM_CELL_TYPES = 3


class World:
    """Helper class for a SOG gridworld"""

    def __init__(self, name, size):
        self.name = str(name)
        self.size_nd = size
        self.size_1d = math.prod(size)
        self.array = np.ndarray(self.size_1d, dtype=np.int32)
        self.foods = {}
        self.agent_position = None
        self.agents = {}

    def get_one_hot_food_location(self, criterion='reward', choice='max'):
        """Return one-hot representation of food locations.
        Args:
            selection (str): How to select a food item.
              max_reward:
              max_nutrition:
              random:

        """
        if choice == 'max':
            choice_func = max
        elif choice == 'min':
            choice_func = min
        elif choice == 'random':
            pass

        if criterion == 'reward':
            eval_func = lambda x: x[1].get_reward_value()
        elif criterion == 'nutrition':
            eval_func = lambda x: x[1].get_nutrition_value()
        else:
            raise ValueError(f"Undefined food selection criterion: {criterion}")

        if self.foods:
            if choice == 'random':
                position = random.choice(self.foods.keys())
            else:
            # pick most rewarding food item
                position, _ = choice_func(self.foods.items(), key=eval_func)
            vec = self.get_food_one_hot(position)
            return vec
        return None


    def place_agent(self, agent, placement_strategy="random"):
        """Place agent in the world

        Args:
            placement_strategy (str, optional): _description_. Defaults to "random".
        """
        if placement_strategy == "random":
            self.agent_position = random.choice(range(self.size_1d))
        elif placement_strategy == "midpoint":
            self.agent_position = self.size_1d // 2
        else:
            raise ValueError(f"Undefined agent placement {placement_strategy}")
        self.agents[self.agent_position] = agent
        self.array[self.agent_position] = WorldTile.AGENT
        self.agent = agent

    def get_food_view(self, view='reward'):
        """
        Return a world array view listing food values

        Args:
            view (str): Whether to return a view of 'reward' or 'nutrition' of food items.
        """
        arr = np.zeros_like(self.array, dtype=np.float32)
        for pos, food_item in self.foods.items():
            if view == 'reward':
                value = food_item.get_reward_value()
            elif view == 'nutrition':
                value = food_item.get_nutrition_value()
            else:
                raise ValueError(f"Undefined world array food view: {view}")
            arr[pos] = value
        return arr


    def __repr__(self):
        return f"world:{self.agent.name} | size:{self.array.shape}"

    def place_food(self, food, location=None):
        """Insert food into a world location

        Args:
            location (int, optional): Food location to place to. Defaults to None.
        """
        if location is None:
            # randomize
            location = np.random.choice(self.get_empty())
        self.foods[location] = food
        self.array[location] = WorldTile.FOOD

    def can_insert_food(self):
        """Whether the array is completely filled with food
           Returns:
            bool: The boolean value
        """
        return np.any(self.array != WorldTile.FOOD)

    def get_empty(self):
        """Return empty world positions
           Returns:
            np.ndarray: Position indexes
        """
        return np.where(self.array == WorldTile.EMPTY)[0]

    def is_empty(self):
        """Whether the array is completely empty
           Returns:
            bool: The boolean value
        """
        return np.all(self.array == WorldTile.EMPTY)

    def has_food(self, loc=None):
        """Checks whether the world contains food

        Args:
            loc (int, optional): Specific location to look for food. If None, checks all locations. Defaults to None.

        Returns:
            The boolean result.
        """
        if loc is None:
            return np.any(self.array == WorldTile.FOOD)
        return loc in self.foods

    def clear_food(self):
        self.array = np.full_like(self.array, WorldTile.EMPTY)
        # throws read-only error with RlLib
        # self.array.fill(WorldTile.EMPTY)
        self.foods = {}

    def reset(self):
        """Reset the world
        """
        self.clear_food()
        self.agent_position = None

    def render(self, movement=None):
        """Print the world
        """
        cells = []
        for x in range(self.size_1d):
            cell = ""
            if x == self.agent_position:
                cell += f"A{self.agents[x].index}"
            if x in self.foods:
                cell += f"F{self.foods[x].get_nutrition_value()}"
            if not cell:
                cell = "<>"
            pad = " " * ((6 - len(cell))//2)
            cell = f"{pad}{cell}{pad}"
            # if len(cell) < 6:
            #     cell = f"   {cell}   "
            if x == movement:
                cell = f"({cell[1:-1]})"
            cells.append(cell)
        return cells

    def attempt_set_food_consumed(self, location):
        """Set food at a location as consumed, if it exists

        Args:
            location (int): Location for food

        Returns:
            A consumed food item or None, if no food exists in that location
        """
        for loc, food in self.foods.items():
            if loc == location:
                # remove food from the world
                del self.foods[loc]
                self.array[location] = WorldTile.EMPTY
                return food
        return None

    def get_food_one_hot(self, food_loc=None):
        """Get one-hot vector of location of input food

        Args:
            food_loc (int): Linear 1D location index. If None, assume all food locations. Defaults to None.
        """
        index = np.zeros_like(self.array)
        if food_loc is None:
            food_loc = list(self.foods.keys())
        if not isinstance(food_loc, list):
            food_loc = [food_loc]
        for loc in food_loc:
            if self.has_food(loc):
                index[loc] = 1.0
        return index
