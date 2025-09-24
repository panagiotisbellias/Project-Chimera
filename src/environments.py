# environments.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseEnvironment(ABC):
    """
    Defines the basic interface (contract) for any single-agent environment.
    This class specifies the minimum capabilities an environment must have.
    """
    
    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """
        Resets the environment to its initial state and returns the first observation.
        This is called at the beginning of an episode.

        Returns:
            Dict[str, Any]: A dictionary representing the initial state of the environment.
        """
        pass

    @abstractmethod
    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies the given action, advances the environment by one step, and returns the new state.
        
        Args:
            action (Dict[str, Any]): The action to be performed by the agent.

        Returns:
            Dict[str, Any]: The new state of the environment after the action is applied.
        """
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Returns a copy of the current environment state without advancing the environment.
        Used for observation.

        Returns:
            Dict[str, Any]: A copy of the current state.
        """
        pass


class BaseMultiAgentEnvironment(BaseEnvironment):
    """
    An extended interface for environments where multiple agents interact simultaneously.
    The 'step' method is specialized to accept a list of actions from multiple agents.
    """
    
    @abstractmethod
    def step(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applies a list of actions, one for each agent.
        Advances the environment by one step and returns a list of new states for each agent.

        Args:
            actions (List[Dict[str, Any]]): A list containing the action of each agent.

        Returns:
            List[Dict[str, Any]]: A list of new states for each agent after the actions are applied.
        """
        pass