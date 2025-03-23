import cv2
import sys
from queue import PriorityQueue

sys.path.append("..")
import PathPlanning.utils as utils
from PathPlanning.planner import Planner

class PlannerAStar(Planner):
    def __init__(self, m, inter=10):
        super().__init__(m)
        self.inter = inter
        self.initialize()

    def initialize(self):
        """Initialize the A* planner."""
        self.queue = PriorityQueue()
        self.parent = {}
        self.h = {}  # Heuristic: estimated distance from node to goal
        self.g = {}  # Cost: distance from start to node
        self.goal_node = None

    def _check_collision(self, n1, n2):
        """
        Check if the line between n1 and n2 collides with any obstacles.

        Args:
            n1 (tuple): Start node (x, y).
            n2 (tuple): End node (x, y).

        Returns:
            bool: True if there is a collision; False otherwise.
        """
        n1_int = utils.pos_int(n1)
        n2_int = utils.pos_int(n2)
        line = utils.Bresenham(n1_int[0], n2_int[0], n1_int[1], n2_int[1])
        for pt in line:
            # Check if the pixel is in collision (obstacle if value < 0.5)
            if self.map[int(pt[1]), int(pt[0])] < 0.5:
                return True
        return False

    def planning(self, start=(100, 200), goal=(375, 520), inter=None, img=None):
        """
        Plan a path from start to goal using the A* algorithm.

        Args:
            start (tuple): Starting coordinate (x, y).
            goal (tuple): Goal coordinate (x, y).
            inter (int, optional): Step interval. Defaults to self.inter.
            img (optional): Image for visualization (unused here).

        Returns:
            list: List of nodes representing the path.
        """
        if inter is None:
            inter = self.inter

        # Convert coordinates to integers.
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))

        # Initialize the planner.
        self.initialize()
        self.queue.put((0, start))
        self.parent[start] = None
        self.g[start] = 0
        self.h[start] = utils.distance(start, goal)

        while not self.queue.empty():
            _, current_node = self.queue.get()
            if current_node == goal or utils.distance(current_node, goal) < inter:
                self.goal_node = current_node
                break

            for direction in [(0, 1), (0, -1), (1, 0), (-1, 0), 
                              (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                next_node = (
                    current_node[0] + direction[0] * inter,
                    current_node[1] + direction[1] * inter,
                )
                # Check boundaries and collision.
                if (next_node[1] < 0 or next_node[1] >= self.map.shape[0] or
                        next_node[0] < 0 or next_node[0] >= self.map.shape[1] or
                        self._check_collision(current_node, next_node)):
                    continue

                new_cost = self.g[current_node] + utils.distance(current_node, next_node)
                if next_node not in self.g or new_cost < self.g[next_node]:
                    self.g[next_node] = new_cost
                    self.h[next_node] = utils.distance(next_node, goal)
                    self.queue.put((self.g[next_node] + self.h[next_node], next_node))
                    self.parent[next_node] = current_node

        # Extract the path from the goal node to the start node.
        path = []
        p = self.goal_node
        if p is None:
            return path

        while p is not None:
            path.insert(0, p)
            p = self.parent[p]

        # Ensure the goal is explicitly included.
        if path[-1] != goal:
            path.append(goal)
        return path
