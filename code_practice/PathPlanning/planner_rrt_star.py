import cv2
import numpy as np
import sys
sys.path.append("..")
import PathPlanning.utils as utils
from PathPlanning.planner import Planner

class PlannerRRTStar(Planner):
    """
    RRT* Planner for path planning.
    
    This class builds an RRT* tree by randomly sampling nodes,
    steering toward them, and performing re-parenting and re-wiring (TODO).
    """
    
    def __init__(self, m, extend_len=20):
        """
        Initialize the PlannerRRTStar.
        
        Args:
            m: The map (obstacle grid).
            extend_len (int): The step size for extending the tree.
        """
        super().__init__(m)
        self.extend_len = extend_len

    def _random_node(self, goal, shape):
        """
        Generate a random node. With 50% probability, return the goal.
        
        Args:
            goal (tuple): The goal coordinates.
            shape (tuple): The dimensions of the map.
            
        Returns:
            tuple: A randomly sampled node (x, y).
        """
        r = np.random.choice(2, 1, p=[0.5, 0.5])
        if r == 1:
            return (float(goal[0]), float(goal[1]))
        else:
            rx = float(np.random.randint(int(shape[1])))
            ry = float(np.random.randint(int(shape[0])))
            return (rx, ry)

    def _nearest_node(self, samp_node):
        """
        Find the nearest node in the tree to the given sample.
        
        Args:
            samp_node (tuple): The sampled node (x, y).
            
        Returns:
            tuple: The nearest node in the tree.
        """
        min_dist = 99999  # Alternatively, float('inf') can be used
        min_node = None
        for n in self.ntree:
            dist = utils.distance(n, samp_node)
            if dist < min_dist:
                min_dist = dist
                min_node = n
        return min_node

    def _check_collision(self, n1, n2):
        """
        Check if the straight-line path between two nodes collides with an obstacle.
        
        Args:
            n1 (tuple): Start node (x, y).
            n2 (tuple): End node (x, y).
            
        Returns:
            bool: True if a collision is detected; False otherwise.
        """
        n1_ = utils.pos_int(n1)
        n2_ = utils.pos_int(n2)
        line = utils.Bresenham(n1_[0], n2_[0], n1_[1], n2_[1])
        for pts in line:
            if self.map[int(pts[1]), int(pts[0])] < 0.5:
                return True
        return False

    def _steer(self, from_node, to_node, extend_len):
        """
        Steer from 'from_node' toward 'to_node' by a fixed extension length.
        
        Args:
            from_node (tuple): Starting node (x, y).
            to_node (tuple): Target node (x, y).
            extend_len (float): Maximum extension length.
            
        Returns:
            tuple: (new_node, distance) if valid; otherwise (False, None).
        """
        vect = np.array(to_node) - np.array(from_node)
        v_len = np.hypot(vect[0], vect[1])
        v_theta = np.arctan2(vect[1], vect[0])
        # Limit the extension to the distance between nodes
        if extend_len > v_len:
            extend_len = v_len
        new_node = (
            from_node[0] + extend_len * np.cos(v_theta),
            from_node[1] + extend_len * np.sin(v_theta)
        )
        # Check boundaries and collision
        if (new_node[1] < 0 or new_node[1] >= self.map.shape[0] or
                new_node[0] < 0 or new_node[0] >= self.map.shape[1] or
                self._check_collision(from_node, new_node)):
            return False, None
        else:
            return new_node, utils.distance(new_node, from_node)

    def _near_nodes(self, new_node, radius=100):
        """
        Returns a list of nodes in the tree that are within a given radius of new_node.
        
        Args:
            new_node (tuple): The node for which to find nearby nodes.
            radius (float): The radius within which to search for neighbors. (default: 100)
        
        Returns:
            list: Nearby nodes (tuples).
        """
        near_nodes = []
        for n in self.ntree:
            if utils.distance(n, new_node) < radius:
                near_nodes.append(n)
        return near_nodes

    def _reparent(self, new_node, near_nodes):
        """
        Re-parent the new_node if a neighbor offers a lower cost path.
        
        Args:
            new_node (tuple): The newly added node.
            near_nodes (list): A list of nearby nodes.
        """
        for node in near_nodes:
            if not self._check_collision(node, new_node):
                new_cost = self.cost[node] + utils.distance(node, new_node)
                if new_cost < self.cost[new_node]:
                    self.ntree[new_node] = node
                    self.cost[new_node] = new_cost

    def _rewire(self, new_node, near_nodes):
        """
        Rewire the nearby nodes if the new_node provides a lower cost path.
        
        Args:
            new_node (tuple): The newly added node.
            near_nodes (list): A list of nearby nodes.
        """
        for node in near_nodes:
            if node == self.ntree[new_node]:
                continue # Skip the parent node
            if not self._check_collision(new_node, node):
                new_cost = self.cost[new_node] + utils.distance(new_node, node)
                if new_cost < self.cost[node]:
                    self.ntree[node] = new_node
                    self.cost[node] = new_cost

    def planning(self, start, goal, extend_len=None, img=None):
        """
        Plan a path from start to goal using RRT*.
        
        Args:
            start (tuple): Starting coordinates (x, y).
            goal (tuple): Goal coordinates (x, y).
            extend_len (float, optional): Step length for tree extension.
            img (numpy.ndarray, optional): Image for visualization.
            
        Returns:
            list: The path from start to goal as a list of nodes.
        """
        if extend_len is None:
            extend_len = self.extend_len

        self.ntree = {start: None}
        self.cost = {start: 0}
        goal_node = None

        for it in range(20000):
            print("\r", it, len(self.ntree), end="")
            samp_node = self._random_node(goal, self.map.shape)
            near_node = self._nearest_node(samp_node)
            new_node, cost = self._steer(near_node, samp_node, extend_len)
            if new_node is not False:
                self.ntree[new_node] = near_node
                self.cost[new_node] = cost + self.cost[near_node]
            else:
                continue

            # Check if the new node is close enough to the goal.
            if utils.distance(new_node, goal) < extend_len:
                goal_node = new_node
                break

            # Re-parent and re-wire the tree
            near_nodes = self._near_nodes(new_node)
            self._reparent(new_node, near_nodes)
            self._rewire(new_node, near_nodes)

            # Visualization: draw the tree and the latest node
            if img is not None:
                for n in self.ntree:
                    if self.ntree[n] is None:
                        continue
                    node = self.ntree[n]
                    cv2.line(img, (int(n[0]), int(n[1])), (int(node[0]), int(node[1])), (0, 1, 0), 1)
                # Highlight the new node
                img_ = img.copy()
                cv2.circle(img_, utils.pos_int(new_node), 5, (0, 0.5, 1), 3)
                img_ = cv2.flip(img_, 0)
                cv2.imshow("Path Planning", img_)
                k = cv2.waitKey(1)
                if k == 27:
                    break

        # Backtrack from the goal node to extract the final path.
        path = []
        n = goal_node
        while n is not None:
            path.insert(0, n)
            n = self.ntree[n]
        path.append(goal)
        return path
