import cv2
import numpy as np
import sys
sys.path.append("..")
import PathPlanning.utils as utils
from PathPlanning.planner import Planner


class PlannerRRT(Planner):
    def __init__(self, m, extend_len=20):
        super().__init__(m)
        self.extend_len = extend_len

    def _random_node(self, goal, shape):
        """
        Randomly choose a node.
        With 50% probability, return the goal node.
        Otherwise, sample a random point within the map dimensions.
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
        Find the nearest node in the tree to the sampled node.
        """
        min_dist = float("inf")
        min_node = None
        for n in self.ntree:
            dist = utils.distance(n, samp_node)
            if dist < min_dist:
                min_dist = dist
                min_node = n
        return min_node

    def _check_collision(self, n1, n2):
        """
        Check if the line between n1 and n2 collides with any obstacles.
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
        Steer from 'from_node' towards 'to_node' by a step length of 'extend_len'.
        Returns the new node and the distance traveled if the path is collision free;
        otherwise returns False, None.
        """
        vect = np.array(to_node) - np.array(from_node)
        v_len = np.hypot(vect[0], vect[1])
        v_theta = np.arctan2(vect[1], vect[0])
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

    def planning(self, start, goal, extend_len=None, img=None):
        """
        Perform RRT planning from start to goal.
        Optionally display the planning process on an image.
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

            # If the nearest node is within extend_len of the goal, consider it reached
            if utils.distance(near_node, goal) < extend_len:
                goal_node = near_node
                break

            # Draw the current tree if an image is provided
            if img is not None:
                for n in self.ntree:
                    if self.ntree[n] is None:
                        continue
                    node = self.ntree[n]
                    cv2.line(img, (int(n[0]), int(n[1])), (int(node[0]), int(node[1])), (0, 1, 0), 2)
                img_ = cv2.flip(img, 0)
                cv2.imshow("Path Planning", img_)
                k = cv2.waitKey(1)
                if k == 27:
                    break

        # Extract the path by backtracking from the goal node
        path = []
        n = goal_node
        while n is not None:
            path.insert(0, n)
            n = self.ntree[n]
        path.append(goal)
        return path
