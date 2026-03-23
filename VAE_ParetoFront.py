import bisect


class ParetoFront:
    def __init__(self):
        self.front = []  # Sorted list of (x, y), minimizing both

    def is_dominated(self, x, y):
        i = bisect.bisect_left(self.front, (x, -float('inf')))
        #print(" Proposed Pt:  (",x,", ",y,") :",i)

        # Check if dominated by existing point
        if len(self.front) == 0 or i==0:
            return False
        elif i < len(self.front) and self.front[i][1] <= y:
            #print("     Rejected: self.front[i][1] =",self.front[i][1], " <= ",y)
            return True
        elif i > 0 and self.front[i-1][1] <= y:
            #print("     Rejected: self.front[i-1][1] =",self.front[i-1][1], " <= ",y)
            return True
        return False
    
    def clear_front(self):
        self.front = []

    def add_points(self, x_pts, y_pts):

        for x, y in zip(x_pts, y_pts):
            #print("\nCurr Pts: ", self.front)
            if self.is_dominated(x, y):
                continue

            # Find insertion index
            i = bisect.bisect_left(self.front, (x, y))
            self.front.insert(i, (x, y))

            # Prune points to the right that are now dominated
            j = i + 1
            while j < len(self.front):
                if self.front[j][1] >= y:
                   del self.front[j]
                else:
                    break


    def get_front(self):
        return self.front


    def get_smooth_pareto_curve(self, num_iters=3):

            point_list = self.get_front()
            pareto_curve = list(point_list) 
            
            for curr_iter in range(num_iters):
                for ctr in range(1,len(pareto_curve)-1):

                    lft_pt = pareto_curve[ctr-1]
                    center_pt = pareto_curve[ctr]
                    rgt_pt = pareto_curve[ctr+1]
                
                    den = rgt_pt[0] - lft_pt[0]
                    if den==0:
                        continue
                    delta_yeq = (center_pt[0] - lft_pt[0])/den * (rgt_pt[1] - lft_pt[1])
                    
                    delta_y12 = (center_pt[1] - lft_pt[1])
                    
                    if delta_y12 < delta_yeq:
                        new_y = lft_pt[1] + delta_yeq
                        pareto_curve[ctr] = (pareto_curve[ctr][0], new_y)

            return pareto_curve
