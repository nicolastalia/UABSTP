class ReplayBuffer:
    def __init__(self):
        self.experiences = []
        self.pre_str = []
        self.visited = []
        self.index = 0
        self.priority = 0
        self.trained = []
        self.size = 0
    
    def add(self,experience):
        try:
            index = self.pre_str.index(str(experience.pre_state))
            self.visited[index] += 1
        except ValueError:           
            self.experiences.append(experience)
            self.pre_str.append(str(experience.pre_state))
            self.index += 1
            self.visited.append(1)
            self.trained.append(0)
            self.size += 1