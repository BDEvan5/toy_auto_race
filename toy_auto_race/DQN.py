
#Hyperparameters
GAMMA = 0.95
LEARNING_RATE = 0.001
h_size = 512
BATCH_SIZE = 64

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.99


class ReplayBufferDQN():
    def __init__(self):
        self.buffer = deque(maxlen=MEMORY_SIZE)
    
    def add(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        # todo: move the tensor bit to the agent file, just return lists for the moment.
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self, obs_space, action_space):
        super(Qnet, self).__init__()
        
        self.fc1 = nn.Linear(obs_space, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, action_space)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.exploration_rate = EXPLORATION_MAX

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      

class DQN:
    def __init__(self, obs_space, action_space, name):
        self.action_space = action_space
        self.obs_space = obs_space
        self.name = name

        self.model = None
        self.target = None
        self.last_out = None

        self.update_steps = 0

    def act(self, obs):
        if random.random() < self.model.exploration_rate:
            return [random.randint(0, self.action_space-1)]
        else: 
            a = self.greedy_action(obs)
            return a

    def get_out(self, obs):
        obs_t = torch.from_numpy(obs).float()
        out = self.model.forward(obs_t)

        return out.detach().numpy()

    def greedy_action(self, obs):
        obs_t = torch.from_numpy(obs).float()
        out = self.model.forward(obs_t)
        self.last_out = out
        a = [out.argmax().item()]
        return a

    def train_modification(self, buffer):
        """This is for single examples with no future values"""
        n_train = 5
        for i in range(n_train):
            if buffer.size() < BATCH_SIZE:
                return
            s, a, r, s_p, done = buffer.memory_sample(BATCH_SIZE)

            target = r.detach().float()
            a = torch.squeeze(a, dim=-1)

            q_vals = self.model.forward(s)
            q_a = q_vals.gather(1, a)

            loss = F.mse_loss(q_a, target)

            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()
        self.update_networks()

    def train_episodes(self, buffer):
        n_train = 1
        for i in range(n_train):
            if buffer.size() < BATCH_SIZE:
                return
            s, a, r, s_p, done = buffer.memory_sample(BATCH_SIZE)

            next_values = self.target.forward(s_p)
            max_vals = torch.max(next_values, dim=1)[0].reshape((BATCH_SIZE, 1))
            g = torch.ones_like(done) * GAMMA
            q_update = r + g * max_vals * done
            q_vals = self.model.forward(s)
            a = torch.squeeze(a, dim=-1)
            q_a = q_vals.gather(1, a)
            loss = F.mse_loss(q_a, q_update.detach())

            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()

        self.update_networks()

    def update_networks(self):
        self.update_steps += 1
        if self.update_steps % 100 == 1: # every 20 eps or so
            self.target.load_state_dict(self.model.state_dict())
        if self.update_steps % 30 == 1:
            self.model.exploration_rate *= EXPLORATION_DECAY 
            self.model.exploration_rate = max(EXPLORATION_MIN, self.model.exploration_rate)

    def save(self, directory="./dqn_saves"):
        filename = self.name
        torch.save(self.model, '%s/%s_model.pth' % (directory, filename))
        torch.save(self.target, '%s/%s_target.pth' % (directory, filename))
        print(f"Saved Agent: {self.name}")

    def load(self, directory="./dqn_saves"):
        filename = self.name
        self.model = torch.load('%s/%s_model.pth' % (directory, filename))
        self.target = torch.load('%s/%s_target.pth' % (directory, filename))
        print(f"Loaded Agent: {self.name}")

    def try_load(self, load=True):
        if load:
            try:
                self.load()
            except:
                print(f"Unable to load model")
                self.create_model()
        else:
            self.create_model()

    def create_model(self):
        self.model = Qnet(self.obs_space, self.action_space)
        self.target = Qnet(self.obs_space, self.action_space)
        print(f"Created new model")


