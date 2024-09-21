import random as rand
import numpy as np
import copy
import util as u


# probabilities
PROB_BLOCKED = 0.28 # probability that a cell in a maze is blocked
PROB_GHOST_PHASE_THROUGH = 0.5 # probability that ghost moves into blocked cell

# cell values
EMPTY = 0
AGENT = 1
GOAL = 2
GHOST = 3
WALL = 4
WALL_WITH_GHOST = 5

# blocked cells
BLOCKED_AGENT = {GHOST, WALL, WALL_WITH_GHOST}
BLOCKED_GHOST = {GOAL}


class MazeObj:
	def __init__(self, maze, agent, goal, ghosts):
		self.maze = maze
		self.agent = agent
		self.goal = goal
		self.ghosts = ghosts


# ============================== MAZE GENERATION ===============================
def set_ghosts(num_ghosts, maze):
	ghosts = []
	while len(ghosts) < num_ghosts:
		# ghosts can spawn anywhere except on the agent or the goal
		r = rand.randint(0, len(maze) - 1)
		c = rand.randint(0, len(maze) - 1)
		if maze[r][c] == AGENT or maze[r][c] == GOAL:
			continue
		
		# empty cell -> ghost
		if maze[r][c] == EMPTY:
			maze[r][c] = GHOST
		# wall -> wall_with_ghost
		elif maze[r][c] == WALL:
			maze[r][c] = WALL_WITH_GHOST

		ghosts.append((r, c))
	return ghosts


def validate_maze(agent, goal, maze):
	return u.A_star(agent, goal, maze, BLOCKED_AGENT)


def _generate_maze(size):
	# create matrix
	maze = np.zeros((size, size), np.int8)

	# fill matrix
	for row in range(size):
		for col in range(size):
			blocked = rand.randint(1, 100) / 100
			if blocked <= PROB_BLOCKED:
				maze[row][col] = WALL
	
	# set agent
	agent = (0, 0)
	maze[agent[0]][agent[1]] = AGENT

	# set goal
	goal = (size - 1, size - 1)
	maze[goal[0]][goal[1]] = GOAL

	return agent, goal, maze


def generate_maze(size, num_ghosts):
	agent, goal, maze = _generate_maze(size)

	# validate maze
	while not validate_maze(agent, goal, maze):
		agent, goal, maze = _generate_maze(size)

	# set ghosts after validating maze
	ghosts = set_ghosts(num_ghosts, maze)

	Maze = MazeObj(maze, agent, goal, ghosts)

	return Maze
# ==============================================================================


# ============================== FUNCTIONALITY =================================
# moves the agent to the specified cell and returns new position of the agent
def move_agent(cell, agent, maze):
	maze[agent[0]][agent[1]] = EMPTY
	maze[cell[0]][cell[1]] = AGENT
	return cell


# ghosts randomly move into any neighboring cell except the goal
def update_ghosts(ghosts, maze):
	for g in range(len(ghosts)):
		r = ghosts[g][0]
		c = ghosts[g][1]
		ghost_moved = False

		# get unblocked neighbors of ghost
		neighbors = u.get_neighbors(ghosts[g], maze, BLOCKED_GHOST)
		n = rand.choice(list(neighbors))

		# update maze with new ghost position
		if maze[n[0]][n[1]] != WALL and maze[n[0]][n[1]] != WALL_WITH_GHOST:
			maze[n[0]][n[1]] = GHOST
			ghost_moved = True
		else:
			phase_through = rand.randint(1, 100) / 100
			if phase_through <= PROB_GHOST_PHASE_THROUGH:
				maze[n[0]][n[1]] = WALL_WITH_GHOST
				ghost_moved = True
		
		# update ghosts list and maze
		if ghost_moved:
			ghosts[g] = n
			# if no more ghosts remain in cell, revert back to original
			if maze[r][c] == GHOST and (r, c) not in ghosts:
					maze[r][c] = EMPTY
			elif maze[r][c] == WALL_WITH_GHOST and (r, c) not in ghosts:
				maze[r][c] = WALL


# agent has failed if a ghost collides with it
def check_fail(agent, maze):
	return (maze[agent[0]][agent[1]] == GHOST or
			maze[agent[0]][agent[1]] == WALL_WITH_GHOST)


# returns agent and whether or not timestep was successful (agent didn't fail)
def step(cell, Maze):
	if cell != None:
		Maze.agent = move_agent(cell, Maze.agent, Maze.maze)
	if check_fail(Maze.agent, Maze.maze):
		return False
	update_ghosts(Maze.ghosts, Maze.maze)
	if check_fail(Maze.agent, Maze.maze):
		return False
	return True


# copy of step function for simulations
# so visualization overrides correct function
def sim_step(cell, Maze):
	if cell != None:
		Maze.agent = move_agent(cell, Maze.agent, Maze.maze)
	if check_fail(Maze.agent, Maze.maze):
		return False
	update_ghosts(Maze.ghosts, Maze.maze)
	if check_fail(Maze.agent, Maze.maze):
		return False
	return True
# ==============================================================================


# ================================== SOLVING ===================================
def agent1(Maze):
	# regenerate maze if no initial path exists
	# only try regenerating a certain number of times
	MAX_REGEN_COUNT = 29
	regen_count = 0
	while regen_count < MAX_REGEN_COUNT:
		if not validate_maze(Maze.agent, Maze.goal, Maze.maze):
			Maze = generate_maze(len(Maze.maze), len(Maze.ghosts))
			regen_count += 1
		else:
			break
	if (not validate_maze(Maze.agent, Maze.goal, Maze.maze) and
			regen_count == MAX_REGEN_COUNT):
		return -1
	
	path = u.A_star(Maze.agent, Maze.goal, Maze.maze, BLOCKED_AGENT)
	for cell in path:
		if not step(cell, Maze):
			return 0
	return 1


# find nearest ghosts from agent's pos
def get_nearest_ghosts(maze, agent, ghosts):
	# add all ghosts that are neighbors
	ng = list(u.get_neighbors(agent, maze, {EMPTY, GOAL, WALL}))

	# if no ghosts are within radius, find closest ghost
	if len(ng) == 0:
		ng.append(None)
		min_dist = -1
		# use Manhattan distance since we are not restricted to visible ghosts
		for g in ghosts:
			dist = abs(agent[0] - g[0]) + abs(agent[1] - g[1])
			if min_dist == -1 or dist < min_dist:
				min_dist = dist
				ng[0] = g
			if min_dist == 2: # atp ghost cannot be neighbor
				break

	return ng


# find nearest visible ghost from agent's pos
def get_nearest_visible_ghost(Maze):
	ng = None # nearest ghost

	# check neighbors first for efficiency
	neighbors = u.get_neighbors(Maze.agent, Maze.maze, BLOCKED_AGENT - {GHOST})
	for n in neighbors:
		if Maze.maze[n[0]][n[1]] == GHOST:
			return n

	png = None # path to nearest ghost
	for g in Maze.ghosts:
		pg = u.A_star(Maze.agent, g, Maze.maze, BLOCKED_AGENT - {GHOST})
		# neighbors already checked, so min dist is 2
		if pg != None and len(pg) == 2:
			return g
		if pg != None and (ng == None or len(pg) < len(png)):
			png = pg
			ng = g
	return ng


# given pos of agent and ghost, compute where to move away
def move_away(a, g, maze):
	r = a[0]
	c = a[1]
	rd = r - g[0]
	cd = c - g[1]

	# if ghost is closer in terms of rows
	if abs(rd) <= abs(cd):
		# move agent up if ghost is below
		if rd < 0:
			if r - 1 >= 0 and maze[r - 1][c] not in BLOCKED_AGENT:
				return (r - 1, c)
		# move agent down if ghost is above
		elif rd > 0:
			if r + 1 < len(maze) and maze[r + 1][c] not in BLOCKED_AGENT:
				return (r + 1, c)
		# if ghost is in same row, move agent down/up whichever is possible
		else:
			if r + 1 < len(maze) and maze[r + 1][c] not in BLOCKED_AGENT: # down
				return (r + 1, c)
			if r - 1 >= 0 and maze[r - 1][c] not in BLOCKED_AGENT: # up
				return (r - 1, c)
	
	# if ghost is closer in terms of cols or if agent cannot move up/down
	# move agent left if ghost is right
	if cd < 0:
		if c - 1 >= 0 and maze[r][c - 1] not in BLOCKED_AGENT:
			return (r, c - 1)
	# move agent right if ghost is left
	elif cd > 0:
		if c + 1 < len(maze) and maze[r][c + 1] not in BLOCKED_AGENT:
			return (r, c + 1)
	# if ghost is in same col, move agent right/left whichever is possible
	else:
		if c + 1 < len(maze) and maze[r][c + 1] not in BLOCKED_AGENT: # right
			return (r, c + 1)
		if c - 1 >= 0 and maze[r][c - 1] not in BLOCKED_AGENT: # left
			return (r, c - 1)

	# if agent cannot move left/right
	# move agent up if ghost is below
	if rd < 0:
		if r - 1 >= 0 and maze[r - 1][c] not in BLOCKED_AGENT:
			return (r - 1, c)
	# move agent down if ghost is above
	elif rd > 0:
		if r + 1 < len(maze) and maze[r + 1][c] not in BLOCKED_AGENT:
			return (r + 1, c)
	# if ghost is in same row, move agent down/up whichever is possible
	else:
		if r + 1 < len(maze) and maze[r + 1][c] not in BLOCKED_AGENT: # down
			return (r + 1, c)
		if r - 1 >= 0 and maze[r - 1][c] not in BLOCKED_AGENT: # up
			return (r - 1, c)

	return None


def agent2(Maze):
	path = u.A_star(Maze.agent, Maze.goal, Maze.maze, BLOCKED_AGENT)

	while Maze.agent != Maze.goal:
		if path != None:
			if not step(path[0], Maze):
				return 0
			path.pop(0)
		else:
			# if all paths are blocked, move away from nearest visible ghost
			ng = get_nearest_visible_ghost(Maze)
			if not step(move_away(Maze.agent, ng, Maze.maze), Maze):
				return 0

		# only replan if didn't move forward last timestep or ghost is in path
		replan = (path == None)
		if path:
			for c in path:
				if Maze.maze[c[0]][c[1]] == GHOST:
					replan = True
					break
		if replan:
			path = u.A_star(Maze.agent, Maze.goal, Maze.maze, BLOCKED_AGENT)

	return 1


# returns whether or not agent 2 survived a given path
# requires a copy of Maze to be passed
def simulate_agent2(path, Maze):
	i = 0
	move_count = 0 # move count for early termination
	max_sim_moves = 10 # number of moves for early termination

	while Maze.agent != Maze.goal:
		# terminate early if reached max move count (agent survived)
		if move_count >= max_sim_moves:
			return 1

		if path != None:
			if not sim_step(path[i], Maze):
				return 0
			i += 1
			move_count += 1
		else:
			# if all paths are blocked, move away from nearest visible ghost
			ng = get_nearest_visible_ghost(Maze)
			if not sim_step(move_away(Maze.agent, ng, Maze.maze), Maze):
				return 0
			move_count += 1

		# only replan if didn't move forward last timestep or ghost is in path
		replan = (path == None)
		if path:
			for c in path:
				if Maze.maze[c[0]][c[1]] == GHOST:
					replan = True
					break
		if replan:
			path = u.A_star(Maze.agent, Maze.goal, Maze.maze, BLOCKED_AGENT)
			i -= i # reset path counter

	return 1


def agent3(Maze):
	sim_count = 3 # number of simulations to run per move
	last_move = None

	while Maze.agent != Maze.goal:
		move_success = {} # times survived for each move
		move_dist = {} # min planned dist to goal from each move (to break ties)

		# for possible moves, simulate agent 2 sim_count times
		for n in u.get_neighbors(Maze.agent, Maze.maze, BLOCKED_AGENT):
			move_success[n] = 0
			if n != last_move: # do not consider paths that backtrack
				# if next move is goal, take it
				if n == Maze.goal:
					step(n, Maze)
					return 1
				# initial path is provided so it doesn't need to be
				# computed sim_count times
				path = u.A_star(n, Maze.goal, Maze.maze, BLOCKED_AGENT)
				# if no path exists, continue to next move
				if path == None:
					continue
				move_dist[n] = len(path)
				for i in range(sim_count):
					# create a copy of Maze to use in simulation
					MazeCopy = copy.deepcopy(Maze)
					move_success[n] += simulate_agent2(path, MazeCopy)
 
		# get moves with max survivability
		max_success = 0
		for move, success in move_success.items():
			if success == sim_count: # max success possible is sim_count
				max_success = success
				break
			if success > max_success:
				max_success = success
		
		if max_success != 0:
			# if tie between moves, pick shortest agent 2 planned dist
			# agent 2 initial planned dist is equal to dist of shortest path
			best_move = None # best move is one with highest success rate
			min_dist = -1
			for move, success in move_success.items():
				if (success == max_success and
						(min_dist == -1 or move_dist[move] < min_dist)):
					min_dist = move_dist[move]
					best_move = move
			
			# take best move
			if not step(best_move, Maze):
				return 0
			last_move = best_move
		else:
			# if no move is survivable, move away from nearest visible ghost
			ng = get_nearest_visible_ghost(Maze)
			# if some ghost is visible, move away
			if ng != None:
				if not step(move_away(Maze.agent, ng, Maze.maze), Maze):
					return 0
	return 1


# given pos of agent and list of ghosts, compute where to move away
def move_away_mult(a, ghosts, maze):
	# if only 1 ghost, follow rules of move_away
	if len(ghosts) == 1:
		return [move_away(a, ghosts[0], maze)]

	possible_moves = list(u.get_neighbors(a, maze, BLOCKED_AGENT))
	
	# ghosts must be neighbors to agent (see get_nearest_ghost)
	if len(possible_moves) == 0: # no moves are possible
		return [None]
	if len(possible_moves) == 1: # only 1 possible move
		return possible_moves
	
	# 2 ghosts, 2 possible moves
	r = a[0]
	c = a[1]
	# if down blocked and right unblocked, go right
	if maze[r + 1][c] in BLOCKED_AGENT and maze[r][c + 1] not in BLOCKED_AGENT:
		return [(r, c + 1)]
	# if right blocked and down unblocked, go down
	if maze[r][c + 1] in BLOCKED_AGENT and maze[r + 1][c] not in BLOCKED_AGENT:
		return [(r + 1, c)]
	# otherwise, return both moves for agent to decide
	return possible_moves


# simulation used in agent 4
# returns whether or not agent survived a given path
# requires a copy of Maze to be passed
def simulate_for_agent4(path, Maze):
	i = 0
	move_count = 0 # move count for early termination
	max_sim_moves = 10 # number of moves for early termination

	while Maze.agent != Maze.goal:
		# terminate early if reached max move count (agent survived)
		if move_count >= max_sim_moves:
			return 1

		# move away if any ghosts neighbor agent
		ng = list(u.get_neighbors(Maze.agent, Maze.maze, {EMPTY, GOAL, WALL}))
		if ng: # some ghost is neighbor
			if not sim_step(move_away_mult(Maze.agent, ng, Maze.maze)[0], Maze):
				return 0
			else:
				continue

		if path != None:
			if not sim_step(path[i], Maze):
				return 0
			i += 1
			move_count += 1
		else:
			# if all paths are blocked, move away from nearest ghosts
			ng = get_nearest_ghosts(Maze.maze, Maze.agent, Maze.ghosts)
			if ng != None:
				if not sim_step(move_away_mult(Maze.agent, ng, Maze.maze)[0],
						Maze):
					return 0
				move_count += 1

		# only replan if didn't move last timestep or ghost is in path
		replan = (path == None)
		if path:
			for c in path:
				if Maze.maze[c[0]][c[1]] == GHOST:
					replan = True
					break
		if replan:
			path = u.A_star_v2(Maze.agent, Maze.goal, Maze.maze,
					BLOCKED_AGENT, {GHOST, WALL_WITH_GHOST})
			i -= i # reset path counter
	return 1


def agent4(Maze):
	sim_count = 3 # number of simulations to run per move
	last_move = None

	while Maze.agent != Maze.goal:
		move_success = {} # times survived for each move
		move_dist = {} # min planned dist to goal from each move (to break ties)
		possible_moves = []

		# move away if any ghosts neighbor agent
		ng = list(u.get_neighbors(Maze.agent, Maze.maze, {EMPTY, GOAL, WALL}))
		if ng: # some ghost is neighbor
			possible_moves = move_away_mult(Maze.agent, ng, Maze.maze)
		if len(possible_moves) == 1:
			if not step(possible_moves[0], Maze):
				return 0
			else:
				continue

		if not possible_moves:
			possible_moves = u.get_neighbors(Maze.agent, Maze.maze,
					BLOCKED_AGENT)

		# for possible moves, simulate agent sim_count times
		for n in possible_moves:
			move_success[n] = 0
			if n != last_move: # do not consider paths that backtrack
				# if next move is goal, take it
				if n == Maze.goal:
					step(n, Maze)
					return 1
				# initial path is provided so it doesn't need to be
				# computed sim_count times
				path = u.A_star_v2(n, Maze.goal, Maze.maze,
						BLOCKED_AGENT, {GHOST, WALL_WITH_GHOST})
				# if no path exists, continue to next move
				if path == None:
					continue
				move_dist[n] = len(path)
				for i in range(sim_count):
					# create a copy of Maze to use in simulation
					MazeCopy = copy.deepcopy(Maze)
					move_success[n] += simulate_for_agent4(path, MazeCopy)
 
		# get moves with max survivability
		max_success = 0
		for move, success in move_success.items():
			if success == sim_count: # max success possible is sim_count
				max_success = success
				break
			if success > max_success:
				max_success = success
		
		if max_success != 0:
			# if tie between moves, pick shortest agent 2 planned dist
			# agent 2 initial planned dist is equal to dist of shortest path
			best_move = None # best move is one with highest success rate
			min_dist = -1
			for move, success in move_success.items():
				if (success == max_success and
						(min_dist == -1 or move_dist[move] < min_dist)):
					min_dist = move_dist[move]
					best_move = move
			
			# take best move
			if not step(best_move, Maze):
				return 0
			last_move = best_move
		else:
			# if no move is survivable, move away from nearest ghosts
			ng = get_nearest_ghosts(Maze.maze, Maze.agent, Maze.ghosts)
			if ng != None:
				if not step(move_away_mult(Maze.agent, ng, Maze.maze)[0], Maze):
					return 0
	return 1


# predicts locations of all ghosts based on last seen ghosts
def predict_ghosts(maze, ghosts, last_ghosts):
	pg = []

	for i in range(len(ghosts)):
		if maze[ghosts[i][0]][ghosts[i][1]] == GHOST: # visible
			pg.append(ghosts[i])
		# if there are no last ghosts (1st run), no prediction can be made
		elif last_ghosts: # hidden
			g = last_ghosts[i]
			# if ghost was visible, then it entered a wall
			if maze[g[0]][g[1]] == EMPTY or maze[g[0]][g[1]] == GHOST:
				# use last position to guess which wall it entered
				walls = u.get_neighbors(g,
						maze, {EMPTY, AGENT, GOAL, GHOST})
				pg.append(rand.choice(list(walls)))
			# if ghost was hidden, either stayed in place or moved to diff wall
			else:
				walls = u.get_neighbors(g, maze, {EMPTY, AGENT, GOAL, GHOST})
				phase_through = rand.randint(1, 100) / 100
				# guess moved
				if walls and phase_through <= PROB_GHOST_PHASE_THROUGH:
					pg.append(rand.choice(list(walls)))
				else: # guess stayed in place
					pg.append(g)

	return pg


def agent5(Maze):
	sim_count = 3 # number of simulations to run per move
	last_move = None

	lg = [] # ghosts in last timestep
	pg = [] # predicted positions of ghosts

	while Maze.agent != Maze.goal:
		move_success = {} # times survived for each move
		move_dist = {} # min planned dist to goal from each move (to break ties)
		possible_moves = []

		# move away if any ghosts neighbor agent
		ng = list(u.get_neighbors(Maze.agent, Maze.maze,
				{EMPTY, GOAL, WALL, WALL_WITH_GHOST}))
		if ng: # some ghost is neighbor
			possible_moves = move_away_mult(Maze.agent, ng, Maze.maze)
		if len(possible_moves) == 1:
			if not step(possible_moves[0], Maze):
				return 0
			else:
				# update last ghosts
				lg = copy.deepcopy(Maze.ghosts)
				continue

		if not possible_moves:
			possible_moves = u.get_neighbors(Maze.agent, Maze.maze,
					BLOCKED_AGENT)

		# predict ghosts
		pg = predict_ghosts(Maze.maze, Maze.ghosts, lg)
		# update last ghosts
		lg = copy.deepcopy(Maze.ghosts)

		# for possible moves, simulate agent sim_count times
		for n in possible_moves:
			move_success[n] = 0
			if n != last_move: # do not consider paths that backtrack
				# if next move is goal, take it
				if n == Maze.goal:
					step(n, Maze)
					return 1
				# initial path is provided so it doesn't need to be
				# computed sim_count times
				path = u.A_star_v2(n, Maze.goal, Maze.maze,
						BLOCKED_AGENT, {GHOST})
				# if no path exists, continue to next move
				if path == None:
					continue
				move_dist[n] = len(path)
				for i in range(sim_count):
					# create a copy of Maze to use in simulation
					# use predicted ghosts
					MazeCopy = copy.deepcopy(Maze)
					MazeCopy.ghosts = copy.deepcopy(pg)
					move_success[n] += simulate_for_agent4(path, MazeCopy)
 
		# get moves with max survivability
		max_success = 0
		for move, success in move_success.items():
			if success == sim_count: # max success possible is sim_count
				max_success = success
				break
			if success > max_success:
				max_success = success
		
		if max_success != 0:
			# if tie between moves, pick shortest agent 2 planned dist
			# agent 2 initial planned dist is equal to dist of shortest path
			best_move = None # best move is one with highest success rate
			min_dist = -1
			for move, success in move_success.items():
				if (success == max_success and
						(min_dist == -1 or move_dist[move] < min_dist)):
					min_dist = move_dist[move]
					best_move = move
			
			# take best move
			if not step(best_move, Maze):
				return 0
			last_move = best_move
		else:
			# if no move is survivable, move away from nearest ghosts
			# use predicted ghosts
			ng = get_nearest_ghosts(Maze.maze, Maze.agent, pg)
			if pg and ng:
				if not step(move_away_mult(Maze.agent, ng, Maze.maze)[0], Maze):
					return 0
	return 1


# returns whether or not agent was successful
def solve(strategy, Maze):
	if strategy == 1:
		return agent1(Maze)
	if strategy == 2:
		return agent2(Maze)
	if strategy == 3:
		return agent3(Maze)
	if strategy == 4:
		return agent4(Maze)
	if strategy == 5:
		return agent5(Maze)


def run_maze(size, num_ghosts, strategy):
	Maze = generate_maze(size, num_ghosts)
	return solve(strategy, Maze)
# ==============================================================================


if __name__ == "__main__":
	# test code
	print(run_maze(10, 7, 5))
