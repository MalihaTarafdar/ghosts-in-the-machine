# module for maze utility functions

import heapq as hq


def get_neighbors(cell, maze, restricted_set):
	row = cell[0]
	col = cell[1]
	neighbors = set()

	# add neighbors that are not out of bounds
	if row != len(maze) - 1: # below
		neighbors.add((row + 1, col))
	if col != len(maze) - 1: # right
		neighbors.add((row, col + 1))
	if row != 0: # above
		neighbors.add((row - 1, col))
	if col != 0: # left
		neighbors.add((row, col - 1))

	# remove neighbors that are unblocked
	neighbors = set(filter(
		lambda x: maze[x[0]][x[1]] not in restricted_set, neighbors))

	return neighbors


def _rec_DFS(s, d, maze, restricted_set, closed_set):
	if s == d: # path from source to dest found
		return True
	closed_set.add(s)

	for n in get_neighbors(s, maze, restricted_set):
		if n not in closed_set:
			# explore down branch
			if _rec_DFS(n, d, maze, restricted_set, closed_set):
				return True

	return False


# returns whether or not path exists from s to d
def DFS(s, d, maze, restricted_set):
	closed_set = set()
	return _rec_DFS(s, d, maze, restricted_set, closed_set)


def get_path(d, prev):
	path = []
	cur = d
	while prev[cur] != None:
		p = prev[cur]
		path.append(cur)
		cur = p
	path.reverse()
	return path


def UFCS(s, d, maze, restricted_set):
	fringe = [] # priority queue with priority g(n)
	prev = {}
	closed_set = set()

	fringe.append((0, s))
	prev[s] = None
	closed_set.add(s)

	while fringe:
		g, cur = hq.heappop(fringe)
		if cur == d:
			return get_path(d, prev)
		else:
			for n in get_neighbors(cur, maze, restricted_set):
				if n not in closed_set:
					hq.heappush(fringe, (g + 1, n))
					closed_set.add(n)
					prev[n] = cur
	
	return None


def A_star(s, d, maze, restricted_set):
	fringe = [] # priority queue with priority f(n) = g(n) + h(n)
	prev = {}
	closed_set = set()

	fringe.append((0, s))
	prev[s] = None
	closed_set.add(s)

	while fringe:
		g, cur = hq.heappop(fringe)
		if cur == d:
			return get_path(d, prev)
		else:
			for n in get_neighbors(cur, maze, restricted_set):
				if n not in closed_set:
					g = g + 1
					# heuristic uses Manhattan distance
					h = abs(n[0] - d[0]) + abs(n[1] - d[1])
					hq.heappush(fringe, (g + h, n))
					closed_set.add(n)
					prev[n] = cur


# uses a different heuristic than v1
def A_star_v2(s, d, maze, restricted_set, ghost_set):
	fringe = [] # priority queue with priority f(n) = g(n) + h(n)
	prev = {}
	closed_set = set()

	fringe.append((0, s))
	prev[s] = None
	closed_set.add(s)

	while fringe:
		g, cur = hq.heappop(fringe)
		if cur == d:
			return get_path(d, prev)
		else:
			for n in get_neighbors(cur, maze, restricted_set):
				if n not in closed_set:
					g = g + 1

					# heuristic is Manhattan distance + num ghosts neighbors
					# cost of cell is higher when ghosts next to it
					n_ghosts = 0
					for c in get_neighbors(n, maze, restricted_set - ghost_set):
						if maze[c[0]][c[1]] in ghost_set:
							n_ghosts += 1
					
					h = abs(n[0] - d[0]) + abs(n[1] - d[1]) + n_ghosts
					hq.heappush(fringe, (g + h, n))
					closed_set.add(n)
					prev[n] = cur
	
	return None
