# script for visualizing maze

import sys
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import maze as m


# delay is required for visualization to update
VIS_DELAY = 0.001 # delay for viewing visualization

# NOTE: visualization may stall with a large number of ghosts


def show_maze(maze):
	global im

	# colormap
	colors = {
		m.EMPTY: 'silver',
		m.AGENT: 'dodgerblue',
		m.GOAL: 'gold',
		m.GHOST: 'red',
		m.WALL: 'black',
		m.WALL_WITH_GHOST: 'darkred'
	}
	cmap = mp.colors.ListedColormap([colors[x] for x in colors.keys()])
	labels = [None] * 6
	labels[m.EMPTY] = "empty"
	labels[m.AGENT] = "agent"
	labels[m.GOAL] = "goal"
	labels[m.GHOST] = "ghost"
	labels[m.WALL] = "wall"
	labels[m.WALL_WITH_GHOST] = "ghost in wall"

	# normalizer
	b = np.sort(range(len(labels))) + 0.5
	b = np.insert(b, 0, np.min(b) - 1.0)
	norm = mp.colors.BoundaryNorm(b, len(labels), clip=True)

	# plot
	fig, ax = plt.subplots()
	im = ax.imshow(maze, cmap=cmap, norm=norm)
	cb = fig.colorbar(im, ticks=range(len(labels)))
	cb.ax.set_yticklabels(labels)

	# grid
	ax.set_xticks(np.arange(-.5, len(maze), 1))
	ax.set_yticks(np.arange(-.5, len(maze), 1))
	ax.tick_params(axis='x', colors='w')
	ax.tick_params(axis='y', colors='w')
	ax.axes.xaxis.set_ticklabels([])
	ax.axes.yaxis.set_ticklabels([])
	ax.grid(color='w', linestyle='-', linewidth=1.4)

	plt.ion()
	plt.pause(VIS_DELAY)


# override maze.py step function allowing for visualization
# returns agent and whether or not timestep was successful (agent didn't fail)
def step(cell, Maze):
	if cell != None:
		Maze.agent = m.move_agent(cell, Maze.agent, Maze.maze)

	if m.check_fail(Maze.agent, Maze.maze):
		im.set_data(Maze.maze) # show maze
		plt.pause(VIS_DELAY)
		return False
	
	m.update_ghosts(Maze.ghosts, Maze.maze)

	if m.check_fail(Maze.agent, Maze.maze):
		im.set_data(Maze.maze) # show maze
		plt.pause(VIS_DELAY)
		return False
	
	im.set_data(Maze.maze) # show maze
	plt.pause(VIS_DELAY)
	return True


# function for ease of running maze visualization
def run_maze(size, num_ghosts, strategy):
	m.step = step

	Maze = m.generate_maze(size, num_ghosts)
	show_maze(Maze.maze)
	print(m.solve(strategy, Maze))

	plt.ioff()
	plt.show()


if __name__ == "__main__":
	if len(sys.argv) != 4:
		print("Usage: python3 visualization.py size num_ghosts strategy")
		sys.exit()
	size = int(sys.argv[1])
	num_ghosts = int(sys.argv[2])
	strategy = int(sys.argv[3])

	run_maze(size, num_ghosts, strategy)
