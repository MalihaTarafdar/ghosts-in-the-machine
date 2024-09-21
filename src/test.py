# script for data collection

import signal
import sys
import csv
import os
import datetime
import maze as m


SIZE = 51 # size of maze
MIN_SURVIVABILITY = 0.01 # min acceptable survivability rate
TIMEOUT = 2400 # seconds before maze times out (40 min)
TIMEOUT_LIMIT = 3 # max number of timeouts


# handler for timeout on hung mazes
def timeout_handler(signum, frame):
	raise Exception("timeout")


# g_i = initial ghost count
# g_f = final ghost count
# g_d = ghost increment
# n_max = max number of mazes per ghost count
def test(g_i, g_f, g_d, n_max, strategy):
	# set signal handler
	signal.signal(signal.SIGALRM, timeout_handler)

	s = {} # success count per number of ghosts
	h = {} # number of hung mazes per number of ghosts
	n = {} # number of mazes run per number of ghosts (including hung)
	p = {} # survivability rate per number of ghosts

	# for loop used for simplicity
	for g in range(g_i, g_f, g_d):
		print("ghost count: " + str(g))
		s[g] = 0
		h[g] = 0
		n[g] = 0
		p[g] = 0

		for i in range(n_max):
			timeout_count = 0

			Maze = m.generate_maze(SIZE, g)
			r = 0

			signal.alarm(TIMEOUT) # enable alarm
			try:
				r += m.solve(strategy, Maze)
			except Exception as e:
				# if timeout occurs, continue to next maze
				if str(e) == "timeout":
					h[g] += 1
					n[g] += 1
					timeout_count += 1
					# stop if reach max number of timeouts
					# likely that most mazes >= g will hang
					if timeout_count >= TIMEOUT_LIMIT:
						if n[g] - h[g] > 0:
							p[g] = s[g] / (n[g] - h[g])
						return s, h, n, p
					continue
			except:
				print("error occurred while solving maze")
				continue
			signal.alarm(0) # disable alarm
			
			# agent 1: if no valid maze generated with cur ghost count, stop
			# likely that higher ghost counts won't generate valid mazes either
			if r == -1:
				if n[g] - h[g] > 0:
					p[g] = s[g] / (n[g] - h[g])
				return s, h, n, p
			
			s[g] += r
			n[g] += 1
		
		# if survivability rate is less than 1%, stop
		if n[g] - h[g] > 0:
			p[g] = s[g] / (n[g] - h[g])
		if p[g] < MIN_SURVIVABILITY:
			return s, h, n, p

	return s, h, n, p


def output_to_csv(s, h, n, p, strategy):
	# fields: num ghosts, success count, num hung mazes, num mazes, success rate
	fields = ["Ghosts", "Success", "Hung", "Total", "Success Rate"]

	# file path
	dt = (str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')
			.replace('.', '-'))
	file_path = "./data/agent%s_%s.csv" % (strategy, dt)
	os.makedirs(os.path.dirname(file_path), exist_ok=True)

	# write to csv
	with open(file_path, 'w', newline='') as csv_file:
		writer = csv.writer(csv_file, delimiter=',')
		writer.writerow(fields) # header
		for g in p.keys():
			line = [g, s[g], h[g], n[g], p[g]]
			writer.writerow(line)


if __name__ == "__main__":
	# allow for CLI input to test multiple agents at once
	if len(sys.argv) != 6:
		print("Usage: python3 test.py g_i g_f g_d n_max strategy")
		sys.exit()
	g_i = int(sys.argv[1])
	g_f = int(sys.argv[2])
	g_d = int(sys.argv[3])
	n_max = int(sys.argv[4])
	strategy = int(sys.argv[5])
	
	s, h, n, p = test(g_i, g_f, g_d, n_max, strategy)
	output_to_csv(s, h, n, p, strategy)
