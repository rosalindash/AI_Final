import hexgrid as hg
import numpy as np
from src.som import Som

# =============== SOM config =============== #
somX = 25
somY = 15
iterations = 10000
sample_size = 26
filename = 'countries_data.csv'

income_colors_dict = {
	"HI": "red",  # High income countries
	"MHI": "orange",  # Medium High income countries
	"LMI": "yellow",  # Medium Low income countries
	"LI": "green"  # Low income countries
}
# ========================================== #
# Helper function to convert umatrix distances into colors
# The darker a neuron is, the higher its distance from neighbors is.


def convert_color(color):
	if 0.75 <= color <= 1:
		return "black"
	if 0.5 <= color <= 0.74:
		return "gray20"
	if 0.25 <= color <= 0.49:
		return "gray40"
	else:
		return "gray60"

# Helper function to print som.get_weight_dicts()
# It shows every data in the som. 


def print_data(data):
		print("# ==================== Matrix Data ==================== #")
		for key in sorted(data.keys()):
			pos = '({},{})'.format(key[0], key[1])
			print('{:10}-->   {:3} sample(s):    {}'.format(pos, len(data[key]), data[key]))


def run():

	# Read first column of the table. It contains and abbreviation for each country
	country_codes = np.genfromtxt(filename, delimiter=',', dtype="str", usecols=(0))
	income_groups = np.genfromtxt(filename, delimiter=',', dtype="str", usecols=(1))

	# Read the data and normalize it.
	data = np.genfromtxt(filename, delimiter=',', usecols=np.arange(2, sample_size+2))
	data = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, data)

	# Initialize the Som with random weights (read som.py description) and train
	som = Som(somX, somY, sample_size, sigma=5, learning_rate=0.5, random_seed=10)
	som.random_weights_init(data)
	som.train_random(data, iterations)  # random training
	
	# print (som.distance_map())
	# print ('\n\n\n\n ============ \n\n\n\n\n\n')
	# print (som.distance_map().T)
	income_colors = [income_colors_dict[x] for x in income_groups]
	colors_dict = dict(zip(country_codes, income_colors))

	# Calculate umatrix colors by converting distance values to color string
	distance = [list(map(convert_color, d)) for d in som.distance_map()]

	# Use hexgrid to display information
	# Som.get_winner_dict() returns a dictionary of type (x,y) -> [country_code]
	# get_winner_dict will map samples to its winners and use the country codes to create a dictionary
	# The hexagonal grid will only display one name and a number representing the total number of samples in that neuron.
	print_data(som.get_winner_dict(data, country_codes))
	hexgrid = hg.SomGrid(somX, somY, som.get_winner_dict(data, country_codes), distance, colors_dict)
	hexgrid.display()

run()





