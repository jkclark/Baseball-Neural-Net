# for some reason, sometimes player names have "," at the end
# so this just runs through the array and removes all commas

import pickle

with open('obj/' + "all_players_list" + '.pkl', 'rb') as f:
        array = pickle.load(f)

length = len(array)
for x in xrange(length):
	if array[x][-1] == ",":
		array[x] = array[x][:-1]
		print "fixed element %d" % x

with open('obj/'+ "all_players_list" + '.pkl', 'wb') as f:
        pickle.dump(array, f, pickle.HIGHEST_PROTOCOL)