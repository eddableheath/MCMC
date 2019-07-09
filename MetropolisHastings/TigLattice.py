# Tig's helpful suggestions on creating lattice

def create_lattice_and_indices(n_dim, extent):
    # This is a simple 1D lattice, can extend to more dimensions by simply iterating
    lattice = np.arange(-extent, extent)
    lattice = np.asarray([lattice for _ in range(n_dim)]).T
    # Creating array of indices
    indices = np.arange(0, 2 * extent)
    indices = np.asarray([indices for _ in range(n_dim)]).T
    return indices, lattice


#def transform_lattice(lattice, basis):
 #   n_dim = len(lattice)

  #  for
   # for index, point in enumerate( lattice ):


def distance(lattice, norm = 2):
    # Norm is just how you want to define your distance
    # e.g norm = 1 is 'city block' distance, moves in integer amounts
    #     norm = 2 is standard distance from a point.
    n_dim = len(lattice)
    extended_shape = ( lattice.shape[0], 1, *lattice.shape[1:] )

    print("lattice shape", lattice.shape)
    diff = lattice.reshape( *extended_shape ) - lattice
    print("difference shape", diff.shape)
    distance = (diff ** norm).sum(2)
    print("distance shape", distance.shape)

    return distance

def points_in_range(lattice, distance, r_cut):
    # Create array of zeros and then assign values in the lattice.
    points = np.zeros(lattice.shape)
    range_condition = distance < r_cut
    points[range_condition] = lattice[ range_condition ]
    # The below function isn't quite right probably but you could look up the where function to find the indices
    indices = np.where( distance[ range_condition ] )
    return indices, points

# Dimension and extent of the lattice
n_dim = 3
extent = 10
# Number of standard deviations for the cutoff
n_dev = 15
# Radius for the cutoff of the Gaussian
r_cut = n_dev * sigma

indices, lattice = create_lattice_and_indices(n_dim, extent)

#d = distance(lattice)
#print(d)
#print(d.shape)

# The easiest way to get all of the numbers within a range in an array is to create an array of booleans
#points_in_range = copy.copy(lattice)

#points_in_range[ lattice >  r_cut ] = 0

#print(points_in_range)
#print(lattice.shape)