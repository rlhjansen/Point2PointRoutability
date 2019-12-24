
from random import randint, shuffle
import queue as Q
import functools
import operator


from .node import Node


class Mesh:
    def __init__(self, size_params, solver="A_star", height=8, terminals=None):

        # initialize the mesh basics
        self.params = size_params + [height]  # parameters of the mesh
        self.platform_params = size_params
        self.terminal_coords = {}  # key:val gX:tuple(terminal_loc)
        self.coord_terminal = {}
        self.terminal_net = {} # key:val gX:set(nA, nB, nC...)
        self.net_terminal = {}  # key:val nX:tuple(g1, g2)
        self.nets = {}  # Live nets
        self.connections = {}  # key:val coord_tuple:tuple(neighbour_tuples)
        self.wire_locs = set()
        self.meshdict = {n:Node(n, '0') for n in params_inp(self.params)}

        if terminals:
            self.place_premade_terminals(terminals_from_list_of_lists(terminals))



    def write_mesh(self, fname):
        """
        writes current terminal configuration to an out-file
        :param fname: filename to save to
        """
        mesh = self.to_base()
        with open(fname, 'w+') as fout:
            for row in mesh:
                fout.write(",".join(row) + "\n")
        fout.close()

    def to_base(self):
        """
        :return: a list of lists of the "ground floor" of the mesh
        """
        x = self.platform_params[0]
        y = self.platform_params[1]
        list = [str(self.meshdict[i+(0,)].get_value()) for i in params_inp(self.platform_params)]
        newbase = [[list[j * x + i] for i in range(x)] for j in
                   range(y)]
        return newbase

    # connects nodes in the mesh to it's neighbouring nodes
    def connect(self):
        """
        adds the connections each node has into the connection dictionary
        """
        for key in self.meshdict.keys():
            neighbour_nodes = tuple([self.meshdict.get(pn) for pn in neighbours(key) if self.meshdict.get(pn, False)])
            # neigbour_coords = tuple([pn for pn in neighbours(key) if self.meshdict.get(pn, False)]) // testing with coords
            self.meshdict[key].connect(neighbour_nodes)

    def disconnect(self):
        for key in self.meshdict.keys():
            neighbour_nodes = tuple([self.meshdict.get(pn) for pn in neighbours(key) if self.meshdict.get(pn, False)])
            # neigbour_coords = tuple([pn for pn in neighbours(key) if self.meshdict.get(pn, False)]) // testing with coords
            self.meshdict[key].disconnect()

    def rand_loc(self):
        """
        :return: random empty location on the "ground floor" of the mesh
        """
        x_pos = randint(1, self.params[0]-2)
        y_pos = randint(1, self.params[1]-2)
        z_pos = 0
        terminal_pos = tuple([x_pos, y_pos, z_pos])
        already_occupied = self.meshdict.get(terminal_pos).get_value()[0] != '0'
        while already_occupied:
            x_pos = randint(1, self.params[0]-2)
            y_pos = randint(1, self.params[1]-2)
            terminal_pos = tuple([x_pos, y_pos, z_pos])
            already_occupied = self.meshdict.get(terminal_pos).get_value()[0] != '0'

        return terminal_pos


    def generate_terminals(self, num):
        """
        places num terminals on the mesh
        """
        self.wipe_terminals()
        for i in range(num):
            rescoords = self.rand_loc()
            self.add_terminal(rescoords, "g"+str(i))


    def wipe_terminals(self):
        """
        remove all terminals from mesh
        """
        self.meshdict = {n:Node(n, '0') for n in params_inp(self.params)}
        self.connect()

        self.terminal_coords = {}  # key:val gX:tuple(terminal_loc)
        self.coord_terminal = {}
        self.terminal_net = {}


    # adds a terminal to the mesh
    def add_terminal(self, coords, terminal_string):
        """
        places the terminal inside the mesh ditionary
        places the terminalstring inside the terminalcoord dictionary
        """
        self.meshdict[coords].set_value(terminal_string)
        self.terminal_coords[terminal_string] = coords
        self.coord_terminal[coords] = terminal_string
        self.terminal_net[terminal_string] = set()


    def place_premade_terminals(self, terminal_pairs):
        terminalcoords, terminals = terminal_pairs
        for n, val in enumerate(terminalcoords):
            self.add_terminal(val[::-1] + (0,), terminals[n])

    def get_terminal_coords(self):
        kv = [[k , v] for k, v in self.terminal_coords.items()]
        k = [kv[i][0] for i in range(len(kv))]
        v = [kv[i][1] for i in range(len(kv))]
        return k, v


    def add_net(self, terminal1, terminal2, n_str):
        self.net_terminal[n_str] = (terminal1, terminal2)
        self.meshdict[self.terminal_coords[terminal1]].add_net(n_str)
        self.meshdict[self.terminal_coords[terminal2]].add_net(n_str)
        self.terminal_net[terminal1].add(n_str)
        self.terminal_net[terminal2].add(n_str)
        self.nets[n_str] = (terminal1, terminal2)



    def generate_nets(self, num):
        AG = list(self.terminal_coords.keys())
        GN = len(AG)-1
        for i in range(num):
            g1, g2, net = AG[randint(0,GN)], AG[randint(0,GN)], 'n'+str(i)
            g1nets = self.terminal_net.get(g1, set())
            g2nets = self.terminal_net.get(g2, set())
            common = (g1nets & g2nets)
            roomleft1 = self.meshdict.get(self.terminal_coords[g1]).has_room()
            roomleft2 = self.meshdict.get(self.terminal_coords[g2]).has_room()
            no_room_left = not (roomleft1 and roomleft2)
            while (common or no_room_left) or (g1==g2):
                g1, g2 = AG[randint(0, GN)], AG[randint(0, GN)]
                if g1 == g2:
                    continue
                g1nets = self.terminal_net.get(g1)
                g2nets = self.terminal_net.get(g2)
                common = g1nets & g2nets
                roomleft1 = self.meshdict.get(self.terminal_coords[g1]).has_room()
                roomleft2 = self.meshdict.get(self.terminal_coords[g2]).has_room()
                no_room_left = not (roomleft1 and roomleft2)
            self.add_net(g1, g2, net)

    def get_random_net_order(self):
        key_list = list(self.net_terminal.keys())
        shuffle(key_list)
        return key_list[:]


    def write_nets(self, abspath):
        with open(abspath, 'w+') as out:
            for netk in self.nets.keys():
                g1, g2 = self.nets.get(netk)
                out.write(','.join([netk,g1,g2])+'\n')

    def read_nets(self, abspath):
        nets = []
        with open(abspath, 'r') as inf:
            for line in inf:
                nets.append(line[:-1].split(','))
        for line in nets:
            net, g1, g2 = line
            self.add_net(g1, g2, net)


    ###########################
    #####   Reset Block   #####
    ###########################
    def wipe_nets(self):
        """
        remove the netlist from class
        """
        for key in self.coord_terminal.keys():
            self.meshdict[key].remove_out_nets()
        for key in self.terminal_net.keys():
            self.terminal_net[key] = set()
        self.net_terminal = {}
        self.reset_nets()
        self.nets = {}


    def reset_nets(self):
        """
        retains netlist connections but resets their placement
        """
        for spot in self.wire_locs:
            self.meshdict[spot].remove_net()
        self.wire_locs = set()


    def __str__(self):
        complete = []
        pars = self.params
        for z in range(pars[2]):
            complete.append("### Layer" + str(z + 1) + "###")
            for y in range(pars[1]):
                vals = [self.meshdict[(x,y,z)].get_value() for x in range(pars[0])]
                transformed_vals = [transform_print(val) for val in vals]
                complete.append(" ".join(transformed_vals))
        return "\n".join(complete)


    def extract_route(self, path_dict, end_loc):
        path = ((),)
        get_loc = path_dict.get(end_loc)[0]
        while path_dict.get(get_loc)[0] != get_loc:

            path = path + (get_loc,)
            get_loc = path_dict.get(get_loc)[0]
        return path[::-1]


    def A_star_max_g(self, net):
        """ finds a path for a net with A-star algorithm, quits searching early if the end-terminal is closed off by its immediate neighbourse.

        in case of ties in nodes to expand by (heuristic+cost to node)
        the node with most steps taken yet is chosen

        :param net: terminal-pair (gX, gY)
        :return: path, length if path founde, else false, false
        """

        q = Q.PriorityQueue()
        count = 0
        end_loc = self.terminal_coords.get(self.net_terminal.get(net)[1])
        if self.meshdict.get(end_loc).is_blocked_in():
            return False, False, count
        start_loc = self.terminal_coords.get(self.net_terminal.get(net)[0])
        if self.meshdict.get(start_loc).is_blocked_in():
            return False, False, count

        path = ((start_loc),)
        manh_d = manhattan(path[-1], end_loc)
        q.put((manh_d, 0, start_loc))
        visited = dict()
        visited[start_loc] = [start_loc, 0]
        while not q.empty():
            count += 1
            k = q.get()
            _, steps, current = k
            for neighbour in self.meshdict.get(current).get_neighbours():
                n_coord = neighbour.get_coord()

                if neighbour.is_occupied():
                    if n_coord == end_loc:
                        # end condition, path found

                        visited[n_coord] = [current, steps]
                        return self.extract_route(visited, n_coord), \
                               visited.get(end_loc)[1], count
                    else:
                        continue
                if n_coord in visited:
                    if visited.get(n_coord)[1] > steps:
                        # checks if current number of steps is lower than
                        # established cost of the node

                        visited[n_coord] = [current, steps]
                        # was - 1 before, not sure why atm
                        q.put((manhattan(n_coord, end_loc) + steps + 1, steps + 1,
                          n_coord))
                else:
                    visited[n_coord] = [current, steps]
                    q.put((manhattan(n_coord, end_loc) + steps + 1, steps + 1,
                           n_coord))
        return False, False, count


    def solve_order(self, net_order, reset=False):
        tot_length = 0
        solved = 0
        nets_solved = []
        tries = 0
        for net in net_order:
            path, length, expansions = self.A_star_max_g(net)
            tries += expansions
            if path:
                self.place(net, path)
                solved += 1
                tot_length += length
                nets_solved.append(net)
        if reset:
            self.reset_nets()
        return [solved, tot_length, tries]



    def get_solution_placement(self, net_order):
        paths = []
        for net in net_order:
            path = self.A_star_max_g(net)[0]
            if path:
                paths.append(path)
            else:
                paths.append( ((),))
        self.reset_nets()
        return paths

    def place(self, net, path):
        for spot in path[:-1]:
            if self.meshdict[spot].set_value(net):
                self.wire_locs.add(spot)
            else:
                raise ValueError("invalid placement")
        return False



def transform_print(val):
    vlen = len(val)
    if val == '0':
        return '___'
    elif val[0] == 'n':
        return ' '*(3-vlen) + val
    elif val[0] == 'g':
        return ' '*(3-vlen) + val
    else:
        raise ValueError("incorrect node value")


def file_to_mesh(fpath, nets):
    """
    :param nets: either a netlist or a number of nets
    :return: a new Mesh
    """
    base = read_mesh(fpath)
    xlen = len(base[0])
    ylen = len(base)
    Newmesh = Mesh([xlen, ylen], terminals=base)
    return Newmesh


def terminals_from_list_of_lists(lol):
    """
    :return: tuple of all terminal coordinates for a mesh and the terminal numbers
    """
    terminal_coords = []
    terminals = []
    for x in range(len(lol)):
        for y in range(len(lol[0])):
            terminal = lol[x][y]
            if terminal[0] == 'g':
                terminal_coords.append((x, y))
                terminals.append(terminal)
    return terminal_coords, terminals


def read_mesh(fpath):
    """
    reads a mesh configuration fom the file at the file path
    :return: list of lists
    """
    base = []
    with open(fpath, 'r') as fin:
        for line in fin:
            base.append(line[:-1].split(','))  # [:-1] so no '\n')
    return base


def manhattan(loc1, loc2):
    """
    :param loc1: tuple, coordinate
    :param loc2: tuple, coordinate
    :return: manhattan distance between the two coordinates
    """
    manh_d = sum([abs(loc1[i] - loc2[i]) for i in range(len(loc1))])
    return manh_d


def params_inp(params):
    """ return all tuples for a mesh of certain size,

    params = (10,10) creates tuples for positions (0, 0), (1, 0), ..., (9,9)

    :return: tuple for every node in a mesh with params
    """
    base = [0]*len(params)
    count = 0
    tot = prodsum(params)
    return tuple([tuple(count_to_pos(c, params)) for c in range(tot)])


def prodsum(iterable):
    """
    :param iterable: list of numbers
    :return: returns the product of all numbers i.e [5,6,2] returns 5*6*2
    """
    return functools.reduce(operator.mul, iterable, 1)


def count_to_pos(count, params):
    """
    :param count: count is the number of the node being made
    :param params: parameters of the mesh
    :return: returns a set of new coordinates to be placed in the meshdict
    """
    base = [0]*len(params)
    for i in range(len(params)):
        base[i] = count // prodsum(params[:i]) % params[i]
    return base


def neighbours(coords):
    """
    :param - tuple: tuple of coordinates of a point in the mesh:
    :return: neighbouring nodes in the mesh
    """
    rl = []
    for i in range(len(coords)):
        temp1 = list(coords)
        temp2 = list(coords)
        temp1[i] -= 1
        temp2[i] += 1
        rl.extend((tuple(temp1), tuple(temp2)))

    return tuple(rl)
