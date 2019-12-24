

def manhattan(vertex1, vertex2):
    """
    :param vertex1: tuple, coordinate
    :param vertex2: tuple, coordinate
    :return: manhattan distance between the two coordinates
    """
    manh_d = sum([abs(vertex1[i] - vertex2[i]) for i in range(len(vertex1))])
    return manh_d



class Node:
    def __init__(self, coord, value):
        self.value = value
        self.coord = coord
        self.neighbours = []
        self.terminal = False
        self.net = False
        self.neighbour_num = 0
        self.set_value(value)
        self.out_nets = set()


    def set_value(self, value):
        if self.is_occupied():
            raise ValueError("node already occupied")
        self.value = value
        if value[0] == 'g':
            self.terminal = True
        elif value[0] == 'n':
            self.net = True
        return True



    def get_value(self):
        """
        :return: string "0", "gX", or "nY"
        """
        return self.value

    def get_neighbours(self):
        return self.neighbours

    def get_neighbour_order_to(self, end_vertex):
        nnl = self.neighbours[:]
        nnl.sort(key=lambda x: manhattan(x.get_coord(), end_vertex))
        return nnl

    def get_coord(self):
        return self.coord

    def is_occupied(self):
        """
        :return: True if node is in use by a net or terminal, else False
        """
        return self.terminal or self.net

    def is_terminal(self):
        return self.terminal

    def is_net(self):
        return self.net

    def get_adjecent_occupied(self):
        """
        :return: number of adjecent nodes that are occupied, either by a terminal
         or by a net
        """
        count = 0
        for adj in self.neighbours:
            if adj.is_occupied():
                count += 1
        return count

    def has_room(self):
        """
        note: for netlist creation

        :return: True if node has room for an additional outgoing net,
        """
        count = self.get_adjecent_occupied() + len(self.out_nets)
        if count < self.neighbour_num:
            return True
        else:
            return False


    def add_net(self, net):
        """
        :param net: adds net to the set of nets allowed at the terminal
        :return:
        """
        if self.is_terminal():
            self.out_nets.add(net)
        else:
            raise ValueError("cannot add net to non-terminal node")
            print("a net should not be added here")

    def connect(self, neighbours):
        """
        :param neighbours: tuple (neighbouring) of Node objects
        :saves: this the list in the node object
        """
        self.neighbours = list(neighbours)
        self.neighbour_num = len(neighbours)

    def disconnect(self):
        self.neighbours = []
        self.neighbour_num = 0


    def remove_out_nets(self):
        """
        sets the outgoing nets (of a terminal-node) to the empty set
        """
        self.out_nets = set()


    def remove_net(self):
        if self.is_terminal():
            raise ValueError("not a net, node ", self.coord, "contains:", self.value )
        else:
            self.value = "0"
            self.net = False


    def is_blocked_in(self):
        """pre-routing, skip routing if terminal is blocked in"""
        for neighbour in self.neighbours:
            if not (neighbour.is_terminal() or neighbour.is_net()):
                return False
        return True
