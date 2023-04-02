import sys
import docplex.cp
from docplex.cp.model import *
from collections import namedtuple

##############################################################################
# Model configuration
##############################################################################

# The number of coils to produce
TUPLE_ORDER = namedtuple("TUPLE_ORDER", ["index", "weight", "color"])
orders = [ TUPLE_ORDER(1, 22, 5),
           TUPLE_ORDER(2,  9, 3),
           TUPLE_ORDER(3,  9, 4),
           TUPLE_ORDER(4,  8, 5),
           TUPLE_ORDER(5,  8, 7),
           TUPLE_ORDER(6,  6, 3),
           TUPLE_ORDER(7,  5, 6),
           TUPLE_ORDER(8,  3, 0),
           TUPLE_ORDER(9,  3, 2),
           TUPLE_ORDER(10, 3, 3),
           TUPLE_ORDER(11, 2, 1),
           TUPLE_ORDER(12, 2, 5)
           ]

NB_SLABS = 12
MAX_COLOR_PER_SLAB = 2

# The total number of slabs available.  In theory this can be unlimited,
# but we impose a reasonable upper bound in order to produce a practical
# optimization model.

# The different slab weights available.
slab_weights = [ 0, 11, 13, 16, 17, 19, 20, 23, 24, 25,
                 26, 27, 28, 29, 30, 33, 34, 40, 43, 45 ]



nb_orders = len(orders)
slabs = range(NB_SLABS)
allcolors = set([ o.color for o in orders ])

# CPO needs lists for pack constraint
order_weights = [ o.weight for o in orders ]

# The heaviest slab
max_slab_weight = max(slab_weights)

# The amount of loss incurred for different amounts of slab use
# The loss will depend on how much less steel is used than the slab
# just large enough to produce the coils.
loss = [ min([sw-use for sw in slab_weights if sw >= use]) for use in range(max_slab_weight+1)]


mdl = CpoModel(name="trucks")

# Which slab is used to produce each coil
production_slab = integer_var_dict(orders, 0, NB_SLABS-1, "production_slab")

# How much of each slab is used
slab_use = integer_var_list(NB_SLABS, 0, max_slab_weight, "slab_use")

# The total loss is
total_loss = sum([element(slab_use[s], loss) for s in slabs])

# The orders are allocated to the slabs with capacity
mdl.add(pack(slab_use, [production_slab[o] for o in orders], order_weights))

# At most MAX_COLOR_PER_SLAB colors per slab
for s in slabs:
   su = 0
   for c in allcolors:
       lo = False
       for o in orders:
           if o.color==c:
               lo = (production_slab[o] == s) | lo
       su += lo
   mdl.add(su <= MAX_COLOR_PER_SLAB)

# Add minimization objective
mdl.add(minimize(total_loss))

print("\nSolving model....")
# Search strategy
mdl.set_search_phases([search_phase([production_slab[o] for o in orders])])

msol = mdl.solve(FailLimit=100000, TimeLimit=10)
