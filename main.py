import numpy as np
import matplotlib.pyplot as plt

from LegendreQuad import LegInt
from LobattoQuad import LobInt
from SimpleRules import MidInt, TrapInt, SimpInt

ff = lambda x: np.cos(x)*np.exp(-x/5)

a = -10
b = 10
ex_sol = 5/13 * (5* np.cosh(2)*np.sin(10) +np.cos(10)*np.sinh(2))

ns = [ 3, 5, 8, 10, 20, 30, 50, 80 ,100, 130, 150, 400, 700 ,1000]
Lob = []
Leg = []
Mid = []
Trap = []
Simp = []

for n in ns:

    Lob_new = np.log10( abs( LobInt(a,b,n,ff) -ex_sol ) )
    Leg_new = np.log10( abs( LegInt(a,b,n,ff) - ex_sol ) )
    Mid_new = np.log10( abs( MidInt(a,b,n,ff) - ex_sol ) )
    Trap_new = np.log10( abs( TrapInt(a,b,n,ff) - ex_sol ) )
    Simp_new = np.log10( abs( SimpInt(a,b,n,ff) - ex_sol ) )
    
    Lob.append( Lob_new )
    Leg.append( Leg_new )
    Mid.append( Mid_new )
    Trap.append(Trap_new)
    Simp.append(Simp_new)
    

plt.plot(ns, Lob, ls = '--', label = 'Lobatto')
plt.plot(ns, Leg, ls = '-.', label = 'Legendre')
plt.plot(ns, Mid, ls = '-', label = 'Midpoint')
plt.plot(ns, Trap, ls = ':', label = 'Trapezoid')
plt.plot(ns, Simp, ls = '-', label = 'Simpson')

plt.xscale('log')

plt.title('Comparison of different integration methods')
plt.xlabel('Number of points in log10 scale')
plt.ylabel('log10 of error')
plt.legend()
# plt.savefig('comparison', dpi = 300)
