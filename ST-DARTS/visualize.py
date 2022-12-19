import sys
import genotypes
from graphviz import Digraph


def plot(genotype, filename):
  g = Digraph(
      format='pdf',
      edge_attr=dict(fontsize='35', fontname="times bold"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='35', height='0.5', width='0.5', penwidth='2', fontname="times bold"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  g.node("x_{t}", fillcolor='darkseagreen2')#'white'
  g.node("h_{t-1}", fillcolor='darkseagreen2')#'white'
  g.node("0", fillcolor='lightblue')#'white'
  g.edge("x_{t}", "0", fillcolor="gray")
  g.edge("h_{t-1}", "0", fillcolor="gray")
  steps = len(genotype)

  for i in range(1, steps + 1):
    g.node(str(i), fillcolor='lightblue')#'white'

  for i, (op, j) in enumerate(genotype):
    g.edge(str(j), str(i + 1), label=op, fillcolor="gray")#, style="dashed")

  g.node("h_{t}", fillcolor='palegoldenrod')#'white'
  for i in range(1, steps + 1):
    g.edge(str(i), "h_{t}", fillcolor="gray")#, style="dashed")

  g.render(filename, view=True)


if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
    sys.exit(1)

  genotype_name = sys.argv[1]
  try:
    genotype = eval('genotypes.{}'.format(genotype_name))
  except AttributeError:
    print("{} is not specified in genotypes.py".format(genotype_name)) 
    sys.exit(1)

  plot(genotype.recurrent, "recurrent")

