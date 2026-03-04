import csv
import sys

sys.setrecursionlimit(20000000)

from Opdrachten_Alexander import LinkedList

kentekens = LinkedList()

with open(sys.argv[1]) as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        kentekens.addFirst(row[0])