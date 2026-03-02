import csv
import sys

sys.setrecursionlimit(20000000)

from LinkedList import LinkedListEmpty

kentekens = LinkedListEmpty()

with open(sys.argv[1]) as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        kentekens = kentekens.addFirst(row[0])