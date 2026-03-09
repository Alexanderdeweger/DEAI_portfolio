import csv
import sys
from pathlib import Path

sys.setrecursionlimit(20000000)

from Opdrachten_Alexander import LinkedList

kentekens = LinkedList()

with open(Path(__file__).with_name("kentekens1000.txt")) as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        kentekens = kentekens.addFirst(row[0])
kentekens = kentekens.sortSimple()
#print(kentekens.toString())
print(kentekens.uniq())

# opdracht 10 lukt niet want het gaat ver boven de recursielimit want de sort methode is O(n^2) en dat wordt veels te groot op een bestand groter dan 1GB