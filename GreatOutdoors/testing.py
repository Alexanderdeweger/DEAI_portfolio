import csv
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from Week4.DA.OpdrachtenAlexander import LinkedList

list = LinkedList()
i = 0
with open(Path(__file__).with_name("INVENTORY_LEVELS-data.csv")) as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if i== 0:
            i = 1
            continue
        test = {'INVENTORY_YEAR': float(row[0])+((int(row[1])-1)/12), 'PRODUCT_NUMBER': int(row[2]), 'INVENTORY_COUNT': int(row[3])}
        list.addLast(test)

output_path = Path(__file__).with_name('tester.csv')

with open(output_path, 'w', newline='') as csvfile:
    fieldnames = ['INVENTORY_YEAR', 'PRODUCT_NUMBER', 'INVENTORY_COUNT']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
    writer.writeheader()
    writer.writerows(list.returnDicts())
