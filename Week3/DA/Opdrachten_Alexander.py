class Node:
    def __init__(self, new_data, next_node):
        self.data = new_data
        self.next = next_node
    

class LinkedList:
    def __init__(self, values= None):
        if (type(values) == type([1])):
            self.head = Node(values[0], None)        
            prev_node = self.head
            next_node = None
            for i in values[1:]:
                prev_node.next = Node(i, next_node)
                prev_node = prev_node.next
        else:
            self.head = None

    

    def toString(self):

        current_node = self.head
        returnable_string = ""
        while current_node is not None:
            returnable_string += str(current_node.data) + " "
            current_node = current_node.next
        return returnable_string
    
    
    def addFirst(self, value):
        if(self.head == None):
            self.head = Node(value, None)
        else:
            current_head = self.head
            self.head = Node(value, current_head)

    def remove(self, value):
        if self.head.data == value:
            self.head = self.head.next
            return
        
        previous_node = self.head
        current_node = self.head.next
        while current_node is not None:
            if current_node.data == value:
                previous_node.next = current_node.next
                return
            else: 
                previous_node = current_node
                current_node = current_node.next

    def find_smallest(self):
        smallest = self.head.data
        current_node = self.head
        while current_node is not None:
            if current_node.data < smallest:
                smallest = current_node.data
            current_node = current_node.next
            
        return smallest
    
    def find_biggest(self):
        biggest = self.head.data
        current_node = self.head
        while current_node is not None:
            if current_node.data > biggest:
                biggest = current_node.data
            current_node = current_node.next
            
        return biggest
    
    def sorting_big_first(self):
        new_list = LinkedList([self.find_smallest()])
        self.remove(self.find_smallest())
        while self.head is not None:
            smallest = self.find_smallest()
            new_list.addFirst(smallest)
            self.remove(smallest)
        return new_list

    def sortSimple(self):
        new_list = LinkedList([self.find_biggest()])
        self.remove(self.find_biggest())
        while self.head is not None:
            biggest = self.find_biggest()
            new_list.addFirst(biggest)
            self.remove(biggest)
        return new_list

    

Lister = LinkedList()

List = LinkedList([3,6,32,6,8,9,5,34,6,8,68,24])
# print(List.toString())
List.addFirst(7)
List.remove(3)
print(List.toString())
List.remove(32)
print(List.toString())
# List_big = List.sorting_big_first()
# print(List_big.toString())
# List_small = List_big.sortSimple()
# print(List_small.toString())
