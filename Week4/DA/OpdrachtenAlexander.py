class Node:
    def __init__(self, new_data, next_node):
        self.data = new_data
        self.next = next_node
    

class LinkedList:
    def __init__(self, values= None):
        if (values is None):
            self.head = None
            self.last = None
        else:
            self.head = Node(values[0], None)      
            self.last = self.head  
            prev_node = self.head
            next_node = None
            for i in values[1:]:
                new_node = Node(i, next_node)
                prev_node.next = new_node
                prev_node = prev_node.next
                self.last = new_node
    

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
            self.last = self.head
        else:
            current_head = self.head
            self.head = Node(value, current_head)
        return
    
    def addLast(self, value):
        if type(value) == Node:
            if self.head == None:
                self.head = value
                self.last = value
            else:
                self.last.next = value
                self.last = value
        else:
            new_node = Node(value, None)
            if self.head == None:
                self.head = new_node
                self.last = new_node
            else:
                self.last.next = new_node
                self.last = new_node
        return


    def remove(self, value):
        if self.head.data == value:
            self.head = self.head.next
            return
        
        previous_node = self.head
        current_node = self.head.next
        while current_node is not None:
            if current_node.data == value:
                if current_node == self.last:
                    self.last = previous_node
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
    
    def uniq(self):
        count = 1
        current_node = self.head
        while current_node is not None:
            if current_node.next is None:
                return count
            if not current_node.data == current_node.next.data:
                count = count + 1
            current_node = current_node.next


    def subList(self, start, end):
        counter = 0
        currentNode = self.head
        newList = LinkedList()
        while (currentNode is not None):
            if(counter >= start and counter < end):
                newList.addLast(currentNode)

            currentNode = currentNode.next
            counter +=1

            if (counter == end):
                return newList
        return LinkedList(['list is empty cant do that'])
    
    def merge(self, list):
        assert type(list) == LinkedList
        current_node = self.head
        comparing_node = list.head
        newList = LinkedList()
        while (current_node is not None and comparing_node is not None):
            if current_node.data > comparing_node.data:
                newList.addLast(comparing_node)
                comparing_node = comparing_node.next
            else:
                newList.addLast(current_node)
                current_node = current_node.next
        if current_node is None:
            while comparing_node is not None:
                newList.addLast(comparing_node)
                comparing_node = comparing_node.next
        elif comparing_node is None:
            while current_node is not None:
                newList.addLast(current_node)
                current_node = current_node.next
        
        return newList

    def length(self):
        if self.head == None:
            return 0
        counter =0
        current_node = self.head
        while current_node is not None:
            counter+= 1
            current_node = current_node.next
        return counter

    def sortMerge(self):
        n = self.length()
        if n == 1:
            return self
        left = self.subList(0, n // 2)
        right = self.subList(n // 2, n)
        return left.sortMerge().merge(right.sortMerge())


List = LinkedList([1,2, 34, 63, 62, 6, 12,123,123,31,51,511]) 
# tijdscomplexiteit is O(n)

List2 = LinkedList([12,123,123,31,51,511])
List.addLast(12)
List.sortMerge()
print(List.sortMerge().toString())
#tijdscomplexitiet van merge is vgm O(n+m) omdat het door allebei de lijsten geheel moet gaan
#tijdscomplexitiet van sortMerge is vgm nlog(n) omdat het splitsen log(n) en je dat voor iedere waarde moet uitvoeren dus n keer
