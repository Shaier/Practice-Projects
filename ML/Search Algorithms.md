# Search Algorithms in Python
```python
#Linear Search

def LinearSearch(lys, element):
    for i in range (len(lys)):
        if lys[i] == element:
            return i
    return -1
print(LinearSearch([1,2,3,4,5,2,1], 2)) #Index of the first occurence

#################################################################################################
#Binary Search
#Requires that the array will be sorted
'''
"If mid is the element we are looking for (best case), we return its index.
If not, we identify which side of mid val is more likely to be on based on whether val is smaller or greater than mid,
and discard the other side of the array. We then recursively or iteratively follow the same steps, choosing a new value 
for mid, comparing it with val and discarding half of the possible matches in each iteration of the algorithm."
'''
def BinarySearch(lys, val):
    first = 0
    last = len(lys)-1
    index = -1
    while (first <= last) and (index == -1):
        mid = (first+last)//2
        if lys[mid] == val:
            index = mid
        else:
            if val<lys[mid]:
                last = mid -1
            else:
                first = mid +1
    return index
BinarySearch([10,20,30,40,50], 20)

#A drawback of this algorithm is that if there are similar terms it does not return the index of the first element,
#but rather the index of the element closest to the middle:
print(BinarySearch([4,4,4,4,4], 4)) #Prints 1
#linear search would print 0

#################################################################################################
#Jump Search
#Needs a sorted array
"instead of searching through the array elements incrementally, we search in jumps"
"lys[0], lys[0+jump], lys[0+2jump], lys[0+3jump]..."
"With each jump, we store the previous value we looked at and its index. When we find a set of values where" \
"lys[i]lys[i+jump], we perform a linear search with lys[i] as the left-most element and lys[i+jump] as the right-most element in our search set:"
import math

def JumpSearch (lys, val):
    length = len(lys)
    jump = int(math.sqrt(length))
    left, right = 0, 0
    while left < length and lys[left] <= val:
        right = min(length - 1, left + jump)
        if lys[left] <= val and lys[right] >= val:
            break
        left += jump;
    if left >= length or lys[left] > val:
        return -1
    right = min(length - 1, right)
    i = left
    while i <= right and lys[i] <= val:
        if lys[i] == val:
            return i
        i += 1
    return -1
print(JumpSearch([1,2,3,4,5,6,7,8,9], 5)) #Prints 4
#################################################################################################

#Fibonacci Search
"uses Fibonacci numbers to calculate the block size or search range in each step."
#The algorithm uses 3 fib. numbers at a time (call them fibM, fibM_minus_1, and fibM_minus_2)
#fibM = fibM_minus_1 + fibM_minus_2
#Start with 0,1,1 for the 3 numbers.
#Then choose the smallest number of the Fib sequence that is greater than or equal to the number of elements in our search array lys

def FibonacciSearch(lys, val):
    fibM_minus_2 = 0
    fibM_minus_1 = 1
    fibM = fibM_minus_1 + fibM_minus_2
    while (fibM < len(lys)):
        fibM_minus_2 = fibM_minus_1
        fibM_minus_1 = fibM
        fibM = fibM_minus_1 + fibM_minus_2
    index = -1;
    while (fibM > 1):
        i = min(index + fibM_minus_2, (len(lys)-1))
        if (lys[i] < val):
            fibM = fibM_minus_1
            fibM_minus_1 = fibM_minus_2
            fibM_minus_2 = fibM - fibM_minus_1
            index = i
        elif (lys[i] > val):
            fibM = fibM_minus_2
            fibM_minus_1 = fibM_minus_1 - fibM_minus_2
            fibM_minus_2 = fibM - fibM_minus_1
        else :
            return i
    if(fibM_minus_1 and index < (len(lys)-1) and lys[index+1] == val):
        return index+1;
    return -1
print(FibonacciSearch([1,2,3,4,5,6,7,8,9,10,11], 6))

#################################################################################################

#Exponential Search

def ExponentialSearch(lys, val):
    if lys[0] == val:
        return 0
    index = 1
    while index < len(lys) and lys[index] <= val:
        index = index * 2
    return BinarySearch(lys[:min(index, len(lys))], val)

print(ExponentialSearch([1,2,3,4,5,6,7,8],3))
#################################################################################################

#Interpolation Search
"Interpolation search calculates the probable position of the element we are searching for using the formula:"
#index = low + [(val-lys[low])*(high-low) / (lys[high]-lys[low])]
"Interpolation search works best on uniformly distributed, sorted arrays. Whereas binary search starts in the middle " \
"and always divides into two, interpolation search calculates the likely position of the element and checks the index, " \
"making it more likely to find the element in a smaller number of iterations.
"
'''
lys - our input array
val - the element we are searching for
index - the probable index of the search element. This is computed to be a higher value when val is closer in value to the element at the end of the array (lys[high]), and lower when val is closer in value to the element at the start of the array (lys[low])
low - the starting index of the array
high - the last index of the array
'''
def InterpolationSearch(lys, val):
    low = 0
    high = (len(lys) - 1)
    while low <= high and val >= lys[low] and val <= lys[high]:
        index = low + int(((float(high - low) / ( lys[high] - lys[low])) * ( val - lys[low])))
        if lys[index] == val:
            return index
        if lys[index] < val:
            low = index + 1;
        else:
            high = index - 1;
    return -1

print(InterpolationSearch([1,2,3,4,5,6,7,8], 6))
#################################################################################################
```
# Resources
https://stackabuse.com/search-algorithms-in-python/










