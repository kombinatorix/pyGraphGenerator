import numpy as np

# Input:
# n = number of vertices

def generateInitialMatrix(numberOfVertices):
    basicMatrix = np.zeros((numberOfVertices, numberOfVertices))
    return basicMatrix

def realizeBinaryValue(f, p):
    bin = 0
    if (f< p):
        bin = 1
    return bin

def realizeBinaryArray(pArray, p):
    binArray = np.zeros_like(pArray)
    for index in range(np.alen(pArray)):
            binArray[index]= realizeBinaryValue(pArray[index],p)
    return binArray

def randomTriangle(p,numberOfVertices):
    n = numberOfVertices - 1
    numberOfEntries = n*(n+1)/2
    pTriangle = np.random.random_sample((numberOfEntries,))
    binTriangle = realizeBinaryArray(pTriangle, p)
    return binTriangle

def randomDiagonal(p,numberOfVertices):
    pDiagonal = np.random.random_sample((numberOfVertices,))
    binDiagonal = realizeBinaryArray(pDiagonal, p)
    return binDiagonal

def fillUpperTriangle(matrix,binTriangle):
    # TODO: Here has to be a check of dimensions
    #print (binTriangle)
    counter=0
    newMatrix = matrix
    for i in range(np.alen(matrix)-1):
        for j in range(i+1):
            #print ("Row: "+ str(j)+ ", Column: "+ str (i+1)+", Index: "+ str(counter))
            newMatrix[j,i+1] = binTriangle[counter]
            counter = counter + 1
    return newMatrix

def fillLowerTriangle(matrix,binTriangle):
    counter=0
    newMatrix = matrix
    for i in range(np.alen(matrix)-1):
        for j in range(i+1):
            #print ("Row: "+ str(j)+ ", Column: "+ str (i+1)+", Index: "+ str(counter))
            newMatrix[i+1,j] = binTriangle[counter]
            counter = counter + 1
    return newMatrix
def fillLowerAndUpperTriangle(matrix,binTriangle):
    counter=0
    newMatrix = matrix
    for i in range(np.alen(matrix)-1):
        for j in range(i+1):
            #print ("Row: "+ str(j)+ ", Column: "+ str (i+1)+", Index: "+ str(counter))
            newMatrix[j,i+1] = binTriangle[counter]
            newMatrix[i+1,j] = binTriangle[counter]
            counter = counter + 1
    return newMatrix
def fillDiagonal(matrix,binDiagonal):
    newMatrix = matrix
    for i in range(np.alen(binDiagonal)):
        newMatrix[i,i] = binDiagonal[i]
    return newMatrix

def generateSimpleUndirectedGraph(numberOfVertices,p):
    adjacencyMatrix = generateInitialMatrix(numberOfVertices)
    binTriangle = randomTriangle(p,numberOfVertices)
    adjacencyMatrix = fillLowerAndUpperTriangle(adjacencyMatrix, binTriangle)
    return adjacencyMatrix

def generateUndirectedGraph(numberOfVertices,p,q):
    adjacencyMatrix = generateSimpleUndirectedGraph(numberOfVertices,p)
    binDiagonal = randomDiagonal(q,numberOfVertices)
    adjacencyMatrix = fillDiagonal(adjacencyMatrix, binDiagonal)
    return adjacencyMatrix

#def generateSimpleDirectedGraph(numberOfVertices,p,q):
#    adjacencyMatrix = generateInitialMatrix(numberOfVertices)
#    binUpperTriangle = randomTriangle(p,numberOfVertices)
#    binLowerTriangle = randomTriangle(q,numberOfVertices)
#    adjacencyMatrix = fillUpperTriangle(adjacencyMatrix, binUpperTriangle)
#    adjacencyMatrix = fillLowerTriangle(adjacencyMatrix, binLowerTriangle)
#    return adjacencyMatrix
#
#def generateDirectedGraph(numberOfVertices,p,q,r):
#    adjacencyMatrix = generateSimpleDirectedGraph(numberOfVertices,p,q)
#    binDiagonal = randomDiagonal(r,numberOfVertices)
#    adjacencyMatrix = fillDiagonal(adjacencyMatrix, binDiagonal)
#    return adjacencyMatrix

def getArrayOfDegrees(adjacencyMatrix):
    numberOfVertices = np.alen(adjacencyMatrix)
    arrayOfDegrees = np.zeros((numberOfVertices,))
    for i in range(numberOfVertices):
        arrayOfDegrees[i] = np.sum(adjacencyMatrix[i])
    return arrayOfDegrees

def findMinimumDegree(arrayOfDegrees):
    return np.min(arrayOfDegrees)

def findMaximumDegree(arrayOfDegrees):
    return np.max(arrayOfDegrees)

def findMeanDegree(arrayOfDegrees):
    return np.mean(arrayOfDegrees)

def findMedianDegree(arrayOfDegrees):
    return np.median(arrayOfDegrees)

def numberOfEdges(arrayOfDegrees):
    return np.sum(arrayOfDegrees)/2

def degreeSequence(arrayOfDegrees):
    return np.sort(arrayOfDegrees)[::-1]
def graphSpectrum(adjacencyMatrix):
    return np.linalg.eigvals(adjacencyMatrix) # TODO: Look np.eig up. It could be easy to determine the number of connected components
    #return np.linalg.eig(adjacencyMatrix) # TODO: Look np.eig up. It could be easy to determine the number of connected components
    
def createDegreeMatrix(arrayOfDegrees):
    n = np.alen(arrayOfDegrees)
    degreeMatrix = np.zeros((n,n))
    for i in range(n):
        degreeMatrix[i,i]=arrayOfDegrees[i]
    return degreeMatrix

def createLaplacian(adjacencyMatrix):
    arrayOfDegrees = getArrayOfDegrees(adjacencyMatrix)
    degreeMatrix = createDegreeMatrix(arrayOfDegrees)
    Laplacian = degreeMatrix-adjacencyMatrix
    return Laplacian

def numberOfCOnnectedComponents(adjacencyMatrix):# Problem with floating point calculation
    Laplacian = createLaplacian(adjacencyMatrix)
    arrayOfEigenvalues = np.linalg.eigvals(Laplacian)
    print(arrayOfEigenvalues)
    numberOfCOnnectedComponents= np.alen(adjacencyMatrix) - np.count_nonzero(arrayOfEigenvalues)
    return numberOfCOnnectedComponents


SUG = generateSimpleUndirectedGraph(10,0.5)
print SUG
AoD = getArrayOfDegrees(SUG)
print ("Minimum degree: "+str(findMinimumDegree(AoD)))
print ("Maximum degree: "+str(findMaximumDegree(AoD)))
print ("Mean degree: "+str(findMeanDegree(AoD)))
print ("Median degree: "+str(findMedianDegree(AoD)))
print ("Number of edges: "+str(numberOfEdges(AoD)))
print ("Degree Sequence: "+str(degreeSequence(AoD)))
print ("Graph spectrum: "+str(graphSpectrum(SUG)))
print ("Number of connected Components: "+ str(numberOfCOnnectedComponents(SUG)))
#print(graphSpectrum(SUG))
#print(graphSpectrum())




