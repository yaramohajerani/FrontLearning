#!/anaconda2/bin/python2.7
u"""
sckikit_canny.py
by Michael Wood (Last Updated by Yara Mohajerani 10/2018)

find path of least resistance through an image

Update History
        10/2018 - Yara: Change input folder to be consistent with
                        other scripts
        09/2018 - Yara: Clean up and add user input
        09/2018 - Michael: written
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.graph import route_through_array
import shapefile
import os
import sys
import getopt
from osgeo import ogr
from osgeo import osr
import urllib
from pyproj import Proj,transform

#############################################################################################
#############################################################################################

#This function to make a list of the labels
def generateLabelList(indir):
    labelList=[]
    for fil in os.listdir(indir):
        if fil[-6:] == 'B8.png' or fil[-6:] == 'B2.png':
            labelList.append(fil[:-4])
    return(labelList)



#############################################################################################
# These functions are to create a list of indices used to find the line label

def obtainSceneCornersProjection(glacier,sceneID,glaciersFolder):
    f=open(glaciersFolder+'/'+glacier+'/'+glacier+' Image Data.csv')
    lines=f.read()
    f.close()
    lines=lines.split('\n')
    for line in lines:
        line=line.split(',')
        if line[1][:-4]==sceneID:
            corners=[]
            projection=int(line[2])
            for i in range(4,12):
                corners.append(float(line[i]))
    return(corners,projection)

def geoCoordsToImagePixels(coords,corners, projection, imageSize):
    coords=reprojectPolygon(coords,3413,projection)
    # fx(x,y) = ax + by + cxy + d
    A=np.array([[corners[6],corners[7],corners[6]*corners[7],1], #lower left corner,
               [corners[4],corners[5],corners[4]*corners[5],1], #lower right corner
               [corners[2], corners[3], corners[2] * corners[3],1], #upper right corner
               [corners[0], corners[1], corners[0] * corners[1],1]]) #upper left corner
    #option 1
    bx = np.array([[imageSize[0]],[0],[0],[imageSize[0]]])
    by = np.array([[0],[0],[imageSize[1]], [imageSize[1]] ])
    Cx=np.dot(np.linalg.inv(A),bx)

    Cy = np.dot(np.linalg.inv(A), by)
    imagePixels=[]
    for coord in coords:
        pixelX=Cx[0]*coord[0] + Cx[1]*coord[1] + Cx[2]*coord[0]*coord[1] + Cx[3]
        pixelY=Cy[0]*coord[0] + Cy[1]*coord[1] + Cy[2]*coord[0]*coord[1] + Cy[3]
        if pixelX>0 and pixelX<imageSize[0]-1 and pixelY>0 and pixelY<imageSize[1]-1:
            imagePixels.append([round(pixelX),round(pixelY)])
    return(np.array(imagePixels))

def reprojectPolygon(polygon,inputCRS,outputCRS):
    inProj = Proj(init='epsg:'+str(inputCRS))
    outProj = Proj(init='epsg:'+str(outputCRS))
    x1,y1 = -11705274.6374,4826473.6922
    x2,y2 = transform(inProj,outProj,x1,y1)
    outputPolygon=[]
    for point in polygon:
        x = point[0]
        y = point[1]
        x2,y2 = transform(inProj,outProj,x,y)
        outputPolygon.append([x2,y2])

    return np.array(outputPolygon)

def seriesToNPoints(series,N):
    #find the total length of the series
    totalDistance=0
    for s in range(len(series[:,0])-1):
        totalDistance+=((series[s,0]-series[s+1,0])**2+(series[s,1]-series[s+1,1])**2)**0.5
    intervalDistance=totalDistance/(N-1)

    #make the list of points
    newSeries=series[0,:]
    currentS = 0
    currentPoint1=series[currentS,:]
    currentPoint2=series[currentS+1,:]
    for p in range(N-2):
        distanceAccrued = 0
        while distanceAccrued<intervalDistance:
            currentLineDistance=((currentPoint1[0]-currentPoint2[0])**2+(currentPoint1[1]-currentPoint2[1])**2)**0.5
            if currentLineDistance<intervalDistance-distanceAccrued:
                distanceAccrued+=currentLineDistance
                currentS+=1
                currentPoint1 = series[currentS, :]
                currentPoint2 = series[currentS + 1, :]
            else:
                distance=intervalDistance-distanceAccrued
                newX=currentPoint1[0]+(distance/currentLineDistance)*(currentPoint2[0]-currentPoint1[0])
                newY = currentPoint1[1] + (distance / currentLineDistance) * (currentPoint2[1] - currentPoint1[1])
                distanceAccrued=intervalDistance+1
                newSeries=np.vstack([newSeries,np.array([newX,newY])])
                currentPoint1=np.array([newX,newY])
    newSeries = np.vstack([newSeries, series[-1,:]])
    return(newSeries)

def fjordBoundaryIndices(glaciersFolder,glacier,corners,projection,imageSize):
    boundary1file=glaciersFolder+'/'+glacier+'/Fjord Boundaries/'+glacier+' Boundary 1 V2.csv'
    boundary1=np.genfromtxt(boundary1file,delimiter=',')
    boundary2file = glaciersFolder + '/' + glacier + '/Fjord Boundaries/' + glacier + ' Boundary 2 V2.csv'
    boundary2 = np.genfromtxt(boundary2file, delimiter=',')

    boundary1=seriesToNPoints(boundary1,1000)
    boundary2 = seriesToNPoints(boundary2, 1000)

    boundary1pixels = geoCoordsToImagePixels(boundary1,corners,projection,imageSize)
    boundary2pixels = geoCoordsToImagePixels(boundary2, corners, projection,imageSize)
    return(boundary1pixels,boundary2pixels)

def plotImageWithBoundaries(image,boundary1pixels,boundary2pixels):
    imArr = np.asarray(image)
    plt.contourf(imArr)
    plt.plot(boundary1pixels[:,0],boundary1pixels[:,1],'w-')
    plt.plot(boundary2pixels[:, 0], boundary2pixels[:, 1], 'w-')
    plt.gca().set_aspect('equal')
    plt.show()

def testBoundaryIndices():
    boundarySide1indices=[]
    boundarySide2indices=[]
    for j in range(30,180,10):
        boundarySide1indices.append([40,j])
        boundarySide2indices.append([160,j])
    return(np.array(boundarySide1indices),np.array(boundarySide2indices))



#############################################################################################
# These functions are to find the most probable front based on the NN solution

def plotImageWithSolutionAndEndpoints(image,solution,startPoint,endPoint,boundary1pixels,boundary2pixels):
    imArr = np.asarray(image)
    C=plt.contourf(imArr)
    plt.colorbar(C)
    plt.plot(startPoint[0],startPoint[1],'w.',markersize=20)
    plt.plot(endPoint[0], endPoint[1], 'w.', markersize=20)
    plt.plot(boundary1pixels[:, 0], boundary1pixels[:, 1], 'w-')
    plt.plot(boundary2pixels[:, 0], boundary2pixels[:, 1], 'w-')
    plt.plot(solution[:,0],solution[:,1],'g-')
    plt.gca().set_aspect('equal')
    plt.show()

def leastCostSolution(imgArr,boundarySide1indices,boundarySide2indices,step):
    weight=1e22
    indices=[]
    for b1 in range(len(boundarySide1indices)):
        if b1 % step==0:
            startPoint = np.array(boundarySide1indices[b1],dtype=int)
            if b1 % step == 0:
                print('    '+str(b1+1)+' of '+str(len(boundarySide1indices))+' indices tested')
            for b2 in range(len(boundarySide2indices)):
                if b2 % step ==0:
                    endPoint = np.array(boundarySide2indices[b2],dtype=int)
                    testIndices, testWeight = route_through_array(imgArr, (startPoint[1], startPoint[0]),\
                        (endPoint[1], endPoint[0]), geometric=True,\
                        fully_connected=True)
                    tmpIndices = np.array(testIndices)
                    testIndices=np.hstack([np.reshape(tmpIndices[:,1],(np.shape(tmpIndices)[0],1)),np.reshape(tmpIndices[:,0],(np.shape(tmpIndices)[0],1))])

                    if testWeight<weight:
                        weight=testWeight
                        indices=testIndices
    return(indices)


def plotImageWithSolution(image,solution):
    imArr = np.asarray(image)
    plt.contourf(imArr)
    plt.plot(solution[:,0],solution[:,1],'w-')
    plt.gca().set_aspect('equal')
    plt.show()

def outputSolutionIndicesPng(imgArr,solutionIndices,outputFolder,label):
    solutionArr=255*np.ones_like(imgArr)
    for i in range(len(solutionIndices)):
        if solutionIndices[i,1]>1 and solutionIndices[i,1]<np.shape(solutionArr)[0]-1 and solutionIndices[i,1]>1 and solutionIndices[i,0]<np.shape(solutionArr)[1]-1:
            solutionArr[solutionIndices[i, 1], solutionIndices[i, 0]] = 0
            solutionArr[solutionIndices[i, 1]+1, solutionIndices[i, 0]+1] = 0
            solutionArr[solutionIndices[i, 1], solutionIndices[i, 0]+1] = 0
            solutionArr[solutionIndices[i, 1]-1, solutionIndices[i, 0]+1] = 0
            solutionArr[solutionIndices[i, 1]+1, solutionIndices[i, 0]] = 0
            solutionArr[solutionIndices[i, 1]-1, solutionIndices[i, 0]] = 0
            solutionArr[solutionIndices[i, 1]+1, solutionIndices[i, 0]-1] = 0
            solutionArr[solutionIndices[i, 1], solutionIndices[i, 0]-1] = 0
            solutionArr[solutionIndices[i, 1]-1, solutionIndices[i, 0]-1] = 0
    outIm=Image.fromarray(solutionArr)
    outIm=outIm.transpose(Image.FLIP_LEFT_RIGHT)
    # plt.imshow(solutionArr)
    # plt.show()
    outIm.save(outputFolder+'/'+label+'_Solution.png')



#############################################################################################
# These functions are to construct a shapefile from the geometric coordinates

def imagePixelsToGeoCoords(pixels, corners, projection, imageSize):

    # fx(x,y) = ax + by + cxy + d
    A = np.array([[imageSize[0], 0, imageSize[0] * 0, 1],  # lower left corner,
                  [0, 0, 0 * 0, 1],  # lower right corner
                  [0, imageSize[1], 0 * imageSize[1], 1],  # upper right corner
                  [imageSize[0], imageSize[1], imageSize[0] * imageSize[1], 1]])  # upper left corner
    # option 1
    bx = np.array([[corners[6]], [corners[4]], [corners[2]], [corners[0]]])
    by = np.array([[corners[7]], [corners[5]], [corners[3]], [corners[1]]])
    Cx = np.dot(np.linalg.inv(A), bx)
    Cy = np.dot(np.linalg.inv(A), by)
    geoCoords = []
    for pixel in pixels:
        geoX = Cx[0] * pixel[0] + Cx[1] * pixel[1] + Cx[2] * pixel[0] * pixel[1] + Cx[3]
        geoY = Cy[0] * pixel[0] + Cy[1] * pixel[1] + Cy[2] * pixel[0] * pixel[1] + Cy[3]
        geoCoords.append([round(geoX), round(geoY)])
    geoCoords = reprojectPolygon(geoCoords, projection,3413)
    return (np.array(geoCoords))

def getPrj(epsg):
    # access projection information
    wkt = urllib.urlopen("http://spatialreference.org/ref/epsg/{0}/prettywkt/".format(str(epsg)))
    remove_spaces = wkt.read().replace(" ", "")
    output = remove_spaces.replace("\n", "")
    return output

def solutionToShapefile(glacier,labels,frontIndices,shapefileOutputFolder, cornersList, projectionList, imageSizeList):
        #output the shapefile
        outputFile = glacier+' Front Profiles'
        w = shapefile.Writer()
        w.field('Glacier', 'C')
        w.field('Scene', 'C')
        for ll in range(len(labels)):
            frontSolution=imagePixelsToGeoCoords(frontIndices[ll],cornersList[ll],projectionList[ll],imageSizeList[ll])
            w.record(glacier,labels[ll])
            output = []
            for c in range(len(frontSolution)):
                output.append([frontSolution[c, 0], frontSolution[c, 1]])
            w.line(parts=[output])
        w.save(shapefileOutputFolder + '/' + outputFile)


        # create the .prj file
        prj = open(shapefileOutputFolder + '/' + outputFile + ".prj", "w")
        epsg = getPrj(3413)
        prj.write(epsg)
        prj.close()


def solutionToCSV(glacier, labels, frontIndices, csvOutputFolder, cornersList, projectionList,imageSizeList):
    for ll in range(len(labels)):
        frontSolution = imagePixelsToGeoCoords(frontIndices[ll], cornersList[ll], projectionList[ll], imageSizeList[ll])
        outputFile = glacier + ' ' + labels[ll] + ' Profile.csv'
        output = []
        for c in range(len(frontSolution)):
            output.append([frontSolution[c, 0], frontSolution[c, 1]])
        output=np.array(output)
        np.savetxt(csvOutputFolder+'/'+outputFile,output,delimiter=',')

def pixelSolutionToCSV(glacier, labels, frontIndices, pixelOutputFolder, cornersList, projectionList, imageSizeList):
    for ll in range(len(labels)):
        frontSolution = frontIndices[ll]
        outputFile = glacier + ' ' + labels[ll] + ' Pixels.csv'
        output = []
        for c in range(len(frontSolution)):
            output.append([frontSolution[c, 0], frontSolution[c, 1]])
        output = np.array(output)
        np.savetxt(pixelOutputFolder + '/' + outputFile, output, delimiter=',')




#############################################################################################
# All of the functions are run here
#-- main function to get user input and make training data
def main():
    #-- Read the system arguments listed after the program
    long_options = ['glaciers=','method=','step=','indir=']
    optlist,arglist = getopt.getopt(sys.argv[1:],'=G:M:S:I:',long_options)

    glacier= 'Helheim'
    method = 'CNN'
    step = 50
    indir = ''
    for opt, arg in optlist:
        if opt in ('-G','--glaciers'):
            glacier = arg
        elif opt in ('-M','--method'):
            method = arg
        elif opt in ('-S','--step'):
            step = np.int(arg)
        elif opt in ('-I','--indir'):
            indir = os.path.expanduser(arg)

    #-- directory setup
    #- current directory
    current_dir = os.path.dirname(os.path.realpath(__file__))
    headDirectory = os.path.join(current_dir,'..','FrontLearning_data')

    glaciersFolder=headDirectory+'/Glaciers'

    #-- if user input not given, set label folder
    if indir == '':
        indir=headDirectory+'/Results/'+glacier+' Results/'+method+'/'+method

    postProcessedOutputFolder=headDirectory+'/Results/'+glacier+' Results/'+method+'/'+method+' Post-Processed '+str(step)
    csvOutputFolder=headDirectory+'/Results/'+glacier+' Results/'+method+'/'+method+' Geo CSVs '+str(step)
    pixelOutputFolder=headDirectory+'/Results/'+glacier+' Results/'+method+'/'+method+' Pixel CSVs '+str(step)
    shapefileOutputFolder=headDirectory+'/Results/'+glacier+' Results/'+method+'/'+method+' Shapefile '+str(step)
    
    #-- make output folders
    if (not os.path.isdir(postProcessedOutputFolder)):
        os.mkdir(postProcessedOutputFolder)
    if (not os.path.isdir(csvOutputFolder)):
        os.mkdir(csvOutputFolder)
    if (not os.path.isdir(pixelOutputFolder)):
        os.mkdir(pixelOutputFolder)
    if (not os.path.isdir(shapefileOutputFolder)):
        os.mkdir(shapefileOutputFolder)

    labelList=generateLabelList(indir)

    frontIndicesList=[]
    cornersList=[]
    projectionList=[]
    imageSizeList=[]
    for label in labelList:
        print('Working on label '+label)
        if ('sobel' in method) or ('Sobel' in method):
            im = Image.open(indir + '/' + label + '.png').transpose(Image.FLIP_LEFT_RIGHT)
        else:
            im=Image.open(indir+'/'+label+'_nothreshold.png').transpose(Image.FLIP_LEFT_RIGHT)

        corners,projection=obtainSceneCornersProjection(glacier,label,glaciersFolder)
        cornersList.append(corners)
        projectionList.append(projection)
        imageSizeList.append(im.size)

        boundary1pixels,boundary2pixels=fjordBoundaryIndices(glaciersFolder,glacier,corners,projection,im.size)
        # plotImageWithBoundaries(im,boundary1pixels,boundary2pixels)

        solutionIndices = leastCostSolution(im,boundary1pixels,boundary2pixels,step)
        frontIndicesList.append(solutionIndices)

        outputSolutionIndicesPng(im,solutionIndices,postProcessedOutputFolder,label)
        # plotImageWithSolution(im,solutionIndices)

    solutionToCSV(glacier, labelList, frontIndicesList, csvOutputFolder, cornersList, projectionList,imageSizeList)
    pixelSolutionToCSV(glacier, labelList, frontIndicesList, pixelOutputFolder, cornersList, projectionList, imageSizeList)
    solutionToShapefile(glacier, labelList, frontIndicesList, shapefileOutputFolder, cornersList, projectionList, imageSizeList)

if __name__ == '__main__':
    main()