u"""
createRotatedTrainingData.py

by Michael Wood (Last update 09/2018 by Yara Mohajerani)
Create the rotated training data

To run, specify the glaciers, coast, and image dimeions; e.g:
"
python createRotatedTrainingData.py --glaciers=Sverdrup,Jakobshavn 
    --coasts=NW,CW --dimensions=2,3
"

Update History
    09/2018 Clean up and add to pipline (Yara Mohajerani)
    04/2018 Written (Michael Wood)
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from osgeo import ogr
from osgeo import osr
from osgeo import gdal
from PIL import Image
from matplotlib import path
from scipy.interpolate import griddata


#-- set rotations for different glaciers
glacierRotationDefintion={('Sverdrup',32621):-144,
                          ('Sverdrup',32620):-144,
                          ('Helheim',32624):103,
                          ('Kangerlussuaq',32625):120,
                          ('Jakobshavn',32622):-85,
                          ('Jakobshavn', 32623): -85}

#code to read in the sample area
def readSampleArea(sampleAreaFolder,Glacier):
    sampleArea=np.genfromtxt(sampleAreaFolder+'/'+Glacier+' Sample Area - ESPG 3413.csv',delimiter=',')
    return sampleArea

#change Julian day to month day
def JDtoMonthDay(year,JD):
    if JD<=31:
        month=1
        day=JD
    else:
        if year%4==0:
            if JD<=60 and JD>31:
                month=2
                day=JD-31
            if JD<=91 and JD>60:
                month=3
                day=JD-60
            if JD<=121 and JD>91:
                month=4
                day=JD-91
            if JD<=152 and JD>121:
                month=5
                day=JD-121
            if JD<=182 and JD>152:
                month=6
                day=JD-152
            if JD<=213 and JD>182:
                month=7
                day=JD-182
            if JD<=244 and JD>213:
                month=8
                day=JD-213
            if JD<=274 and JD>244:
                month=9
                day=JD-244
            if JD<=305 and JD>274:
                month=10
                day=JD-274
            if JD<=335 and JD>305:
                month=11
                day=JD-305
            if JD<=366 and JD>335:
                month=12
                day=JD-335
        else:
            if JD <= 59 and JD>31:
                month = 2
                day = JD - 31
            if JD <= 90 and JD>59:
                month = 3
                day = JD - 59
            if JD <= 120 and JD>90:
                month = 4
                day = JD - 90
            if JD <= 151 and JD>120:
                month = 5
                day = JD - 120
            if JD <= 181 and JD>151:
                month = 6
                day = JD - 151
            if JD <= 212 and JD>181:
                month = 7
                day = JD - 181
            if JD <= 243 and JD>212:
                month = 8
                day = JD - 212
            if JD <= 273 and JD>243:
                month = 9
                day = JD - 243
            if JD <= 304 and JD>273:
                month = 10
                day = JD - 273
            if JD <= 334 and JD>304:
                month = 11
                day = JD - 304
            if JD <= 365 and JD>334:
                month = 12
                day = JD - 334
    return (month,day)

#code to get the list of fronts and their images
def frontAndImageLists(Glacier,frontsFolder,satelliteImageryFolder):
    ignoreSLCoff=True
    ignoreBadScenes=True
    fronts=os.listdir(frontsFolder)
    frontsList=[]
    imageList=[]
    for front in fronts:
        frontParts=front.split('-')
        if len(frontParts)>1:
            month,day=JDtoMonthDay(int(frontParts[0]),int(frontParts[1]))
            ymdString=str(frontParts[0])+'{:02}'.format(month)+'{:02}'.format(day)
            for imageFile in os.listdir(satelliteImageryFolder):
                if imageFile[-4:]=='.TIF':
                    imageFileParts=imageFile.split('_')
                    if len(imageFileParts)>2:
                        strTest=imageFileParts[3]
                        if strTest==ymdString:
                            if ignoreSLCoff:
                                if int(frontParts[0])>2002 and int(frontParts[0])<2014 and imageFile[:4]=='LE07':
                                    useImage=False
                                else:
                                    useImage=True
                            else:
                                useImage=True
                            if useImage:
                                frontsList.append(front)
                                imageList.append(imageFile)
    return(frontsList,imageList)

#code to reproject a point
def reprojectPoint(x,y,inputCRS,outputCRS):
    source = osr.SpatialReference()
    source.ImportFromEPSG(inputCRS)
    target = osr.SpatialReference()
    target.ImportFromEPSG(outputCRS)
    transform = osr.CoordinateTransformation(source, target)
    WKTinput = "POINT (" + str(x) + " " + str(y) + ")"
    newPoint = ogr.CreateGeometryFromWkt(WKTinput)
    newPoint.Transform(transform)
    outPoint = newPoint.ExportToWkt()
    outPoint = outPoint.split(' ')
    outX=float(outPoint[1][1:])
    outY=float(outPoint[2][:-1])
    return outX,outY

#code to reproject a polygon
def reprojectPolygon(polygon,inputCRS,outputCRS):
    source = osr.SpatialReference()
    source.ImportFromEPSG(inputCRS)
    target = osr.SpatialReference()
    target.ImportFromEPSG(outputCRS)
    transform = osr.CoordinateTransformation(source, target)
    outputPolygon=[]
    for point in polygon:
        x = point[0]
        y = point[1]
        WKTinput = "POINT (" + str(x) + " " + str(y) + ")"
        newPoint = ogr.CreateGeometryFromWkt(WKTinput)
        newPoint.Transform(transform)
        outPoint = newPoint.ExportToWkt()
        outPoint = outPoint.split(' ')
        outputPolygon.append([float(outPoint[1][1:]),float(outPoint[2][:-1])])
    return np.array(outputPolygon)

#code to get map extent from fronts
def getMapExtentFromFronts(frontList,frontFolder,sateliteImageryFolder,satelliteImageFile):
    minX = 1e22
    maxX = -1e22
    minY = 1e22
    maxY = -1e22
    for frontFile in frontList:
        front=np.genfromtxt(frontFolder+'/'+frontFile,delimiter=',')
        if np.min(front[:,0])<minX:
            minX=np.min(front[:,0])
        if np.max(front[:, 0]) > maxX:
            maxX=np.max(front[:,0])
        if np.min(front[:, 1]) < minY:
            minY=np.min(front[:,1])
        if np.max(front[:, 1]) > maxY:
            maxY=np.max(front[:,1])
    ds = gdal.Open(sateliteImageryFolder + '/' + satelliteImageFile)
    prj = ds.GetProjection()
    srs = osr.SpatialReference(wkt=prj)
    prjName = srs.GetAttrValue('projcs')
    espg = int('326' + prjName.split()[-1][:2])
    minX, minY = reprojectPoint(minX - 100, minY - 100, 3413, espg)
    maxX, maxY = reprojectPoint(maxX + 100, maxY + 100, 3413, espg)

    ds=None
    buffer=1000
    return(espg,[minX-buffer,maxX+buffer,minY-buffer,maxY+buffer])

#add room to extents for the rotation
def addToExtentsForRotation(extent):
    minX=np.copy(extent[0])
    maxX=np.copy(extent[1])
    minY=np.copy(extent[2])
    maxY=np.copy(extent[3])

    xRange = maxX - minX
    yRange = maxY - minY
    if yRange > xRange:
        addX = (yRange - xRange) / 2
        maxX += addX
        minX = minX - addX
    if xRange > yRange:
        addY = (xRange - yRange) / 2
        maxY += addY
        minY = minY - addY
    rangeB = maxY - minY
    maxX+=1*rangeB
    minX-=1*rangeB
    maxY+=1*rangeB
    minY-=1*rangeB
    return ([minX, maxX, minY, maxY])

#get rotation angle from the fronts
def rotationAngleFromFronts(frontList,frontFolder,sateliteImageryFolder,satelliteImageFile):
    # ds = gdal.Open(sateliteImageryFolder + '/' + satelliteImageFile)
    # prj = ds.GetProjection()
    # srs = osr.SpatialReference(wkt=prj)
    # prjName = srs.GetAttrValue('projcs')
    # espg = int('326' + prjName.split()[-1][:2])
    # ds=None
    # for frontFile in frontList:
    #     front = np.genfromtxt(frontFolder + '/' + frontFile, delimiter=',')
    #     front = reprojectPolygon(front,3413,espg)
    rotationAngle=float(90)
    return(rotationAngle)

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

def newExtentFromRotation(extent,rotationAngle):
    ulX=extent[0]
    ulY=extent[3]
    llX=extent[0]
    llY=extent[2]
    lrX=extent[1]
    lrY=extent[2]
    urX=extent[1]
    urY=extent[3]

    #this is for the -90 test angle
    ul=[llX,llY]
    ll=[lrX,lrY]
    lr=[urX,urY]
    ur=[ulX,urY]
    return([ul,ll,lr,ur])

#code to plot base satellite imagery
def baseImageryArray(sateliteImageryFolder,satelliteImageFile,extent):
    minX=extent[0]
    maxX=extent[1]
    minY=extent[2]
    maxY=extent[3]

    ds = gdal.Open(sateliteImageryFolder+'/'+satelliteImageFile)
    sceneArray = np.array(ds.GetRasterBand(1).ReadAsArray())
    rows, cols = sceneArray.shape
    transform=ds.GetGeoTransform()
    ds = None

    minSceneX=transform[0]
    maxSceneX=transform[0]+transform[1]*cols
    maxSceneY=transform[3]
    minSceneY=transform[3]+transform[5]*rows

    xStep = (maxSceneX - minSceneX) / cols
    yStep = (maxSceneY - minSceneY) / rows
    minXindex = int((minX - minSceneX) / xStep)
    maxXindex = int((maxX - minSceneX) / xStep)
    maxYindex = int((maxSceneY - minY) / yStep)
    minYindex = int((maxSceneY - maxY) / yStep)

    sceneArrayX=np.arange(transform[0],transform[0]+transform[1]*cols,transform[1])
    sceneArrayY=np.arange(transform[3],transform[3]+transform[5]*rows,transform[5])

    sceneArrayX=sceneArrayX[minXindex:maxXindex]
    sceneArrayY=sceneArrayY[minYindex:maxYindex]
    sceneArray = sceneArray[minYindex:maxYindex, minXindex:maxXindex]
    return(sceneArrayX,sceneArrayY,sceneArray)

#rotate and cut the base imagery
def rotateAndCut(sceneArrayX,sceneArrayY,sceneArray,rotationAngle,extent,front,sampleArea,imageDimensions):
    #center the front
    front[:,0]-=np.mean(sceneArrayX)
    front[:,1]-=np.mean(sceneArrayY)

    # center the sample Area
    sampleArea[:, 0] -= np.mean(sceneArrayX)
    sampleArea[:, 1] -= np.mean(sceneArrayY)

    #change extents to new coords and see what they will be in the new rotation
    minX=extent[0]-np.mean(sceneArrayX)
    maxX=extent[1]-np.mean(sceneArrayX)
    minY = extent[2]-np.mean(sceneArrayY)
    maxY = extent[3]-np.mean(sceneArrayY)
    corners=np.array([[minX,minY],[maxX,minY],[maxX,maxY],[minX,maxY]])
    newMinX=1e22
    newMaxX=-1e22
    newMinY = 1e22
    newMaxY = -1e22
    for corner in corners:
        rotatedX=np.cos(np.deg2rad(rotationAngle))*corner[0] - np.sin(np.deg2rad(rotationAngle))*corner[1]
        rotatedY=np.sin(np.deg2rad(rotationAngle)) * corner[0] + np.cos(np.deg2rad(rotationAngle)) * corner[1]
        if rotatedX<newMinX:
            newMinX=rotatedX
        if rotatedX>newMaxX:
            newMaxX=rotatedX
        if rotatedY<newMinY:
            newMinY=rotatedY
        if rotatedY>newMaxY:
            newMaxY=rotatedY

    #make the shape match the image dimensions
    yxRatio=imageDimensions[1]/float(imageDimensions[0])
    xRange=newMaxX-newMinX
    yRange=newMaxY-newMinY
    if yRange<yxRatio*xRange:
        addition=yRange-yxRatio*xRange
        newMaxY-=addition/2
        newMinY+=addition/2
        print('    Y Range Extended')

    # center the array on (0,0)
    sceneXshift=np.mean(sceneArrayX)
    sceneYshift=np.mean(sceneArrayY)
    sceneArrayX -= sceneXshift
    sceneArrayY -= sceneYshift
    newSceneArrayX=np.arange(newMinX,newMaxX,15)
    newSceneArrayY=np.arange(newMinY,newMaxY,15)
    NewSceneArrayX,NewSceneArrayY=np.meshgrid(newSceneArrayX,newSceneArrayY)

    rotatedScenePoints=[]
    rotatedSceneValues=[]
    for i in range(len(sceneArrayX)):
        for j in range(len(sceneArrayY)):
            rotatedX = np.cos(np.deg2rad(-rotationAngle)) * sceneArrayX[i] - np.sin(np.deg2rad(-rotationAngle)) * sceneArrayX[j]
            rotatedY = np.sin(np.deg2rad(-rotationAngle)) * sceneArrayX[i] + np.cos(np.deg2rad(-rotationAngle)) * sceneArrayX[j]
            if rotatedX>=newMinX and rotatedX<=newMaxX and rotatedY>=newMinY and rotatedY<=newMaxY:
                rotatedScenePoints.append([rotatedX,rotatedY])
                rotatedSceneValues.append([sceneArray[j,i]])
    rotatedScenePoints=np.array(rotatedScenePoints)
    rotatedSceneValues=np.array(rotatedSceneValues)
    newSceneArray=griddata(np.array(rotatedScenePoints),np.array(rotatedSceneValues),(NewSceneArrayX,NewSceneArrayY),method='cubic')

    #rotate the front
    rotatedFront=np.zeros_like(front)
    for i in range(np.shape(front)[0]):
            rotatedFront[i,0] = np.cos(np.deg2rad(rotationAngle)) * front[i,0] - np.sin(np.deg2rad(rotationAngle)) * front[i,1]
            rotatedFront[i,1] = np.sin(np.deg2rad(rotationAngle)) * front[i,0] + np.cos(np.deg2rad(rotationAngle)) * front[i,1]

    # rotate the sample area
    rotatedSampleArea = np.zeros_like(sampleArea)
    for i in range(np.shape(sampleArea)[0]):
        rotatedSampleArea[i, 0] = np.cos(np.deg2rad(rotationAngle)) * sampleArea[i, 0] - np.sin(np.deg2rad(rotationAngle)) * sampleArea[i, 1]
        rotatedSampleArea[i, 1] = np.sin(np.deg2rad(rotationAngle)) * sampleArea[i, 0] + np.cos(np.deg2rad(rotationAngle)) * sampleArea[i, 1]

    newSceneArray=newSceneArray[:,:,0]
    newExtent=(newMinX,newMaxX,newMinY,newMaxY)

    #regain a list of preserved corners (uL,uR,lR,lL)
    preservedCorners=np.array([[np.min(rotatedScenePoints[:, 0]), np.min(rotatedScenePoints[:, 1])],
                      [np.max(rotatedScenePoints[:, 0]), np.min(rotatedScenePoints[:, 1])],
                      [np.max(rotatedScenePoints[:, 0]), np.max(rotatedScenePoints[:, 1])],
                      [np.min(rotatedScenePoints[:, 0]), np.max(rotatedScenePoints[:, 1])]])
    for pC in range(len(preservedCorners)):
        x=np.copy(preservedCorners[pC,0])
        y=np.copy(preservedCorners[pC,1])
        preservedCorners[pC, 0] = np.cos(np.deg2rad(rotationAngle)) * x - np.sin(np.deg2rad(rotationAngle)) * y
        preservedCorners[pC, 1] = np.sin(np.deg2rad(rotationAngle)) * x + np.cos(np.deg2rad(rotationAngle)) * y
    preservedCorners[:,0]+=sceneXshift
    preservedCorners[:,1]+=sceneYshift

    return(newSceneArray,newExtent,rotatedFront,rotatedSampleArea,preservedCorners)

#save image anf mask
def saveRotatedImage(outputFolder,outputFile,newSceneArray,newExtent):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(2, 3)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(newSceneArray, cmap=plt.get_cmap('gray'), interpolation='bicubic', extent=newExtent)
    fig.savefig(outputFolder + '/' + outputFile)
    plt.close(fig)

def saveRotatedFront(outputFolder,outputFile,imFolder,imFile,rotatedFront,extent,pix=3):
    img=Image.open(imFolder+'/'+imFile)
    w, h = img.size
    new_im=Image.new('RGB',(w,h),'white')
    pixelArr=np.array(new_im)
    for f in range(np.shape(rotatedFront)[0]):
        pX = (w / (extent[1] - extent[0])) * (rotatedFront[f, 0] - extent[0])
        pY = (-h / (extent[3] - extent[2])) * (rotatedFront[f, 1] - extent[2]) + h
        if pX>1 and pX<w-1 and pY>1 and pY<h-1:
            pixelArr[pY,pX,:]=0
            if pix == 3:
                pixelArr[pY-1,pX-1,:] = 0
                pixelArr[pY,pX-1,:] = 0
                pixelArr[pY+1,pX-1,:] = 0
                pixelArr[pY-1,pX,:] = 0
                pixelArr[pY+1,pX,:] = 0
                pixelArr[pY-1,pX+1,:] = 0
                pixelArr[pY,pX+1,:] = 0
                pixelArr[pY-1,pX+1,:] = 0
    frontIm=Image.fromarray(pixelArr)
    new_im.paste(frontIm)
    new_im.save(outputFolder+'/'+outputFile)

def plotBaseImagery(outputFolder,outputFile,sceneArray,extent,bigExtent,front):
    brightnessFactor=2.2
    sceneArray = (sceneArray/float(65535))**(1/brightnessFactor)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 8)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(sceneArray, cmap=plt.get_cmap('gray'), interpolation='bicubic',extent=bigExtent)
    corners = np.array([[extent[0], extent[2]], [extent[1], extent[2]], [extent[1], extent[3]], [extent[0], extent[3]]])
    plt.plot(front[:,0],front[:,1],'y-')
    plt.plot(corners[:,0],corners[:,1],'y-')
    # plt.show()
    fig.savefig(outputFolder + '/' + outputFile)
    plt.close(fig)


#-- main function to get user input and make training data
def main():
    #-- Read the system arguments listed after the program
    long_options = ['glaciers=','coasts=','dimensions=']
    optlist,arglist = getopt.getopt(sys.argv[1:],'=G:C:D:',long_options)

    Glaciers=['Sverdrup','Jakobshavn']
    coasts=['NW','CW']
    imageDimensions=(2,3)
    for opt, arg in optlist:
        if opt in ('-G','--glaciers'):
            Glaciers = arg.split(',')
        elif opt in ('-C','--coasts'):
            coasts = arg.split(',')
        elif opt in ('-D','--dimensions'):
            dims = np.array(arg.split(','),dtype=int)
            imageDimensions = (dims[0],dims[1])

    #-- directory setup
    #- current directory
    current_dir = os.path.dirname(os.path.realpath(__file__))
    main_dir = os.path.join(current_dir,'..','FrontLearning_data')

    for g in range(len(Glaciers)):
        Glacier=Glaciers[g]
        coast=coasts[g]
        #############################################################################
        #Steps in the process
        #############################################################################
        #Step 1: Read in the sample area for the glacier
        sampleAreaFolder=os.path.join(main_dir,'Regions/'+coast+'/Glaciers/'+Glacier+'/Retreat/Sample Areas')
        sampleArea=readSampleArea(sampleAreaFolder,Glacier)

        #Step 2: Get list of fronts and corresponding satellite images
        frontsFolder=os.path.join(main_dir,'Regions/'+coast+'/Glaciers/'+Glacier+'/Retreat/Front Locations/3413')
        satelliteImageryFolder=os.path.join(main_dir,'Greenland/Satellite Imagery/'+Glacier+'/Bands')
        frontList,imageList=frontAndImageLists(Glacier,frontsFolder,satelliteImageryFolder)

        outputFolder=os.path.join(main_dir,Glacier)

        #Step 3: Get a list of rotated images and the rotated front pixels
        metaDataOutput='Front File,Image File,Projection,Rotation Angle,ulX,ulY,urX,urY,lrX,lrY,llX,llY\n'
        rotatedImagesList=[]
        rotatedFrontPixelsList=[]
        outputFiles=[]
        N=len(imageList)
        for s in range(N):
            print('Preparing Image '+str(s+1)+' of '+str(N))
            satelliteImageFile=imageList[s]
            frontFile=frontList[s]
            print(satelliteImageFile)

            #get the rotation angle
            # rotationAngle = rotationAngleFromFronts(frontList, frontsFolder, satelliteImageryFolder, satelliteImageFile)
            espg, extent = getMapExtentFromFronts(frontList, frontsFolder, satelliteImageryFolder, satelliteImageFile)
            rotationAngle = glacierRotationDefintion[(Glacier,espg)]
            bigExtent=addToExtentsForRotation(extent)

            #get the front
            front = np.genfromtxt(frontsFolder + '/' + frontFile, delimiter=',')
            p=path.Path(sampleArea)
            indices=p.contains_points(front)
            front=front[indices,:]
            front=seriesToNPoints(front,1000)
            front=reprojectPolygon(front,3413,espg)

            #reproject the sample area
            sampleAreaCopy = seriesToNPoints(sampleArea,1000)
            sampleAreaCopy = reprojectPolygon(sampleAreaCopy, 3413, espg)

            outputFile=satelliteImageFile[:-4]+'_Subset.png'
            sceneArrayX, sceneArrayY, sceneArray=baseImageryArray(satelliteImageryFolder,satelliteImageFile,bigExtent)
            # plotBaseImagery(outputFolder+'/Test', outputFile, sceneArray, extent, bigExtent,front)
            newSceneArray, newExtent, rotatedFront, rotatedSampleArea, preservedCorners=rotateAndCut(sceneArrayX, sceneArrayY, sceneArray, rotationAngle,extent,front,sampleAreaCopy,imageDimensions)

            #save the results
            saveRotatedImage(outputFolder + '/Small Images', outputFile, newSceneArray, newExtent)
            outputFile = satelliteImageFile[:-4] + '_Front.png'
            outputSAfile = satelliteImageFile[:-4] + '_SampleArea.png'
            imFile = satelliteImageFile[:-4] + '_Subset.png'
            saveRotatedFront(outputFolder+'/Front Mask',outputFile,outputFolder+'/Small Images',imFile,rotatedFront,newExtent)
            saveRotatedFront(outputFolder + '/Sample Areas', outputSAfile, outputFolder + '/Small Images', imFile, rotatedSampleArea,newExtent)

            metaDataOutput+=frontFile+','+satelliteImageFile+','+str(espg)+','+str(rotationAngle)+','
            metaDataOutput+=str(preservedCorners[0,0])+','+str(preservedCorners[0,1])+','+str(preservedCorners[1,0])+','+str(preservedCorners[1,1])
            metaDataOutput+=str(preservedCorners[2,0])+','+str(preservedCorners[2,1])+','+str(preservedCorners[3,0])+','+str(preservedCorners[3,1])+'\n'




        #Step 4: Write out the meta data
        f=open(outputFolder+'/'+Glacier+' Image Data.csv','w')
        f.write(metaDataOutput[:-1])
        f.close()

    

if __name__ == '__main__':
    main()
